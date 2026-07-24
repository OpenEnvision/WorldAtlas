"""NVIDIA GEN3C model adapter for WorldAtlas Arena.

Converts WorldArena ``camera_path`` tokens into canonical per-frame camera poses
and launches ``gen3c_runner`` for single-sample inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from worldarena.benchmark.annotations import resolve_prompt_contract
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.synthetic_camera import SUPPORTED_SYNTHETIC_CAMERA_TOKENS
from worldarena.common.checkpoints import filter_checkpoint_env_overrides, hf_local_dir
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
)
from worldarena.models.adapters.gen3c_runner import (
    _build_env as _build_gen3c_env,
    _checkpoint_layout,
    _link_repo_alias,
    _prepare_checkpoint_root,
    _require_repo_layout,
)


# Compatibility-only values accepted by GEN3C's upstream CLI.  When a
# WorldArena camera_path is present, these presets do not define the geometry;
# gen3c_runner builds the canonical per-frame path directly.
GEN3C_CAMERA_TRAJECTORIES: dict[str, tuple[str, str]] = {
    "push_in": ("zoom_in", "center_facing"),
    "pull_out": ("zoom_out", "center_facing"),
    "move_left": ("left", "no_rotation"),
    "move_right": ("right", "no_rotation"),
    "orbit_left": ("clockwise", "center_facing"),
    "orbit_right": ("counterclockwise", "center_facing"),
    "pan_left": ("left", "trajectory_aligned"),
    "pan_right": ("right", "trajectory_aligned"),
    "tilt_up": ("up", "center_facing"),
    "tilt_down": ("down", "center_facing"),
    "pedestal_up": ("up", "no_rotation"),
    "pedestal_down": ("down", "no_rotation"),
    "roll_cw": ("clockwise", "trajectory_aligned"),
    "roll_ccw": ("counterclockwise", "trajectory_aligned"),
    "fixed": ("left", "no_rotation"),
}


def _generation_for_suite(generation: dict[str, Any], suite: str) -> dict[str, Any]:
    resolved = dict(generation)
    for key, value in generation.items():
        if not key.endswith("_by_suite") or not isinstance(value, dict):
            continue
        base_key = key[: -len("_by_suite")]
        if suite in value:
            resolved[base_key] = value[suite]
    return resolved


def _camera_path_for_sample(sample: BenchmarkSample) -> list[str]:
    camera_path = [str(token).strip() for token in sample.camera_path if str(token).strip()]
    if camera_path:
        return camera_path
    contract = resolve_prompt_contract(sample.annotation_path)
    fallback = contract.get("camera_path")
    if isinstance(fallback, list):
        normalized = [str(token).strip() for token in fallback if str(token).strip()]
        if normalized:
            return normalized
    if sample.modality == "image":
        return ["fixed"]
    return []


def _generation_for_camera_token(
    generation: dict[str, Any],
    sample: BenchmarkSample,
    *,
    camera_path: list[str],
    camera_token: str,
) -> dict[str, Any]:
    resolved = _generation_for_suite(generation, sample.suite)
    use_sample_camera_path = bool(resolved.get("use_sample_camera_path", sample.suite == "image_static"))
    if not use_sample_camera_path or sample.suite != "image_static":
        return resolved

    primary_token = camera_token.strip().lower()
    normalized_path = [
        str(token).strip().lower().replace("-", "_").replace(" ", "_")
        for token in camera_path
        if str(token).strip()
    ]
    unsupported = [
        token for token in normalized_path if token not in SUPPORTED_SYNTHETIC_CAMERA_TOKENS
    ]
    if unsupported:
        raise ValueError(f"unsupported GEN3C camera token(s): {unsupported!r}")

    trajectory, camera_rotation = GEN3C_CAMERA_TRAJECTORIES.get(
        primary_token,
        (str(resolved.get("trajectory", "left")), str(resolved.get("camera_rotation", "center_facing"))),
    )
    resolved["trajectory"] = trajectory
    resolved["camera_rotation"] = camera_rotation
    if primary_token == "fixed":
        resolved["movement_distance"] = 0.0
    resolved["camera_path"] = normalized_path
    resolved["camera_path_primary"] = primary_token
    resolved["camera_path_json"] = json.dumps(normalized_path)
    resolved["camera_control_source"] = "worldarena_canonical_c2w"
    return resolved


def _generation_for_sample(generation: dict[str, Any], sample: BenchmarkSample) -> dict[str, Any]:
    camera_path = _camera_path_for_sample(sample)
    if not camera_path:
        return _generation_for_suite(generation, sample.suite)
    return _generation_for_camera_token(
        generation,
        sample,
        camera_path=camera_path,
        camera_token=camera_path[0],
    )


class Gen3CAdapter(ModelAdapter):
    """Generate trajectory-controlled videos via the GEN3C upstream repo."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        if self.config.repo_root is None:
            raise ValueError("GEN3C adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("GEN3C adapter requires checkpoint_dir")

        generation = dict(self.config.generation)
        num_gpus = int(generation.get("num_gpus", 1))
        if num_gpus != 1:
            raise ValueError(
                "GEN3C persistent batch runner currently requires generation.num_gpus=1"
            )
        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError("GEN3C batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        repo_root = self.config.repo_root.resolve()
        _require_repo_layout(repo_root)

        with TemporaryDirectory(prefix="gen3c_persistent_", dir=spec_dir) as temp_dir_raw:
            work_root = Path(temp_dir_raw)
            workspace_root = work_root / "workspace"
            workspace_root.mkdir(parents=True, exist_ok=True)
            checkpoints_root = _prepare_checkpoint_root(
                work_root,
                _checkpoint_layout(self.config.checkpoint_dir),
            )
            moge_checkpoint = hf_local_dir("Ruicheng/moge-vitl", required=True) / "model.pt"
            if not moge_checkpoint.is_file():
                raise FileNotFoundError(f"GEN3C MoGe checkpoint not found: {moge_checkpoint}")
            _link_repo_alias(workspace_root, "Ruicheng/moge-vitl", moge_checkpoint)

            batch_input_path = work_root / "batch_inputs.jsonl"
            error_root = work_root / "errors"
            error_root.mkdir(parents=True, exist_ok=True)
            generation_by_sample: dict[str, dict[str, Any]] = {}
            with batch_input_path.open("w", encoding="utf-8") as handle:
                for index, request in enumerate(requests):
                    sample_generation = _generation_for_sample(generation, request.sample)
                    generation_by_sample[request.sample.sample_id] = sample_generation
                    payload = {
                        "sample_id": request.sample.sample_id,
                        "prompt": request.prompt,
                        "visual_input": str(request.conditioning_image.expanduser().resolve()),
                        "output_path": str(request.output_path.expanduser().resolve()),
                        "error_path": str(error_root / f"{index}.txt"),
                        "camera_path": sample_generation.get("camera_path"),
                        "trajectory": sample_generation.get("trajectory"),
                        "camera_rotation": sample_generation.get("camera_rotation"),
                        "movement_distance": sample_generation.get("movement_distance"),
                        "filter_points_threshold": sample_generation.get(
                            "filter_points_threshold"
                        ),
                        "foreground_masking": sample_generation.get(
                            "foreground_masking", False
                        ),
                        "negative_prompt": sample_generation.get("negative_prompt"),
                        "num_video_frames": sample_generation.get("num_video_frames"),
                        "height": sample_generation.get("height"),
                        "width": sample_generation.get("width"),
                        "fps": sample_generation.get("fps"),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            command = [
                self.config.python_bin,
                "-m",
                "worldarena.models.adapters.gen3c_persistent_runner",
                "--checkpoint_dir",
                str(checkpoints_root),
                "--batch_input_path",
                str(batch_input_path),
            ]
            extend_command_with_options(
                command,
                payload=generation,
                value_options={
                    "num_video_frames": int,
                    "height": int,
                    "width": int,
                    "fps": int,
                    "seed": int,
                    "guidance": float,
                    "num_steps": int,
                    "trajectory": str,
                    "camera_rotation": str,
                    "movement_distance": float,
                    "filter_points_threshold": float,
                    "negative_prompt": str,
                    "prompt_upsampler_dir": str,
                },
                flag_options=(
                    "save_buffer",
                    "foreground_masking",
                    "offload_diffusion_transformer",
                    "offload_tokenizer",
                    "offload_text_encoder_model",
                    "offload_prompt_upsampler",
                    "offload_guardrail_models",
                    "disable_prompt_encoder",
                ),
            )
            env = _build_gen3c_env(repo_root, workspace_root)
            env.update(filter_checkpoint_env_overrides(self.config.env))
            pythonpaths = [str(project_root)]
            if env.get("PYTHONPATH"):
                pythonpaths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            run_command(command, cwd=workspace_root, env=env)

            results: dict[str, dict[str, Any]] = {}
            for index, request in enumerate(requests):
                error_path = error_root / f"{index}.txt"
                sample_generation = generation_by_sample[request.sample.sample_id]
                if request.output_path.is_file() and request.output_path.stat().st_size > 0:
                    results[request.sample.sample_id] = {
                        "status": "generated",
                        "prediction_path": str(request.output_path),
                        "prompt": request.prompt,
                        "command": command,
                        "camera_path": sample_generation.get("camera_path"),
                        "persistent_engine": True,
                        "checkpoint_load_count": 1,
                    }
                else:
                    error = (
                        error_path.read_text(encoding="utf-8", errors="replace")
                        if error_path.exists()
                        else f"GEN3C output was not written: {request.output_path}"
                    )
                    results[request.sample.sample_id] = {
                        "status": "failed",
                        "error": error,
                        "prediction_path": str(request.output_path),
                        "prompt": request.prompt,
                        "command": command,
                        "persistent_engine": True,
                        "checkpoint_load_count": 1,
                    }
            return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if self.config.repo_root is None:
            raise ValueError("GEN3C adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("GEN3C adapter requires checkpoint_dir")

        generation = _generation_for_sample(self.config.generation, sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.gen3c_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--checkpoint_dir",
            str(self.config.checkpoint_dir),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--sample_name",
            sample.prediction_stem,
            "--prompt",
            prompt,
        ]
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "num_gpus": int,
                "num_video_frames": int,
                "height": int,
                "width": int,
                "fps": int,
                "seed": int,
                "guidance": float,
                "num_steps": int,
                "trajectory": str,
                "camera_rotation": str,
                "movement_distance": float,
                "camera_path_json": str,
                "filter_points_threshold": float,
                "negative_prompt": str,
                "prompt_upsampler_dir": str,
            },
            flag_options=(
                "save_buffer",
                "foreground_masking",
                "offload_diffusion_transformer",
                "offload_tokenizer",
                "offload_text_encoder_model",
                "offload_prompt_upsampler",
                "offload_guardrail_models",
                "disable_prompt_encoder",
            ),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"GEN3C output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
        }
