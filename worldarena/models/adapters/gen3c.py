from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.annotations import resolve_prompt_contract
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
    run_sequential_generate_batch,
)


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
    if resolved.get("multi_trajectory"):
        # Lyra/Gen3C official static demo uses a fixed six-trajectory sweep.
        # Do not let a WorldArena camera token override trajectory/rotation in
        # that mode, otherwise the generated views diverge from the upstream
        # static_sdg.sh defaults.
        resolved["camera_path"] = camera_path
        resolved["camera_path_primary"] = primary_token
        return resolved

    trajectory, camera_rotation = GEN3C_CAMERA_TRAJECTORIES.get(
        primary_token,
        (str(resolved.get("trajectory", "left")), str(resolved.get("camera_rotation", "center_facing"))),
    )
    resolved["trajectory"] = trajectory
    resolved["camera_rotation"] = camera_rotation
    if primary_token == "fixed":
        resolved["movement_distance"] = 0.0
    resolved["camera_path"] = camera_path
    resolved["camera_path_primary"] = primary_token
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
    def supports_batch_generation(self) -> bool:
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        return run_sequential_generate_batch(self, requests)

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
