"""WonderJourney model adapter for WorldAtlas Arena.

WonderJourney generates exploratory 3D scene videos from structured prompts.
Shares prompt/trajectory helpers with WonderWorld.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.checkpoints import (
    project_root as worldarena_project_root,
    resolve_checkpoint_path,
)
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
)
from worldarena.models.adapters.wonderworld import (
    _generation_for_suite,
    _structured_prompts_for_sample,
)
from worldarena.models.adapters.wonderjourney_runner import resolve_wonderjourney_camera_control
from worldarena.models.config import ModelRuntimeConfig


def _project_root(config: ModelRuntimeConfig) -> Path:
    del config
    return worldarena_project_root()


def _resolved_midas_checkpoint(
    config: ModelRuntimeConfig,
    generation: dict[str, Any],
) -> Path:
    del config
    configured_value = generation.get("midas_checkpoint")
    checkpoint_path = resolve_checkpoint_path(
        configured_value,
        kind="file",
        required=configured_value is not None,
    )
    if checkpoint_path is None:
        raise ValueError("WonderJourney adapter requires generation.midas_checkpoint")
    return checkpoint_path


def _resolved_checkpointish_option(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if Path(text).is_absolute() or text.startswith("ckpt/"):
        path = resolve_checkpoint_path(text, kind="any", required=False)
        return str(path) if path is not None else None
    return text


def _sequence_option(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    try:
        return json.dumps(list(value))
    except TypeError:
        return json.dumps([value])


def _generation_with_camera_control(
    sample: BenchmarkSample,
    generation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, object] | None]:
    resolved = dict(generation)
    if sample.suite != "image_static" or not bool(resolved.get("use_sample_camera_path", True)):
        return resolved, None
    control = resolve_wonderjourney_camera_control(
        list(sample.camera_path),
        frames=int(resolved.get("frames", 50)),
        num_scenes=int(resolved.get("num_scenes", 1)),
        num_keyframes=int(resolved.get("num_keyframes", 2)),
    )
    for key in ("frames", "num_scenes", "num_keyframes", "rotation_path"):
        resolved[key] = control[key]
    return resolved, control


class WonderJourneyAdapter(ModelAdapter):
    """Generate exploratory scene videos via the WonderJourney upstream repo."""

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
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("WonderJourney adapter requires repo_root")

        project_root = _project_root(self.config)
        spec_dir = requests[0].output_path.parent / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"wonderjourney_{uuid.uuid4().hex}.jsonl"
        results_path = spec_path.with_suffix(".results.json")
        rows: list[dict[str, Any]] = []
        prepared_results: dict[str, dict[str, Any]] = {}
        metadata_by_sample: dict[str, dict[str, Any]] = {}

        for request in requests:
            sample = request.sample
            output_path = request.output_path
            generation, camera_control = _generation_with_camera_control(
                sample,
                _generation_for_suite(self.config.generation, sample.suite),
            )
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_prefix = str(generation.get("prompt_prefix", ""))
                prompt_suffix = str(generation.get("prompt_suffix", ""))
                prompt_text = f"{prompt_prefix}{request.prompt}{prompt_suffix}".strip()
                if not prompt_text:
                    prompt_text = str(sample.prompt_current or sample.prompt_target or "").strip()

                structured = _structured_prompts_for_sample(
                    sample,
                    prompt_text=prompt_text,
                    generation=generation,
                )
                midas_checkpoint = _resolved_midas_checkpoint(self.config, generation)

                command = [
                    self.config.python_bin,
                    "-m",
                    "worldarena.models.adapters.wonderjourney_runner",
                    "--repo_root",
                    str(repo_root),
                    "--entrypoint",
                    str(self.config.entrypoint or "run.py"),
                    "--conditioning_image",
                    str(request.conditioning_image.expanduser().resolve()),
                    "--output_path",
                    str(output_path),
                    "--sample_name",
                    output_path.stem,
                    "--prompt",
                    prompt_text,
                    "--scene_name",
                    str(structured["scene_name"]),
                    "--style_prompt",
                    str(structured["style_prompt"]),
                    "--negative_prompt",
                    str(structured["negative_prompt"]),
                    "--midas_checkpoint",
                    str(midas_checkpoint),
                ]
                background_prompt = str(structured["background_prompt"])
                if background_prompt:
                    command.extend(["--background_prompt", background_prompt])
                entities = [str(entity) for entity in structured["entities"]]
                if entities:
                    command.extend(["--entities", *entities])

                stable_diffusion_checkpoint = _resolved_checkpointish_option(
                    generation.get("stable_diffusion_checkpoint")
                )
                if stable_diffusion_checkpoint is not None:
                    command.extend(["--stable_diffusion_checkpoint", stable_diffusion_checkpoint])
                rotation_path = _sequence_option(generation.get("rotation_path"))
                if rotation_path is not None:
                    command.extend(["--rotation_path", rotation_path])
                if camera_control is not None:
                    command.extend(
                        ["--camera_actions", json.dumps(camera_control["camera_actions"])]
                    )

                extend_command_with_options(
                    command,
                    payload=generation,
                    value_options={
                        "seed": int,
                        "frames": int,
                        "num_scenes": int,
                        "num_keyframes": int,
                        "save_fps": int,
                        "run_timeout": float,
                        "depth_model": str,
                        "stable_diffusion_revision": str,
                        "camera_speed": float,
                        "rotation_range": float,
                        "init_focal_length": float,
                        "inpainting_resolution_gen": int,
                        "inpainting_resolution_interp": int,
                        "kf2_upsample_coef": int,
                        "fg_depth_range": float,
                        "depth_shift": float,
                        "regenerate_times": int,
                        "num_finetune_decoder_steps": int,
                        "num_finetune_decoder_steps_interp": int,
                        "camera_speed_multiplier_rotation": float,
                    },
                    bool_value_options=(
                        "use_gpt",
                        "debug",
                        "skip_interp",
                        "skip_gen",
                        "enable_regenerate",
                        "finetune_decoder_gen",
                        "finetune_decoder_interp",
                        "finetune_depth_model",
                        "keep_work_dir",
                        "archive_sidecars",
                    ),
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "args": command[3:],
                        "prediction_path": str(output_path),
                        "prompt": prompt_text,
                    }
                )
                metadata_by_sample[sample.sample_id] = {
                    "command": command,
                    "prediction_path": str(output_path),
                    "prompt": prompt_text,
                    "scene_name": structured["scene_name"],
                    "entities": structured["entities"],
                    "entity_source": structured["entity_source"],
                    "style_prompt": structured["style_prompt"],
                    "background_prompt": structured["background_prompt"],
                    "negative_prompt": structured["negative_prompt"],
                    "content_prompt": structured["content_prompt"],
                }
                if camera_control is not None:
                    metadata_by_sample[sample.sample_id].update(
                        {
                            "camera_path": list(sample.camera_path),
                            "camera_actions": list(camera_control["camera_actions"]),
                            "wonderjourney_motion_codes": list(camera_control["rotation_path"]),
                            "camera_control_source": "sample.camera_path",
                            "camera_control_frames_per_segment": int(camera_control["frames"]),
                            "camera_control_total_frame_budget": int(
                                camera_control["total_frame_budget"]
                            ),
                        }
                    )
            except Exception as exc:
                prepared_results[sample.sample_id] = {
                    "status": "failed",
                    "error": str(exc),
                    "prediction_path": str(output_path),
                    "prompt": request.prompt,
                }

        loaded_results: dict[str, dict[str, Any]] = {}
        if rows:
            with spec_path.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False))
                    file.write(chr(10))
            command = [
                self.config.python_bin,
                "-m",
                "worldarena.models.adapters.wonderjourney_runner",
                "--batch_spec",
                str(spec_path),
                "--batch_results",
                str(results_path),
            ]
            env = build_env(extra_pythonpaths=[project_root], overrides=self.config.env)
            run_command(command, cwd=project_root, env=env)
            if results_path.exists():
                loaded_results = json.loads(results_path.read_text(encoding="utf-8"))

        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            sample_id = request.sample.sample_id
            payload = dict(prepared_results.get(sample_id) or loaded_results.get(sample_id) or {})
            if not payload:
                payload = {
                    "status": "failed",
                    "error": "batch generation returned no result for this sample",
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                }
            if payload.get("status") == "generated":
                payload.update(metadata_by_sample.get(sample_id, {}))
                reverse_path = request.output_path.with_name(f"{request.output_path.stem}.reverse{request.output_path.suffix}")
                keyframes_path = request.output_path.with_name(f"{request.output_path.stem}.keyframes{request.output_path.suffix}")
                if reverse_path.exists():
                    payload["reverse_video_path"] = str(reverse_path)
                if keyframes_path.exists():
                    payload["keyframes_video_path"] = str(keyframes_path)
            results[sample_id] = payload
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("WonderJourney adapter requires repo_root")

        generation, camera_control = _generation_with_camera_control(
            sample,
            _generation_for_suite(self.config.generation, sample.suite),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = _project_root(self.config)

        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()
        if not prompt_text:
            prompt_text = str(sample.prompt_current or sample.prompt_target or "").strip()

        structured = _structured_prompts_for_sample(
            sample,
            prompt_text=prompt_text,
            generation=generation,
        )
        midas_checkpoint = _resolved_midas_checkpoint(self.config, generation)

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.wonderjourney_runner",
            "--repo_root",
            str(repo_root),
            "--entrypoint",
            str(self.config.entrypoint or "run.py"),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--sample_name",
            output_path.stem,
            "--prompt",
            prompt_text,
            "--scene_name",
            str(structured["scene_name"]),
            "--style_prompt",
            str(structured["style_prompt"]),
            "--negative_prompt",
            str(structured["negative_prompt"]),
            "--midas_checkpoint",
            str(midas_checkpoint),
        ]
        background_prompt = str(structured["background_prompt"])
        if background_prompt:
            command.extend(["--background_prompt", background_prompt])
        entities = [str(entity) for entity in structured["entities"]]
        if entities:
            command.extend(["--entities", *entities])

        stable_diffusion_checkpoint = _resolved_checkpointish_option(
            generation.get("stable_diffusion_checkpoint")
        )
        if stable_diffusion_checkpoint is not None:
            command.extend(["--stable_diffusion_checkpoint", stable_diffusion_checkpoint])
        rotation_path = _sequence_option(generation.get("rotation_path"))
        if rotation_path is not None:
            command.extend(["--rotation_path", rotation_path])
        if camera_control is not None:
            command.extend(["--camera_actions", json.dumps(camera_control["camera_actions"])])

        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "seed": int,
                "frames": int,
                "num_scenes": int,
                "num_keyframes": int,
                "save_fps": int,
                "run_timeout": float,
                "depth_model": str,
                "stable_diffusion_revision": str,
                "camera_speed": float,
                "rotation_range": float,
                "init_focal_length": float,
                "inpainting_resolution_gen": int,
                "inpainting_resolution_interp": int,
                "kf2_upsample_coef": int,
                "fg_depth_range": float,
                "depth_shift": float,
                "regenerate_times": int,
                "num_finetune_decoder_steps": int,
                "num_finetune_decoder_steps_interp": int,
                "camera_speed_multiplier_rotation": float,
            },
            bool_value_options=(
                "use_gpt",
                "debug",
                "skip_interp",
                "skip_gen",
                "enable_regenerate",
                "finetune_decoder_gen",
                "finetune_decoder_interp",
                "finetune_depth_model",
                "keep_work_dir",
                "archive_sidecars",
            ),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"WonderJourney output was not written: {output_path}")

        payload: dict[str, Any] = {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
            "scene_name": structured["scene_name"],
            "entities": structured["entities"],
            "entity_source": structured["entity_source"],
            "style_prompt": structured["style_prompt"],
            "background_prompt": structured["background_prompt"],
            "negative_prompt": structured["negative_prompt"],
            "content_prompt": structured["content_prompt"],
        }
        if camera_control is not None:
            payload.update(
                {
                    "camera_path": list(sample.camera_path),
                    "camera_actions": list(camera_control["camera_actions"]),
                    "wonderjourney_motion_codes": list(camera_control["rotation_path"]),
                    "camera_control_source": "sample.camera_path",
                    "camera_control_frames_per_segment": int(camera_control["frames"]),
                    "camera_control_total_frame_budget": int(
                        camera_control["total_frame_budget"]
                    ),
                }
            )
        reverse_path = output_path.with_name(f"{output_path.stem}.reverse{output_path.suffix}")
        keyframes_path = output_path.with_name(f"{output_path.stem}.keyframes{output_path.suffix}")
        if reverse_path.exists():
            payload["reverse_video_path"] = str(reverse_path)
        if keyframes_path.exists():
            payload["keyframes_video_path"] = str(keyframes_path)
        return payload


__all__ = ["WonderJourneyAdapter"]
