from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
    run_sequential_generate_batch,
)
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample


ASTRA_DEFAULT_CAM_TYPE = 4


def _generation_for_suite(generation: dict[str, Any], suite: str) -> dict[str, Any]:
    resolved = dict(generation)
    for key, value in generation.items():
        if not key.endswith("_by_suite") or not isinstance(value, dict):
            continue
        base_key = key[: -len("_by_suite")]
        if suite in value:
            resolved[base_key] = value[suite]
    return resolved


def _astra_cam_type_for_sample(
    sample: BenchmarkSample,
    generation: dict[str, Any],
) -> tuple[int, str, list[str]]:
    explicit_cam_type = int(generation.get("cam_type", ASTRA_DEFAULT_CAM_TYPE))
    if not bool(generation.get("cam_type_from_sample", True)):
        return explicit_cam_type, "config", []

    camera_path = [str(token).strip().lower() for token in _camera_path_for_sample(sample) if str(token).strip()]
    if not camera_path:
        return explicit_cam_type, "config", []

    has_forward = any(token in {"push_in"} for token in camera_path)
    has_left = any(token in {"pan_left", "move_left", "orbit_left"} for token in camera_path)
    has_right = any(token in {"pan_right", "move_right", "orbit_right"} for token in camera_path)

    if has_forward and has_left and has_right:
        return 6, "sample_camera_path", camera_path
    if has_left and has_right:
        return 7, "sample_camera_path", camera_path
    if has_forward and has_left:
        return 4, "sample_camera_path", camera_path
    if has_forward and has_right:
        return 5, "sample_camera_path", camera_path
    if has_forward:
        return 1, "sample_camera_path", camera_path
    if has_left:
        return 2, "sample_camera_path", camera_path
    if has_right:
        return 3, "sample_camera_path", camera_path
    return explicit_cam_type, "config", camera_path


class AstraAdapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        return run_sequential_generate_batch(self, requests)

    def prompt_for(self, sample: BenchmarkSample) -> str:
        prompt_target = str(sample.prompt_target).strip()
        if prompt_target:
            return prompt_target
        return str(sample.prompt_current).strip()

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("Astra adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Astra adapter requires checkpoint_dir")

        generation = _generation_for_suite(self.config.generation, sample.suite)
        cam_type, cam_type_source, camera_path = _astra_cam_type_for_sample(sample, generation)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]

        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.astra_runner",
            "--repo_root",
            str(repo_root),
            "--checkpoint_dir",
            str(self.config.checkpoint_dir),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--prompt",
            prompt_text,
            "--cam_type",
            str(int(cam_type)),
        ]
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "wan_model_path": str,
                "dit_path": str,
                "start_frame": int,
                "initial_condition_frames": int,
                "frames_per_generation": int,
                "total_frames_to_generate": int,
                "max_history_frames": int,
                "device": str,
                "modality_type": str,
                "scene_info_path": str,
                "camera_guidance_scale": float,
                "text_guidance_scale": float,
                "moe_num_experts": int,
                "moe_top_k": int,
                "moe_hidden_dim": int,
                "gpu_index": int,
                "cuda_visible_devices": str,
            },
            flag_options=("use_gt_prompt", "add_icons"),
        )
        if generation.get("use_camera_cfg", False):
            command.extend(["--use_camera_cfg", "true"])
        if generation.get("use_real_poses", False):
            command.extend(["--use_real_poses", "true"])

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"Astra output was not written: {output_path}")

        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
            "cam_type": int(cam_type),
            "cam_type_source": cam_type_source,
            "camera_path": list(camera_path),
        }


__all__ = [
    "ASTRA_DEFAULT_CAM_TYPE",
    "AstraAdapter",
    "_astra_cam_type_for_sample",
    "_generation_for_suite",
]
