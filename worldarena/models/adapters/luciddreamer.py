"""LucidDreamer model adapter for WorldAtlas Arena.

Maps ``camera_path`` tokens to LucidDreamer render presets and launches
``luciddreamer_runner`` or ``luciddreamer_batch_runner``.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.synthetic_camera import SUPPORTED_SYNTHETIC_CAMERA_TOKENS
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
)

DEFAULT_CAMERA_PATH_RENDER_PRESET = {
    "fixed": "llff",
    "move_left": "llff",
    "move_right": "llff",
    "orbit_left": "llff",
    "orbit_right": "llff",
    "pan_left": "headbanging",
    "pan_right": "headbanging",
    "pedestal_down": "headbanging",
    "pedestal_up": "headbanging",
    "pull_out": "back_and_forth",
    "push_in": "back_and_forth",
    "roll_ccw": "headbanging_circle",
    "roll_cw": "headbanging_circle",
    "tilt_down": "headbanging",
    "tilt_up": "headbanging",
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
    return [str(token).strip().lower() for token in sample.camera_path if str(token).strip()]


def _generation_for_sample(generation: dict[str, Any], sample: BenchmarkSample) -> dict[str, Any]:
    resolved = _generation_for_suite(generation, sample.suite)
    camera_path = _camera_path_for_sample(sample)
    resolved["_camera_path"] = camera_path
    resolved["_camera_path_primary"] = camera_path[0] if camera_path else None
    resolved["_campath_render_source"] = "config"

    if not bool(resolved.get("use_camera_path_render", True)) or not camera_path:
        return resolved

    unsupported = [token for token in camera_path if token not in SUPPORTED_SYNTHETIC_CAMERA_TOKENS]
    if unsupported:
        raise ValueError(f"unsupported LucidDreamer camera token(s): {unsupported!r}")
    resolved["campath_render"] = "worldarena_camera_path"
    resolved["_campath_render_source"] = "worldarena_camera_path"
    return resolved


class LucidDreamerAdapter(ModelAdapter):
    """Generate 3D-consistent videos via the LucidDreamer upstream repo."""

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
            raise ValueError("LucidDreamer adapter requires repo_root")

        project_root = Path(__file__).resolve().parents[3]
        runner_path = project_root / "worldarena" / "models" / "adapters" / "luciddreamer_batch_runner.py"
        if not runner_path.exists():
            raise FileNotFoundError(f"LucidDreamer batch runner not found: {runner_path}")

        with tempfile.TemporaryDirectory(prefix="worldarena_luciddreamer_batch_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            requests_path = temp_dir / "requests.jsonl"
            results_path = temp_dir / "results.jsonl"
            with requests_path.open("w", encoding="utf-8") as file:
                for request in requests:
                    generation = _generation_for_sample(self.config.generation, request.sample)
                    prompt_prefix = str(generation.get("prompt_prefix", ""))
                    prompt_suffix = str(generation.get("prompt_suffix", ""))
                    prompt_text = f"{prompt_prefix}{request.prompt}{prompt_suffix}".strip()
                    payload = {
                        "sample_id": request.sample.sample_id,
                        "suite": request.sample.suite,
                        "camera_path": generation.get("_camera_path") or [],
                        "camera_path_primary": generation.get("_camera_path_primary"),
                        "campath_render_source": generation.get("_campath_render_source", "config"),
                        "conditioning_image": str(request.conditioning_image.expanduser().resolve()),
                        "output_path": str(request.output_path.expanduser().resolve()),
                        "prompt": prompt_text,
                        "negative_prompt": str(generation.get("negative_prompt", "")),
                        "campath_gen": str(generation.get("campath_gen", "lookdown")),
                        "campath_render": str(generation.get("campath_render", "llff")),
                        "model_name": generation.get("model_name"),
                        "seed": int(generation.get("seed", 1)),
                        "diff_steps": int(generation.get("diff_steps", 50)),
                        "input_size": int(generation.get("input_size", 512)),
                        "keep_work_dir": bool(generation.get("keep_work_dir", False)),
                        "min_seconds": float(generation.get("min_seconds", 5.0)),
                        "camera_path_frames": int(generation.get("camera_path_frames", 300)),
                        "camera_path_translation_scale": float(
                            generation.get("camera_path_translation_scale", 1.0)
                        ),
                    }
                    file.write(json.dumps(payload, ensure_ascii=False) + "\n")

            command = [
                self.config.python_bin,
                "-u",
                str(runner_path),
                "--repo_root",
                str(repo_root),
                "--requests_jsonl",
                str(requests_path),
                "--results_jsonl",
                str(results_path),
            ]

            env = build_env(
                extra_pythonpaths=[project_root],
                overrides=self.config.env,
            )
            run_command(command, cwd=repo_root, env=env)

            results: dict[str, dict[str, Any]] = {}
            if not results_path.exists():
                raise FileNotFoundError(f"LucidDreamer batch runner did not write results: {results_path}")
            with results_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    sample_id = str(payload.get("sample_id", ""))
                    if sample_id:
                        results[sample_id] = payload
            return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if sample.suite == "image_static" and sample.camera_path:
            result = self.generate_batch(
                [
                    PreparedGenerationRequest(
                        sample=sample,
                        conditioning_image=conditioning_image,
                        output_path=output_path,
                        prompt=prompt,
                    )
                ]
            )[sample.sample_id]
            if result.get("status") != "generated":
                raise RuntimeError(str(result.get("error") or "LucidDreamer batch generation failed"))
            return result

        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("LucidDreamer adapter requires repo_root")

        generation = _generation_for_sample(self.config.generation, sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]

        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.luciddreamer_runner",
            "--repo_root",
            str(repo_root),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--prompt",
            prompt_text,
            "--campath_gen",
            str(generation.get("campath_gen", "lookdown")),
            "--campath_render",
            str(generation.get("campath_render", "llff")),
        ]
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "negative_prompt": str,
                "model_name": str,
                "seed": int,
                "diff_steps": int,
            },
            bool_value_options=("keep_work_dir",),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"LucidDreamer output was not written: {output_path}")

        ply_path = output_path.with_suffix(".ply")
        depth_video_path = output_path.with_name(f"{output_path.stem}.depth{output_path.suffix}")
        payload: dict[str, Any] = {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
            "camera_path": generation.get("_camera_path") or [],
            "camera_path_primary": generation.get("_camera_path_primary"),
            "campath_render_source": generation.get("_campath_render_source", "config"),
            "campath_gen": str(generation.get("campath_gen", "lookdown")),
            "campath_render": str(generation.get("campath_render", "llff")),
        }
        if ply_path.exists():
            payload["gaussian_splat_path"] = str(ply_path)
        if depth_video_path.exists():
            payload["depth_video_path"] = str(depth_video_path)
        return payload


__all__ = [
    "DEFAULT_CAMERA_PATH_RENDER_PRESET",
    "LucidDreamerAdapter",
    "_generation_for_sample",
    "_generation_for_suite",
]
