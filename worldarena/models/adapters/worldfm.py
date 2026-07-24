"""WorldFM model adapter for WorldAtlas Arena.

WorldFM renders camera-guided videos from panoramas. The adapter resolves pose
annotations and delegates batch inference to ``worldfm_batch_runner``.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, run_command
from worldarena.models.adapters.pose_synthesis import resolve_pose_annotation_path


def _clean_prompt(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _panorama_prompt_for_sample(sample: BenchmarkSample, generation: dict[str, Any]) -> str:
    mode = str(generation.get("panorama_prompt_mode", "prompt_current")).strip().lower()
    if mode in {"", "none", "off", "false"}:
        return ""
    if mode in {"prompt_current", "current", "scene", "scene_current"}:
        return _clean_prompt(sample.prompt_current or sample.prompt_target)
    if mode in {"prompt_target", "target", "future"}:
        return _clean_prompt(sample.prompt_target or sample.prompt_current)
    if mode in {"prompt_sequence", "sequence"}:
        return _clean_prompt(" ".join(sample.prompt_sequence) or sample.prompt_target or sample.prompt_current)
    raise ValueError(f"unsupported WorldFM panorama_prompt_mode: {mode}")


class WorldFMAdapter(ModelAdapter):
    """Generate panorama-based world videos via the WorldFM upstream repo."""

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
            raise ValueError("WorldFM adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("WorldFM adapter requires checkpoint_dir")

        generation = self.config.generation
        project_root = Path(__file__).resolve().parents[3]
        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError("WorldFM batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, dict[str, Any]] = {}
        runnable_requests: list[PreparedGenerationRequest] = []
        pose_by_sample: dict[str, tuple[str, Path]] = {}
        panorama_prompt_by_sample: dict[str, str] = {}
        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"worldfm_{uuid4().hex}.jsonl"

        with spec_path.open("w", encoding="utf-8") as file:
            for request in requests:
                try:
                    pose_annotation_path, pose_source = resolve_pose_annotation_path(
                        self.config,
                        request.sample,
                        request.conditioning_image.expanduser().resolve(),
                        frame_count=int(generation.get("trajectory_frames", generation.get("num_frames", 81))),
                        namespace="worldfm",
                        focal_scale=float(generation.get("synthetic_focal_scale", 0.6)),
                    )
                except Exception as exc:
                    results[request.sample.sample_id] = {
                        "status": "failed",
                        "error": str(exc),
                        "prediction_path": str(request.output_path),
                        "prompt": request.prompt,
                    }
                    continue
                pose_by_sample[request.sample.sample_id] = (pose_source, pose_annotation_path)
                panorama_prompt = _panorama_prompt_for_sample(request.sample, generation)
                panorama_negative_prompt = str(generation.get("panorama_negative_prompt", ""))
                panorama_prompt_by_sample[request.sample.sample_id] = panorama_prompt
                runnable_requests.append(request)
                file.write(
                    json.dumps(
                        {
                            "sample_id": request.sample.sample_id,
                            "prediction_stem": request.sample.prediction_stem,
                            "conditioning_image": str(request.conditioning_image.expanduser().resolve()),
                            "output_path": str(request.output_path),
                            "prompt": request.prompt,
                            "panorama_prompt": panorama_prompt,
                            "panorama_negative_prompt": panorama_negative_prompt,
                            "annotation_path": str(pose_annotation_path),
                            "pose_source": pose_source,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        command = [
            sys.executable,
            "-m",
            "worldarena.models.adapters.worldfm_batch_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--checkpoint_dir",
            str(self.config.checkpoint_dir),
            "--batch_spec_path",
            str(spec_path),
            "--model_filename",
            str(generation.get("model_filename", "worldfm_2-step.pth")),
            "--vae_subdir",
            str(generation.get("vae_subdir", "vae")),
            "--step",
            str(int(generation.get("step", 2))),
            "--image_size",
            str(int(generation.get("image_size", 512))),
            "--cfg_scale",
            str(float(generation.get("cfg_scale", 4.5))),
            "--render_size",
            str(int(generation.get("render_size", 512))),
            "--fps",
            str(int(generation.get("fps", 30))),
            "--gpu_index",
            str(int(generation.get("gpu_index", 0))),
            "--video_crf",
            str(int(generation.get("video_crf", 14))),
            "--video_preset",
            str(generation.get("video_preset", "medium")),
        ]
        if bool(generation.get("cache_condition_latents", False)):
            command.append("--cache_condition_latents")
        for option_name in ("config_path", "hw_path", "moge_path", "moge_pretrained"):
            option_value = generation.get(option_name)
            if option_value:
                command.extend([f"--{option_name}", str(option_value)])

        cli_error: str | None = None
        if runnable_requests:
            env = build_env(
                extra_pythonpaths=[project_root],
                overrides=self.config.env,
            )
            try:
                run_command(command, cwd=self.config.repo_root.resolve(), env=env)
            except subprocess.CalledProcessError as exc:
                cli_error = f"WorldFM batch runner exited with code {exc.returncode}"

        for request in runnable_requests:
            pose_source, pose_annotation_path = pose_by_sample[request.sample.sample_id]
            panorama_prompt = panorama_prompt_by_sample.get(request.sample.sample_id, "")
            panorama_negative_prompt = str(generation.get("panorama_negative_prompt", ""))
            if request.output_path.exists():
                results[request.sample.sample_id] = {
                    "status": "generated",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                    "panorama_prompt": panorama_prompt,
                    "panorama_negative_prompt": panorama_negative_prompt,
                    "pose_source": pose_source,
                    "pose_annotation_path": str(pose_annotation_path),
                }
            else:
                results[request.sample.sample_id] = {
                    "status": "failed",
                    "error": cli_error or f"WorldFM output was not written: {request.output_path}",
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                    "panorama_prompt": panorama_prompt,
                    "panorama_negative_prompt": panorama_negative_prompt,
                    "pose_source": pose_source,
                    "pose_annotation_path": str(pose_annotation_path),
                }
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        payload = self.generate_batch(
            [
                PreparedGenerationRequest(
                    sample=sample,
                    conditioning_image=conditioning_image,
                    output_path=output_path,
                    prompt=prompt,
                )
            ]
        ).get(sample.sample_id, {})
        if payload.get("status") == "failed":
            raise RuntimeError(str(payload.get("error") or "WorldFM generation failed"))
        payload.pop("status", None)
        payload.pop("error", None)
        return payload
