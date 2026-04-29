from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, extend_command_with_options, run_command


_YUME_OPTION_NAMES = (
    "mode",
    "caption_model_dir",
    "fps",
    "sample_steps",
    "sample_num",
    "frame_zero",
    "shift",
    "seed",
    "resolution",
    "gpu_index",
    "refine_from_image",
    "load_caption_model",
    "memory_optimization",
    "vae_memory_optimization",
    "camera_movement1",
    "camera_movement2",
)


def _suite_value(payload: dict[str, Any], name: str, suite: str) -> Any:
    by_suite = payload.get(f"{name}_by_suite")
    if isinstance(by_suite, dict) and suite in by_suite:
        return by_suite[suite]
    return payload.get(name)


def _generation_payload_for_sample(generation: dict[str, Any], sample: BenchmarkSample) -> dict[str, Any]:
    payload = dict(generation)
    for option_name in _YUME_OPTION_NAMES:
        resolved = _suite_value(generation, option_name, sample.suite)
        if resolved is not None:
            payload[option_name] = resolved
    return payload


class YumeAdapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if self.config.repo_root is None:
            raise ValueError("YUME adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("YUME adapter requires checkpoint_dir")
        if not requests:
            return {}

        generation = self.config.generation
        project_root = Path(__file__).resolve().parents[3]
        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError("YUME batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"yume_{uuid4().hex}.jsonl"
        with spec_path.open("w", encoding="utf-8") as file:
            for request in requests:
                file.write(
                    json.dumps(
                        {
                            "sample_id": request.sample.sample_id,
                            "prediction_stem": request.sample.prediction_stem,
                            "suite": request.sample.suite,
                            "conditioning_image": str(request.conditioning_image.expanduser().resolve()),
                            "output_path": str(request.output_path),
                            "prompt": request.prompt,
                            "generation": _generation_payload_for_sample(generation, request.sample),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.yume_batch_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--checkpoint_dir",
            str(self.config.checkpoint_dir),
            "--entrypoint",
            str(self.config.entrypoint or "webapp_single_gpu.py"),
            "--batch_spec_path",
            str(spec_path),
        ]
        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)

        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            if request.output_path.exists():
                results[request.sample.sample_id] = {
                    "status": "generated",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                }
            else:
                results[request.sample.sample_id] = {
                    "status": "failed",
                    "error": f"YUME batch output was not written: {request.output_path}",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
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
            raise ValueError("YUME adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("YUME adapter requires checkpoint_dir")

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]

        runner_payload = _generation_payload_for_sample(generation, sample)

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.yume_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--entrypoint",
            str(self.config.entrypoint or "webapp_single_gpu.py"),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--prompt",
            prompt,
        ]
        command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])

        extend_command_with_options(
            command,
            payload=runner_payload,
            value_options={
                "mode": str,
                "caption_model_dir": str,
                "fps": int,
                "sample_steps": int,
                "sample_num": int,
                "frame_zero": int,
                "shift": float,
                "seed": int,
                "resolution": str,
                "gpu_index": int,
                "camera_movement1": str,
                "camera_movement2": str,
            },
            bool_value_options=(
                "refine_from_image",
                "load_caption_model",
                "memory_optimization",
                "vae_memory_optimization",
            ),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"YUME output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
        }


__all__ = ["YumeAdapter"]
