from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
)
from worldarena.models.adapters.gen3c import _generation_for_sample


def _run_lyra1_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                command,
                check=True,
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            log_file.flush()
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
            raise RuntimeError(
                f"Lyra-1 command failed with exit code {exc.returncode}; log: {log_path}\n{tail}"
            ) from exc


def _view_indices_arg(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value)


class Lyra1Adapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        if self.config.repo_root is None:
            raise ValueError("Lyra-1 adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Lyra-1 adapter requires checkpoint_dir")

        generations = [
            _generation_for_sample(self.config.generation, request.sample)
            for request in requests
        ]
        generation = dict(generations[0])

        project_root = Path(__file__).resolve().parents[3]
        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError("Lyra-1 batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"lyra1_{uuid4().hex}.jsonl"
        with spec_path.open("w", encoding="utf-8") as file:
            for request, request_generation in zip(requests, generations):
                file.write(
                    json.dumps(
                        {
                            "sample_id": request.sample.sample_id,
                            "prediction_stem": request.sample.prediction_stem,
                            "conditioning_image": str(request.conditioning_image.expanduser().resolve()),
                            "output_path": str(request.output_path),
                            "prompt": request.prompt,
                            "generation": request_generation,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.lyra1_batch_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--checkpoint_dir",
            str(self.config.checkpoint_dir),
            "--batch_spec_path",
            str(spec_path),
        ]
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "num_gpus": int,
                "stage1_num_gpus": int,
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
                "total_movement_distance_factor": float,
                "filter_points_threshold": float,
                "center_depth_quantile_value": float,
                "target_index_subsample": int,
                "output_view_index": int,
                "negative_prompt": str,
            },
            flag_options=(
                "foreground_masking",
                "center_depth_quantile",
                "multi_trajectory",
                "offload_diffusion_transformer",
                "offload_tokenizer",
                "offload_text_encoder_model",
                "offload_prompt_upsampler",
                "offload_guardrail_models",
                "disable_guardrail",
                "disable_prompt_encoder",
                "skip_stage2",
            ),
        )
        static_view_indices = _view_indices_arg(generation.get("static_view_indices_fixed"))
        if static_view_indices:
            command.extend(["--static_view_indices_fixed", static_view_indices])

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        log_path = spec_path.with_suffix(".log")
        _run_lyra1_command(command, cwd=project_root, env=env, log_path=log_path)

        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            if request.output_path.exists():
                results[request.sample.sample_id] = {
                    "status": "generated",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                    "generation_log": str(log_path),
                    "batch_spec_path": str(spec_path),
                }
            else:
                results[request.sample.sample_id] = {
                    "status": "failed",
                    "error": f"Lyra-1 output was not written: {request.output_path}",
                    "prompt": request.prompt,
                    "generation_log": str(log_path),
                    "batch_spec_path": str(spec_path),
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
            raise ValueError("Lyra-1 adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Lyra-1 adapter requires checkpoint_dir")

        generation = _generation_for_sample(self.config.generation, sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.lyra1_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--sample_name",
            sample.prediction_stem,
            "--prompt",
            prompt,
        ]
        command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])

        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "num_gpus": int,
                "stage1_num_gpus": int,
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
                "total_movement_distance_factor": float,
                "filter_points_threshold": float,
                "center_depth_quantile_value": float,
                "target_index_subsample": int,
                "output_view_index": int,
                "negative_prompt": str,
            },
            flag_options=(
                "foreground_masking",
                "center_depth_quantile",
                "multi_trajectory",
                "offload_diffusion_transformer",
                "offload_tokenizer",
                "offload_text_encoder_model",
                "offload_prompt_upsampler",
                "offload_guardrail_models",
                "disable_guardrail",
                "disable_prompt_encoder",
                "skip_stage2",
            ),
        )
        static_view_indices = _view_indices_arg(generation.get("static_view_indices_fixed"))
        if static_view_indices:
            command.extend(["--static_view_indices_fixed", static_view_indices])

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"Lyra-1 output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
        }
