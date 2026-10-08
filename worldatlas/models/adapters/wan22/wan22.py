from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldatlas.models.adapters.common import build_env, extend_command_with_options, worldatlas_project_root
from worldatlas.models.adapters.external_batch import run_logged_command


def _runner_command(python_bin: str, generation: dict[str, Any]) -> list[str]:
    module_name = "worldatlas.models.adapters.wan22.wan22_batch_runner"
    nproc_per_node = int(generation.get("nproc_per_node", 1) or 1)
    if nproc_per_node <= 1:
        return [python_bin, "-m", module_name]

    # Invoke torch.distributed.run through the model interpreter. A standalone
    # `torchrun` shebang (`#!/usr/bin/env python3.10`) can pick up the Hope
    # orchestrator env after launch_model_batch_8gpu.sh prepends it to PATH.
    command = [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(nproc_per_node),
    ]
    master_port = generation.get("master_port")
    if master_port is not None:
        command.extend(["--master-port", str(int(master_port))])
    command.extend(["-m", module_name])
    return command


def _run_wan22_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    run_logged_command(command, cwd=cwd, env=env, log_path=log_path, label="Wan2.2")


class Wan22Adapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

    def batch_checkpoint_load_policy(self) -> str:
        """The runner builds one distributed Wan pipeline per batch spec."""
        return "load_once"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        if self.config.repo_root is None:
            raise ValueError("Wan2.2 adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Wan2.2 adapter requires checkpoint_dir")

        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError("Wan2.2 batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"wan22_{uuid4().hex}.jsonl"
        with spec_path.open("w", encoding="utf-8") as file:
            for request in requests:
                file.write(
                    json.dumps(
                        {
                            "sample_id": request.sample.sample_id,
                            "suite": request.sample.suite,
                            "prediction_stem": request.sample.prediction_stem,
                            "conditioning_image": str(
                                request.conditioning_image.expanduser().resolve()
                            ),
                            "output_path": str(request.output_path.expanduser().resolve()),
                            "prompt": request.prompt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
        )

        generation = self.config.generation
        command = _runner_command(self.config.python_bin, generation)
        command.extend(
            [
                "--repo_root",
                str(self.config.repo_root),
                "--checkpoint_dir",
                str(self.config.checkpoint_dir),
                "--batch_spec_path",
                str(spec_path),
            ]
        )
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "task": str,
                "size": str,
                "frame_num": int,
                "min_seconds": float,
                "sample_solver": str,
                "sample_steps": int,
                "sample_shift": float,
                "base_seed": int,
                "negative_prompt": str,
                "device_id": int,
                "ulysses_size": int,
            },
            bool_value_options=("offload_model",),
            flag_options=("t5_cpu", "convert_model_dtype", "t5_fsdp", "dit_fsdp"),
        )
        sample_guide_scale = generation.get("sample_guide_scale")
        if sample_guide_scale is not None:
            command.append("--sample_guide_scale")
            if isinstance(sample_guide_scale, (list, tuple)):
                command.extend(str(float(value)) for value in sample_guide_scale)
            else:
                command.append(str(float(sample_guide_scale)))

        project_root = worldatlas_project_root()
        extra_pythonpaths = [project_root]
        if self.config.repo_root is not None:
            extra_pythonpaths.insert(0, self.config.repo_root)
        env = build_env(
            extra_pythonpaths=extra_pythonpaths,
            overrides=self.config.env,
        )
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        log_path = spec_path.with_suffix(".log")
        _run_wan22_command(command, cwd=project_root, env=env, log_path=log_path)

        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            if request.output_path.is_file() and request.output_path.stat().st_size > 0:
                results[request.sample.sample_id] = {
                    "status": "generated",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                    "batch_spec_path": str(spec_path),
                    "generation_log": str(log_path),
                }
            else:
                results[request.sample.sample_id] = {
                    "status": "failed",
                    "error": f"Wan2.2 output was not written: {request.output_path}",
                    "prompt": request.prompt,
                    "batch_spec_path": str(spec_path),
                    "generation_log": str(log_path),
                }
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
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
        if result.get("status") == "failed":
            raise RuntimeError(str(result.get("error", "Wan2.2 generation failed")))
        return result
