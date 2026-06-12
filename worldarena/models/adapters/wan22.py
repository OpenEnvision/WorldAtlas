from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, extend_command_with_options


def _run_wan22_command(
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
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            )
            raise RuntimeError(
                f"Wan2.2 command failed with exit code {exc.returncode}; log: {log_path}\n{tail}"
            ) from exc


class Wan22Adapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

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
        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.wan22_batch_runner",
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
            },
            bool_value_options=("offload_model",),
            flag_options=("t5_cpu", "convert_model_dtype"),
        )
        sample_guide_scale = generation.get("sample_guide_scale")
        if sample_guide_scale is not None:
            command.append("--sample_guide_scale")
            if isinstance(sample_guide_scale, (list, tuple)):
                command.extend(str(float(value)) for value in sample_guide_scale)
            else:
                command.append(str(float(sample_guide_scale)))

        project_root = Path(__file__).resolve().parents[3]
        env = build_env(
            extra_pythonpaths=[project_root],
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
