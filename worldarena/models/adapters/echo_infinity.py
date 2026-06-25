from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, extend_command_with_options


def _run_echo_infinity_command(
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
                f"Echo-Infinity command failed with exit code {exc.returncode}; "
                f"log: {log_path}\n{tail}"
            ) from exc


class EchoInfinityAdapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        if self.config.repo_root is None:
            raise ValueError("EchoInfinity adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("EchoInfinity adapter requires checkpoint_dir")

        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError(
                "EchoInfinity batch generation requires a single output directory"
            )
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"echo_infinity_{uuid4().hex}.jsonl"
        with spec_path.open("w", encoding="utf-8") as file:
            for request in requests:
                file.write(
                    json.dumps(
                        {
                            "sample_id": request.sample.sample_id,
                            "suite": request.sample.suite,
                            "prediction_stem": request.sample.prediction_stem,
                            "output_path": str(
                                request.output_path.expanduser().resolve()
                            ),
                            "prompt": request.prompt,
                            "conditioning_image": (
                                str(request.conditioning_image.expanduser().resolve())
                                if request.conditioning_image is not None
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        repo_root = Path(self.config.repo_root).expanduser().resolve()
        checkpoint_dir = Path(self.config.checkpoint_dir).expanduser().resolve()
        ckpt_path = checkpoint_dir / "echo_infinity.pt"

        generation = self.config.generation or {}
        num_gpus = int(generation.get("num_gpus", 1))

        project_root = Path(__file__).resolve().parents[3]
        command = [
            self.config.python_bin,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(num_gpus),
            "-m",
            "worldarena.models.adapters.echo_infinity_batch_runner",
            "--repo_root",
            str(repo_root),
            "--batch_spec_path",
            str(spec_path),
            "--ckpt_path",
            str(ckpt_path),
        ]
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "num_output_frames": int,
                "seed": int,
                "fps": int,
            },
        )

        # PYTHONPATH must include both:
        #   - project_root: so torchrun can find worldarena.models.adapters.echo_infinity_batch_runner
        #   - repo_root: so the batch runner can import pipeline, utils, wan from Echo-Infinity
        env = build_env(
            extra_pythonpaths=[project_root, repo_root],
            overrides=self.config.env,
        )
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("NCCL_CROSS_NIC", "1")
        env.setdefault("NCCL_TIMEOUT", "3600")

        log_path = spec_path.with_suffix(".log")
        _run_echo_infinity_command(command, cwd=project_root, env=env, log_path=log_path)

        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            conditioning_path = str(request.conditioning_image.expanduser().resolve())
            if request.output_path.is_file() and request.output_path.stat().st_size > 0:
                results[request.sample.sample_id] = {
                    "status": "generated",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "conditioning_image": conditioning_path,
                    "conditioning_input": conditioning_path,
                    "prompt": request.prompt,
                    "batch_spec_path": str(spec_path),
                    "generation_log": str(log_path),
                    "echo_infinity_i2v_conditioning": "kv_cache_priming",
                    "echo_infinity_i2v_decode": "conditioned_decode_trim_prefix",
                }
            else:
                results[request.sample.sample_id] = {
                    "status": "failed",
                    "error": f"Echo-Infinity output was not written: {request.output_path}",
                    "conditioning_image": conditioning_path,
                    "conditioning_input": conditioning_path,
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
            raise RuntimeError(
                str(result.get("error", "Echo-Infinity generation failed"))
            )
        return result
