"""Base class for adapters that delegate batch generation to external runner modules.

Subclasses set ``runner_module`` to a ``*_batch_runner`` entry point and inherit
JSONL spec writing, logging, and per-sample result collection.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.common.progress import log_progress
from worldatlas.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldatlas.models.adapters.common import build_env, worldatlas_project_root

_MAX_TEE_LINE_CHARS = 400


def run_logged_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    label: str,
) -> None:
    """Run a subprocess, appending stdout to a log file and teeing compact lines to stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = dict(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                display = line
                if len(display) > _MAX_TEE_LINE_CHARS:
                    display = display[:_MAX_TEE_LINE_CHARS].rstrip() + "...\n"
                sys.stdout.write(display)
                sys.stdout.flush()
            returncode = process.wait()
        if returncode:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            )
            raise RuntimeError(
                f"{label} command failed with exit code {returncode}; log: {log_path}\n{tail}"
            )
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise


class ExternalBatchAdapter(ModelAdapter):
    """Run many samples through one long-lived external batch runner subprocess."""

    runner_module: str = ""
    label: str = "external model"

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        return {
            "sample_id": request.sample.sample_id,
            "suite": request.sample.suite,
            "prediction_stem": request.sample.prediction_stem,
            "conditioning_image": str(request.conditioning_image.expanduser().resolve()),
            "output_path": str(request.output_path.expanduser().resolve()),
            "prompt": request.prompt,
        }

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        if not self.runner_module:
            raise ValueError(f"{self.__class__.__name__} must define runner_module")
        if self.config.repo_root is None:
            raise ValueError(f"{self.label} adapter requires repo_root")
        command = [
            self.config.python_bin,
            "-m",
            self.runner_module,
            "--repo_root",
            str(self.config.repo_root),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ]
        if self.config.checkpoint_dir is not None:
            command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])
        return command

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        if self.config.repo_root is None:
            raise ValueError(f"{self.label} adapter requires repo_root")

        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError(f"{self.label} batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)
        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"{self.config.family.lower().replace('.', '_')}_{uuid4().hex}.jsonl"
        generation_path = spec_path.with_suffix(".generation.json")
        generation_path.write_text(
            json.dumps(self.config.generation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with spec_path.open("w", encoding="utf-8") as file:
            for request in requests:
                file.write(json.dumps(self.batch_spec_payload(request), ensure_ascii=False) + "\n")

        project_root = worldatlas_project_root()
        env = build_env(
            extra_pythonpaths=[project_root, self.config.repo_root],
            overrides=self.config.env,
        )
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        command = self._build_command(spec_path, generation_path)
        log_path = spec_path.with_suffix(".log")
        log_progress(
            "external_runner_start",
            label=self.label,
            samples=len(requests),
            runner=self.runner_module,
            log=str(log_path),
        )
        run_logged_command(
            command,
            cwd=project_root,
            env=env,
            log_path=log_path,
            label=self.label,
        )
        log_progress(
            "external_runner_done",
            label=self.label,
            samples=len(requests),
            log=str(log_path),
        )

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
                    "error": f"{self.label} output was not written: {request.output_path}",
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
            raise RuntimeError(str(result.get("error", f"{self.label} generation failed")))
        return result

