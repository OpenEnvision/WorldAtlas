from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class LongLiveAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.longlive_batch_runner"
    label = "LongLive"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        if request.sample.prompt_sequence:
            payload["prompt_sequence"] = list(request.sample.prompt_sequence)
        return payload

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        if self.config.repo_root is None:
            raise ValueError("LongLive adapter requires repo_root")
        return [
            self.config.python_bin,
            "-m",
            self.runner_module,
            "--repo_root",
            str(self.config.repo_root),
            "--entrypoint",
            str(self.config.entrypoint or "inference.py"),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ]
