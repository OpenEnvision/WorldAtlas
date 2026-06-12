from __future__ import annotations

from pathlib import Path

from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class LTX23Adapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.ltx23_batch_runner"
    label = "LTX-2.3"

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        if self.config.checkpoint_dir is None:
            raise ValueError("LTX-2.3 adapter requires checkpoint_dir")
        return super()._build_command(spec_path, generation_path)
