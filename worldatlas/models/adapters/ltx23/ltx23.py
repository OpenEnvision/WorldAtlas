from __future__ import annotations

from pathlib import Path

from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


class LTX23Adapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.ltx23.ltx23_batch_runner"
    label = "LTX-2.3"

    def batch_checkpoint_load_policy(self) -> str:
        """The LTX process reuses its pipeline for every row in the shard."""
        return "load_once"

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        if self.config.checkpoint_dir is None:
            raise ValueError("LTX-2.3 adapter requires checkpoint_dir")
        return super()._build_command(spec_path, generation_path)
