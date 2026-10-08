"""Cosmos3-Nano adapter using the pinned WorldFoundry pipeline.

WorldAtlas runs the pipeline in a dedicated model environment and keeps one
loaded checkpoint per shard. The environment must import both projects.
"""

from __future__ import annotations

from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


class Cosmos3Adapter(ExternalBatchAdapter):
    """Generate videos with WorldFoundry's native Cosmos3 pipeline."""

    runner_module = "worldatlas.models.adapters.cosmos3.cosmos3_batch_runner"
    label = "Cosmos3"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"
