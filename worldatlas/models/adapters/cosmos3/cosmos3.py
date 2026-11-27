"""Cosmos3 adapter using the official NVIDIA cosmos-framework repository.

WorldAtlas runs ``cosmos_framework.scripts.inference`` inside the dedicated
Cosmos3 environment and loads one checkpoint per shard across the full batch.
"""

from __future__ import annotations

from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


class Cosmos3Adapter(ExternalBatchAdapter):
    """Generate videos with NVIDIA's official cosmos-framework inference CLI."""

    runner_module = "worldatlas.models.adapters.cosmos3.cosmos3_batch_runner"
    label = "Cosmos3"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"
