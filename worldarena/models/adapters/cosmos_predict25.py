"""Cosmos-Predict2.5 model adapter for WorldAtlas Arena."""

from __future__ import annotations

from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class CosmosPredict25Adapter(ExternalBatchAdapter):
    """Generate videos via NVIDIA Cosmos-Predict2.5 batch runner."""

    runner_module = "worldarena.models.adapters.cosmos_predict25_batch_runner"
    label = "Cosmos-Predict2.5"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"
