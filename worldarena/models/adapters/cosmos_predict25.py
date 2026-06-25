from __future__ import annotations

from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class CosmosPredict25Adapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.cosmos_predict25_batch_runner"
    label = "Cosmos-Predict2.5"
