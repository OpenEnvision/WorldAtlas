from __future__ import annotations

from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class CosmosFrameworkAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.cosmos_framework_batch_runner"
    label = "Cosmos Framework"
