from __future__ import annotations

from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class HeliosAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.helios_batch_runner"
    label = "Helios"


__all__ = ["HeliosAdapter"]
