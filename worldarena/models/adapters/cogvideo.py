from __future__ import annotations

from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class CogVideoAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.cogvideo_batch_runner"
    label = "CogVideo"
