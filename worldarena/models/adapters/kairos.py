from __future__ import annotations

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class KairosAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.kairos_batch_runner"
    label = "Kairos Sensenova"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, object]:
        payload = super().batch_spec_payload(request)
        payload["input_image"] = str(request.conditioning_image.expanduser().resolve())
        return payload


__all__ = ["KairosAdapter"]
