from __future__ import annotations

from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class StableVirtualCameraAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.stable_virtual_camera_batch_runner"
    label = "Stable Virtual Camera"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        payload.update(
            {
                "camera_path": list(request.sample.camera_path),
                "width": request.sample.width,
                "height": request.sample.height,
            }
        )
        return payload
