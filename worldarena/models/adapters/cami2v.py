"""WorldArena adapter for the official ZGCTroy/CamI2V implementation."""

from __future__ import annotations

from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter
from worldarena.models.adapters.camera_i2v_common import normalize_camera_path


class CamI2VAdapter(ExternalBatchAdapter):
    """Run image suites through one long-lived official CamI2V process."""

    runner_module = "worldarena.models.adapters.cami2v_batch_runner"
    label = "CamI2V"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        payload.update(
            {
                "camera_path": normalize_camera_path(request.sample.camera_path),
                "generation_mode": request.sample.generation_mode,
                "source_width": request.sample.width,
                "source_height": request.sample.height,
            }
        )
        return payload


__all__ = ["CamI2VAdapter"]
