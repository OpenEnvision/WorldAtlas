from __future__ import annotations

from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.camera_runtime import camera_annotation_for_conditioning_image
from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class SanaWMAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.sana_wm_batch_runner"
    label = "Sana-WM"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        generation = self.config.generation
        annotation_path, pose_source = camera_annotation_for_conditioning_image(
            request.sample,
            request.conditioning_image.expanduser().resolve(),
            frame_count=int(generation.get("num_frames", 161)),
            namespace="sana_wm",
            focal_scale=float(generation.get("synthetic_focal_scale", 0.6)),
        )
        payload["annotation_path"] = str(annotation_path)
        payload["pose_source"] = pose_source
        return payload
