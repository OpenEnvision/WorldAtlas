from __future__ import annotations

from typing import Any

from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.camera_runtime import camera_annotation_for_conditioning_image
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


class WarpAsHistoryAdapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.warp_as_history.warp_as_history_batch_runner"
    label = "Warp-as-History"

    def batch_checkpoint_load_policy(self) -> str:
        """The external runner keeps the Warp pipeline resident per shard."""
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        generation = self.config.generation
        annotation_path, pose_source = camera_annotation_for_conditioning_image(
            request.sample,
            request.conditioning_image.expanduser().resolve(),
            frame_count=int(generation.get("num_frames", 81)),
            namespace="warp_as_history",
            focal_scale=float(generation.get("synthetic_focal_scale", 0.6)),
        )
        payload["annotation_path"] = str(annotation_path)
        payload["pose_source"] = pose_source
        return payload
