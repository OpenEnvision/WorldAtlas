"""SolarWM's camera-conditioned H3 Stage2 model (distinct from native H3)."""

from __future__ import annotations

import math
from typing import Any

from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.camera_runtime import camera_annotation_for_conditioning_image
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


def source_frame_count(suite: str, generation: dict[str, Any]) -> int:
    if suite not in {"image_static", "image_dynamic", "memory_loop"}:
        raise ValueError(f"Unsupported SolarWM-H3 suite: {suite}")
    fps = float(generation.get("fps", 24))
    key = "memory_duration_seconds" if suite == "memory_loop" else "duration_seconds"
    seconds = float(generation.get(key, 60 if suite == "memory_loop" else 5))
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("SolarWM-H3 FPS and duration must be positive and finite")
    frames = math.ceil(fps * seconds)
    if frames < 22:
        raise ValueError("SolarWM-H3 requires at least 22 source frames")
    return frames


class SolarWMH3Adapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.solarwm.solarwm_h3_batch_runner"
    label = "SolarWM-H3"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        frames = source_frame_count(request.sample.suite, self.config.generation)
        annotation, source = camera_annotation_for_conditioning_image(
            request.sample,
            request.conditioning_image.expanduser().resolve(),
            frame_count=frames,
            namespace="solarwm_h3",
            focal_scale=float(self.config.generation.get("synthetic_focal_scale", 0.6)),
        )
        payload.update(annotation_path=str(annotation), pose_source=source, source_frame_count=frames)
        return payload
