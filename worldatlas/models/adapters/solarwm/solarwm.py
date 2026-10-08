"""SolarWM Stage2 adapter for WorldAtlas.

SolarWM consumes canonical raw-WebDataset rows rather than loose image files.
The adapter therefore adds the resolved WorldAtlas camera annotation to each
batch row; :mod:`solarwm_batch_runner` materializes the small bridge dataset and
launches the official ``solarwm infer`` entrypoint once for the whole shard.
"""

from __future__ import annotations

import math
from typing import Any

from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.camera_runtime import camera_annotation_for_conditioning_image
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


SOLARWM_DEFAULT_FPS = 16.0
SOLARWM_DEFAULT_FRAMES = 81


def resolve_source_frame_count(generation: dict[str, Any]) -> int:
    """Resolve the camera-backed source length used by SolarWM inference."""
    fps = float(generation.get("fps", SOLARWM_DEFAULT_FPS))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"SolarWM fps must be positive and finite, got {fps!r}")

    target_seconds = generation.get("target_duration_seconds")
    if target_seconds is None:
        frame_count = int(generation.get("num_frames", SOLARWM_DEFAULT_FRAMES))
    else:
        seconds = float(target_seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(
                "SolarWM target_duration_seconds must be positive and finite, "
                f"got {seconds!r}"
            )
        frame_count = int(math.ceil(seconds * fps))

    # The released Stage2 checkpoint is trained on 81-frame chunks. Longer
    # camera tracks are handled by its autoregressive camera-length route.
    if frame_count < SOLARWM_DEFAULT_FRAMES:
        raise ValueError(
            f"SolarWM Stage2 requires at least {SOLARWM_DEFAULT_FRAMES} source frames, "
            f"got {frame_count}"
        )
    return frame_count


class SolarWMAdapter(ExternalBatchAdapter):
    """Run SolarWM-5B Stage2 with WorldAtlas text, image, and camera inputs."""

    runner_module = "worldatlas.models.adapters.solarwm.solarwm_batch_runner"
    label = "SolarWM"

    def batch_checkpoint_load_policy(self) -> str:
        """One official inference launch loads the checkpoint for the whole shard."""
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        generation = self.config.generation
        frame_count = resolve_source_frame_count(generation)
        annotation_path, pose_source = camera_annotation_for_conditioning_image(
            request.sample,
            request.conditioning_image.expanduser().resolve(),
            frame_count=frame_count,
            namespace="solarwm",
            focal_scale=float(generation.get("synthetic_focal_scale", 0.6)),
        )
        payload.update(
            {
                "annotation_path": str(annotation_path),
                "pose_source": pose_source,
                "source_frame_count": frame_count,
            }
        )
        return payload


__all__ = ["SolarWMAdapter", "resolve_source_frame_count"]
