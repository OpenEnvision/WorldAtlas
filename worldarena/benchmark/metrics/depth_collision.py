"""Metric implementations for depth-based collision detection."""

from __future__ import annotations

from typing import Any

from worldarena.benchmark.depth_collision_backend import compute_depth_collision
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.common.video_io import read_video_frames


DEPTH_COLLISION_BACKENDS = {"video_depth_anything"}


class DepthCollisionMetric(Metric):
    """Detect implausible depth overlaps that suggest interpenetrating geometry."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Store runtime options such as frame_source and depth-model settings."""
        super().__init__("depth_collision", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the Video Depth Anything collision-detection backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="video_depth_anything",
            supported_backends=DEPTH_COLLISION_BACKENDS,
        )

    def _frames(self, prediction: LoadedMedia) -> tuple[list[Any], dict[str, Any]]:
        """Load sampled protocol frames or the full decoded video for depth analysis."""
        frame_source = str(self.runtime.get("frame_source") or "sampled")
        if frame_source == "full_video":
            frames = read_video_frames(prediction.path)
            return frames, {
                "frame_source": frame_source,
                "frame_indices": list(range(len(frames))),
                "sampled_frame_indices": list(prediction.sampled_frame_indices),
            }
        if frame_source != "sampled":
            raise ValueError("depth_collision frame_source must be 'sampled' or 'full_video'")
        return list(prediction.frames), {
            "frame_source": frame_source,
            "frame_indices": list(prediction.sampled_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Estimate depth-collision rate over the selected prediction frames."""
        del sample
        del reference

        backend = self._resolve_backend()
        details: dict[str, Any] = {
            "prediction_path": prediction.path,
            "native_frame_count": prediction.native_frame_count,
        }
        if prediction.modality != "video":
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error="depth_collision requires a video prediction",
            )
        try:
            frames, frame_details = self._frames(prediction)
            details.update(frame_details)
            payload = compute_depth_collision(frames, runtime=self.runtime)
            details.update(payload.get("details", {}))
            raw = payload.get("raw")
            raw_float = float(raw) if isinstance(raw, (int, float)) else None
            return MetricOutput(
                raw=raw_float,
                normalized=normalize_score(raw_float, self.normalization),
                backend=str(payload.get("backend") or backend),
                details=details,
                eligibility_status="diagnostic",
                error=payload.get("error"),
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


__all__ = ["DEPTH_COLLISION_BACKENDS", "DepthCollisionMetric"]
