"""Prediction-only temporal flickering metric (VBench C.2.3)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, clamp01, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput

TEMPORAL_FLICKERING_BACKENDS = {"pixel_mae"}
PIXEL_RANGE = 255.0


def _align_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Resize ``right`` to match ``left`` when resolutions differ."""
    if left.shape == right.shape:
        return right
    return cv2.resize(
        right,
        (left.shape[1], left.shape[0]),
        interpolation=cv2.INTER_AREA,
    )


def _pairwise_frame_mae(left: np.ndarray, right: np.ndarray) -> float:
    """Mean absolute error between two RGB frames on the 0–255 scale."""
    aligned = _align_frame(left, right)
    diff = np.abs(left.astype(np.float32) - aligned.astype(np.float32))
    return float(np.mean(diff))


def compute_temporal_flickering_score(frames: list[np.ndarray]) -> dict[str, Any]:
    """Compute VBench-style temporal flickering from consecutive-frame pixel MAE."""
    if len(frames) < 2:
        raise ValueError("not enough frames for temporal flickering metric")

    pair_mae = [
        _pairwise_frame_mae(left, right)
        for left, right in zip(frames[:-1], frames[1:])
    ]
    mean_mae = float(np.mean(pair_mae))
    raw = clamp01((PIXEL_RANGE - mean_mae) / PIXEL_RANGE)
    return {
        "raw": raw,
        "details": {
            "frame_pair_count": len(pair_mae),
            "mean_pairwise_mae": round(mean_mae, 4),
            "pairwise_mae": [round(value, 4) for value in pair_mae],
        },
    }


class TemporalFlickeringMetric(Metric):
    """Score inter-frame pixel stability without a reference video."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Store backend and normalization for temporal flickering."""
        super().__init__("temporal_flickering", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the pixel-MAE temporal flickering backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="pixel_mae",
            supported_backends=TEMPORAL_FLICKERING_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Score prediction-only temporal flickering on protocol-sampled frames."""
        del sample
        del reference

        backend = self._resolve_backend()
        frames = list(prediction.frames or prediction.anchor_frames)
        details = {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }
        try:
            result = compute_temporal_flickering_score(frames)
            details.update(result["details"])
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


__all__ = ["TemporalFlickeringMetric", "compute_temporal_flickering_score"]
