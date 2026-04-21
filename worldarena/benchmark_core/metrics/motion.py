from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from worldarena.benchmark_core.media import LoadedMedia
from worldarena.benchmark_core.metrics.base import Metric, clamp01, normalize_score
from worldarena.benchmark_core.schemas import BenchmarkSample, MetricOutput


def _flow_magnitudes(frames: list[np.ndarray]) -> list[float]:
    if len(frames) < 2:
        return []
    magnitudes = []
    for left, right in zip(frames[:-1], frames[1:]):
        left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            left_gray,
            right_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        magnitudes.append(float(np.mean(magnitude)))
    return magnitudes


def _motion_difference_fallback(frames: list[np.ndarray]) -> list[float]:
    if len(frames) < 2:
        return []
    magnitudes = []
    for left, right in zip(frames[:-1], frames[1:]):
        diff = np.abs(right.astype(np.float32) - left.astype(np.float32))
        magnitudes.append(float(np.mean(diff) / 255.0))
    return magnitudes


def _resolve_flow_backend(backend: str) -> str:
    if backend == "auto":
        return "farneback"
    return backend


class MotionMagnitudeMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("motion_magnitude", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = _resolve_flow_backend(self.backend)
        calculator = _flow_magnitudes if backend == "farneback" else _motion_difference_fallback
        pred_values = calculator(prediction.frames)
        ref_values = calculator(reference.frames)
        if not pred_values or not ref_values:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                error="not enough frames for motion metric",
            )
        pred_median = float(np.median(pred_values))
        ref_median = float(np.median(ref_values))
        raw = clamp01(1.0 - abs(pred_median - ref_median) / max(ref_median, 1.0))
        return MetricOutput(
            raw=raw,
            normalized=normalize_score(raw, self.normalization),
            backend=backend,
            details={
                "prediction_median": round(pred_median, 4),
                "reference_median": round(ref_median, 4),
            },
        )


class MotionSmoothnessMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("motion_smoothness", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = _resolve_flow_backend(self.backend)
        calculator = _flow_magnitudes if backend == "farneback" else _motion_difference_fallback
        pred_values = calculator(prediction.frames)
        ref_values = calculator(reference.frames)
        if len(pred_values) < 2 or len(ref_values) < 2:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                error="not enough frame transitions for smoothness metric",
            )
        pred_jitter = float(np.mean(np.abs(np.diff(pred_values))))
        ref_jitter = float(np.mean(np.abs(np.diff(ref_values))))
        raw = clamp01(1.0 - abs(pred_jitter - ref_jitter) / max(ref_jitter, 1.0))
        return MetricOutput(
            raw=raw,
            normalized=normalize_score(raw, self.normalization),
            backend=backend,
            details={
                "prediction_jitter": round(pred_jitter, 4),
                "reference_jitter": round(ref_jitter, 4),
            },
        )
