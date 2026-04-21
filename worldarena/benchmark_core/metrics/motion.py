from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from worldarena.benchmark_core.media import LoadedMedia
from worldarena.benchmark_core.metrics.base import (
    Metric,
    clamp01,
    normalize_score,
    resolve_metric_backend,
)
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


DISABLED_MOTION_BACKENDS = {"pixel_diff"}


def _resolve_flow_backend(metric_name: str, backend: str) -> str:
    return resolve_metric_backend(
        metric_name=metric_name,
        configured_backend=backend,
        auto_backend="farneback",
        supported_backends={"farneback"},
        disabled_backends=DISABLED_MOTION_BACKENDS,
    )


def _distribution_details(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean": round(float(np.mean(values)), 4),
        f"{prefix}_median": round(float(np.median(values)), 4),
        f"{prefix}_p90": round(float(np.percentile(values, 90)), 4),
    }


class MotionMagnitudeMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("motion_magnitude", backend, normalization)
        _resolve_flow_backend(self.name, self.backend)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = _resolve_flow_backend(self.name, self.backend)
        pred_values = _flow_magnitudes(prediction.frames)
        ref_values = _flow_magnitudes(reference.frames)
        if not pred_values or not ref_values:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details={
                    "sampled_frame_indices": list(prediction.sampled_frame_indices),
                },
                error="not enough frames for motion metric",
            )
        pred_median = float(np.median(pred_values))
        ref_median = float(np.median(ref_values))
        eps = 1e-6
        tau = float(self.normalization.get("tau_log_ratio", 0.35))
        log_ratio = float(np.log((pred_median + eps) / (ref_median + eps)))
        raw = clamp01(float(np.exp(-abs(log_ratio) / tau)))
        details = {
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
            **_distribution_details("prediction", pred_values),
            **_distribution_details("reference", ref_values),
            "log_ratio": round(log_ratio, 4),
        }
        return MetricOutput(
            raw=raw,
            normalized=normalize_score(raw, self.normalization),
            backend=backend,
            details=details,
        )


class MotionSmoothnessMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("motion_smoothness", backend, normalization)
        _resolve_flow_backend(self.name, self.backend)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = _resolve_flow_backend(self.name, self.backend)
        pred_values = _flow_magnitudes(prediction.frames)
        ref_values = _flow_magnitudes(reference.frames)
        if len(pred_values) < 2 or len(ref_values) < 2:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details={
                    "sampled_frame_indices": list(prediction.sampled_frame_indices),
                },
                error="not enough frame transitions for smoothness metric",
            )
        pred_jitter = float(np.mean(np.abs(np.diff(pred_values))))
        ref_jitter = float(np.mean(np.abs(np.diff(ref_values))))
        eps = 1e-6
        tau = float(self.normalization.get("tau_log_ratio", 0.5))
        log_ratio = float(np.log((pred_jitter + eps) / (ref_jitter + eps)))
        raw = clamp01(float(np.exp(-abs(log_ratio) / tau)))
        return MetricOutput(
            raw=raw,
            normalized=normalize_score(raw, self.normalization),
            backend=backend,
            details={
                "sampled_frame_indices": list(prediction.sampled_frame_indices),
                "prediction_jitter": round(pred_jitter, 4),
                "reference_jitter": round(ref_jitter, 4),
                "log_ratio": round(log_ratio, 4),
            },
        )
