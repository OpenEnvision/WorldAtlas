"""Metric implementations for motion quality and dynamics."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import (
    Metric,
    clamp01,
    normalize_score,
    resolve_metric_backend,
)
from worldarena.benchmark.official_runtime import (
    compute_motion_magnitude as compute_official_motion_magnitude,
    compute_motion_smoothness as compute_official_motion_smoothness,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


def _flow_magnitudes(frames: list[np.ndarray]) -> list[float]:
    """Compute mean Farneback optical-flow magnitude for each consecutive frame pair."""
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
    """Pick official SEA-RAFT/VFI backends or fall back to Farneback flow."""
    auto_backend = "sea_raft" if metric_name == "motion_magnitude" else "vfi_mamba"
    supported_backends = {"farneback", auto_backend}
    return resolve_metric_backend(
        metric_name=metric_name,
        configured_backend=backend,
        auto_backend=auto_backend,
        supported_backends=supported_backends,
        disabled_backends=DISABLED_MOTION_BACKENDS,
    )


def _distribution_details(prefix: str, values: list[float]) -> dict[str, float]:
    """Summarize flow statistics used when comparing prediction to reference motion."""
    return {
        f"{prefix}_mean": round(float(np.mean(values)), 4),
        f"{prefix}_median": round(float(np.median(values)), 4),
        f"{prefix}_p90": round(float(np.percentile(values, 90)), 4),
    }


class MotionMagnitudeMetric(Metric):
    """Compare overall motion amount between prediction and reference videos."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Validate backend early and store official-runtime overrides."""
        super().__init__("motion_magnitude", backend, normalization)
        self.runtime = dict(runtime or {})
        _resolve_flow_backend(self.name, self.backend)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Score median motion magnitude similarity via SEA-RAFT or Farneback fallback."""
        del sample

        backend = _resolve_flow_backend(self.name, self.backend)
        if backend == "sea_raft":
            try:
                official_result = compute_official_motion_magnitude(
                    prediction.path,
                    normalization=self.normalization,
                    runtime=self.runtime,
                )
                details = {
                    "frame_indices": list(prediction.protocol_frame_indices),
                    "sampled_frame_indices": list(prediction.sampled_frame_indices),
                    **official_result["details"],
                }
                return MetricOutput(
                    raw=official_result["raw"],
                    normalized=official_result["normalized"],
                    backend=str(official_result["backend"]),
                    details=details,
                )
            except Exception as exc:  # pragma: no cover - heavy runtime failures
                return MetricOutput(
                    raw=None,
                    normalized=None,
                    backend=backend,
                    details={
                        "frame_indices": list(prediction.protocol_frame_indices),
                        "sampled_frame_indices": list(prediction.sampled_frame_indices),
                    },
                    error=str(exc),
                )
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
    """Compare temporal jitter in motion magnitude against a reference video."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Validate backend early and store official-runtime overrides."""
        super().__init__("motion_smoothness", backend, normalization)
        self.runtime = dict(runtime or {})
        _resolve_flow_backend(self.name, self.backend)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Penalize excess frame-to-frame motion variation relative to reference."""
        del sample

        backend = _resolve_flow_backend(self.name, self.backend)
        if backend == "vfi_mamba":
            try:
                official_result = compute_official_motion_smoothness(
                    prediction.path,
                    normalization=self.normalization,
                    runtime=self.runtime,
                )
                details = {
                    "frame_indices": list(prediction.protocol_frame_indices),
                    "sampled_frame_indices": list(prediction.sampled_frame_indices),
                    **official_result["details"],
                }
                return MetricOutput(
                    raw=official_result["raw"],
                    normalized=official_result["normalized"],
                    backend=str(official_result["backend"]),
                    details=details,
                )
            except Exception as exc:  # pragma: no cover - heavy runtime failures
                return MetricOutput(
                    raw=None,
                    normalized=None,
                    backend=backend,
                    details={
                        "frame_indices": list(prediction.protocol_frame_indices),
                        "sampled_frame_indices": list(prediction.sampled_frame_indices),
                    },
                    error=str(exc),
                )
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


__all__ = ["MotionMagnitudeMetric", "MotionSmoothnessMetric"]
