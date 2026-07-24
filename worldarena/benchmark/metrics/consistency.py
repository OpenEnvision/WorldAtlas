"""Metric implementations for photometric and temporal consistency."""

from __future__ import annotations

import json
from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.official import OfficialMetric
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.gvgc_geometry_backend import (
    DEFAULT_BACKEND as DEFAULT_GVGC_GEOMETRY_BACKEND,
    compute_gvgc_geometry,
)
from worldarena.benchmark.official_runtime import (
    compute_optical_flow_aepe,
    compute_reprojection_error,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


REPROJECTION_ERROR_BACKENDS = {"droid_slam"}
OPTICAL_FLOW_AEPE_BACKENDS = {"sea_raft"}
GVGC_GEOMETRY_BACKENDS = {DEFAULT_GVGC_GEOMETRY_BACKEND}
GVGC_GEOMETRY_METRICS = {
    "epipolar_error",
    "epipolar_inlier_rate",
    "feature_match_coverage",
}

_GVGC_GEOMETRY_CACHE_KEY: tuple[Any, ...] | None = None
_GVGC_GEOMETRY_CACHE_PAYLOAD: dict[str, Any] | None = None
_GVGC_GEOMETRY_CACHE_FRAMES: list[Any] | None = None


def _shared_gvgc_geometry_payload(
    prediction: LoadedMedia,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Reuse one geometry analysis across the three per-sample sub-metrics."""
    global _GVGC_GEOMETRY_CACHE_FRAMES, _GVGC_GEOMETRY_CACHE_KEY, _GVGC_GEOMETRY_CACHE_PAYLOAD

    runtime_key = json.dumps(runtime, sort_keys=True, separators=(",", ":"), default=str)
    cache_key = (
        prediction.path,
        prediction.native_frame_count,
        tuple(prediction.sampled_frame_indices),
        runtime_key,
    )
    if (
        prediction.frames is not _GVGC_GEOMETRY_CACHE_FRAMES
        or cache_key != _GVGC_GEOMETRY_CACHE_KEY
        or _GVGC_GEOMETRY_CACHE_PAYLOAD is None
    ):
        _GVGC_GEOMETRY_CACHE_PAYLOAD = compute_gvgc_geometry(
            list(prediction.frames),
            runtime=runtime,
        )
        _GVGC_GEOMETRY_CACHE_FRAMES = prediction.frames
        _GVGC_GEOMETRY_CACHE_KEY = cache_key
    return _GVGC_GEOMETRY_CACHE_PAYLOAD


class ReprojectionErrorMetric(OfficialMetric):
    """Score multi-view geometric consistency via DROID-SLAM reprojection error."""

    supported_backends = REPROJECTION_ERROR_BACKENDS
    auto_backend = "droid_slam"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Attach optional official-runtime overrides for reprojection scoring."""
        super().__init__("reprojection_error", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Run DROID-SLAM reprojection error on the generated video path."""
        del sample
        del reference

        backend = self._resolve_backend()
        details = self._base_details(prediction)
        try:
            result = compute_reprojection_error(
                prediction.path,
                normalization=self.normalization,
                runtime=self.runtime,
            )
            details.update(result["details"])
            return MetricOutput(
                raw=result["raw"],
                normalized=result["normalized"],
                backend=str(result["backend"]),
                details=details,
            )
        except Exception as exc:  # pragma: no cover - heavy runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


class OpticalFlowAepeMetric(OfficialMetric):
    """Score photometric consistency via SEA-RAFT average end-point error."""

    supported_backends = OPTICAL_FLOW_AEPE_BACKENDS
    auto_backend = "sea_raft"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Attach optional official-runtime overrides for optical-flow AEPE."""
        super().__init__("optical_flow_aepe", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Run SEA-RAFT optical-flow AEPE on the generated video path."""
        del sample
        del reference

        backend = self._resolve_backend()
        details = self._base_details(prediction)
        try:
            result = compute_optical_flow_aepe(
                prediction.path,
                normalization=self.normalization,
                runtime=self.runtime,
            )
            details.update(result["details"])
            return MetricOutput(
                raw=result["raw"],
                normalized=result["normalized"],
                backend=str(result["backend"]),
                details=details,
            )
        except Exception as exc:  # pragma: no cover - heavy runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


class GvgcGeometryMetric(Metric):
    """Expose GVGC epipolar and feature-match geometry scores from one backend pass."""

    def __init__(
        self,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Register one GVGC geometry sub-metric and shared runtime options."""
        if metric_name not in GVGC_GEOMETRY_METRICS:
            raise KeyError(f"unsupported GVGC geometry metric: {metric_name}")
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the default GVGC geometry backend for epipolar metrics."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend=DEFAULT_GVGC_GEOMETRY_BACKEND,
            supported_backends=GVGC_GEOMETRY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Extract one GVGC geometry sub-metric from a shared video analysis pass."""
        del sample
        del reference

        backend = self._resolve_backend()
        details: dict[str, Any] = {
            "prediction_path": prediction.path,
            "native_frame_count": prediction.native_frame_count,
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }
        if prediction.modality != "video":
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=f"{self.name} requires a video prediction",
            )

        try:
            payload = _shared_gvgc_geometry_payload(prediction, self.runtime)
            details.update(payload.get("details", {}))
            raw_payload = dict(payload.get("raw") or {})
            raw_value = raw_payload.get(self.name)
            raw_float = float(raw_value) if isinstance(raw_value, (int, float)) else None
            return MetricOutput(
                raw=raw_float,
                normalized=normalize_score(raw_float, self.normalization),
                backend=str(payload.get("backend") or backend),
                details=details,
                eligibility_status="diagnostic",
                error=payload.get("error") if raw_float is None else None,
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


__all__ = [
    "GVGC_GEOMETRY_METRICS",
    "GvgcGeometryMetric",
    "OpticalFlowAepeMetric",
    "ReprojectionErrorMetric",
]
