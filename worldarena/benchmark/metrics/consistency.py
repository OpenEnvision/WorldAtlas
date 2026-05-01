from __future__ import annotations

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


class ReprojectionErrorMetric(OfficialMetric):
    supported_backends = REPROJECTION_ERROR_BACKENDS
    auto_backend = "droid_slam"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("reprojection_error", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
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
    supported_backends = OPTICAL_FLOW_AEPE_BACKENDS
    auto_backend = "sea_raft"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("optical_flow_aepe", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
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
    def __init__(
        self,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        if metric_name not in GVGC_GEOMETRY_METRICS:
            raise KeyError(f"unsupported GVGC geometry metric: {metric_name}")
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
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
            payload = compute_gvgc_geometry(list(prediction.frames), runtime=self.runtime)
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
