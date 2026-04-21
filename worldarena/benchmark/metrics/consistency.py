from __future__ import annotations

from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.official import OfficialMetric
from worldarena.benchmark.official_runtime import (
    compute_optical_flow_aepe,
    compute_reprojection_error,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


REPROJECTION_ERROR_BACKENDS = {"droid_slam"}
OPTICAL_FLOW_AEPE_BACKENDS = {"sea_raft"}


class ReprojectionErrorMetric(OfficialMetric):
    supported_backends = REPROJECTION_ERROR_BACKENDS
    auto_backend = "droid_slam"

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("reprojection_error", backend, normalization)

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

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("optical_flow_aepe", backend, normalization)

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


__all__ = ["OpticalFlowAepeMetric", "ReprojectionErrorMetric"]
