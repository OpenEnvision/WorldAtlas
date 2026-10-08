"""TRAJAN trajectory-reconstruction motion-quality metric."""

from __future__ import annotations

from typing import Any

from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.metrics.base import Metric, resolve_metric_backend
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput
from worldatlas.benchmark.trajan_backend import compute_trajan_average_jaccard


TRAJAN_BACKENDS = {"trajan"}


class TrajanMetric(Metric):
    """Measure per-video motion realism with TRAJAN reconstruction AJ."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("trajan", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="trajan",
            supported_backends=TRAJAN_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del reference
        backend = self._resolve_backend()
        if prediction.modality != "video":
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details={"prediction_path": prediction.path},
                eligibility_status="not_applicable",
                error="trajan requires a video prediction",
            )
        result = compute_trajan_average_jaccard(
            video_path=prediction.path,
            sample_id=sample.sample_id,
            normalization=self.normalization,
            runtime=self.runtime,
        )
        return MetricOutput(
            raw=float(result["raw"]),
            normalized=result["normalized"],
            backend=str(result["backend"]),
            details=dict(result["details"]),
        )


__all__ = ["TRAJAN_BACKENDS", "TrajanMetric"]
