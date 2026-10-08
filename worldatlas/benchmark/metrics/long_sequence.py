"""Metric implementations for long-sequence temporal coherence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput
from worldatlas.benchmark.long_sequence_runtime import (
    LONG_SEQUENCE_DIMENSION_MAP,
    compute_long_sequence_metrics,
)


LONG_SEQUENCE_BACKENDS = {"long_sequence"}
ALL_LONG_SEQUENCE_METRICS = tuple(LONG_SEQUENCE_DIMENSION_MAP)
LONG_SEQUENCE_QUALITY_METRICS = (
    "long_sequence_aesthetic_quality",
    "long_sequence_imaging_quality",
)
_LONG_SEQUENCE_RESULT_CACHE: dict[str, dict[str, Any]] = {}


def _cache_key(
    prediction_path: str,
    *,
    runtime: dict[str, Any],
    metric_names: tuple[str, ...],
) -> str:
    """Key long-sequence results by video file identity, runtime, and metric group."""
    path = Path(prediction_path).expanduser().resolve()
    from worldatlas.benchmark.integrity import digest_json, path_content
    return digest_json({'video': path_content(path), 'runtime': runtime, 'metric_names': metric_names})


def _metric_group(metric_name: str) -> tuple[str, ...]:
    """Batch aesthetic and imaging quality metrics into one backend call."""
    if metric_name in LONG_SEQUENCE_QUALITY_METRICS:
        return LONG_SEQUENCE_QUALITY_METRICS
    return (metric_name,)


def _long_sequence_result(
    prediction_path: str,
    *,
    runtime: dict[str, Any],
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compute or reuse the shared long-sequence runtime payload for a video."""
    key = _cache_key(prediction_path, runtime=runtime, metric_names=metric_names)
    cached = _LONG_SEQUENCE_RESULT_CACHE.get(key)
    if cached is not None:
        return cached
    payload = compute_long_sequence_metrics(
        prediction_path,
        metric_names=metric_names,
        runtime=runtime,
    )
    _LONG_SEQUENCE_RESULT_CACHE[key] = payload
    return payload


class LongSequenceMetric(Metric):
    """Surface one long-horizon diagnostic metric from the shared runtime backend."""

    def __init__(
        self,
        *,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Register a named long-sequence sub-metric and its runtime overrides."""
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the bundled long-sequence evaluation backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="long_sequence",
            supported_backends=LONG_SEQUENCE_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Extract one long-sequence score from the cached backend payload."""
        del reference

        backend = self._resolve_backend()
        details = {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
            "dimension": LONG_SEQUENCE_DIMENSION_MAP[self.name],
            "duration_seconds": sample.duration_seconds,
        }
        try:
            metric_names = _metric_group(self.name)
            result = _long_sequence_result(
                prediction.path,
                runtime=self.runtime,
                metric_names=metric_names,
            )
            raw = result["metrics"].get(self.name)
            if raw is None:
                detail_error = result.get("metric_details", {}).get(self.name, {}).get("error")
                raise RuntimeError(
                    detail_error or f"long-sequence backend did not return a score for {self.name}"
                )
            details.update(result.get("metric_details", {}).get(self.name, {}))
            details["source_root"] = result.get("source_root")
            details["runtime_device"] = result.get("runtime_device")
            return MetricOutput(
                raw=float(raw),
                normalized=normalize_score(float(raw), self.normalization),
                backend=str(result["backend"]),
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:  # pragma: no cover - heavy runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


__all__ = [
    "ALL_LONG_SEQUENCE_METRICS",
    "LongSequenceMetric",
]
