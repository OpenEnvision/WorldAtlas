from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.benchmark.long_sequence_runtime import (
    LONG_SEQUENCE_DIMENSION_MAP,
    compute_long_sequence_metrics,
)


LONG_SEQUENCE_BACKENDS = {"long_sequence"}
ALL_LONG_SEQUENCE_METRICS = tuple(LONG_SEQUENCE_DIMENSION_MAP)
_LONG_SEQUENCE_RESULT_CACHE: dict[tuple[str, int, int, str], dict[str, Any]] = {}


def _cache_key(prediction_path: str, *, runtime: dict[str, Any]) -> tuple[str, int, int, str]:
    path = Path(prediction_path).expanduser().resolve()
    stat = path.stat()
    runtime_key = repr(sorted(runtime.items()))
    return (
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        runtime_key,
    )


def _long_sequence_result(prediction_path: str, *, runtime: dict[str, Any]) -> dict[str, Any]:
    key = _cache_key(prediction_path, runtime=runtime)
    cached = _LONG_SEQUENCE_RESULT_CACHE.get(key)
    if cached is not None:
        return cached
    payload = compute_long_sequence_metrics(
        prediction_path,
        metric_names=ALL_LONG_SEQUENCE_METRICS,
        runtime=runtime,
    )
    _LONG_SEQUENCE_RESULT_CACHE[key] = payload
    return payload


class LongSequenceMetric(Metric):
    def __init__(
        self,
        *,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
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
        del reference

        backend = self._resolve_backend()
        details = {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
            "dimension": LONG_SEQUENCE_DIMENSION_MAP[self.name],
            "duration_seconds": sample.duration_seconds,
        }
        try:
            result = _long_sequence_result(prediction.path, runtime=self.runtime)
            raw = result["metrics"].get(self.name)
            if raw is None:
                raise RuntimeError(f"long-sequence backend did not return a score for {self.name}")
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
