from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.pulse_runtime import compute_pulse_metrics
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


TEMPORAL_CALIBRATION_BACKENDS = {"pulse_of_motion"}
_PULSE_RESULT_CACHE: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}


def _cache_key(
    prediction_path: str,
    *,
    fallback_meta_fps: float | None,
    runtime: dict[str, Any],
) -> tuple[str, str, int, int, str]:
    path = Path(prediction_path).expanduser().resolve()
    stat = path.stat()
    runtime_key = repr(sorted(runtime.items()))
    return (
        str(path),
        "none" if fallback_meta_fps is None else f"{float(fallback_meta_fps):.6f}",
        stat.st_mtime_ns,
        stat.st_size,
        runtime_key,
    )


def _pulse_result(
    prediction_path: str,
    *,
    fallback_meta_fps: float | None,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    key = _cache_key(
        prediction_path,
        fallback_meta_fps=fallback_meta_fps,
        runtime=runtime,
    )
    cached = _PULSE_RESULT_CACHE.get(key)
    if cached is not None:
        return cached
    payload = compute_pulse_metrics(
        prediction_path,
        runtime=runtime,
        fallback_meta_fps=fallback_meta_fps,
    )
    _PULSE_RESULT_CACHE[key] = payload
    return payload


class TemporalCalibrationMetric(Metric):
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
            auto_backend="pulse_of_motion",
            supported_backends=TEMPORAL_CALIBRATION_BACKENDS,
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
        }
        try:
            result = _pulse_result(
                prediction.path,
                fallback_meta_fps=sample.fps,
                runtime=self.runtime,
            )
            raw = float(result["metrics"][self.name])
            details.update(result["details"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
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


__all__ = ["TemporalCalibrationMetric"]
