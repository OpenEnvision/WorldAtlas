"""Temporal coherence metric implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldatlas.benchmark.pulse_runtime import compute_pulse_metrics
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput


TEMPORAL_CALIBRATION_BACKENDS = {"pulse_of_motion"}
_PULSE_RESULT_CACHE: dict[str, dict[str, Any]] = {}


def _cache_key(
    prediction_path: str,
    *,
    fallback_meta_fps: float | None,
    runtime: dict[str, Any],
) -> str:
    """Build a cache key from video identity, metadata FPS, and runtime options."""
    path = Path(prediction_path).expanduser().resolve()
    from worldatlas.benchmark.integrity import digest_json, path_content
    return digest_json({'video': path_content(path), 'fallback_meta_fps': fallback_meta_fps, 'runtime': runtime})


def _pulse_result(
    prediction_path: str,
    *,
    fallback_meta_fps: float | None,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Run Pulse-of-Motion once per video and memoize the full metric bundle."""
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
    """Expose Pulse-of-Motion FPS calibration sub-metrics from one backend pass."""

    def __init__(
        self,
        *,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Bind one physical-FPS sub-metric name and shared Pulse runtime options."""
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the Pulse-of-Motion temporal calibration backend."""
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
        """Return one cached Pulse FPS metric for the prediction video."""
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
