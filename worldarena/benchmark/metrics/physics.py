"""Physics plausibility metric implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.physics_backend import evaluate_generated_video
from worldarena.benchmark.physics_manifest import (
    OPTIONAL_PHYSICS_METRICS,
    REQUIRED_CORE_METRICS,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


PHYSICS_METRIC_NAMES: set[str] = set(REQUIRED_CORE_METRICS) | set(OPTIONAL_PHYSICS_METRICS)
PHYSICS_METRIC_BACKENDS = {"simulator_reference"}
_PHYSICS_RESULT_CACHE: dict[tuple[str, str, int, int], dict[str, Any]] = {}


def is_physics_metric(metric_name: str) -> bool:
    """Return whether a metric name belongs to the physics simulator contract."""
    return metric_name in PHYSICS_METRIC_NAMES


def _cache_key(sample: BenchmarkSample, prediction_path: str) -> tuple[str, str, int, int]:
    """Key physics simulator results by sample id and prediction file identity."""
    path = Path(prediction_path)
    stat = path.stat()
    return sample.sample_id, str(path.resolve()), stat.st_mtime_ns, stat.st_size


def _physics_result(sample: BenchmarkSample, prediction: LoadedMedia) -> dict[str, Any]:
    """Run the physics simulator once per sample/video pair and cache all sub-metrics."""
    key = _cache_key(sample, prediction.path)
    cached = _PHYSICS_RESULT_CACHE.get(key)
    if cached is not None:
        return cached
    payload = evaluate_generated_video(
        sample=sample,
        generated_video_path=prediction.path,
    )
    _PHYSICS_RESULT_CACHE[key] = payload
    return payload


class PhysicsMetric(Metric):
    """Expose one physics-simulator sub-metric from a shared evaluation pass."""

    def __init__(self, metric_name: str, backend: str, normalization: dict[str, Any]) -> None:
        """Register a named physics simulator metric for later extraction."""
        super().__init__(metric_name, backend, normalization)

    def _resolve_backend(self) -> str:
        """Require the simulator-reference physics evaluation backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="simulator_reference",
            supported_backends=PHYSICS_METRIC_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Return one cached physics simulator metric when physics_case metadata exists."""
        del reference

        if not sample.physics_case:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=self._resolve_backend(),
                details={"reason": "sample does not define physics_case"},
                error="physics metadata missing from sample",
            )

        backend = self._resolve_backend()
        try:
            payload = _physics_result(sample, prediction)
        except Exception as exc:  # pragma: no cover - subprocess and env failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details={"prediction_path": prediction.path},
                error=str(exc),
            )

        raw_value = payload.get(self.name)
        not_applicable = dict(payload.get("not_applicable_metrics") or {})
        metric_errors = dict(payload.get("simulator_metric_errors") or {})
        raw = float(raw_value) if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool) else None
        details = {
            "prediction_path": prediction.path,
            "report_path": payload.get("report_path"),
            "world_gt_root": payload.get("world_gt_root"),
            "physics_status": payload.get("physics_status"),
            "pred_world_status": payload.get("pred_world_status"),
            "world_gt_status": payload.get("world_gt_status"),
            "metric_contract": payload.get("metric_contract"),
            "missing_world_artifacts": payload.get("missing_world_artifacts", []),
            "success": payload.get("success"),
        }
        if "missing_required_metrics" in payload:
            details["missing_required_metrics"] = payload["missing_required_metrics"]
        metric_details = dict(payload.get("simulator_metric_details") or {})
        if self.name in metric_details:
            details["simulator_metric"] = metric_details[self.name]

        if self.name in not_applicable:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="not_applicable",
                error=str(not_applicable[self.name]),
            )

        return MetricOutput(
            raw=raw,
            normalized=normalize_score(raw, self.normalization),
            backend=backend,
            details=details,
            error=metric_errors.get(self.name) or payload.get("error"),
        )
