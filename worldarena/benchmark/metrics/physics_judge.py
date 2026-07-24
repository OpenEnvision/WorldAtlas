"""LLM-judge physics metric implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.physics_judge_backend import (
    PHYSICS_JUDGE_METRICS,
    evaluate_physics_judge,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


PHYSICS_JUDGE_BACKENDS = {"openai_judge", "gemini_judge"}
_PHYSICS_JUDGE_RESULT_CACHE: dict[tuple[str, str, int, int, str, str], dict[str, Any]] = {}


def _cache_key(
    *,
    sample: BenchmarkSample,
    prediction: LoadedMedia,
    backend: str,
    runtime: dict[str, Any],
) -> tuple[str, str, int, int, str, str]:
    """Key judge calls by sample, video identity, backend, and runtime fingerprint."""
    path = Path(prediction.path)
    stat = path.stat()
    runtime_token = json.dumps(runtime, sort_keys=True, default=str)
    return sample.sample_id, str(path.resolve()), stat.st_mtime_ns, stat.st_size, backend, runtime_token


class PhysicsJudgeMetric(Metric):
    """Expose one LLM physics-judge sub-score from a shared provider evaluation."""

    def __init__(self, metric_name: str, backend: str, normalization: dict[str, Any], runtime: dict[str, Any]) -> None:
        """Bind a judge sub-metric name with provider runtime and normalization."""
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime)

    def _resolve_backend(self) -> str:
        """Select OpenAI or Gemini judge backend, honoring runtime default_backend."""
        auto_backend = str(self.runtime.get("default_backend", "openai_judge"))
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend=auto_backend,
            supported_backends=PHYSICS_JUDGE_BACKENDS,
        )

    def _payload(
        self,
        *,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
        backend: str,
    ) -> dict[str, Any]:
        """Call the physics judge once and cache the full multi-axis response."""
        key = _cache_key(sample=sample, prediction=prediction, backend=backend, runtime=self.runtime)
        cached = _PHYSICS_JUDGE_RESULT_CACHE.get(key)
        if cached is not None:
            return cached
        payload = evaluate_physics_judge(
            sample=sample,
            prediction=prediction,
            reference=reference,
            backend=backend,
            runtime=self.runtime,
        )
        _PHYSICS_JUDGE_RESULT_CACHE[key] = payload
        return payload

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Return one physics-judge axis score when physics_spec metadata is present."""
        if not sample.physics_spec:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=self._resolve_backend(),
                details={"reason": "sample does not define physics_spec"},
                error="physics judge metadata missing from sample",
            )

        backend = self._resolve_backend()
        payload = self._payload(sample=sample, prediction=prediction, reference=reference, backend=backend)
        raw_value = payload.get(self.name)
        raw = float(raw_value) if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool) else None
        details = {
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "judge_card": payload.get("judge_card"),
            "confidence": payload.get("confidence"),
            "failures": payload.get("failures", []),
            "reasoning": payload.get("reasoning", ""),
            "cache_hit": payload.get("cache_hit", False),
            "success": payload.get("success", False),
        }
        if "provider_response_text" in payload:
            details["provider_response_text"] = payload["provider_response_text"]
        return MetricOutput(
            raw=raw,
            normalized=normalize_score(raw, self.normalization),
            backend=backend,
            details=details,
            error=payload.get("error"),
        )


__all__ = ["PHYSICS_JUDGE_METRICS", "PhysicsJudgeMetric"]
