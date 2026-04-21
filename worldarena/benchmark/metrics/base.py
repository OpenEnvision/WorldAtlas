from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_score(value: float | None, spec: dict[str, Any]) -> float | None:
    if value is None:
        return None
    lower = float(spec.get("lower", 0.0))
    upper = float(spec.get("upper", 1.0))
    higher_is_better = bool(spec.get("higher_is_better", True))
    if np.isclose(upper, lower):
        return clamp01(value)
    if higher_is_better:
        scaled = (float(value) - lower) / (upper - lower)
    else:
        scaled = (upper - float(value)) / (upper - lower)
    return clamp01(scaled)


def resolve_metric_backend(
    *,
    metric_name: str,
    configured_backend: str,
    auto_backend: str,
    supported_backends: set[str],
    disabled_backends: set[str] | None = None,
) -> str:
    if configured_backend == "auto":
        return auto_backend

    if disabled_backends and configured_backend in disabled_backends:
        raise ValueError(
            f"{metric_name} backend '{configured_backend}' is disabled because fallback "
            "metrics are not allowed"
        )

    if configured_backend not in supported_backends:
        supported = ", ".join(sorted({"auto", *supported_backends}))
        raise ValueError(
            f"unsupported backend '{configured_backend}' for {metric_name}; "
            f"supported values: {supported}"
        )

    return configured_backend


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def rgb_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    histograms = []
    for channel in range(frame.shape[2]):
        counts, _ = np.histogram(frame[:, :, channel], bins=bins, range=(0, 255), density=True)
        histograms.append(counts.astype(np.float32))
    return np.concatenate(histograms, axis=0)


class Metric(ABC):
    name: str

    def __init__(self, name: str, backend: str, normalization: dict[str, Any]) -> None:
        self.name = name
        self.backend = backend
        self.normalization = normalization

    @abstractmethod
    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        raise NotImplementedError
