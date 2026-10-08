"""Base metric types, score normalization, and backend resolution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput


def clamp01(value: float) -> float:
    """Clamp finite measurements only; NaN must not turn into a perfect 1."""
    if not np.isfinite(float(value)):
        raise ValueError('Cannot clamp a non-finite metric value')
    return max(0.0, min(1.0, float(value)))


_LINEAR_BOUND_KEYS = ("lower", "upper", "empirical_min", "empirical_max", "avg")
_ERROR_FROM = {"raw", "complement", "one_plus_inverse"}


def recover_error(value: float, spec: dict[str, Any]) -> float:
    """Recover a non-negative error from a raw metric value."""
    error_from = spec.get("error_from")
    if error_from is None:
        error_from = "raw" if not bool(spec.get("higher_is_better", True)) else "complement"
    if error_from not in _ERROR_FROM:
        supported = ", ".join(sorted(_ERROR_FROM))
        raise ValueError(f"unsupported error_from '{error_from}'; supported values: {supported}")
    raw = float(value)
    if not np.isfinite(raw):
        raise ValueError('Normalization requires finite values and a positive error_ref')
    if error_from == "raw":
        error = raw
    elif error_from == "complement":
        error = 1.0 - raw
    elif raw <= 0.0:
        return float("inf")
    else:
        error = (1.0 - raw) / raw
    return max(0.0, error)


def normalize_score(value: float | None, spec: dict[str, Any]) -> float | None:
    """Scale a raw metric value to [0, 1] using normalization spec."""
    if value is None:
        return None
    if spec.get("error_ref") is not None:
        if any(key in spec for key in _LINEAR_BOUND_KEYS):
            raise ValueError('error_ref cannot be combined with linear or z-score bounds')
        error_ref = float(spec["error_ref"])
        if not np.isfinite(float(value)) or not np.isfinite(error_ref) or error_ref <= 0.0:
            raise ValueError('Normalization requires finite values and a positive error_ref')
        error = recover_error(float(value), spec)
        if not np.isfinite(error):
            return 0.0
        return clamp01(1.0 / (1.0 + error / error_ref))
    lower = float(spec.get("lower", 0.0))
    upper = float(spec.get("upper", 1.0))
    higher_is_better = bool(spec.get("higher_is_better", True))
    if not all(np.isfinite(v) for v in (value, lower, upper)) or upper <= lower:
        raise ValueError('Normalization requires finite values and upper > lower')
    if higher_is_better:
        scaled = (float(value) - lower) / (upper - lower)
    else:
        scaled = (upper - float(value)) / (upper - lower)
    return clamp01(scaled)


def apply_display_error_ref(value: float | None, spec: dict[str, Any]) -> float | None:
    """Apply a frozen error_ref display scale, ignoring linear/empirical companions."""
    if value is None or spec.get("error_ref") is None:
        return value
    display_spec = {
        "error_ref": spec["error_ref"],
        "error_from": spec.get("error_from") or "complement",
        "higher_is_better": True,
    }
    return normalize_score(float(value), display_spec)


def resolve_metric_backend(
    *,
    metric_name: str,
    configured_backend: str,
    auto_backend: str,
    supported_backends: set[str],
    disabled_backends: set[str] | None = None,
) -> str:
    """Resolve which backend to use for a metric."""
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
    """Compute cosine similarity between two vectors."""
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def rgb_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    """Concatenate per-channel normalized RGB histograms into one feature vector."""
    histograms = []
    for channel in range(frame.shape[2]):
        counts, _ = np.histogram(frame[:, :, channel], bins=bins, range=(0, 255), density=True)
        histograms.append(counts.astype(np.float32))
    return np.concatenate(histograms, axis=0)


class Metric(ABC):
    """Abstract benchmark metric with backend selection and score normalization."""

    name: str
    # Metrics must explicitly opt out before the runner can omit reference media.
    requires_reference: bool = True

    def __init__(self, name: str, backend: str, normalization: dict[str, Any]) -> None:
        """Bind metric identity, backend override, and normalization spec."""
        self.name = name
        self.backend = backend
        self.normalization = normalization

    @abstractmethod
    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia | None,
    ) -> MetricOutput:
        """Score one prediction against its reference for a benchmark sample."""
        raise NotImplementedError
