from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import cv2
import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, clamp01, cosine_similarity, normalize_score, resolve_metric_backend
from worldarena.benchmark.quality_backends import (
    _configure_pyiqa_cache,
    _frame_to_float_tensor,
    _prepare_quality_runtime_compat,
    _quality_device,
    _to_float,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


BRIGHTNESS_BACKENDS = {"luminance_temporal"}
COLOR_TEMPERATURE_BACKENDS = {"hsv_hue_drift"}
SHARPNESS_BACKENDS = {"tenengrad", "tenengrad_brisque"}
BRIGHTNESS_DISTRIBUTION_BACKENDS = {"distribution"}
COLOR_TEMPERATURE_CONSTRAINT_BACKENDS = {"hue_distribution"}
SHARPNESS_VECTOR_BACKENDS = {"tenengrad_vector"}


def _protocol_details(prediction: LoadedMedia) -> dict[str, Any]:
    return {
        "frame_indices": list(prediction.protocol_frame_indices),
        "sampled_frame_indices": list(prediction.sampled_frame_indices),
    }


def _temporal_frames(prediction: LoadedMedia) -> list[np.ndarray]:
    return list(prediction.frames or prediction.anchor_frames)


def _exp_similarity(error: float, tau: float) -> float:
    return clamp01(float(np.exp(-max(float(error), 0.0) / max(float(tau), 1e-8))))


def _softmax_transform(value: float, lam: float) -> float:
    lam = max(float(lam), 1e-8)
    return clamp01(float((np.exp(lam * clamp01(value)) - 1.0) / (np.exp(lam) - 1.0)))


def _log_transform(value: float, factor: float) -> float:
    factor = max(float(factor), 1e-8)
    return clamp01(float(np.log1p(factor * clamp01(value)) / np.log1p(factor)))


def _normalized_exp_weights(count: int, decay: float) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.float32)
    distances = np.arange(1, count + 1, dtype=np.float32)
    weights = np.exp(-float(decay) * distances)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full((count,), 1.0 / count, dtype=np.float32)
    return (weights / total).astype(np.float32)


def _luminance(frame: np.ndarray) -> float:
    rgb = frame.astype(np.float32) / 255.0
    y = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return float(np.mean(y))


def _brightness_distribution_vector(frame: np.ndarray, dark_threshold: int, bright_threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    total = float(gray.size)
    dark = float(np.count_nonzero(gray < int(dark_threshold))) / total
    bright = float(np.count_nonzero(gray >= int(bright_threshold))) / total
    mid = max(0.0, 1.0 - dark - bright)
    return np.asarray([dark, mid, bright], dtype=np.float32)


def _hue_distribution_vector(frame: np.ndarray, bins: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    counts, _ = np.histogram(hsv[:, :, 0], bins=int(bins), range=(0, 180), weights=saturation)
    vector = counts.astype(np.float32)
    total = float(np.sum(vector))
    if total <= 1e-8:
        return np.full((bins,), 1.0 / bins, dtype=np.float32)
    return vector / total


def _tenengrad_vector(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.asarray([float(np.sum(np.abs(grad_x))), float(np.sum(np.abs(grad_y)))], dtype=np.float32)


def compute_brightness_consistency(
    frames: Sequence[np.ndarray],
    *,
    absolute_tau: float = 0.20,
    relative_tau: float = 0.35,
    min_luminance: float = 0.02,
) -> dict[str, Any]:
    if len(frames) < 2:
        raise ValueError("brightness_consistency requires at least two frames")

    luminance = [_luminance(frame) for frame in frames]
    absolute_deltas = [
        abs(luminance[index + 1] - luminance[index])
        for index in range(len(luminance) - 1)
    ]
    relative_deltas = [
        absolute_deltas[index]
        / max(luminance[index], luminance[index + 1], float(min_luminance), 1e-8)
        for index in range(len(absolute_deltas))
    ]
    mean_absolute_delta = float(np.mean(absolute_deltas))
    mean_relative_delta = float(np.mean(relative_deltas))
    absolute_score = _exp_similarity(mean_absolute_delta, absolute_tau)
    relative_score = _exp_similarity(mean_relative_delta, relative_tau)
    raw = float(np.sqrt(absolute_score * relative_score))
    return {
        "raw": raw,
        "details": {
            "frame_luminance": [round(value, 6) for value in luminance],
            "absolute_deltas": [round(value, 6) for value in absolute_deltas],
            "relative_deltas": [round(value, 6) for value in relative_deltas],
            "mean_absolute_delta": round(mean_absolute_delta, 6),
            "mean_relative_delta": round(mean_relative_delta, 6),
            "absolute_tau": float(absolute_tau),
            "relative_tau": float(relative_tau),
            "min_luminance": float(min_luminance),
        },
    }


def compute_brightness_distribution_consistency(
    frames: Sequence[np.ndarray],
    *,
    dark_threshold: int = 85,
    bright_threshold: int = 170,
    temporal_decay: float = 0.05,
    softmax_lambda: float = 4.0,
) -> dict[str, Any]:
    if len(frames) < 2:
        raise ValueError("brightness_distribution_consistency requires at least two frames")

    vectors = [
        _brightness_distribution_vector(frame, dark_threshold, bright_threshold)
        for frame in frames
    ]
    similarities = [clamp01(cosine_similarity(vector, vectors[0])) for vector in vectors[1:]]
    transformed = [_softmax_transform(value, softmax_lambda) for value in similarities]
    weights = _normalized_exp_weights(len(transformed), temporal_decay)
    raw = float(np.sum(weights * np.asarray(transformed, dtype=np.float32)))
    return {
        "raw": raw,
        "details": {
            "brightness_vectors": [[round(float(item), 6) for item in vector] for vector in vectors],
            "first_frame_similarities": [round(value, 6) for value in similarities],
            "transformed_similarities": [round(value, 6) for value in transformed],
            "temporal_weights": [round(float(value), 6) for value in weights],
            "dark_threshold": int(dark_threshold),
            "bright_threshold": int(bright_threshold),
            "temporal_decay": float(temporal_decay),
            "softmax_lambda": float(softmax_lambda),
        },
    }


def _weighted_hue(frame: np.ndarray, *, saturation_power: float) -> tuple[float, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * (2.0 * np.pi / 180.0)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    weights = np.power(saturation, float(saturation_power)) * value
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-8:
        return 0.0, 0.0
    x = float(np.sum(weights * np.cos(hue)) / weight_sum)
    y = float(np.sum(weights * np.sin(hue)) / weight_sum)
    angle = float(np.mod(np.arctan2(y, x), 2.0 * np.pi))
    reliability = float(np.mean(weights))
    return angle, reliability


def _circular_distance(left: float, right: float) -> float:
    delta = abs(float(left) - float(right))
    delta = min(delta, 2.0 * np.pi - delta)
    return float(delta / np.pi)


def compute_color_temperature_consistency(
    frames: Sequence[np.ndarray],
    *,
    hue_tau: float = 0.12,
    saturation_power: float = 1.0,
    min_saturation_weight: float = 0.0001,
) -> dict[str, Any]:
    if len(frames) < 2:
        raise ValueError("color_temperature_consistency requires at least two frames")

    hues: list[float] = []
    reliabilities: list[float] = []
    for frame in frames:
        hue, reliability = _weighted_hue(frame, saturation_power=saturation_power)
        hues.append(hue)
        reliabilities.append(reliability)

    pair_distances: list[float] = []
    pair_weights: list[float] = []
    for index in range(len(hues) - 1):
        pair_weight = min(reliabilities[index], reliabilities[index + 1])
        if pair_weight >= float(min_saturation_weight):
            pair_distances.append(_circular_distance(hues[index], hues[index + 1]))
            pair_weights.append(pair_weight)

    if not pair_distances:
        raise ValueError(
            "color_temperature_consistency has no sufficiently saturated frame pairs"
        )

    mean_hue_distance = float(np.average(pair_distances, weights=pair_weights))
    raw = _exp_similarity(mean_hue_distance, hue_tau)
    return {
        "raw": raw,
        "details": {
            "frame_hues_degrees": [round(float(np.degrees(value)), 4) for value in hues],
            "frame_hue_reliability": [round(value, 6) for value in reliabilities],
            "pair_hue_distances": [round(value, 6) for value in pair_distances],
            "pair_weights": [round(value, 6) for value in pair_weights],
            "mean_hue_distance": round(mean_hue_distance, 6),
            "hue_tau": float(hue_tau),
            "saturation_power": float(saturation_power),
            "min_saturation_weight": float(min_saturation_weight),
        },
    }


def compute_color_temperature_constraint(
    frames: Sequence[np.ndarray],
    *,
    hue_bins: int = 7,
    temporal_decay: float = 0.10,
    softmax_lambda: float = 4.0,
) -> dict[str, Any]:
    if len(frames) < 2:
        raise ValueError("color_temperature_constraint requires at least two frames")

    vectors = [_hue_distribution_vector(frame, hue_bins) for frame in frames]
    averaged_similarities: list[float] = []
    for index in range(1, len(vectors)):
        first_similarity = clamp01(cosine_similarity(vectors[index], vectors[0]))
        previous_similarity = clamp01(cosine_similarity(vectors[index], vectors[index - 1]))
        averaged_similarities.append((first_similarity + previous_similarity) * 0.5)

    transformed = [_softmax_transform(value, softmax_lambda) for value in averaged_similarities]
    weights = _normalized_exp_weights(len(transformed), temporal_decay)
    raw = float(np.sum(weights * np.asarray(transformed, dtype=np.float32)))
    return {
        "raw": raw,
        "details": {
            "hue_vectors": [[round(float(item), 6) for item in vector] for vector in vectors],
            "averaged_similarities": [round(value, 6) for value in averaged_similarities],
            "transformed_similarities": [round(value, 6) for value in transformed],
            "temporal_weights": [round(float(value), 6) for value in weights],
            "hue_bins": int(hue_bins),
            "temporal_decay": float(temporal_decay),
            "softmax_lambda": float(softmax_lambda),
        },
    }


def _tenengrad(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(grad_x * grad_x + grad_y * grad_y))


@lru_cache(maxsize=1)
def _load_brisque_backend() -> tuple[Any, str] | None:
    try:
        _prepare_quality_runtime_compat(enable_clip_cache_compat=False)
        import pyiqa
    except (ImportError, ModuleNotFoundError):
        return None

    _configure_pyiqa_cache()
    device = _quality_device()
    metric = pyiqa.create_metric("brisque").to(device)
    metric.eval()
    return metric, device


def _brisque_frame_scores(frames: Sequence[np.ndarray]) -> list[float]:
    backend = _load_brisque_backend()
    if backend is None:
        raise ModuleNotFoundError("pyiqa BRISQUE backend is unavailable")
    metric, device = backend
    scores: list[float] = []
    for frame in frames:
        score = _to_float(metric(_frame_to_float_tensor(frame, device)))
        if not np.isfinite(score):
            raise ValueError("pyiqa BRISQUE produced a non-finite score")
        scores.append(score)
    return scores


def compute_sharpness_retention(
    frames: Sequence[np.ndarray],
    *,
    use_brisque_gate: bool,
    baseline_frame_count: int = 2,
    texture_floor: float = 0.00001,
    brisque_good: float = 30.0,
    brisque_bad: float = 80.0,
) -> dict[str, Any]:
    if len(frames) < 2:
        raise ValueError("sharpness_retention requires at least two frames")

    tenengrad_scores = [_tenengrad(frame) for frame in frames]
    baseline_count = max(1, min(int(baseline_frame_count), len(tenengrad_scores) - 1))
    baseline = float(np.median(tenengrad_scores[:baseline_count]))
    if baseline <= float(texture_floor):
        raise ValueError(
            "sharpness_retention baseline texture is below the configured texture floor"
        )

    retention_scores = [
        clamp01(score / baseline)
        for score in tenengrad_scores[baseline_count:]
    ]
    tenengrad_retention = float(np.mean(retention_scores)) if retention_scores else 1.0
    brisque_scores: list[float] = []
    brisque_gate = 1.0
    if use_brisque_gate:
        brisque_scores = _brisque_frame_scores(frames)
        if not all(np.isfinite(score) for score in brisque_scores):
            raise ValueError("BRISQUE gate received non-finite scores")
        mean_brisque = float(np.mean(brisque_scores))
        brisque_gate = clamp01((float(brisque_bad) - mean_brisque) / max(float(brisque_bad) - float(brisque_good), 1e-8))
    raw = clamp01(tenengrad_retention * brisque_gate)
    return {
        "raw": raw,
        "details": {
            "tenengrad_scores": [round(value, 8) for value in tenengrad_scores],
            "baseline_frame_count": baseline_count,
            "baseline_tenengrad": round(baseline, 8),
            "retention_scores": [round(value, 6) for value in retention_scores],
            "tenengrad_retention": round(tenengrad_retention, 6),
            "texture_floor": float(texture_floor),
            "brisque_gate_enabled": bool(use_brisque_gate),
            "brisque_scores": [round(value, 4) for value in brisque_scores],
            "brisque_gate": round(brisque_gate, 6),
            "brisque_good": float(brisque_good),
            "brisque_bad": float(brisque_bad),
        },
    }


def compute_sharpness_vector_retention(
    frames: Sequence[np.ndarray],
    *,
    log_factor: float = 12.0,
    temporal_decay: float = 0.05,
) -> dict[str, Any]:
    if len(frames) < 2:
        raise ValueError("sharpness_vector_retention requires at least two frames")

    vectors = [_tenengrad_vector(frame) for frame in frames]
    similarities = [clamp01(cosine_similarity(vector, vectors[0])) for vector in vectors[1:]]
    transformed = [_log_transform(value, log_factor) for value in similarities]
    weights = _normalized_exp_weights(len(transformed), temporal_decay)
    raw = float(np.sum(weights * np.asarray(transformed, dtype=np.float32)))
    return {
        "raw": raw,
        "details": {
            "tenengrad_vectors": [[round(float(item), 6) for item in vector] for vector in vectors],
            "first_frame_similarities": [round(value, 6) for value in similarities],
            "transformed_similarities": [round(value, 6) for value in transformed],
            "temporal_weights": [round(float(value), 6) for value in weights],
            "log_factor": float(log_factor),
            "temporal_decay": float(temporal_decay),
        },
    }


class BrightnessConsistencyMetric(Metric):
    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("brightness_consistency", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="luminance_temporal",
            supported_backends=BRIGHTNESS_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample
        del reference
        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        try:
            result = compute_brightness_consistency(
                _temporal_frames(prediction),
                absolute_tau=float(self.runtime.get("absolute_tau", 0.20)),
                relative_tau=float(self.runtime.get("relative_tau", 0.35)),
                min_luminance=float(self.runtime.get("min_luminance", 0.02)),
            )
            details.update(result["details"])
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


class BrightnessDistributionConsistencyMetric(Metric):
    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("brightness_distribution_consistency", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="distribution",
            supported_backends=BRIGHTNESS_DISTRIBUTION_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample
        del reference
        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        try:
            result = compute_brightness_distribution_consistency(
                _temporal_frames(prediction),
                dark_threshold=int(self.runtime.get("dark_threshold", 85)),
                bright_threshold=int(self.runtime.get("bright_threshold", 170)),
                temporal_decay=float(self.runtime.get("temporal_decay", 0.05)),
                softmax_lambda=float(self.runtime.get("softmax_lambda", 4.0)),
            )
            details.update(result["details"])
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


class ColorTemperatureConsistencyMetric(Metric):
    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("color_temperature_consistency", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="hsv_hue_drift",
            supported_backends=COLOR_TEMPERATURE_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample
        del reference
        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        try:
            result = compute_color_temperature_consistency(
                _temporal_frames(prediction),
                hue_tau=float(self.runtime.get("hue_tau", 0.12)),
                saturation_power=float(self.runtime.get("saturation_power", 1.0)),
                min_saturation_weight=float(self.runtime.get("min_saturation_weight", 0.0001)),
            )
            details.update(result["details"])
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


class ColorTemperatureConstraintMetric(Metric):
    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("color_temperature_constraint", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="hue_distribution",
            supported_backends=COLOR_TEMPERATURE_CONSTRAINT_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample
        del reference
        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        try:
            result = compute_color_temperature_constraint(
                _temporal_frames(prediction),
                hue_bins=int(self.runtime.get("hue_bins", 7)),
                temporal_decay=float(self.runtime.get("temporal_decay", 0.10)),
                softmax_lambda=float(self.runtime.get("softmax_lambda", 4.0)),
            )
            details.update(result["details"])
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


class SharpnessRetentionMetric(Metric):
    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("sharpness_retention", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="tenengrad_brisque",
            supported_backends=SHARPNESS_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample
        del reference
        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        try:
            result = compute_sharpness_retention(
                _temporal_frames(prediction),
                use_brisque_gate=backend == "tenengrad_brisque",
                baseline_frame_count=int(self.runtime.get("baseline_frame_count", 2)),
                texture_floor=float(self.runtime.get("texture_floor", 0.00001)),
                brisque_good=float(self.runtime.get("brisque_good", 30.0)),
                brisque_bad=float(self.runtime.get("brisque_bad", 80.0)),
            )
            details.update(result["details"])
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


class SharpnessVectorRetentionMetric(Metric):
    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("sharpness_vector_retention", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="tenengrad_vector",
            supported_backends=SHARPNESS_VECTOR_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample
        del reference
        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        try:
            result = compute_sharpness_vector_retention(
                _temporal_frames(prediction),
                log_factor=float(self.runtime.get("log_factor", 12.0)),
                temporal_decay=float(self.runtime.get("temporal_decay", 0.05)),
            )
            details.update(result["details"])
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


__all__ = [
    "BrightnessConsistencyMetric",
    "BrightnessDistributionConsistencyMetric",
    "ColorTemperatureConstraintMetric",
    "ColorTemperatureConsistencyMetric",
    "SharpnessRetentionMetric",
    "SharpnessVectorRetentionMetric",
    "compute_brightness_consistency",
    "compute_brightness_distribution_consistency",
    "compute_color_temperature_consistency",
    "compute_color_temperature_constraint",
    "compute_sharpness_retention",
    "compute_sharpness_vector_retention",
]
