from __future__ import annotations

from typing import Any

import numpy as np

from worldarena.benchmark.consistency_backends import dinov2_image_features
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, clamp01, cosine_similarity, normalize_score, resolve_metric_backend
from worldarena.benchmark.metrics.legacy import _resize_like, _ssim_rgb
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


MEMORY_REVISIT_BACKENDS = {"dinov2"}


def _protocol_details(prediction: LoadedMedia) -> dict[str, Any]:
    return {
        "frame_indices": list(prediction.protocol_frame_indices),
        "sampled_frame_indices": list(prediction.sampled_frame_indices),
    }


def _feature_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return clamp01((cosine_similarity(left, right) + 1.0) / 2.0)


def _mse_similarity(left: np.ndarray, right: np.ndarray, *, tau: float) -> tuple[float, float]:
    left_float = left.astype(np.float32) / 255.0
    right_float = right.astype(np.float32) / 255.0
    mse = float(np.mean((left_float - right_float) ** 2))
    return mse, 1.0 - clamp01(mse / max(float(tau), 1e-6))


class MemoryRevisitConsistencyMetric(Metric):
    """Score whether a 360-degree revisit returns to the same generated scene."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("memory_revisit_consistency", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="dinov2",
            supported_backends=MEMORY_REVISIT_BACKENDS,
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
        frames = prediction.frames
        if len(frames) < 2:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error="not enough frames for memory revisit consistency",
            )

        try:
            start = frames[0]
            end = _resize_like(frames[-1], start)
            features = dinov2_image_features(frames)

            dino_revisit = _feature_similarity(features[0], features[-1])
            ssim_revisit = _ssim_rgb(start, end)
            mse, mse_revisit = _mse_similarity(
                start,
                end,
                tau=float(self.runtime.get("mse_tau", 0.08)),
            )

            closure_weights = dict(self.runtime.get("closure_weights", {}))
            dino_weight = float(closure_weights.get("dino", 0.55))
            ssim_weight = float(closure_weights.get("ssim", 0.25))
            mse_weight = float(closure_weights.get("mse", 0.20))
            weight_sum = max(dino_weight + ssim_weight + mse_weight, 1e-6)
            closure_score = (
                dino_weight * dino_revisit
                + ssim_weight * ssim_revisit
                + mse_weight * mse_revisit
            ) / weight_sum

            quarter_indices = sorted(
                {
                    int(round((len(frames) - 1) * fraction))
                    for fraction in (0.25, 0.5, 0.75)
                }
            )
            quarter_similarities = [
                _feature_similarity(features[0], features[index])
                for index in quarter_indices
                if 0 < index < len(frames) - 1
            ]
            mean_quarter_displacement = (
                float(np.mean([1.0 - value for value in quarter_similarities]))
                if quarter_similarities
                else 0.0
            )
            lower = float(self.runtime.get("progress_lower", 0.03))
            upper = float(self.runtime.get("progress_upper", 0.18))
            progress_score = clamp01((mean_quarter_displacement - lower) / max(upper - lower, 1e-6))
            progress_gate_weight = clamp01(float(self.runtime.get("progress_gate_weight", 0.35)))
            raw = closure_score * ((1.0 - progress_gate_weight) + progress_gate_weight * progress_score)

            details.update(
                {
                    "dino_revisit": round(dino_revisit, 4),
                    "ssim_revisit": round(ssim_revisit, 4),
                    "mse": round(mse, 6),
                    "mse_revisit": round(mse_revisit, 4),
                    "closure_score": round(float(closure_score), 4),
                    "quarter_indices": quarter_indices,
                    "quarter_dino_similarities": [round(value, 4) for value in quarter_similarities],
                    "mean_quarter_displacement": round(mean_quarter_displacement, 4),
                    "progress_score": round(progress_score, 4),
                    "progress_gate_weight": progress_gate_weight,
                }
            )
            return MetricOutput(
                raw=float(raw),
                normalized=normalize_score(float(raw), self.normalization) if self.normalization else float(raw),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


__all__ = ["MemoryRevisitConsistencyMetric"]
