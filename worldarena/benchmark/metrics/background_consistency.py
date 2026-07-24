"""Prediction-only background consistency via consecutive-frame CLIP cosine similarity."""

from __future__ import annotations

from typing import Any

import numpy as np

from worldarena.benchmark.clip_backend import clip_image_features
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, clamp01, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput

BACKGROUND_CONSISTENCY_BACKENDS = {"clip_vit_b32"}
DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def _consecutive_cosine_similarities(features: np.ndarray) -> list[float]:
    """Compute cosine similarity between each adjacent feature pair."""
    if len(features) < 2:
        return []
    return [float(np.dot(features[index], features[index + 1])) for index in range(len(features) - 1)]


def compute_background_consistency_score(
    frames: list[np.ndarray],
    *,
    model_name: str = DEFAULT_CLIP_MODEL_NAME,
) -> dict[str, Any]:
    """Compute mean consecutive-frame CLIP cosine similarity."""
    if len(frames) < 2:
        raise ValueError("not enough frames for background consistency metric")

    features, loader_name = clip_image_features(frames, model_name=model_name)
    pair_similarities = _consecutive_cosine_similarities(features)
    raw = clamp01(float(np.mean(pair_similarities)))
    return {
        "raw": raw,
        "clip_loader": loader_name,
        "clip_model_name": model_name,
        "details": {
            "frame_pair_count": len(pair_similarities),
            "mean_pairwise_cosine": round(raw, 4),
            "pairwise_cosine": [round(value, 4) for value in pair_similarities],
        },
    }


class BackgroundConsistencyMetric(Metric):
    """Score scene/background stability from consecutive-frame CLIP embeddings."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Store CLIP backend options for background consistency."""
        super().__init__("background_consistency", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the CLIP ViT-B/32 background-consistency backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="clip_vit_b32",
            supported_backends=BACKGROUND_CONSISTENCY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Score prediction-only background consistency on protocol-sampled frames."""
        del sample
        del reference

        backend = self._resolve_backend()
        frames = list(prediction.frames or prediction.anchor_frames)
        model_name = str(self.runtime.get("model_name", DEFAULT_CLIP_MODEL_NAME))
        details = {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
            "clip_model_name": model_name,
        }
        try:
            result = compute_background_consistency_score(frames, model_name=model_name)
            details.update(result["details"])
            details["clip_loader"] = result["clip_loader"]
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
            )
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


__all__ = ["BackgroundConsistencyMetric", "compute_background_consistency_score"]
