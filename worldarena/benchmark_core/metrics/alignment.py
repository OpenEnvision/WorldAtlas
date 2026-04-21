from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from worldarena.benchmark_core.media import LoadedMedia
from worldarena.benchmark_core.metrics.base import Metric, clamp01, cosine_similarity, normalize_score, rgb_histogram
from worldarena.benchmark_core.schemas import BenchmarkSample, MetricOutput


@lru_cache(maxsize=1)
def _load_clip_backend() -> tuple[Any, Any, Any] | None:
    try:
        import torch
        from transformers import AutoProcessor, CLIPModel
    except ModuleNotFoundError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model.eval()
    return model, processor, device


def _clip_image_text_similarity(frame: np.ndarray, text: str) -> float:
    backend = _load_clip_backend()
    if backend is None:
        raise ModuleNotFoundError("transformers/torch not available")

    import torch

    model, processor, device = backend
    inputs = processor(
        text=[text],
        images=[Image.fromarray(frame)],
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    image_vector = image_features[0].detach().cpu().numpy()
    text_vector = text_features[0].detach().cpu().numpy()
    return clamp01((cosine_similarity(image_vector, text_vector) + 1.0) / 2.0)


def _clip_image_image_similarity(left: np.ndarray, right: np.ndarray) -> float:
    backend = _load_clip_backend()
    if backend is None:
        raise ModuleNotFoundError("transformers/torch not available")

    import torch

    model, processor, device = backend
    inputs = processor(
        images=[Image.fromarray(left), Image.fromarray(right)],
        return_tensors="pt",
        padding=True,
    )
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=pixel_values)
    left_vector = image_features[0].detach().cpu().numpy()
    right_vector = image_features[1].detach().cpu().numpy()
    return clamp01((cosine_similarity(left_vector, right_vector) + 1.0) / 2.0)


def _reference_proxy_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return clamp01((cosine_similarity(rgb_histogram(left), rgb_histogram(right)) + 1.0) / 2.0)


class AlignmentMetric(Metric):
    def _resolve_backend(self) -> str:
        if self.backend == "auto":
            return "clip" if _load_clip_backend() is not None else "reference_proxy"
        return self.backend


class SceneAlignmentMetric(AlignmentMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("scene_alignment", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = self._resolve_backend()
        try:
            if backend == "clip":
                scores = [
                    _clip_image_text_similarity(frame, sample.prompt_target)
                    for frame in prediction.anchor_frames
                ]
            else:
                paired = zip(prediction.anchor_frames, reference.anchor_frames)
                scores = [_reference_proxy_similarity(left, right) for left, right in paired]
            raw = float(np.mean(scores)) if scores else None
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details={"frame_scores": [round(value, 4) for value in scores]},
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                error=str(exc),
            )


class DynamicAlignmentMetric(AlignmentMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("dynamic_alignment", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        return SceneAlignmentMetric(backend=self.backend, normalization=self.normalization).compute(
            sample=sample,
            prediction=prediction,
            reference=reference,
        )


class RolloutQualityMetric(AlignmentMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("rollout_quality", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = self._resolve_backend()
        try:
            paired = list(zip(prediction.anchor_frames, reference.anchor_frames))
            if backend == "clip":
                scores = [_clip_image_image_similarity(left, right) for left, right in paired]
            else:
                scores = [_reference_proxy_similarity(left, right) for left, right in paired]
            raw = float(np.mean(scores)) if scores else None
            decay = None
            if scores:
                decay = round(scores[-1] - scores[0], 4)
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details={
                    "frame_scores": [round(value, 4) for value in scores],
                    "late_minus_early": decay,
                },
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                error=str(exc),
            )
