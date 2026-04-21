from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from worldarena.benchmark_core.media import LoadedMedia
from worldarena.benchmark_core.metrics.base import (
    Metric,
    clamp01,
    cosine_similarity,
    normalize_score,
    resolve_metric_backend,
)
from worldarena.benchmark_core.reference_runtime import (
    compute_reference_prompt_alignment,
)
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


PROMPT_ALIGNMENT_BACKENDS = {"reference", "clip", "reference_proxy"}
IMAGE_ALIGNMENT_BACKENDS = {"clip", "reference_proxy"}
DISABLED_ALIGNMENT_BACKENDS = {"reference_proxy"}
WEAK_SUPERVISION_WARNING = (
    "prompt alignment is diagnostic-only because the sample is self-conditioned "
    "or lacks explicit instruction supervision"
)


class AlignmentMetric(Metric):
    def __init__(self, name: str, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__(name, backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="clip",
            supported_backends=IMAGE_ALIGNMENT_BACKENDS,
            disabled_backends=DISABLED_ALIGNMENT_BACKENDS,
        )

    def _protocol_details(self, prediction: LoadedMedia) -> dict[str, Any]:
        return {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }


class PromptAlignmentMetric(AlignmentMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("prompt_alignment", backend, normalization)

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="reference",
            supported_backends=PROMPT_ALIGNMENT_BACKENDS,
            disabled_backends=DISABLED_ALIGNMENT_BACKENDS,
        )

    def _resolve_eligibility(self, sample: BenchmarkSample) -> tuple[str, str | None]:
        if not sample.has_instruction or sample.conditioning_strategy == "self":
            return "weak_supervision", WEAK_SUPERVISION_WARNING
        return "official", None

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = self._resolve_backend()
        eligibility_status, supervision_warning = self._resolve_eligibility(sample)
        try:
            details = self._protocol_details(prediction)
            if backend == "reference":
                reference_result = compute_reference_prompt_alignment(
                    prediction.anchor_frames,
                    sample.prompt_target,
                )
                raw = reference_result["raw"]
                details.update(reference_result["details"])
                if supervision_warning:
                    details["supervision_warning"] = supervision_warning
                return MetricOutput(
                    raw=raw,
                    normalized=normalize_score(raw, self.normalization),
                    backend=str(reference_result["backend"]),
                    details=details,
                    eligibility_status=eligibility_status,
                )
            scores = [
                _clip_image_text_similarity(frame, sample.prompt_target)
                for frame in prediction.anchor_frames
            ]
            raw = float(np.mean(scores)) if scores else None
            details["frame_scores"] = [round(value, 4) for value in scores]
            if supervision_warning:
                details["supervision_warning"] = supervision_warning
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status=eligibility_status,
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            details = self._protocol_details(prediction)
            if supervision_warning:
                details["supervision_warning"] = supervision_warning
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status=eligibility_status,
                error=str(exc),
            )


class ReferenceAppearanceAlignmentMetric(AlignmentMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("reference_appearance_alignment", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample

        backend = self._resolve_backend()
        try:
            paired = list(zip(prediction.anchor_frames, reference.anchor_frames))
            scores = [_clip_image_image_similarity(left, right) for left, right in paired]
            raw = float(np.mean(scores)) if scores else None
            details = self._protocol_details(prediction)
            details["frame_scores"] = [round(value, 4) for value in scores]
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=self._protocol_details(prediction),
                eligibility_status="diagnostic",
                error=str(exc),
            )


class AnchorConsistencyMetric(AlignmentMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("anchor_consistency", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample

        backend = self._resolve_backend()
        try:
            paired = list(zip(prediction.anchor_frames, reference.anchor_frames))
            scores = [_clip_image_image_similarity(left, right) for left, right in paired]
            raw = float(np.mean(scores)) if scores else None
            details = self._protocol_details(prediction)
            details["frame_scores"] = [round(value, 4) for value in scores]
            details["late_minus_early"] = round(scores[-1] - scores[0], 4) if scores else None
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=self._protocol_details(prediction),
                eligibility_status="diagnostic",
                error=str(exc),
            )


class SceneAlignmentMetric(PromptAlignmentMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__(backend=backend, normalization=normalization)
        self.name = "scene_alignment"


class DynamicAlignmentMetric(PromptAlignmentMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__(backend=backend, normalization=normalization)
        self.name = "dynamic_alignment"


class RolloutQualityMetric(AnchorConsistencyMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__(backend=backend, normalization=normalization)
        self.name = "rollout_quality"
