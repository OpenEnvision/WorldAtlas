"""Metric implementations for prompt and semantic alignment."""

from __future__ import annotations

from typing import Any

import numpy as np

from worldarena.benchmark.clip_backend import clip_image_features, clip_text_features
from worldarena.benchmark.media import LoadedMedia, protocol_frame_details
from worldarena.benchmark.metrics.base import (
    Metric,
    clamp01,
    cosine_similarity,
    normalize_score,
    resolve_metric_backend,
)
from worldarena.benchmark.reference_runtime import (
    compute_reference_prompt_alignment,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


def _clip_image_text_similarity(frame: np.ndarray, text: str) -> tuple[float, str]:
    """Map CLIP image–text cosine similarity to a [0, 1] alignment score."""
    image_features, loader_name = clip_image_features([frame])
    text_features, _ = clip_text_features([text])
    score = clamp01((cosine_similarity(image_features[0], text_features[0]) + 1.0) / 2.0)
    return score, loader_name


PROMPT_ALIGNMENT_BACKENDS = {"reference", "clip", "reference_proxy"}
DISABLED_ALIGNMENT_BACKENDS = {"reference_proxy"}
PROMPT_NOT_APPLICABLE_REASON = (
    "prompt_alignment requires explicit instruction supervision and non-self conditioning"
)
PROMPT_INVALID_PREDICTION_REASON = "prompt_alignment requires a video prediction"


class AlignmentMetric(Metric):
    """Shared helpers for prompt- and scene-level alignment metrics."""

    def __init__(self, name: str, backend: str, normalization: dict[str, Any]) -> None:
        """Validate backend choice at construction time."""
        super().__init__(name, backend, normalization)
        self._resolve_backend()

    def _protocol_details(self, prediction: LoadedMedia) -> dict[str, Any]:
        """Attach protocol frame indices and prediction modality to metric details."""
        details = protocol_frame_details(prediction)
        details["prediction_modality"] = prediction.modality
        return details


class PromptAlignmentMetric(AlignmentMetric):
    """Score how well generated video frames match the target instruction."""

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        """Register prompt_alignment with backend validation."""
        super().__init__("prompt_alignment", backend, normalization)

    def _resolve_backend(self) -> str:
        """Pick reference or CLIP backend, rejecting disabled proxy fallbacks."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="reference",
            supported_backends=PROMPT_ALIGNMENT_BACKENDS,
            disabled_backends=DISABLED_ALIGNMENT_BACKENDS,
        )

    def _is_applicable(self, sample: BenchmarkSample) -> bool:
        """Return whether the sample has an external instruction to score against."""
        return bool(sample.has_instruction and sample.conditioning_strategy != "self")

    def _not_applicable_output(
        self,
        *,
        backend: str,
        prediction: LoadedMedia,
        error: str = PROMPT_NOT_APPLICABLE_REASON,
    ) -> MetricOutput:
        """Build a skipped output when prompt alignment cannot be evaluated."""
        return MetricOutput(
            raw=None,
            normalized=None,
            backend=backend,
            details=self._protocol_details(prediction),
            eligibility_status="not_applicable",
            error=error,
        )

    def _resolve_eligibility(self, sample: BenchmarkSample) -> tuple[str, str | None]:
        """Classify whether this sample is officially eligible for prompt alignment."""
        if not sample.has_instruction or sample.conditioning_strategy == "self":
            return "not_applicable", PROMPT_NOT_APPLICABLE_REASON
        return "official", None

    def _invalid_prediction_output(
        self,
        *,
        backend: str,
        prediction: LoadedMedia,
    ) -> MetricOutput:
        """Return an error output when the prediction is not a scorable video."""
        return MetricOutput(
            raw=None,
            normalized=None,
            backend=backend,
            details=self._protocol_details(prediction),
            eligibility_status="official",
            error=PROMPT_INVALID_PREDICTION_REASON,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Average per-frame prompt alignment via reference runtime or CLIP."""
        backend = self._resolve_backend()
        eligibility_status, not_applicable_reason = self._resolve_eligibility(sample)
        if eligibility_status == "not_applicable":
            return self._not_applicable_output(
                backend=backend,
                prediction=prediction,
                error=str(not_applicable_reason),
            )
        if prediction.modality != "video":
            return self._invalid_prediction_output(
                backend=backend,
                prediction=prediction,
            )
        prompt_target = sample.prompt_target.strip()
        if not prompt_target:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=self._protocol_details(prediction),
                eligibility_status="official",
                error="prompt target is unavailable for this sample",
            )
        try:
            details = self._protocol_details(prediction)
            if backend == "reference":
                reference_result = compute_reference_prompt_alignment(
                    prediction.anchor_frames,
                    prompt_target,
                )
                raw = reference_result["raw"]
                details.update(reference_result["details"])
                return MetricOutput(
                    raw=raw,
                    normalized=normalize_score(raw, self.normalization),
                    backend=str(reference_result["backend"]),
                    details=details,
                    eligibility_status="official",
                )
            scores = [
                _clip_image_text_similarity(frame, prompt_target)
                for frame in prediction.anchor_frames
            ]
            score_values = [score for score, _ in scores]
            raw = float(np.mean(score_values)) if score_values else None
            details["frame_scores"] = [round(value, 4) for value in score_values]
            if scores:
                details["clip_loader"] = scores[0][1]
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="official",
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=self._protocol_details(prediction),
                eligibility_status="official",
                error=str(exc),
            )


class SceneAlignmentMetric(PromptAlignmentMetric):
    """Alias metric that reuses prompt alignment under the scene_alignment name."""

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        """Point scene_alignment at the shared prompt-alignment implementation."""
        super().__init__(backend=backend, normalization=normalization)
        self.name = "scene_alignment"


class DynamicAlignmentMetric(PromptAlignmentMetric):
    """Alias metric that reuses prompt alignment under the dynamic_alignment name."""

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        """Point dynamic_alignment at the shared prompt-alignment implementation."""
        super().__init__(backend=backend, normalization=normalization)
        self.name = "dynamic_alignment"
