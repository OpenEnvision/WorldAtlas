from __future__ import annotations

from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.quality_backends import compute_musiq_scores, compute_quality_component_scores
from worldarena.benchmark.reference_runtime import compute_reference_perceptual_quality
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


QUALITY_BACKENDS = {"reference", "pyiqa"}
IMAGE_QUALITY_BACKENDS = {"musiq"}
DISABLED_QUALITY_BACKENDS = {"heuristic"}


class PerceptualQualityMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("perceptual_quality", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="reference",
            supported_backends=QUALITY_BACKENDS,
            disabled_backends=DISABLED_QUALITY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = self._resolve_backend()
        details = {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }
        try:
            if backend == "reference":
                reference_result = compute_reference_perceptual_quality(prediction.anchor_frames)
                raw = reference_result["raw"]
                details.update(reference_result["details"])
                return MetricOutput(
                    raw=raw,
                    normalized=normalize_score(raw, self.normalization),
                    backend=str(reference_result["backend"]),
                    details=details,
                )
            if backend == "pyiqa":
                quality_result = compute_quality_component_scores(
                    prediction.anchor_frames,
                    component_name="clip_iqa",
                )
                raw = quality_result["raw"]
                details.update(
                    {
                        "frame_scores": list(quality_result["frame_scores"]),
                        "native_frame_scores": list(quality_result["native_frame_scores"]),
                        "native_score_range": list(quality_result["native_score_range"]),
                    }
                )
                return MetricOutput(
                    raw=raw,
                    normalized=normalize_score(raw, self.normalization),
                    backend=str(quality_result["backend"]),
                    details=details,
                )
            raise RuntimeError(f"unsupported quality backend resolved at runtime: {backend}")
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


class SubjectiveQualityMetric(PerceptualQualityMetric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__(backend=backend, normalization=normalization)
        self.name = "subjective_quality"


class ImageQualityMetric(Metric):
    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("image_quality", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="musiq",
            supported_backends=IMAGE_QUALITY_BACKENDS,
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
        details = {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }
        try:
            quality_result = compute_musiq_scores(
                prediction.frames or prediction.anchor_frames,
                model_path=self.runtime.get("model_path"),
                preprocess_mode=str(self.runtime.get("preprocess_mode", "longer")),
            )
            raw = quality_result["raw"]
            details.update(
                {
                    "frame_scores": list(quality_result["frame_scores"]),
                    "native_frame_scores": list(quality_result["native_frame_scores"]),
                    "native_score_range": list(quality_result["native_score_range"]),
                    "preprocess_mode": quality_result["preprocess_mode"],
                    "weights_path": quality_result["weights_path"],
                }
            )
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=str(quality_result["backend"]),
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
