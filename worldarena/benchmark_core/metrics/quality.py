from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from worldarena.benchmark_core.media import LoadedMedia
from worldarena.benchmark_core.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark_core.reference_runtime import compute_reference_perceptual_quality
from worldarena.benchmark_core.schemas import BenchmarkSample, MetricOutput


@lru_cache(maxsize=1)
def _load_pyiqa_backend() -> Any | None:
    try:
        import pyiqa
    except ModuleNotFoundError:
        return None
    return pyiqa


QUALITY_BACKENDS = {"reference", "pyiqa"}
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
                import torch

                pyiqa = _load_pyiqa_backend()
                if pyiqa is None:
                    raise ModuleNotFoundError("pyiqa is not available")
                metric = pyiqa.create_metric("clipiqa+")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                metric = metric.to(device)
                frame_scores = []
                for frame in prediction.anchor_frames:
                    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    tensor = tensor.to(device)
                    score = float(metric(tensor).item())
                    frame_scores.append(score)
                raw = float(np.mean(frame_scores)) if frame_scores else None
            details["frame_scores"] = [round(value, 4) for value in frame_scores]
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
            )
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
