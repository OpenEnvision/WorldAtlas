from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from worldarena.benchmark_core.media import LoadedMedia
from worldarena.benchmark_core.metrics.base import Metric, clamp01, normalize_score
from worldarena.benchmark_core.schemas import BenchmarkSample, MetricOutput


@lru_cache(maxsize=1)
def _load_pyiqa_backend() -> Any | None:
    try:
        import pyiqa
    except ModuleNotFoundError:
        return None
    return pyiqa


def _heuristic_quality(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
    contrast = gray.std()
    exposure = 1.0 - min(abs(float(gray.mean()) / 255.0 - 0.5) * 2.0, 1.0)
    sharpness_score = clamp01(sharpness / 600.0)
    contrast_score = clamp01(contrast / 64.0)
    return round(0.5 * sharpness_score + 0.3 * contrast_score + 0.2 * exposure, 4)


class SubjectiveQualityMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("subjective_quality", backend, normalization)

    def _resolve_backend(self) -> str:
        if self.backend == "auto":
            return "pyiqa" if _load_pyiqa_backend() is not None else "heuristic"
        return self.backend

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = self._resolve_backend()
        try:
            if backend == "pyiqa":
                import torch

                pyiqa = _load_pyiqa_backend()
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
            else:
                frame_scores = [_heuristic_quality(frame) for frame in prediction.anchor_frames]
                raw = float(np.mean(frame_scores)) if frame_scores else None
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details={"frame_scores": [round(value, 4) for value in frame_scores]},
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                error=str(exc),
            )
