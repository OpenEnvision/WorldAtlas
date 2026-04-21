from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import (
    Metric,
    clamp01,
    normalize_score,
    resolve_metric_backend,
)
from worldarena.benchmark.reference_runtime import compute_reference_style_consistency
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


@lru_cache(maxsize=1)
def _load_vgg_backend() -> tuple[Any, Any, Any] | None:
    try:
        import torch
        from torchvision import transforms
        from torchvision.models import VGG19_Weights, vgg19
    except ModuleNotFoundError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:19].to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    preprocessing = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, preprocessing, device


def _gram_similarity(left: np.ndarray, right: np.ndarray) -> float:
    backend = _load_vgg_backend()
    if backend is None:
        raise ModuleNotFoundError("torchvision/torch not available")

    import torch

    model, preprocessing, device = backend
    left_tensor = preprocessing(left).unsqueeze(0).to(device)
    right_tensor = preprocessing(right).unsqueeze(0).to(device)
    with torch.no_grad():
        left_features = model(left_tensor)
        right_features = model(right_tensor)
    left_features = left_features.squeeze(0)
    right_features = right_features.squeeze(0)
    left_gram = left_features.flatten(1) @ left_features.flatten(1).T
    right_gram = right_features.flatten(1) @ right_features.flatten(1).T
    distance = torch.mean(torch.abs(left_gram - right_gram)).item()
    return clamp01(1.0 / (1.0 + distance))


STYLE_BACKENDS = {"reference", "vgg19"}
DISABLED_STYLE_BACKENDS = {"histogram"}


class StyleConsistencyMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("style_consistency", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="reference",
            supported_backends=STYLE_BACKENDS,
            disabled_backends=DISABLED_STYLE_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        backend = self._resolve_backend()
        try:
            if backend == "reference":
                reference_result = compute_reference_style_consistency(
                    reference.anchor_frames,
                    prediction.anchor_frames,
                )
                raw = reference_result["raw"]
                return MetricOutput(
                    raw=raw,
                    normalized=normalize_score(raw, self.normalization),
                    backend=str(reference_result["backend"]),
                    details=reference_result["details"],
                )
            paired = zip(prediction.anchor_frames, reference.anchor_frames)
            scores = [_gram_similarity(left, right) for left, right in paired]
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
