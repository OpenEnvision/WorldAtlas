"""VGG Gram-matrix style consistency metric."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import numpy as np
from PIL import Image

from worldarena.benchmark.media import LoadedMedia, protocol_frame_details
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.common.checkpoints import checkpoint_env


GRAM_EMPIRICAL_MIN = 0.0
GRAM_EMPIRICAL_MAX = 0.007
GRAM_STYLE_LAYERS = {"conv_1", "conv_2", "conv_3", "conv_4", "conv_5"}


@lru_cache(maxsize=1)
def _load_gram_backend() -> tuple[Any, Any, Any, Any, Any, Any, Any, str] | None:
    """Lazily load VGG19 and preprocessing used for gram-matrix distances."""
    try:
        os.environ.update(checkpoint_env())
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import transforms
        from torchvision.models import VGG19_Weights, vgg19
    except (ImportError, ModuleNotFoundError):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 512 if torch.cuda.is_available() else 128
    weights = VGG19_Weights.DEFAULT
    model = vgg19(weights=weights).features.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    preprocessing = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )
    normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225], device=device)
    return torch, nn, F, model, preprocessing, normalization_mean, normalization_std, device


def _frame_to_image(frame: np.ndarray) -> Image.Image:
    """Convert an RGB ndarray into a PIL image for torchvision preprocessing."""
    return Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")


def _gram_matrix(features: Any) -> Any:
    """Normalize feature Gram matrices for style scoring."""
    batch, channels, height, width = features.size()
    flattened = features.view(batch, channels, height * width)
    gram = flattened @ flattened.transpose(1, 2)
    return gram.div(channels * height * width)


def _gram_distances(
    reference_frame: np.ndarray,
    rendered_frames: list[np.ndarray],
) -> tuple[list[float], list[list[float]]]:
    """Sum MSE between VGG Gram matrices at configured style layers."""
    backend = _load_gram_backend()
    if backend is None:
        raise ModuleNotFoundError("torchvision/torch not available for gram_matrix")
    if not rendered_frames:
        return [], []

    torch, nn, F, model, preprocessing, normalization_mean, normalization_std, device = backend
    reference_tensor = preprocessing(_frame_to_image(reference_frame)).unsqueeze(0).to(device, torch.float)
    rendered_tensor = torch.stack(
        [preprocessing(_frame_to_image(frame)) for frame in rendered_frames],
        dim=0,
    ).to(device, torch.float)
    mean = normalization_mean.view(1, -1, 1, 1)
    std = normalization_std.view(1, -1, 1, 1)
    reference_tensor = (reference_tensor - mean) / std
    rendered_tensor = (rendered_tensor - mean) / std

    conv_index = 0
    layer_distances_per_frame: list[list[float]] = [[] for _ in rendered_frames]
    with torch.inference_mode():
        for layer in model.children():
            reference_tensor = layer(reference_tensor)
            rendered_tensor = layer(rendered_tensor)
            if isinstance(layer, nn.Conv2d):
                conv_index += 1
                if f"conv_{conv_index}" in GRAM_STYLE_LAYERS:
                    reference_gram = _gram_matrix(reference_tensor)
                    rendered_grams = _gram_matrix(rendered_tensor)
                    layer_distances = F.mse_loss(
                        rendered_grams,
                        reference_gram.expand_as(rendered_grams),
                        reduction="none",
                    ).mean(dim=(1, 2))
                    for frame_index, layer_distance in enumerate(layer_distances.detach().cpu().tolist()):
                        layer_distances_per_frame[frame_index].append(float(layer_distance))
                if conv_index >= len(GRAM_STYLE_LAYERS):
                    break

    native_distances = [float(sum(layer_distances)) for layer_distances in layer_distances_per_frame]
    return native_distances, layer_distances_per_frame


def _gram_distance(
    reference_frame: np.ndarray,
    rendered_frame: np.ndarray,
) -> tuple[float, list[float]]:
    """Sum MSE between VGG Gram matrices at configured style layers."""
    native_distances, layer_distances = _gram_distances(reference_frame, [rendered_frame])
    return native_distances[0], layer_distances[0]


def _round(values: list[float], digits: int = 6) -> list[float]:
    """Round float lists for compact metric detail payloads."""
    return [round(float(value), digits) for value in values]


def _gram_selected_frames(
    prediction_frames: list[np.ndarray],
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Pair the first anchor frame against all later anchors."""
    if not prediction_frames:
        return None, []
    return prediction_frames[0], list(prediction_frames[1:])


STYLE_BACKENDS = {"gram_matrix"}
DISABLED_STYLE_BACKENDS = {"histogram", "reference", "vgg19"}


class StyleConsistencyMetric(Metric):
    """Measure style drift as VGG Gram distance from the first prediction anchor."""

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        """Validate gram-matrix backend selection."""
        super().__init__("style_consistency", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Allow gram_matrix while blocking legacy histogram backends."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="gram_matrix",
            supported_backends=STYLE_BACKENDS,
            disabled_backends=DISABLED_STYLE_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Average Gram-matrix distances from the first anchor to later anchors."""
        del sample
        backend = self._resolve_backend()
        protocol_details = protocol_frame_details(prediction)
        reference_frame, selected_frames = _gram_selected_frames(prediction.anchor_frames)
        details: dict[str, object] = {
            **protocol_details,
            "style_reference_policy": "prediction_first_anchor",
            "pairing_policy": "first_anchor_to_later_anchors",
            "reference_frame_count": 1 if reference_frame is not None else 0,
            "prediction_frame_count": len(prediction.anchor_frames),
            "ground_truth_reference_frame_count": len(reference.anchor_frames),
            "paired_frame_count": len(selected_frames),
            "selected_frame_count": len(selected_frames),
            "native_direction": "lower_is_better",
            "native_score_range": [
                GRAM_EMPIRICAL_MIN,
                GRAM_EMPIRICAL_MAX,
            ],
            "gram_metric": "gram_matrix",
            "worldscore_metric": "gram_matrix",
            "gram_style_layers": sorted(GRAM_STYLE_LAYERS),
        }

        if reference_frame is None or not selected_frames:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend="gram_matrix",
                details={
                    **details,
                    "frame_scores": [],
                    "native_frame_distances": [],
                    "native_layer_distances": [],
                },
                error="style_consistency requires at least two prediction anchor frames",
            )

        try:
            native_distances, native_layer_distances = _gram_distances(
                reference_frame,
                selected_frames,
            )
            normalized_frame_scores = [
                normalize_score(native_distance, self.normalization) or 0.0
                for native_distance in native_distances
            ]
            layer_distances_per_frame = [
                _round(layer_distances)
                for layer_distances in native_layer_distances
            ]

            raw = float(np.mean(native_distances)) if native_distances else None
            if raw is not None:
                raw = float(
                    max(
                        GRAM_EMPIRICAL_MIN,
                        min(GRAM_EMPIRICAL_MAX, raw),
                    )
                )
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details={
                    **details,
                    "frame_scores": _round(normalized_frame_scores),
                    "native_frame_distances": _round(native_distances),
                    "native_layer_distances": layer_distances_per_frame,
                },
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )
