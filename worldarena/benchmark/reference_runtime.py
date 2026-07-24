"""Reference media runtime for metrics that compare against GT references."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, Sequence

import numpy as np
from PIL import Image

from worldarena.benchmark.clip_backend import (
    clip_backend_available,
    clip_image_features,
    clip_text_features,
    load_clip_backend,
)
from worldarena.benchmark.quality_backends import (
    REQUIRED_REFERENCE_QUALITY_COMPONENTS,
    compute_quality_component_scores,
    quality_component_available,
)
from worldarena.common.checkpoints import checkpoint_env


STYLE_CAPTURE_INDICES = (0, 2, 5, 7, 10)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _round(values: list[float]) -> list[float]:
    return [round(value, 4) for value in values]


def _paired_reference_prediction_frames(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    if not reference_frames or not prediction_frames:
        return [], "empty"
    if len(reference_frames) == len(prediction_frames):
        return list(zip(reference_frames, prediction_frames)), "aligned"
    if len(reference_frames) == 1:
        reference_frame = reference_frames[0]
        return [(reference_frame, prediction_frame) for prediction_frame in prediction_frames], "broadcast_reference"
    if len(prediction_frames) == 1:
        prediction_frame = prediction_frames[0]
        return [(reference_frame, prediction_frame) for reference_frame in reference_frames], "broadcast_prediction"
    pair_count = min(len(reference_frames), len(prediction_frames))
    return list(zip(reference_frames[:pair_count], prediction_frames[:pair_count])), "truncated"


def _frame_to_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame)


@lru_cache(maxsize=1)
def _reference_prompt_alignment_backend_name() -> str | None:
    """Reference prompt alignment backend name -> str | None."""
    backend = load_clip_backend()
    if backend is None:
        return None
    loader_name = backend[0]
    return f"clipscore_{loader_name}"


@lru_cache(maxsize=1)
def _load_reference_style_backend() -> tuple[str, Any, Any, str] | None:
    """Load reference style backend -> tuple[str, Any, Any, str] | None."""
    try:
        os.environ.update(checkpoint_env())
        import torch
        from torchvision import transforms
        from torchvision.models import VGG19_Weights, vgg19
    except (ImportError, ModuleNotFoundError):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 512 if device == "cuda" else 128
    preprocessing = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return "vgg19_gram", model, preprocessing, device


def reference_prompt_alignment_available() -> bool:
    """Reference prompt alignment available -> bool."""
    return clip_backend_available()


def reference_style_consistency_available() -> bool:
    """Reference style consistency available -> bool."""
    return _load_reference_style_backend() is not None


def reference_perceptual_quality_available() -> bool:
    """Reference perceptual quality available -> bool."""
    return all(
        quality_component_available(component_name)
        for component_name in REQUIRED_REFERENCE_QUALITY_COMPONENTS
    )


def _gram_matrix(features: Any) -> Any:
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    return (flattened @ flattened.transpose(1, 2)) / float(channels * height * width)


def _style_distance(
    reference_frame: np.ndarray,
    prediction_frame: np.ndarray,
) -> tuple[str, float, list[float]]:
    backend_payload = _load_reference_style_backend()
    if backend_payload is None:
        raise ModuleNotFoundError("torchvision/torch not available for style consistency")

    import torch
    import torch.nn.functional as F

    backend_name, model, preprocessing, device = backend_payload
    reference_tensor = preprocessing(_frame_to_image(reference_frame)).unsqueeze(0).to(device)
    prediction_tensor = preprocessing(_frame_to_image(prediction_frame)).unsqueeze(0).to(device)
    layer_distances: list[float] = []

    with torch.inference_mode():
        for index, layer in enumerate(model):
            reference_tensor = layer(reference_tensor)
            prediction_tensor = layer(prediction_tensor)
            if index in STYLE_CAPTURE_INDICES:
                layer_distance = F.mse_loss(
                    _gram_matrix(reference_tensor),
                    _gram_matrix(prediction_tensor),
                ).item()
                layer_distances.append(float(layer_distance))
            if index >= STYLE_CAPTURE_INDICES[-1]:
                break

    native_distance = float(np.mean(layer_distances)) if layer_distances else 0.0
    return backend_name, native_distance, layer_distances


def compute_reference_prompt_alignment(
    frames: Sequence[np.ndarray],
    prompt: str,
) -> dict[str, Any]:
    backend_name = _reference_prompt_alignment_backend_name()
    if backend_name is None:
        raise ModuleNotFoundError("CLIPScore backend is unavailable")
    native_upper = 100.0
    if not frames:
        return {
            "raw": None,
            "backend": backend_name,
            "details": {
                "frame_scores": [],
                "native_frame_scores": [],
                "native_score_range": [0.0, native_upper],
            },
        }

    image_features, loader_name = clip_image_features(frames)
    text_features, _ = clip_text_features([prompt])
    text_feature = text_features[0]
    native_scores = [
        max(100.0 * float(np.dot(image_feature, text_feature)), 0.0)
        for image_feature in image_features
    ]
    frame_scores = [clamp01(native_score / native_upper) for native_score in native_scores]

    return {
        "raw": _mean(frame_scores),
        "backend": backend_name,
        "details": {
            "clip_loader": loader_name,
            "frame_scores": _round(frame_scores),
            "native_frame_scores": _round(native_scores),
            "native_score_range": [0.0, native_upper],
        },
    }


def compute_reference_style_consistency(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
) -> dict[str, Any]:
    native_distances: list[float] = []
    frame_scores: list[float] = []
    layer_distances_per_frame: list[list[float]] = []
    backend_name = "vgg19_gram"
    frame_pairs, pairing_policy = _paired_reference_prediction_frames(
        reference_frames,
        prediction_frames,
    )

    for reference_frame, prediction_frame in frame_pairs:
        backend_name, native_distance, layer_distances = _style_distance(
            reference_frame,
            prediction_frame,
        )
        native_distances.append(native_distance)
        frame_scores.append(clamp01(1.0 / (1.0 + native_distance)))
        layer_distances_per_frame.append(_round(layer_distances))

    return {
        "raw": _mean(frame_scores),
        "backend": backend_name,
        "details": {
            "frame_scores": _round(frame_scores),
            "native_frame_distances": _round(native_distances),
            "native_layer_distances": layer_distances_per_frame,
            "native_direction": "lower_is_better",
            "reference_frame_count": len(reference_frames),
            "prediction_frame_count": len(prediction_frames),
            "paired_frame_count": len(frame_pairs),
            "pairing_policy": pairing_policy,
        },
    }


def compute_reference_perceptual_quality(
    frames: Sequence[np.ndarray],
) -> dict[str, Any]:
    frame_component_scores: list[list[float]] = [[] for _ in frames]
    components: dict[str, dict[str, Any]] = {}

    for component_name in REQUIRED_REFERENCE_QUALITY_COMPONENTS:
        component_result = compute_quality_component_scores(
            frames,
            component_name=component_name,
        )
        normalized_scores = list(component_result["frame_scores"])
        for index, normalized_score in enumerate(normalized_scores):
            frame_component_scores[index].append(float(normalized_score))
        components[component_name] = {
            "backend": component_result["backend"],
            "frame_scores": normalized_scores,
            "native_frame_scores": list(component_result["native_frame_scores"]),
            "native_score_range": list(component_result["native_score_range"]),
        }

    frame_scores = [
        float(np.mean(component_scores))
        for component_scores in frame_component_scores
        if component_scores
    ]
    return {
        "raw": _mean(frame_scores),
        "backend": "pyiqa_stack",
        "details": {
            "frame_scores": _round(frame_scores),
            "required_components": list(REQUIRED_REFERENCE_QUALITY_COMPONENTS),
            "components": components,
        },
    }
