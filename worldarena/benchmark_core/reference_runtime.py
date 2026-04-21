from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from PIL import Image

from worldarena.benchmark_core.metrics.base import clamp01


STYLE_CAPTURE_INDICES = (0, 2, 5, 7, 10)
REFERENCE_QUALITY_COMPONENTS = {
    "clip_iqa": ("clipiqa_plus", "clipiqa+", 1.0),
    "aesthetic": ("laion_aes", "laion_aes", 10.0),
}
REQUIRED_REFERENCE_QUALITY_COMPONENTS = ("clip_iqa", "aesthetic")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _round(values: list[float]) -> list[float]:
    return [round(value, 4) for value in values]


def _frame_to_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame)


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


@lru_cache(maxsize=1)
def _load_reference_clipscore_backend() -> tuple[str, Any, str] | None:
    try:
        import torch
        from torchmetrics.multimodal.clip_score import CLIPScore
    except (ImportError, ModuleNotFoundError):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    metric.eval()
    return "clipscore_torchmetrics", metric, device


@lru_cache(maxsize=1)
def _load_reference_style_backend() -> tuple[str, Any, Any, str] | None:
    try:
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


@lru_cache(maxsize=2)
def _load_reference_quality_component(
    component_name: str,
) -> tuple[str, Any, str, float] | None:
    component_spec = REFERENCE_QUALITY_COMPONENTS.get(component_name)
    if component_spec is None:
        return None

    backend_name, metric_name, scale = component_spec
    try:
        import pyiqa
        import torch
    except (ImportError, ModuleNotFoundError):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = pyiqa.create_metric(metric_name).to(device)
    return backend_name, metric, device, scale


def reference_prompt_alignment_available() -> bool:
    return _load_reference_clipscore_backend() is not None


def reference_style_consistency_available() -> bool:
    return _load_reference_style_backend() is not None


def reference_perceptual_quality_available() -> bool:
    return all(
        _load_reference_quality_component(component_name) is not None
        for component_name in REQUIRED_REFERENCE_QUALITY_COMPONENTS
    )


def _frame_to_uint8_tensor(frame: np.ndarray, device: str) -> Any:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(frame))
    return tensor.permute(2, 0, 1).to(device=device, dtype=torch.uint8)


def _frame_to_float_tensor(frame: np.ndarray, device: str) -> Any:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(frame))
    tensor = tensor.permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device)


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
    backend_payload = _load_reference_clipscore_backend()
    if backend_payload is None:
        raise ModuleNotFoundError("CLIPScore backend is unavailable")

    try:
        import torch
    except ModuleNotFoundError:
        torch = None  # type: ignore[assignment]

    backend_name, metric, device = backend_payload
    native_scores: list[float] = []
    frame_scores: list[float] = []
    native_upper = 100.0

    context = torch.inference_mode() if torch is not None else nullcontext()
    with context:
        for frame in frames:
            image = _frame_to_uint8_tensor(frame, device)
            native_score = _to_float(metric(image, prompt))
            metric.reset()
            native_scores.append(native_score)
            frame_scores.append(clamp01(native_score / native_upper))

    return {
        "raw": _mean(frame_scores),
        "backend": backend_name,
        "details": {
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

    for reference_frame, prediction_frame in zip(reference_frames, prediction_frames):
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
        },
    }


def compute_reference_perceptual_quality(
    frames: Sequence[np.ndarray],
) -> dict[str, Any]:
    component_payloads: dict[str, tuple[str, Any, str, float]] = {}
    missing_components: list[str] = []
    for component_name in REQUIRED_REFERENCE_QUALITY_COMPONENTS:
        payload = _load_reference_quality_component(component_name)
        if payload is None:
            missing_components.append(component_name)
            continue
        component_payloads[component_name] = payload
    if missing_components:
        missing = ", ".join(sorted(missing_components))
        raise ModuleNotFoundError(
            "reference perceptual-quality backends are unavailable for required "
            f"components: {missing}"
        )

    frame_component_scores: list[list[float]] = [[] for _ in frames]
    components: dict[str, dict[str, Any]] = {}

    for component_name, payload in component_payloads.items():
        backend_name, metric, device, scale = payload
        native_scores: list[float] = []
        normalized_scores: list[float] = []
        for index, frame in enumerate(frames):
            tensor = _frame_to_float_tensor(frame, device)
            native_score = _to_float(metric(tensor))
            normalized_score = clamp01(native_score / scale)
            native_scores.append(native_score)
            normalized_scores.append(normalized_score)
            frame_component_scores[index].append(normalized_score)
        components[component_name] = {
            "backend": backend_name,
            "frame_scores": _round(normalized_scores),
            "native_frame_scores": _round(native_scores),
            "native_score_range": [0.0, scale],
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
