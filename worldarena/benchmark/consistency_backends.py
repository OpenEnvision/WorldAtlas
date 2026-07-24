"""Backend implementations for photometric and temporal consistency metrics."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from PIL import Image

from worldarena.common.checkpoints import checkpoint_env, checkpoint_root, hf_local_dir
from worldarena.benchmark.metrics.base import cosine_similarity


@lru_cache(maxsize=1)
def _load_dinov2_backend() -> tuple[Any, Any, str] | None:
    """Load dinov2 backend -> tuple[Any, Any, str] | None."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except (ImportError, ModuleNotFoundError):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = hf_local_dir("facebook/dinov2-base", required=True)
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).to(device)
    model.eval()
    return model, processor, device


def dinov2_image_features(frames: Sequence[np.ndarray]) -> np.ndarray:
    backend = _load_dinov2_backend()
    if backend is None:
        raise ModuleNotFoundError("transformers/torch not available")

    import torch
    import torch.nn.functional as F

    model, processor, device = backend
    inputs = processor(images=[Image.fromarray(frame) for frame in frames], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = outputs.last_hidden_state[:, 0, :]
        features = F.normalize(features, dim=-1)
    return features.detach().cpu().numpy()


@lru_cache(maxsize=1)
def _load_dreamsim_backend() -> tuple[Any, Any, str] | None:
    """Load dreamsim backend -> tuple[Any, Any, str] | None."""
    try:
        os.environ.update(checkpoint_env())
        import torch
        from dreamsim import dreamsim
    except (ImportError, ModuleNotFoundError):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = dreamsim(
        pretrained=True,
        cache_dir=str(checkpoint_root()),
    )
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model, preprocess, device


def dreamsim_image_features(frames: Sequence[np.ndarray]) -> np.ndarray:
    backend = _load_dreamsim_backend()
    if backend is None:
        raise ModuleNotFoundError("dreamsim/torch not available")

    import torch
    import torch.nn.functional as F

    model, preprocess, device = backend
    images = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames]).to(device)
    with torch.inference_mode():
        features = model.embed(images)
        features = F.normalize(features, dim=-1)
    return features.detach().cpu().numpy()


def transition_scores(features: np.ndarray) -> list[float]:
    if features.shape[0] < 2:
        return []
    first = features[0]
    previous = first
    scores: list[float] = []
    for current in features[1:]:
        scores.append(
            float(
                (
                    max(0.0, cosine_similarity(previous, current))
                    + max(0.0, cosine_similarity(first, current))
                )
                / 2.0
            )
        )
        previous = current
    return scores


__all__ = [
    "dinov2_image_features",
    "dreamsim_image_features",
    "transition_scores",
]
