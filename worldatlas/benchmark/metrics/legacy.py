"""Shared frame helpers retained for object-permanence and memory metrics."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from worldatlas.benchmark.consistency_backends import (
    dinov2_image_features as _shared_dinov2_image_features,
)
from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.metrics.base import clamp01, normalize_score


def _protocol_details(prediction: LoadedMedia) -> dict[str, Any]:
    """Collect protocol and sampled frame indices for legacy frame metrics."""
    return {
        "frame_indices": list(prediction.protocol_frame_indices),
        "sampled_frame_indices": list(prediction.sampled_frame_indices),
    }


def _normalize_or_identity(value: float | None, spec: dict[str, Any]) -> float | None:
    """Apply normalization when configured, otherwise return the raw float."""
    if value is None:
        return None
    if not spec:
        return float(value)
    return normalize_score(float(value), spec)


def _resize_like(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize a frame to match another frame's height and width when needed."""
    if frame.shape[:2] == target.shape[:2]:
        return frame
    target_size = (target.shape[1], target.shape[0])
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)


def _dinov2_image_features(frames: list[np.ndarray]) -> np.ndarray:
    """Delegate DINOv2 feature extraction to the shared consistency backend."""
    return _shared_dinov2_image_features(frames)


def _ssim_channel(left: np.ndarray, right: np.ndarray) -> float:
    """Gaussian-window SSIM for a single grayscale channel."""
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    left = left.astype(np.float32)
    right = right.astype(np.float32)
    mu_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right, (11, 11), 1.5)

    mu_left_sq = mu_left * mu_left
    mu_right_sq = mu_right * mu_right
    mu_cross = mu_left * mu_right

    sigma_left_sq = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu_left_sq
    sigma_right_sq = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu_right_sq
    sigma_cross = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu_cross

    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (sigma_left_sq + sigma_right_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(ssim_map))


def _ssim_rgb(left: np.ndarray, right: np.ndarray) -> float:
    """Average per-channel SSIM across RGB and clamp to [0, 1]."""
    scores = [
        _ssim_channel(left[:, :, channel], right[:, :, channel])
        for channel in range(left.shape[2])
    ]
    return clamp01(float(np.mean(scores)))


__all__ = [
    "_dinov2_image_features",
    "_normalize_or_identity",
    "_protocol_details",
    "_resize_like",
    "_ssim_rgb",
]
