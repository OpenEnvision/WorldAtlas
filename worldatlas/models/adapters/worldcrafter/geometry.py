"""WorldCrafter frame geometry shared by the adapter and lightweight runner."""

from __future__ import annotations

import math
from typing import Any


FRAMES_PER_CHUNK = 33
BASE_VAE_TEMPORAL_SCALE = 4


def output_frame_count(model_type: str, camera_frames: int) -> int:
    """Base trims decoded RGB to the VAE's 4n+1 frame length."""
    if model_type == "fast":
        return camera_frames
    if model_type == "base":
        return ((camera_frames - 1) // BASE_VAE_TEMPORAL_SCALE) * BASE_VAE_TEMPORAL_SCALE + 1
    raise ValueError(f"Unsupported WorldCrafter model type: {model_type}")


def rollout_geometry(generation: dict[str, Any]) -> tuple[int, int, float]:
    fps = int(generation.get("fps", 16))
    seconds = float(generation.get("target_duration_seconds", 5))
    if fps <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("WorldCrafter fps and target duration must be positive and finite")
    chunks = math.ceil(seconds * fps / FRAMES_PER_CHUNK)
    frames = chunks * FRAMES_PER_CHUNK
    return chunks, frames, frames / fps
