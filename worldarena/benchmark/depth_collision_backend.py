"""Depth-based collision detection backend for physics-related metrics."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from worldarena.common.checkpoints import (
    checkpoint_env,
    checkpoint_path,
    resolve_checkpoint_path,
    resolve_project_path,
)


DEFAULT_DEPTH_THRESHOLD = 1.0
DEFAULT_CENTER_CROP_RATIO = 0.2


def _default_device() -> str:
    """Default device -> str."""
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_video_depth_anything_root(repo_root: str | os.PathLike[str] | None) -> Path:
    path = resolve_project_path(repo_root or "thirdparty/Video-Depth-Anything")
    if path is None or not path.is_dir():
        raise FileNotFoundError(f"Video-Depth-Anything source root not found: {path}")
    return path


def _resolve_video_depth_anything_checkpoint(
    checkpoint_value: str | os.PathLike[str] | None,
    *,
    encoder: str,
) -> Path:
    if checkpoint_value is not None:
        resolved = resolve_checkpoint_path(checkpoint_value, kind="file", required=True)
        if resolved is None:
            raise FileNotFoundError("Video-Depth-Anything checkpoint path is missing")
        return resolved
    return checkpoint_path(
        "Video-Depth-Anything",
        f"metric_video_depth_anything_{encoder}.pth",
        kind="file",
        required=True,
    )


@lru_cache(maxsize=4)
def _load_video_depth_anything(
    *,
    encoder: str,
    repo_root: str,
    checkpoint_file: str,
    device: str,
) -> Any:
    os.environ.update(checkpoint_env())
    source_root = Path(repo_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import torch
    from video_depth_anything.video_depth import VideoDepthAnything

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }
    try:
        config = model_configs[encoder]
    except KeyError as exc:
        supported = ", ".join(sorted(model_configs))
        raise ValueError(f"unsupported Video-Depth-Anything encoder '{encoder}'; supported: {supported}") from exc

    model = VideoDepthAnything(**config, metric=True)
    state = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _thresholds(runtime: dict[str, Any]) -> list[float]:
    configured = runtime.get("depth_thresholds")
    if configured is None:
        configured = [runtime.get("depth_threshold", DEFAULT_DEPTH_THRESHOLD)]
    thresholds = [float(value) for value in configured]
    if not thresholds:
        raise ValueError("depth_collision requires at least one depth threshold")
    return thresholds


def _center_crop_mask(height: int, width: int, ratio: float) -> np.ndarray:
    crop_h = max(1, int(round(height * ratio)))
    crop_w = max(1, int(round(width * ratio)))
    start_h = max(0, (height - crop_h) // 2)
    start_w = max(0, (width - crop_w) // 2)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[start_h : start_h + crop_h, start_w : start_w + crop_w] = 1.0
    return mask


def predict_metric_depths(
    frames: Sequence[np.ndarray],
    *,
    runtime: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = dict(runtime or {})
    encoder = str(payload.get("encoder") or "vits")
    device = str(payload.get("device") or _default_device())
    source_root = _resolve_video_depth_anything_root(payload.get("repo_root"))
    checkpoint_file = _resolve_video_depth_anything_checkpoint(
        payload.get("checkpoint_path"),
        encoder=encoder,
    )
    model = _load_video_depth_anything(
        encoder=encoder,
        repo_root=str(source_root),
        checkpoint_file=str(checkpoint_file),
        device=device,
    )

    import torch

    frame_array = np.asarray(frames, dtype=np.uint8)
    with torch.inference_mode():
        depths, _ = model.infer_video_depth(
            frame_array,
            target_fps=float(payload.get("target_fps", -1)),
            input_size=int(payload.get("input_size", 256)),
            device=device,
            fp32=bool(payload.get("fp32", False)),
        )
    details = {
        "backend": "video_depth_anything",
        "encoder": encoder,
        "device": device,
        "repo_root": str(source_root),
        "checkpoint_path": str(checkpoint_file),
    }
    return np.asarray(depths, dtype=np.float32), details


def compute_depth_collision(
    frames: Sequence[np.ndarray],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(runtime or {})
    if not frames:
        return {
            "raw": None,
            "backend": "video_depth_anything",
            "details": {"frame_count": 0},
            "error": "no prediction frames available for depth collision",
        }

    thresholds = _thresholds(payload)
    center_crop_ratio = float(payload.get("center_crop_ratio", DEFAULT_CENTER_CROP_RATIO))
    if not 0.0 < center_crop_ratio <= 1.0:
        raise ValueError("depth_collision center_crop_ratio must be in (0, 1]")

    depths, backend_details = predict_metric_depths(frames, runtime=payload)
    if depths.ndim != 3:
        raise ValueError(f"expected depth tensor with shape (T, H, W), got {depths.shape}")

    mask = _center_crop_mask(depths.shape[1], depths.shape[2], center_crop_ratio)
    mask_sum = float(mask.sum())
    center_depths = [float(np.sum(depth_map * mask) / mask_sum) for depth_map in depths]
    threshold_details: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        hit_indices = [index for index, value in enumerate(center_depths) if value < threshold]
        threshold_details[str(threshold)] = {
            "collision": bool(hit_indices),
            "first_collision_frame": hit_indices[0] if hit_indices else None,
            "collision_frames": hit_indices,
        }

    primary = threshold_details[str(thresholds[0])]
    raw = 1.0 if primary["collision"] else 0.0
    return {
        "raw": raw,
        "backend": "video_depth_anything",
        "details": {
            **backend_details,
            "frame_count": len(frames),
            "depth_shape": list(depths.shape),
            "center_crop_ratio": center_crop_ratio,
            "depth_threshold": thresholds[0],
            "depth_thresholds": thresholds,
            "center_crop_mean_depths": [round(value, 4) for value in center_depths],
            "threshold_results": threshold_details,
            "collision": bool(primary["collision"]),
            "first_collision_frame": primary["first_collision_frame"],
        },
        "error": None,
    }


__all__ = [
    "compute_depth_collision",
    "predict_metric_depths",
]
