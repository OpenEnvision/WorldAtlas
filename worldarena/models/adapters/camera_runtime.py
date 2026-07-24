"""Shared camera-annotation resolution for camera-conditioned model adapters.

When a sample lacks on-disk poses, adapters synthesize trajectories from
``camera_path`` tokens and cache the resulting ``poses.npy`` / ``intrinsics.npy``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.checkpoints import project_root
from worldarena.common.media import probe_image
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample
from worldarena.models.adapters.pose_synthesis import (
    _synthetic_image_poses,
    _synthetic_intrinsics_sequence,
    local_pose_annotation_path,
)


def camera_annotation_for_conditioning_image(
    sample: BenchmarkSample,
    conditioning_image: Path,
    *,
    frame_count: int,
    namespace: str,
    focal_scale: float = 0.6,
) -> tuple[Path, str]:
    """Return a pose annotation directory and whether it came from GT or synthesis."""
    local_annotation = local_pose_annotation_path(sample)
    if local_annotation is not None:
        return local_annotation, "annotation"

    image_probe = probe_image(conditioning_image)
    image_width = int(image_probe["width"])
    image_height = int(image_probe["height"])
    camera_path = _camera_path_for_sample(sample) or ["fixed"]
    cache_payload: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "relative_path": sample.relative_path,
        "annotation_path": sample.annotation_path,
        "camera_path": camera_path,
        "frame_count": int(frame_count),
        "image_width": image_width,
        "image_height": image_height,
        "focal_scale": float(focal_scale),
        "namespace": namespace,
    }
    key = hashlib.sha1(
        json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    annotation_root = project_root() / "cache" / "model_runtime" / namespace / "synthetic_pose" / (
        f"{sample.prediction_stem}--{key}"
    )
    poses_path = annotation_root / "poses.npy"
    intrinsics_path = annotation_root / "intrinsics.npy"
    if poses_path.exists() and intrinsics_path.exists():
        return annotation_root, "synthetic"

    annotation_root.mkdir(parents=True, exist_ok=True)
    np.save(
        poses_path,
        _synthetic_image_poses(camera_path, frame_count=frame_count).astype(np.float32),
    )
    np.save(
        intrinsics_path,
        _synthetic_intrinsics_sequence(
            frame_count,
            image_width=image_width,
            image_height=image_height,
            focal_scale=focal_scale,
        ).astype(np.float32),
    )
    return annotation_root, "synthetic"
