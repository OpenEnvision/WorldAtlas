"""Synthetic camera pose and intrinsics generation for image_static samples.

Used when benchmark samples specify ``camera_path`` tokens but no ``poses.npy``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices
from worldarena.common.annotation_index import split_annotation_reference
from worldarena.common.checkpoints import project_root
from worldarena.common.media import probe_image
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample
from worldarena.models.config import ModelRuntimeConfig


def local_pose_annotation_path(sample: BenchmarkSample) -> Path | None:
    """Return the on-disk annotation dir when poses/intrinsics sidecars exist."""
    if not sample.annotation_path or split_annotation_reference(sample.annotation_path) is not None:
        return None
    candidate = Path(sample.annotation_path).expanduser().resolve()
    if (candidate / "poses.npy").exists() and (candidate / "intrinsics.npy").exists():
        return candidate
    return None


def _synthetic_image_poses(
    camera_path: list[str],
    *,
    frame_count: int,
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Build poses while preserving rotations and scaling translations.

    Camera-path tokens are defined in LingBot's unit scale. Adapters whose
    reconstructed scene uses a different coordinate scale can supply a
    model-native ``translation_scale`` without changing rotation semantics.
    """
    translation_scale = float(translation_scale)
    if not np.isfinite(translation_scale) or translation_scale < 0.0:
        raise ValueError(f"translation_scale must be finite and non-negative, got {translation_scale!r}")

    poses = synthetic_camera_matrices(
        camera_path,
        target_frames=max(int(frame_count), 1),
    ).matrices.copy()
    poses[:, :3, 3] *= translation_scale
    return poses.astype(np.float32)


def _synthetic_intrinsics_sequence(
    frame_count: int,
    *,
    image_width: int,
    image_height: int,
    focal_scale: float,
) -> np.ndarray:
    focal_length = float(max(image_width, image_height)) * float(focal_scale)
    intrinsics = np.asarray(
        [focal_length, focal_length, image_width / 2.0, image_height / 2.0],
        dtype=np.float32,
    )
    return np.repeat(intrinsics[None], repeats=max(int(frame_count), 1), axis=0)


def resolve_pose_annotation_path(
    config: ModelRuntimeConfig,
    sample: BenchmarkSample,
    conditioning_image: Path,
    *,
    frame_count: int,
    namespace: str,
    focal_scale: float,
    translation_scale: float = 1.0,
) -> tuple[Path, str]:
    """Resolve GT annotations or build cached synthetic poses for one sample."""
    local_annotation = local_pose_annotation_path(sample)
    if local_annotation is not None:
        return local_annotation, "annotation"
    if sample.modality != "image":
        raise FileNotFoundError(
            f"{config.name} requires poses.npy/intrinsics.npy for non-image sample {sample.sample_id}"
        )
    try:
        image_probe = probe_image(conditioning_image)
    except Exception as exc:
        raise FileNotFoundError(
            f"{config.name} requires poses.npy/intrinsics.npy or an image conditioning asset "
            f"for sample {sample.sample_id}"
        ) from exc
    image_width = int(image_probe["width"])
    image_height = int(image_probe["height"])
    camera_path = _camera_path_for_sample(sample) or ["fixed"]
    translation_scale = float(translation_scale)
    if not np.isfinite(translation_scale) or translation_scale < 0.0:
        raise ValueError(f"translation_scale must be finite and non-negative, got {translation_scale!r}")
    cache_payload: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "relative_path": sample.relative_path,
        "annotation_path": sample.annotation_path,
        "camera_path": camera_path,
        "frame_count": int(frame_count),
        "image_width": image_width,
        "image_height": image_height,
        "focal_scale": float(focal_scale),
        "translation_scale": translation_scale,
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
        _synthetic_image_poses(
            camera_path,
            frame_count=frame_count,
            translation_scale=translation_scale,
        ).astype(np.float32),
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
