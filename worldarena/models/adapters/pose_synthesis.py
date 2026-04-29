from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.benchmark.annotations import (
    _matrix_to_pose_vector,
    _pose_vector_to_matrix,
    _resample_pose_vectors,
)
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.annotation_index import split_annotation_reference
from worldarena.common.checkpoints import project_root
from worldarena.common.media import probe_image
from worldarena.models.adapters.lingbot_world import _camera_delta, _camera_path_for_sample
from worldarena.models.config import ModelRuntimeConfig


def local_pose_annotation_path(sample: BenchmarkSample) -> Path | None:
    if not sample.annotation_path or split_annotation_reference(sample.annotation_path) is not None:
        return None
    candidate = Path(sample.annotation_path).expanduser().resolve()
    if (candidate / "poses.npy").exists() and (candidate / "intrinsics.npy").exists():
        return candidate
    return None


def _interpolate_pose_segment(start_pose: np.ndarray, end_pose: np.ndarray, *, frames: int) -> np.ndarray:
    pose_vectors = np.stack(
        [
            _matrix_to_pose_vector(start_pose.astype(np.float32)),
            _matrix_to_pose_vector(end_pose.astype(np.float32)),
        ],
        axis=0,
    ).astype(np.float32)
    interpolated = _resample_pose_vectors(
        pose_vectors,
        np.asarray([0.0, 1.0], dtype=np.float32),
        frames,
    )
    return np.stack([_pose_vector_to_matrix(row) for row in interpolated], axis=0).astype(np.float32)


def _synthetic_image_poses(camera_path: list[str], *, frame_count: int) -> np.ndarray:
    target_frames = max(int(frame_count), 1)
    if target_frames <= 1:
        return np.repeat(np.eye(4, dtype=np.float32)[None], repeats=target_frames, axis=0)

    normalized_path = [token for token in camera_path if token] or ["fixed"]
    keyframes: list[np.ndarray] = [np.eye(4, dtype=np.float32)]
    current_pose = np.eye(4, dtype=np.float32)
    for token in normalized_path:
        current_pose = (current_pose @ _camera_delta(token)).astype(np.float32)
        keyframes.append(current_pose.copy())

    segment_count = max(1, len(normalized_path))
    remaining_intervals = target_frames - 1
    base_intervals = remaining_intervals // segment_count
    remainder = remaining_intervals % segment_count
    segments: list[np.ndarray] = []
    for index in range(segment_count):
        segment_intervals = base_intervals + (1 if index < remainder else 0)
        segment = _interpolate_pose_segment(
            keyframes[index],
            keyframes[index + 1],
            frames=segment_intervals + 1,
        )
        if index:
            segment = segment[1:]
        segments.append(segment)
    return np.concatenate(segments, axis=0).astype(np.float32)


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
) -> tuple[Path, str]:
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
    cache_payload: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "relative_path": sample.relative_path,
        "annotation_path": sample.annotation_path,
        "camera_path": camera_path,
        "frame_count": int(frame_count),
        "image_width": image_width,
        "image_height": image_height,
        "focal_scale": float(focal_scale),
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
