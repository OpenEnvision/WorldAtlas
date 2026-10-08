"""Synthesize camera trajectories from ``camera_path`` tokens for metrics without GT poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from worldatlas.benchmark.annotations import (
    _matrix_to_pose_vector,
    _pose_vector_to_matrix,
    _resample_pose_vectors,
)
from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.benchmark.taxonomy import is_camera_path_suite


SYNTHETIC_CAMERA_ROTATION_DEGREES = 30.0
SYNTHETIC_CAMERA_TILT_DEGREES = 20.0
SYNTHETIC_CAMERA_ROLL_DEGREES = 15.0
SYNTHETIC_CAMERA_ORBIT_RADIUS = 1.0
SYNTHETIC_CAMERA_FORWARD_STEP = 1.0
SYNTHETIC_CAMERA_LATERAL_STEP = 0.5
SYNTHETIC_CAMERA_VERTICAL_STEP = 0.35

_TOKEN_ALIASES = {
    "dolly_in": "push_in",
    "dolly_out": "pull_out",
    "zoom_in": "push_in",
    "zoom_out": "pull_out",
    "truck_left": "move_left",
    "truck_right": "move_right",
    "stay": "fixed",
}

SUPPORTED_SYNTHETIC_CAMERA_TOKENS = frozenset(
    {
        "fixed",
        "push_in",
        "pull_out",
        "move_left",
        "move_right",
        "pedestal_up",
        "pedestal_down",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "roll_cw",
        "roll_ccw",
        "orbit_left",
        "orbit_right",
    }
)


@dataclass(frozen=True, slots=True)
class SyntheticCameraTrajectory:
    matrices: np.ndarray
    scale: float
    camera_path: tuple[str, ...]
    source: str = "synthetic_camera_path"


def _normalize_camera_path(camera_path: list[str] | tuple[str, ...]) -> list[str]:
    tokens: list[str] = []
    for value in camera_path:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        token = _TOKEN_ALIASES.get(token, token)
        if token:
            tokens.append(token)
    return tokens or ["fixed"]


def _translation_matrix(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray([x, y, z], dtype=np.float32)
    return matrix


def _rotation_matrix_x(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_z(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _rigid_transform(rotation: np.ndarray | None = None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    if rotation is not None:
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float32)
    return matrix


def _camera_delta(token: str, runtime: dict[str, Any]) -> np.ndarray:
    rotation_degrees = float(runtime.get("synthetic_camera_rotation_degrees", SYNTHETIC_CAMERA_ROTATION_DEGREES))
    tilt_degrees = float(runtime.get("synthetic_camera_tilt_degrees", SYNTHETIC_CAMERA_TILT_DEGREES))
    roll_degrees = float(runtime.get("synthetic_camera_roll_degrees", SYNTHETIC_CAMERA_ROLL_DEGREES))
    orbit_radius = float(runtime.get("synthetic_camera_orbit_radius", SYNTHETIC_CAMERA_ORBIT_RADIUS))
    forward_step = float(runtime.get("synthetic_camera_forward_step", SYNTHETIC_CAMERA_FORWARD_STEP))
    lateral_step = float(runtime.get("synthetic_camera_lateral_step", SYNTHETIC_CAMERA_LATERAL_STEP))
    vertical_step = float(runtime.get("synthetic_camera_vertical_step", SYNTHETIC_CAMERA_VERTICAL_STEP))

    if token == "fixed":
        return np.eye(4, dtype=np.float32)
    if token == "push_in":
        return _translation_matrix(z=forward_step)
    if token == "pull_out":
        return _translation_matrix(z=-forward_step)
    if token == "move_left":
        return _translation_matrix(x=-lateral_step)
    if token == "move_right":
        return _translation_matrix(x=lateral_step)
    if token == "pedestal_up":
        return _translation_matrix(y=vertical_step)
    if token == "pedestal_down":
        return _translation_matrix(y=-vertical_step)
    if token == "pan_left":
        return _rigid_transform(rotation=_rotation_matrix_y(-rotation_degrees))
    if token == "pan_right":
        return _rigid_transform(rotation=_rotation_matrix_y(rotation_degrees))
    if token == "tilt_up":
        return _rigid_transform(rotation=_rotation_matrix_x(tilt_degrees))
    if token == "tilt_down":
        return _rigid_transform(rotation=_rotation_matrix_x(-tilt_degrees))
    if token == "roll_cw":
        return _rigid_transform(rotation=_rotation_matrix_z(-roll_degrees))
    if token == "roll_ccw":
        return _rigid_transform(rotation=_rotation_matrix_z(roll_degrees))
    if token == "orbit_left":
        return (
            _translation_matrix(z=orbit_radius)
            @ _rigid_transform(rotation=_rotation_matrix_y(rotation_degrees))
            @ _translation_matrix(z=-orbit_radius)
        ).astype(np.float32)
    if token == "orbit_right":
        return (
            _translation_matrix(z=orbit_radius)
            @ _rigid_transform(rotation=_rotation_matrix_y(-rotation_degrees))
            @ _translation_matrix(z=-orbit_radius)
        ).astype(np.float32)
    raise ValueError(f"unsupported synthetic camera token: {token!r}")


def camera_delta(token: str, runtime: dict[str, Any] | None = None) -> np.ndarray:
    """Return the rigid delta applied by one normalized camera-path token."""
    normalized = _normalize_camera_path([token])[0]
    return _camera_delta(normalized, dict(runtime or {}))


def segment_interval_counts(segment_count: int, target_frames: int) -> tuple[int, ...]:
    """Split frame intervals across trajectory segments as evenly as possible."""
    segments = max(int(segment_count), 1)
    remaining = max(int(target_frames), 1) - 1
    base = remaining // segments
    remainder = remaining % segments
    return tuple(base + (1 if index < remainder else 0) for index in range(segments))


def segment_frame_bounds(segment_count: int, target_frames: int) -> tuple[int, ...]:
    """Return the frame index of every keyframe boundary, including start and end."""
    bounds = [0]
    for intervals in segment_interval_counts(segment_count, target_frames):
        bounds.append(bounds[-1] + intervals)
    return tuple(bounds)


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


def synthetic_camera_matrices(
    camera_path: list[str] | tuple[str, ...],
    *,
    target_frames: int,
    runtime: dict[str, Any] | None = None,
) -> SyntheticCameraTrajectory:
    runtime = dict(runtime or {})
    frame_count = max(int(target_frames), 1)
    normalized_path = _normalize_camera_path(camera_path)
    if frame_count <= 1:
        matrices = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=frame_count, axis=0)
        return SyntheticCameraTrajectory(
            matrices=matrices,
            scale=float(runtime.get("synthetic_camera_scale", 1.0)),
            camera_path=tuple(normalized_path),
        )

    keyframes: list[np.ndarray] = [np.eye(4, dtype=np.float32)]
    current_pose = np.eye(4, dtype=np.float32)
    for token in normalized_path:
        current_pose = (current_pose @ _camera_delta(token, runtime)).astype(np.float32)
        keyframes.append(current_pose.copy())

    segment_count = max(1, len(normalized_path))
    interval_counts = segment_interval_counts(segment_count, frame_count)

    segments: list[np.ndarray] = []
    for index in range(segment_count):
        segment_intervals = interval_counts[index]
        segment = _interpolate_pose_segment(
            keyframes[index],
            keyframes[index + 1],
            frames=segment_intervals + 1,
        )
        if index:
            segment = segment[1:]
        segments.append(segment)
    matrices = np.concatenate(segments, axis=0).astype(np.float32)
    if len(matrices) != frame_count:
        raise ValueError(f"expected {frame_count} synthetic camera poses, got {len(matrices)}")
    return SyntheticCameraTrajectory(
        matrices=matrices,
        scale=float(runtime.get("synthetic_camera_scale", 1.0)),
        camera_path=tuple(normalized_path),
    )


def synthetic_camera_matrices_for_sample(
    sample: BenchmarkSample,
    *,
    target_frames: int,
    runtime: dict[str, Any] | None = None,
) -> SyntheticCameraTrajectory | None:
    if not is_camera_path_suite(sample.suite):
        return None
    if sample.modality != "image" or sample.generation_mode != "static":
        return None
    if not sample.camera_path:
        return None
    return synthetic_camera_matrices(
        list(sample.camera_path),
        target_frames=target_frames,
        runtime=runtime,
    )


__all__ = [
    "SUPPORTED_SYNTHETIC_CAMERA_TOKENS",
    "SyntheticCameraTrajectory",
    "camera_delta",
    "segment_frame_bounds",
    "segment_interval_counts",
    "synthetic_camera_matrices",
    "synthetic_camera_matrices_for_sample",
]
