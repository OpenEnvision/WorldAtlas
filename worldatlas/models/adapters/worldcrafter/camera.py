"""WorldCrafter camera controls calibrated to its interactive demo defaults.

The demo moves one metre or turns 15 degrees per 33-frame chunk at 16 fps.
Memory tasks retain their prescribed total turn angles so the named loops close.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from worldatlas.benchmark.synthetic_camera import _normalize_camera_path
from worldatlas.common.checkpoints import project_root
from worldatlas.models.adapters.lingbot_world.common import _camera_path_for_sample
from worldatlas.models.adapters.pose_synthesis import local_pose_annotation_path

RECIPE = "worldcrafter_official_demo_v1"
CHUNK_FRAMES = 33
TRANSLATION_PER_CHUNK = 1.0
ROTATION_DEGREES_PER_CHUNK = 15.0


def rotation(axis: int, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def horizontal_axis(pose: np.ndarray, axis: int) -> np.ndarray:
    direction = pose[:3, axis].copy()
    direction[1] = 0
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        other = pose[:3, 0 if axis == 2 else 2]
        direction = np.array([-other[2], 0., other[0]])
        if axis == 0:
            direction *= -1
        norm = np.linalg.norm(direction)
    return direction / norm


def pose_at(start: np.ndarray, token: str, distance: float, degrees: float, fraction: float) -> np.ndarray:
    pose = start.copy()
    if token == "fixed":
        return pose
    if token in ("push_in", "pull_out", "move_left", "move_right", "pedestal_up", "pedestal_down"):
        if token.startswith("pedestal_"):
            direction = np.array([0., -1. if token == "pedestal_up" else 1., 0.])
        else:
            direction = horizontal_axis(start, 2 if token in ("push_in", "pull_out") else 0)
            if token in ("pull_out", "move_left"):
                direction *= -1
        pose[:3, 3] += direction * distance * fraction
    elif token in ("pan_left", "pan_right", "tilt_up", "tilt_down", "roll_cw", "roll_ccw"):
        sign = -1 if token in ("pan_left", "tilt_down", "roll_cw") else 1
        axis = 1 if token.startswith("pan_") else (0 if token.startswith("tilt_") else 2)
        turn = rotation(axis, sign * degrees * fraction)
        pose[:3, :3] = turn @ start[:3, :3] if axis == 1 else start[:3, :3] @ turn
    elif token in ("orbit_left", "orbit_right"):
        # The demo has no orbit button. Match BOTH default travel and turn rates
        # with a circular path: radius = linear speed / angular speed.
        radius = distance / np.deg2rad(degrees)
        sign = 1 if token == "orbit_left" else -1
        turn = rotation(1, sign * degrees * fraction)
        forward = horizontal_axis(start, 2)
        center = start[:3, 3] + radius * forward
        pose[:3, 3] = center - radius * (turn @ forward)
        pose[:3, :3] = turn @ start[:3, :3]
    else:
        raise ValueError(f"Unsupported WorldCrafter demo camera action: {token}")
    return pose


def demo_camera_trajectory(camera_path, *, frame_count: int, suite: str, video_frame_count: int | None = None):
    if frame_count < CHUNK_FRAMES or frame_count % CHUNK_FRAMES:
        raise ValueError("WorldCrafter requires complete 33-frame chunks")
    if suite not in ("image_static", "image_dynamic", "memory_loop"):
        raise ValueError(f"Unsupported WorldCrafter suite: {suite}")
    tokens = _normalize_camera_path(camera_path)
    memory = suite == "memory_loop"
    video_frames = frame_count if video_frame_count is None else int(video_frame_count)
    if not 1 < video_frames <= frame_count:
        raise ValueError("Invalid WorldCrafter output frame count")
    # The image demo excludes the logical endpoint; Memory explicitly ends at
    # the returned pose. Equal segment durations preserve inverse translations.
    # Base's VAE trims up to three final camera frames. Complete a memory loop
    # on the final visible frame, then hold the endpoint through trimmed poses.
    logical_intervals = video_frames - 1 if memory else frame_count
    segment_chunks = logical_intervals / CHUNK_FRAMES / len(tokens)
    distance = TRANSLATION_PER_CHUNK * segment_chunks
    start = np.eye(4, dtype=np.float64)
    starts, angles = [], []
    for token in tokens:
        degrees = (20. if token.startswith("tilt_") else 15. if token.startswith("roll_") else 30.) if memory else ROTATION_DEGREES_PER_CHUNK * segment_chunks
        starts.append(start.copy())
        angles.append(degrees)
        start = pose_at(start, token, distance, degrees, 1.)
    poses = []
    for frame in range(frame_count):
        progress = min(frame, logical_intervals) / logical_intervals * len(tokens)
        index = min(int(progress), len(tokens) - 1)
        poses.append(pose_at(starts[index], tokens[index], distance, angles[index], progress - index))
    poses = np.asarray(poses, dtype=np.float32)
    metadata = dict(recipe=RECIPE, camera_path=tokens, frame_count=frame_count, video_frame_count=video_frames, suite=suite,
        fps=16, frames_per_chunk=CHUNK_FRAMES, translation_metres_per_chunk=TRANSLATION_PER_CHUNK,
        image_rotation_degrees_per_chunk=ROTATION_DEGREES_PER_CHUNK,
        memory_preserves_task_turn_angles=memory, segment_chunks=segment_chunks,
        translation_per_segment=distance, rotation_degrees_per_segment=angles,
        endpoint_sampling="include" if memory else "exclude",
        coordinate_convention="global metric c2w; x right, y down, z forward",
        orbit_policy="circular orbit matching demo linear and angular speeds",
        logical_endpoint_c2w=start.tolist())
    return poses, metadata


def demo_camera_annotation(sample, *, frame_count: int, video_frame_count: int | None = None):
    annotation = local_pose_annotation_path(sample)
    if annotation is not None:
        return annotation, "annotation", dict(recipe=RECIPE, authoritative_poses_preserved=True)
    poses, metadata = demo_camera_trajectory(_camera_path_for_sample(sample) or ["fixed"],
        frame_count=frame_count, suite=sample.suite, video_frame_count=video_frame_count)
    metadata["sample_id"] = sample.sample_id
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    folder = project_root() / "cache/model_runtime/worldcrafter" / RECIPE / f"{sample.prediction_stem}--{key}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "poses.npy"
    if target.is_file():
        if not np.array_equal(np.load(target, allow_pickle=False), poses):
            raise ValueError(f"Changed cached WorldCrafter demo trajectory: {target}")
    else:
        temporary = folder / f"poses.{uuid.uuid4().hex}.npy"
        np.save(temporary, poses, allow_pickle=False)
        temporary.replace(target)
    metadata["poses_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (folder / "camera_recipe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return folder, "synthetic_worldcrafter_demo_v1", metadata
