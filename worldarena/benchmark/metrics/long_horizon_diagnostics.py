"""Diagnostic metrics for long-horizon video evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, clamp01, normalize_score, resolve_metric_backend
from worldarena.benchmark.quality_backends import compute_musiq_scores, compute_quality_component_scores
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.common.video_io import read_video_frames


LONG_HORIZON_ACTION_METRICS = {
    "keystroke_strict_action_accuracy",
    "keystroke_partial_action_accuracy",
    "keystroke_traj_score",
    "keystroke_natet",
    "keystroke_nater",
    "keystroke_action_score",
}
LONG_HORIZON_VISUAL_METRICS = {
    "rollout_aesthetic_drift",
    "rollout_imaging_drift",
}
LONG_HORIZON_MEMORY_METRICS = {
    "scene_memory_f1",
}
LONG_HORIZON_DIAGNOSTIC_METRICS = (
    LONG_HORIZON_ACTION_METRICS | LONG_HORIZON_VISUAL_METRICS | LONG_HORIZON_MEMORY_METRICS
)

ACTION_BACKENDS = {"pose_action"}
VISUAL_BACKENDS = {"quality_drift"}
MEMORY_BACKENDS = {"point_cloud_f1"}

MOVE_PARTS = frozenset({"forward", "back", "left", "right"})
LOOK_PARTS = frozenset({"camera_l", "camera_r", "camera_up", "camera_down"})
ALL_ACTION_PARTS = MOVE_PARTS | LOOK_PARTS
NOOP_LABELS = {"", "idle", "none", "no_op", "noop", "no-op", "nomove", "stationary", "fixed"}
ACTION_ALIASES = {
    "w": "forward",
    "forward": "forward",
    "forwards": "forward",
    "move_forward": "forward",
    "backward": "back",
    "backwards": "back",
    "back": "back",
    "s": "back",
    "a": "left",
    "left": "left",
    "move_left": "left",
    "d": "right",
    "right": "right",
    "move_right": "right",
    "j": "camera_l",
    "look_left": "camera_l",
    "turn_left": "camera_l",
    "yaw_left": "camera_l",
    "yaw-": "camera_l",
    "camera_left": "camera_l",
    "camera_l": "camera_l",
    "l": "camera_r",
    "look_right": "camera_r",
    "turn_right": "camera_r",
    "yaw_right": "camera_r",
    "yaw+": "camera_r",
    "camera_right": "camera_r",
    "camera_r": "camera_r",
    "i": "camera_up",
    "look_up": "camera_up",
    "pitch_up": "camera_up",
    "pitch+": "camera_up",
    "camera_up": "camera_up",
    "k": "camera_down",
    "look_down": "camera_down",
    "pitch_down": "camera_down",
    "pitch-": "camera_down",
    "camera_down": "camera_down",
}


def _not_applicable(metric_name: str, backend: str, reason: str, **details: Any) -> MetricOutput:
    """Build a skipped diagnostic output for long-horizon metrics."""
    return MetricOutput(
        raw=None,
        normalized=None,
        backend=backend,
        details=details,
        eligibility_status="not_applicable",
        error=reason,
    )


def _round_list(values: Sequence[float], digits: int = 6) -> list[float]:
    """Round numeric detail lists for stable JSON serialization."""
    return [round(float(value), digits) for value in values]


def _normalize_action_text(value: Any) -> str:
    """Canonicalize raw action labels into comparable lowercase tokens."""
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def action_parts_from_label(value: Any) -> frozenset[str]:
    """Parse a text or alias label into canonical move/look action parts."""
    label = _normalize_action_text(value)
    if label in NOOP_LABELS:
        return frozenset()

    parts: set[str] = set()
    mutable = label
    for raw_token in sorted(ACTION_ALIASES, key=len, reverse=True):
        if len(raw_token) == 1:
            continue
        canonical = ACTION_ALIASES[raw_token]
        token = raw_token.replace("-", "_")
        if token in mutable:
            parts.add(canonical)
            mutable = mutable.replace(token, "_")

    for token in mutable.split("_"):
        canonical = ACTION_ALIASES.get(token)
        if canonical is not None:
            parts.add(canonical)
    return frozenset(parts)


def _label_from_keyboard_mouse(keyboard_row: Any, mouse_row: Any) -> frozenset[str]:
    """Decode one keyboard/mouse row into move and camera action parts."""
    keyboard = np.asarray(keyboard_row or [], dtype=np.float32)
    mouse = np.asarray(mouse_row or [], dtype=np.float32)
    parts: set[str] = set()

    if keyboard.shape[0] >= 4:
        if keyboard[0] > 0.5:
            parts.add("forward")
        if keyboard[1] > 0.5:
            parts.add("back")
        if keyboard[2] > 0.5:
            parts.add("left")
        if keyboard[3] > 0.5:
            parts.add("right")

    # Matrix-Game-2 TempleRun-like rows use a different seven-column layout.
    if keyboard.shape[0] >= 7 and not parts:
        if keyboard[3] > 0.5:
            parts.add("camera_l")
        if keyboard[4] > 0.5:
            parts.add("camera_r")
        if keyboard[5] > 0.5:
            parts.add("left")
        if keyboard[6] > 0.5:
            parts.add("right")

    if mouse.shape[0] >= 2:
        if mouse[0] > 1e-6:
            parts.add("camera_up")
        elif mouse[0] < -1e-6:
            parts.add("camera_down")
        if mouse[1] > 1e-6:
            parts.add("camera_r")
        elif mouse[1] < -1e-6:
            parts.add("camera_l")
    return frozenset(parts)


def _parts_from_frame_action(value: Any) -> frozenset[str]:
    """Expand structured frame action dicts into canonical action parts."""
    if not isinstance(value, dict):
        return action_parts_from_label(value)
    parts: set[str] = set()
    parts.update(action_parts_from_label(value.get("move")))
    parts.update(action_parts_from_label(value.get("view")))
    parts.update(action_parts_from_label(value.get("action")))
    return frozenset(parts)


def _pad_or_trim_parts(parts: list[frozenset[str]], frame_count: int) -> list[frozenset[str]]:
    """Pad short action timelines or trim them to the target frame count."""
    if frame_count <= 0:
        return []
    if not parts:
        return [frozenset() for _ in range(frame_count)]
    if len(parts) >= frame_count:
        return parts[:frame_count]
    return parts + [parts[-1] for _ in range(frame_count - len(parts))]


def action_parts_from_payload(payload: dict[str, Any], frame_count: int) -> list[frozenset[str]]:
    """Expand action JSON payloads into per-frame action-part timelines."""
    keyboard_rows = list(payload.get("keyboard_condition") or [])
    mouse_rows = list(payload.get("mouse_condition") or [])
    if keyboard_rows or mouse_rows:
        total_rows = max(len(keyboard_rows), len(mouse_rows), int(payload.get("total_frames") or 0), frame_count)
        parts = [
            _label_from_keyboard_mouse(
                keyboard_rows[index] if index < len(keyboard_rows) else None,
                mouse_rows[index] if index < len(mouse_rows) else None,
            )
            for index in range(total_rows)
        ]
        return _pad_or_trim_parts(parts, frame_count)

    frame_actions = payload.get("frame_actions")
    if isinstance(frame_actions, list) and frame_actions:
        return _pad_or_trim_parts([_parts_from_frame_action(item) for item in frame_actions], frame_count)

    actions = [item for item in (payload.get("actions") or []) if str(item).strip()]
    if actions:
        frames_per_action = int(payload.get("frames_per_action") or math.ceil(frame_count / max(len(actions), 1)))
        expanded: list[frozenset[str]] = []
        for action in actions:
            expanded.extend([action_parts_from_label(action)] * max(frames_per_action, 1))
        return _pad_or_trim_parts(expanded, frame_count)

    return []


def _candidate_action_payload_paths(prediction_path: Path) -> list[Path]:
    """List likely action sidecar paths beside a prediction video."""
    stem = prediction_path.stem
    parent = prediction_path.resolve().parent
    return [
        parent / f"{stem}_actions.json",
        parent / "actions" / f"{stem}_actions.json",
        parent / "_actions" / f"{stem}_actions.json",
    ]


def _load_action_payload_for_prediction(
    sample: BenchmarkSample,
    prediction_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load action JSON from generation records or local sidecars."""
    try:
        from worldarena.benchmark.metrics.realtime import _find_generation_record, _load_action_payload

        record = _find_generation_record(sample, prediction_path)
        payload = _load_action_payload(record, prediction_path)
        if payload is not None:
            return payload, "generation_record"
    except Exception:
        pass

    for path in _candidate_action_payload_paths(prediction_path):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload, str(path)
    return None, None


def _load_pose_array(path: Path) -> np.ndarray:
    """Load a camera pose trajectory as an [N, 4, 4] float32 array."""
    if path.is_dir():
        for name in ("poses_c2w.npy", "poses.npy", "camera_poses.npy"):
            candidate = path / name
            if candidate.exists():
                return _load_pose_array(candidate)
        raise FileNotFoundError(f"pose directory has no supported pose file: {path}")

    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        for key in ("poses_c2w", "poses", "data"):
            if key in payload:
                poses = np.asarray(payload[key], dtype=np.float32)
                break
        else:
            raise KeyError(f"pose archive has no poses_c2w/poses/data array: {path}")
    else:
        poses = np.asarray(np.load(path), dtype=np.float32)

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected pose array [N,4,4], got {poses.shape} from {path}")
    return poses.astype(np.float32)


def _candidate_pose_paths(prediction_path: Path, payload: dict[str, Any] | None) -> list[Path]:
    """Enumerate pose sidecar paths from payload hints and naming conventions."""
    paths: list[Path] = []
    base_dir = prediction_path.resolve().parent
    for key in (
        "prediction_pose_path",
        "pose_path",
        "poses_path",
        "vipe_annotation_path",
        "prediction_annotation_path",
    ):
        value = (payload or {}).get(key)
        if not value:
            continue
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        paths.append(candidate)

    stem = prediction_path.stem
    paths.extend(
        [
            base_dir / "annotations_vipe" / f"{stem}_vipe_ann",
            base_dir / f"{stem}_vipe_ann",
            base_dir / f"{stem}_poses_c2w.npy",
            base_dir / f"{stem}_poses.npy",
        ]
    )
    return paths


def _load_prediction_poses(
    prediction_path: Path,
    payload: dict[str, Any] | None,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve and load the first available prediction pose sidecar."""
    for path in _candidate_pose_paths(prediction_path, payload):
        if not path.exists():
            continue
        return _load_pose_array(path), str(path)
    return None, None


def _rotation_matrix_x(angle_degrees: float) -> np.ndarray:
    """Build a 3x3 rotation matrix about the X axis in degrees."""
    radians = math.radians(angle_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float32,
    )


def _rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    """Build a 3x3 rotation matrix about the Y axis in degrees."""
    radians = math.radians(angle_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
        dtype=np.float32,
    )


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-8)


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized quaternion to a 3x3 rotation matrix."""
    qx, qy, qz, qw = (np.asarray(quaternion, dtype=np.float64) / max(float(np.linalg.norm(quaternion)), 1e-8))
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _slerp(start: np.ndarray, end: np.ndarray, ratio: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    start_unit = np.asarray(start, dtype=np.float32) / max(float(np.linalg.norm(start)), 1e-8)
    end_unit = np.asarray(end, dtype=np.float32) / max(float(np.linalg.norm(end)), 1e-8)
    dot = float(np.clip(np.dot(start_unit, end_unit), -1.0, 1.0))
    if dot < 0.0:
        end_unit = -end_unit
        dot = -dot
    if dot > 0.9995:
        blended = start_unit + ratio * (end_unit - start_unit)
        return (blended / max(float(np.linalg.norm(blended)), 1e-8)).astype(np.float32)
    theta_0 = math.acos(dot)
    theta = theta_0 * ratio
    sin_theta_0 = math.sin(theta_0)
    return (
        math.sin(theta_0 - theta) / sin_theta_0 * start_unit
        + math.sin(theta) / sin_theta_0 * end_unit
    ).astype(np.float32)


def _resample_pose_sequence_by_index(poses: np.ndarray, target_count: int) -> np.ndarray:
    """Resample poses in time with SLERP rotations and linear translation."""
    if len(poses) == target_count:
        return poses.astype(np.float32)
    if target_count <= 1:
        return poses[:1].astype(np.float32)
    source_positions = np.linspace(0.0, 1.0, num=len(poses), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_count, dtype=np.float32)
    translations = np.stack(
        [
            np.interp(target_positions, source_positions, poses[:, axis, 3])
            for axis in range(3)
        ],
        axis=1,
    ).astype(np.float32)
    quaternions = np.stack([_matrix_to_quaternion(pose[:3, :3]) for pose in poses], axis=0)
    rotations: list[np.ndarray] = []
    for target in target_positions:
        upper = int(np.searchsorted(source_positions, target, side="right"))
        lower = max(0, upper - 1)
        upper = min(upper, len(source_positions) - 1)
        if lower == upper or source_positions[upper] == source_positions[lower]:
            rotations.append(_quaternion_to_matrix(quaternions[lower]))
            continue
        ratio = float((target - source_positions[lower]) / (source_positions[upper] - source_positions[lower]))
        rotations.append(_quaternion_to_matrix(_slerp(quaternions[lower], quaternions[upper], ratio)))
    result = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=target_count, axis=0)
    result[:, :3, :3] = np.stack(rotations, axis=0)
    result[:, :3, 3] = translations
    return result.astype(np.float32)


def _pitch_yaw_from_rotation(rotation: np.ndarray) -> tuple[float, float]:
    """Extract camera pitch and yaw in degrees from a rotation matrix."""
    forward = np.asarray(rotation, dtype=np.float64) @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    horizontal = max(math.sqrt(float(forward[0] ** 2 + forward[2] ** 2)), 1e-8)
    pitch = math.degrees(math.atan2(float(-forward[1]), horizontal))
    return pitch, yaw


def _rotation_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Measure the geodesic angle between two rotation matrices in degrees."""
    delta = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _pose_parts_from_relative_pose(
    relative_pose: np.ndarray,
    *,
    move_threshold: float,
    rotation_threshold_degrees: float,
    axis_threshold: float,
) -> frozenset[str]:
    """Discretize a relative pose into move and look action parts."""
    translation = np.asarray(relative_pose[:3, 3], dtype=np.float64)
    magnitude = float(np.linalg.norm(translation))
    parts: set[str] = set()
    if magnitude > move_threshold:
        axis_cosine = 0.5
        if translation[2] / magnitude > axis_cosine and abs(float(translation[2])) > axis_threshold:
            parts.add("forward")
        if translation[2] / magnitude < -axis_cosine and abs(float(translation[2])) > axis_threshold:
            parts.add("back")
        if translation[0] / magnitude > axis_cosine and abs(float(translation[0])) > axis_threshold:
            parts.add("right")
        if translation[0] / magnitude < -axis_cosine and abs(float(translation[0])) > axis_threshold:
            parts.add("left")

    pitch, yaw = _pitch_yaw_from_rotation(relative_pose[:3, :3])
    if yaw > rotation_threshold_degrees:
        parts.add("camera_r")
    elif yaw < -rotation_threshold_degrees:
        parts.add("camera_l")
    if pitch > rotation_threshold_degrees:
        parts.add("camera_up")
    elif pitch < -rotation_threshold_degrees:
        parts.add("camera_down")
    return frozenset(parts)


def strict_partial_accuracy(
    predicted: Sequence[frozenset[str]],
    ground_truth: Sequence[frozenset[str]],
) -> dict[str, float]:
    """Compute strict, partial, move-only, and look-only action accuracies."""
    count = min(len(predicted), len(ground_truth))
    if count <= 0:
        raise ValueError("strict/partial accuracy requires at least one aligned frame")
    strict_hits = 0
    partial_hits = 0
    move_hits = 0
    look_hits = 0
    move_total = 0
    look_total = 0
    for pred, gt in zip(predicted[:count], ground_truth[:count]):
        if pred == gt:
            strict_hits += 1
        if pred & gt or (not pred and not gt):
            partial_hits += 1
        gt_move = gt & MOVE_PARTS
        gt_look = gt & LOOK_PARTS
        if gt_move:
            move_total += 1
            if (pred & MOVE_PARTS) == gt_move:
                move_hits += 1
        if gt_look:
            look_total += 1
            if (pred & LOOK_PARTS) == gt_look:
                look_hits += 1
    return {
        "strict_accuracy": strict_hits / count,
        "partial_accuracy": partial_hits / count,
        "move_accuracy": move_hits / move_total if move_total else 1.0,
        "look_accuracy": look_hits / look_total if look_total else 1.0,
        "aligned_frame_count": float(count),
    }


def discretize_pose_actions(
    poses_c2w: np.ndarray,
    ground_truth: Sequence[frozenset[str]],
    *,
    latent_stride: int = 4,
    move_thresholds: Sequence[float] = (0.002, 0.005, 0.01),
    rotation_threshold_degrees: float = 0.2,
) -> dict[str, Any]:
    """Sweep move thresholds to discretize poses into action labels."""
    action_count = len(ground_truth)
    if len(poses_c2w) < 2 or action_count <= 0:
        raise ValueError("pose action discretization requires poses and ground-truth actions")

    stride = max(int(latent_stride), 1)
    pose_count = len(poses_c2w)
    latent_count = min(max((pose_count - 1) // stride, 1), max(math.ceil(action_count / stride), 1))
    relative_poses: list[np.ndarray] = []
    for latent_index in range(latent_count):
        start = min(latent_index * stride, pose_count - 1)
        end = min(start + stride, pose_count - 1)
        relative_poses.append(np.linalg.inv(poses_c2w[start]) @ poses_c2w[end])

    best_predictions: list[frozenset[str]] = []
    best_threshold = None
    best_accuracy = -1.0
    sweep: dict[str, float] = {}
    axis_threshold = min(float(value) for value in move_thresholds) / 2.0 * stride
    rot_threshold = float(rotation_threshold_degrees) * stride
    for base_threshold in move_thresholds:
        threshold = float(base_threshold) * stride
        latent_predictions = [
            _pose_parts_from_relative_pose(
                relative_pose,
                move_threshold=threshold,
                rotation_threshold_degrees=rot_threshold,
                axis_threshold=axis_threshold,
            )
            for relative_pose in relative_poses
        ]
        predictions: list[frozenset[str]] = []
        for item in latent_predictions:
            predictions.extend([item] * stride)
        predictions = _pad_or_trim_parts(predictions, action_count)
        strict = strict_partial_accuracy(predictions, ground_truth)["strict_accuracy"]
        sweep[str(threshold)] = strict
        if strict > best_accuracy:
            best_accuracy = strict
            best_threshold = threshold
            best_predictions = predictions

    return {
        "predicted_parts": best_predictions,
        "threshold": best_threshold,
        "threshold_sweep": sweep,
        **strict_partial_accuracy(best_predictions, ground_truth),
    }


def _action_segments(parts: Sequence[frozenset[str]]) -> list[tuple[int, int, frozenset[str]]]:
    """Split a per-frame action timeline into contiguous constant segments."""
    if not parts:
        return []
    segments: list[tuple[int, int, frozenset[str]]] = []
    start = 0
    previous = parts[0]
    for index, current in enumerate(parts[1:], start=1):
        if current == previous:
            continue
        segments.append((start, index, previous))
        start = index
        previous = current
    segments.append((start, len(parts), previous))
    return segments


def _segment_point_allocations(segment_count: int, total_points: int) -> list[int]:
    """Distribute trajectory sample points across action segments."""
    segment_count = max(int(segment_count), 1)
    total_points = max(int(total_points), segment_count * 2)
    base = max(total_points // segment_count, 2)
    allocations = [base for _ in range(segment_count)]
    for index in range(total_points - base * segment_count):
        allocations[index % segment_count] += 1
    return allocations


def _resample_poses_by_arclength(poses: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample pose positions and rotations uniformly along path length."""
    count = max(int(count), 1)
    if len(poses) == 1:
        return (
            np.repeat(poses[0, :3, 3][None], repeats=count, axis=0).astype(np.float32),
            np.repeat(poses[0, :3, :3][None], repeats=count, axis=0).astype(np.float32),
        )
    positions = poses[:, :3, 3].astype(np.float32)
    distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)]).astype(np.float32)
    if float(cumulative[-1]) <= 1e-8:
        cumulative = np.linspace(0.0, 1.0, num=len(poses), dtype=np.float32)
    targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), num=count, dtype=np.float32)
    sampled_positions = np.stack(
        [np.interp(targets, cumulative, positions[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)

    quaternions = np.stack([_matrix_to_quaternion(pose[:3, :3]) for pose in poses], axis=0)
    sampled_rotations: list[np.ndarray] = []
    for target in targets:
        upper = int(np.searchsorted(cumulative, target, side="right"))
        lower = max(0, upper - 1)
        upper = min(upper, len(cumulative) - 1)
        if lower == upper or cumulative[upper] == cumulative[lower]:
            sampled_rotations.append(_quaternion_to_matrix(quaternions[lower]))
            continue
        ratio = float((target - cumulative[lower]) / (cumulative[upper] - cumulative[lower]))
        sampled_rotations.append(_quaternion_to_matrix(_slerp(quaternions[lower], quaternions[upper], ratio)))
    return sampled_positions, np.stack(sampled_rotations, axis=0).astype(np.float32)


def _ideal_segment_poses(predicted_segment: np.ndarray, parts: frozenset[str]) -> np.ndarray:
    """Synthesize ideal poses for an action segment from predicted motion."""
    start_pose = predicted_segment[0]
    end_pose = predicted_segment[-1]
    count = len(predicted_segment)
    result = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=count, axis=0)

    local_direction = np.zeros(3, dtype=np.float32)
    if "forward" in parts:
        local_direction += np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if "back" in parts:
        local_direction += np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    if "right" in parts:
        local_direction += np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if "left" in parts:
        local_direction += np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)

    translation_length = float(np.linalg.norm(end_pose[:3, 3] - start_pose[:3, 3]))
    if float(np.linalg.norm(local_direction)) > 1e-8 and translation_length > 0.0:
        local_direction = local_direction / max(float(np.linalg.norm(local_direction)), 1e-8)
        world_direction = start_pose[:3, :3] @ local_direction
        positions = start_pose[:3, 3] + np.linspace(0.0, translation_length, num=count, dtype=np.float32)[:, None] * world_direction
    else:
        positions = np.repeat(start_pose[:3, 3][None], repeats=count, axis=0)

    relative_rotation = start_pose[:3, :3].T @ end_pose[:3, :3]
    pred_pitch, pred_yaw = _pitch_yaw_from_rotation(relative_rotation)
    yaw = 0.0
    pitch = 0.0
    if "camera_r" in parts:
        yaw += abs(pred_yaw)
    if "camera_l" in parts:
        yaw -= abs(pred_yaw)
    if "camera_up" in parts:
        pitch += abs(pred_pitch)
    if "camera_down" in parts:
        pitch -= abs(pred_pitch)
    target_rotation = start_pose[:3, :3] @ _rotation_matrix_y(yaw) @ _rotation_matrix_x(pitch)
    start_quaternion = _matrix_to_quaternion(start_pose[:3, :3])
    target_quaternion = _matrix_to_quaternion(target_rotation)
    rotations = np.stack(
        [
            _quaternion_to_matrix(_slerp(start_quaternion, target_quaternion, float(alpha)))
            for alpha in np.linspace(0.0, 1.0, num=count, dtype=np.float32)
        ],
        axis=0,
    )
    result[:, :3, 3] = positions
    result[:, :3, :3] = rotations
    return result.astype(np.float32)


def trajectory_score(
    poses_c2w: np.ndarray,
    ground_truth: Sequence[frozenset[str]],
    *,
    total_points: int = 60,
    min_path_length: float = 0.5,
    min_rotation_degrees: float = 10.0,
) -> dict[str, float]:
    """Score trajectory error against idealized action segments with normalized ATE."""
    if len(ground_truth) <= 0 or len(poses_c2w) < 2:
        raise ValueError("trajectory score requires poses and ground-truth actions")
    poses = _resample_pose_sequence_by_index(np.asarray(poses_c2w, dtype=np.float32), len(ground_truth) + 1)
    segments = _action_segments(ground_truth)
    allocations = _segment_point_allocations(len(segments), total_points)
    pred_positions: list[np.ndarray] = []
    pred_rotations: list[np.ndarray] = []
    gt_positions: list[np.ndarray] = []
    gt_rotations: list[np.ndarray] = []
    for allocation, (start, end, parts) in zip(allocations, segments):
        predicted_segment = poses[start : end + 1]
        if len(predicted_segment) < 2:
            continue
        ideal_segment = _ideal_segment_poses(predicted_segment, parts)
        pred_pos, pred_rot = _resample_poses_by_arclength(predicted_segment, allocation)
        ideal_pos, ideal_rot = _resample_poses_by_arclength(ideal_segment, allocation)
        pred_positions.append(pred_pos)
        pred_rotations.append(pred_rot)
        gt_positions.append(ideal_pos)
        gt_rotations.append(ideal_rot)

    if not pred_positions:
        raise ValueError("trajectory score found no evaluable action segments")

    pred_pos_all = np.concatenate(pred_positions, axis=0)
    gt_pos_all = np.concatenate(gt_positions, axis=0)
    pred_rot_all = np.concatenate(pred_rotations, axis=0)
    gt_rot_all = np.concatenate(gt_rotations, axis=0)
    ate_t = float(np.mean(np.linalg.norm(pred_pos_all - gt_pos_all, axis=1)))
    rot_errors = [
        _rotation_angle_degrees(left, right)
        for left, right in zip(pred_rot_all, gt_rot_all)
    ]
    ate_r = float(np.mean(rot_errors))
    path_length = float(np.sum(np.linalg.norm(poses[1:, :3, 3] - poses[:-1, :3, 3], axis=1)))
    total_rotation = float(
        np.sum(
            [
                _rotation_angle_degrees(poses[index, :3, :3], poses[index + 1, :3, :3])
                for index in range(len(poses) - 1)
            ]
        )
    )
    natet = clamp01(ate_t / max(path_length, float(min_path_length)))
    nater = clamp01(ate_r / max(total_rotation, float(min_rotation_degrees)))
    score = 1.0 - (natet + nater) / 2.0
    return {
        "traj_score": clamp01(score),
        "natet": natet,
        "nater": nater,
        "ate_t": ate_t,
        "ate_r_degrees": ate_r,
        "path_length": path_length,
        "total_rotation_degrees": total_rotation,
    }


def compute_action_payload_metrics(
    poses_c2w: np.ndarray,
    action_payload: dict[str, Any],
    *,
    frame_count: int,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine discretized action accuracy and trajectory scores."""
    runtime = dict(runtime or {})
    gt_parts = action_parts_from_payload(action_payload, frame_count)
    if not gt_parts:
        raise ValueError("action payload does not contain usable frame actions")
    poses = np.asarray(poses_c2w, dtype=np.float32)
    discretized = discretize_pose_actions(
        poses,
        gt_parts,
        latent_stride=int(runtime.get("latent_stride", 4)),
        move_thresholds=tuple(float(value) for value in runtime.get("move_thresholds", [0.002, 0.005, 0.01])),
        rotation_threshold_degrees=float(runtime.get("rotation_threshold_degrees", 0.2)),
    )
    trajectory = trajectory_score(
        poses,
        gt_parts,
        total_points=int(runtime.get("trajectory_points", 60)),
        min_path_length=float(runtime.get("min_path_length", 0.5)),
        min_rotation_degrees=float(runtime.get("min_rotation_degrees", 10.0)),
    )
    action_score = (
        float(discretized["strict_accuracy"])
        + float(discretized["partial_accuracy"])
        + float(trajectory["traj_score"])
    ) / 3.0
    return {
        **discretized,
        **trajectory,
        "action_score": action_score,
        "gt_frame_count": len(gt_parts),
    }


def rollout_visual_score_from_components(
    aesthetic_score: float,
    imaging_score: float,
    aesthetic_drift: float,
    imaging_drift: float,
) -> float:
    """Combine absolute quality and inverted drift into WorldOdyssey S_visual."""
    return clamp01(
        (
            float(aesthetic_score)
            + (1.0 - float(aesthetic_drift))
            + float(imaging_score)
            + (1.0 - float(imaging_drift))
        )
        / 4.0
    )


def segment_drift(scores: Sequence[float], *, segment_count: int = 10) -> dict[str, Any]:
    """Measure relative quality drop between best and worst temporal score segments."""
    values = [float(value) for value in scores if np.isfinite(float(value))]
    if not values:
        raise ValueError("segment drift requires at least one finite score")
    count = max(1, min(int(segment_count), len(values)))
    segments = np.array_split(np.asarray(values, dtype=np.float32), count)
    segment_means = [float(np.mean(segment)) for segment in segments if len(segment)]
    best = max(segment_means)
    worst = min(segment_means)
    drift = 0.0 if abs(best) <= 1e-8 else (best - worst) / abs(best)
    return {
        "drift": clamp01(drift),
        "best_segment_mean": best,
        "worst_segment_mean": worst,
        "segment_means": segment_means,
        "segment_count": len(segment_means),
    }


_VISUAL_CACHE: dict[tuple[str, int, int, str], dict[str, Any]] = {}


def _visual_cache_key(prediction_path: Path, runtime: dict[str, Any]) -> tuple[str, int, int, str]:
    """Key cached visual-drift results by video identity and runtime options."""
    stat = prediction_path.stat()
    relevant = {
        "max_frames": runtime.get("max_frames"),
        "segment_count": runtime.get("segment_count"),
        "musiq_model_path": runtime.get("musiq_model_path"),
        "imaging_quality_preprocessing_mode": runtime.get("imaging_quality_preprocessing_mode"),
    }
    return (str(prediction_path.resolve()), stat.st_mtime_ns, stat.st_size, repr(sorted(relevant.items())))


def _sample_frames_for_visual(frames: list[np.ndarray], *, max_frames: int | None) -> tuple[list[np.ndarray], list[int]]:
    """Uniformly subsample decoded frames for quality drift analysis."""
    if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return frames, list(range(len(frames)))
    indices = np.rint(np.linspace(0, len(frames) - 1, num=max_frames)).astype(np.int64).tolist()
    return [frames[index] for index in indices], [int(index) for index in indices]


def compute_visual_drift_metrics(
    prediction_path: Path,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score aesthetic/imaging drift and combined rollout visual quality."""
    runtime = dict(runtime or {})
    key = _visual_cache_key(prediction_path, runtime)
    cached = _VISUAL_CACHE.get(key)
    if cached is not None:
        return cached

    frames, sampled_indices = _sample_frames_for_visual(
        read_video_frames(prediction_path),
        max_frames=None if runtime.get("max_frames") is None else int(runtime.get("max_frames")),
    )
    aesthetic = compute_quality_component_scores(frames, component_name="aesthetic")
    imaging = compute_musiq_scores(
        frames,
        model_path=None if runtime.get("musiq_model_path") is None else str(runtime.get("musiq_model_path")),
        preprocess_mode=str(runtime.get("imaging_quality_preprocessing_mode") or "longer"),
    )
    segment_count = int(runtime.get("segment_count", 10))
    aesthetic_drift = segment_drift(aesthetic.get("frame_scores") or [], segment_count=segment_count)
    imaging_drift = segment_drift(imaging.get("frame_scores") or [], segment_count=segment_count)
    aesthetic_raw = float(aesthetic["raw"])
    imaging_raw = float(imaging["raw"])
    visual_score = rollout_visual_score_from_components(
        aesthetic_raw,
        imaging_raw,
        float(aesthetic_drift["drift"]),
        float(imaging_drift["drift"]),
    )
    result = {
        "backend": "quality_drift",
        "sampled_frame_indices": sampled_indices,
        "frame_count": len(frames),
        "aesthetic_score": aesthetic_raw,
        "imaging_score": imaging_raw,
        "aesthetic_drift": aesthetic_drift,
        "imaging_drift": imaging_drift,
        "visual_score": visual_score,
        "aesthetic_backend": aesthetic.get("backend"),
        "imaging_backend": imaging.get("backend"),
    }
    _VISUAL_CACHE[key] = result
    return result


def _finite_points(points: np.ndarray) -> np.ndarray:
    """Drop non-finite rows from a point cloud array."""
    array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mask = np.isfinite(array).all(axis=1)
    return array[mask]


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Average points that fall into the same voxel grid cell."""
    points = _finite_points(points)
    if points.size == 0 or voxel_size <= 0.0:
        return points
    coords = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(coords, axis=0, return_inverse=True)
    result = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.zeros((int(inverse.max()) + 1,), dtype=np.float64)
    np.add.at(result, inverse, points)
    np.add.at(counts, inverse, 1.0)
    return (result / np.maximum(counts[:, None], 1.0)).astype(np.float32)


def _nearest_distances(query: np.ndarray, target: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
    """Compute nearest-neighbor distances from query to target point clouds."""
    query = _finite_points(query)
    target = _finite_points(target)
    if len(query) == 0 or len(target) == 0:
        return np.full((len(query),), np.inf, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        return cKDTree(target).query(query, k=1, workers=-1)[0].astype(np.float32)
    except Exception:
        distances: list[np.ndarray] = []
        for start in range(0, len(query), max(int(chunk_size), 1)):
            chunk = query[start : start + chunk_size]
            diff = chunk[:, None, :] - target[None, :, :]
            distances.append(np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)).astype(np.float32))
        return np.concatenate(distances, axis=0) if distances else np.asarray([], dtype=np.float32)


def _nearest_indices(query: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-neighbor distances and indices for rigid alignment."""
    query = _finite_points(query)
    target = _finite_points(target)
    if not len(query) or not len(target):
        raise ValueError("nearest-neighbor alignment requires non-empty point clouds")
    try:
        from scipy.spatial import cKDTree

        distance, index = cKDTree(target).query(query, k=1, workers=-1)
        return distance.astype(np.float64), index.astype(np.int64)
    except Exception:
        distance_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []
        for start in range(0, len(query), 2048):
            chunk = query[start : start + 2048]
            squared = np.sum((chunk[:, None, :] - target[None, :, :]) ** 2, axis=2)
            index = np.argmin(squared, axis=1)
            index_parts.append(index)
            distance_parts.append(np.sqrt(squared[np.arange(len(chunk)), index]))
        return np.concatenate(distance_parts), np.concatenate(index_parts)


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the proper SE(3) transform mapping source points to target points."""
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def trimmed_icp_align(
    source: np.ndarray,
    target: np.ndarray,
    *,
    trim_fraction: float = 0.80,
    max_iterations: int = 20,
    tolerance: float = 1e-5,
    max_translation_ratio: float = 0.10,
    max_rotation_degrees: float = 10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply bounded trimmed point-to-point ICP without scale fitting."""
    moving = _finite_points(source).astype(np.float64)
    fixed = _finite_points(target).astype(np.float64)
    if len(moving) < 3 or len(fixed) < 3:
        raise ValueError("trimmed ICP requires at least three points per cloud")
    original = moving.copy()
    total_rotation = np.eye(3, dtype=np.float64)
    total_translation = np.zeros(3, dtype=np.float64)
    previous_error = float("inf")
    iterations = 0
    keep_fraction = min(max(float(trim_fraction), 0.1), 1.0)
    for iteration in range(max(int(max_iterations), 1)):
        distance, index = _nearest_indices(moving, fixed)
        keep_count = max(3, int(math.ceil(len(distance) * keep_fraction)))
        keep = np.argpartition(distance, keep_count - 1)[:keep_count]
        rotation, translation = _rigid_transform(moving[keep], fixed[index[keep]])
        moving = (rotation @ moving.T).T + translation
        total_translation = rotation @ total_translation + translation
        total_rotation = rotation @ total_rotation
        error = float(np.mean(distance[keep]))
        iterations = iteration + 1
        if abs(previous_error - error) <= float(tolerance):
            break
        previous_error = error

    diagonal = float(np.linalg.norm(np.ptp(fixed, axis=0)))
    translation_norm = float(np.linalg.norm(total_translation))
    rotation_degrees = math.degrees(
        math.acos(float(np.clip((np.trace(total_rotation) - 1.0) * 0.5, -1.0, 1.0)))
    )
    accepted = bool(
        translation_norm <= max(diagonal * float(max_translation_ratio), 1e-8)
        and rotation_degrees <= float(max_rotation_degrees)
    )
    return (moving if accepted else original).astype(np.float32), {
        "alignment": "trimmed_icp" if accepted else "trimmed_icp_rejected_identity",
        "alignment_iterations": iterations,
        "alignment_trim_fraction": keep_fraction,
        "alignment_translation": translation_norm,
        "alignment_translation_ratio": translation_norm / max(diagonal, 1e-8),
        "alignment_rotation_degrees": rotation_degrees,
        "alignment_accepted": accepted,
    }


def scene_memory_f1(
    observation_points: np.ndarray,
    revisit_points: np.ndarray,
    *,
    distance_threshold: float,
) -> dict[str, float]:
    """Score revisit retention versus hallucination with an F1-style point metric."""
    obs = _finite_points(observation_points)
    rev = _finite_points(revisit_points)
    if len(obs) == 0 or len(rev) == 0:
        raise ValueError("scene memory F1 requires non-empty observation and revisit point clouds")
    obs_to_rev = _nearest_distances(obs, rev)
    rev_to_obs = _nearest_distances(rev, obs)
    threshold = max(float(distance_threshold), 1e-8)
    retention = float(np.mean(obs_to_rev < threshold))
    hallucination = float(np.mean(rev_to_obs >= threshold))
    precision = 1.0 - hallucination
    denominator = precision + retention
    f1 = 0.0 if denominator <= 1e-8 else 2.0 * precision * retention / denominator
    return {
        "scene_memory_f1": clamp01(f1),
        "retention": clamp01(retention),
        "hallucination": clamp01(hallucination),
        "precision": clamp01(precision),
        "distance_threshold": threshold,
    }


def _load_point_cloud(path: Path) -> np.ndarray:
    """Load XYZ points from npy, npz, or text point-cloud sidecars."""
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        for key in ("points", "point_cloud", "pobs", "prev", "observation", "revisit"):
            if key in payload:
                return _finite_points(payload[key])
        raise KeyError(f"point-cloud archive has no supported point array: {path}")
    if path.suffix.lower() == ".npy":
        return _finite_points(np.load(path))
    return _finite_points(np.loadtxt(path, dtype=np.float32))


def _explicit_scene_memory_clouds(
    prediction_path: Path,
    action_payload: dict[str, Any] | None,
    runtime: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Load observation/revisit clouds from explicit paths or NPZ sidecars."""
    base_dir = prediction_path.resolve().parent
    payload = dict(action_payload or {})
    explicit_obs = runtime.get("observation_point_cloud_path") or payload.get("observation_point_cloud_path")
    explicit_rev = runtime.get("revisit_point_cloud_path") or payload.get("revisit_point_cloud_path")
    if explicit_obs and explicit_rev:
        obs_path = Path(str(explicit_obs)).expanduser()
        rev_path = Path(str(explicit_rev)).expanduser()
        if not obs_path.is_absolute():
            obs_path = base_dir / obs_path
        if not rev_path.is_absolute():
            rev_path = base_dir / rev_path
        return _load_point_cloud(obs_path), _load_point_cloud(rev_path), f"{obs_path}::{rev_path}"

    sidecar = runtime.get("scene_memory_npz_path") or payload.get("scene_memory_npz_path")
    candidates = []
    if sidecar:
        candidate = Path(str(sidecar)).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        candidates.append(candidate)
    candidates.append(base_dir / f"{prediction_path.stem}_scene_memory.npz")

    for candidate in candidates:
        if not candidate.exists():
            continue
        data = np.load(candidate)
        obs_key = "pobs" if "pobs" in data else "observation"
        rev_key = "prev" if "prev" in data else "revisit"
        if obs_key in data and rev_key in data:
            return _finite_points(data[obs_key]), _finite_points(data[rev_key]), str(candidate)
    return None


def _resolve_vipe_dir(prediction_path: Path, action_payload: dict[str, Any] | None, runtime: dict[str, Any]) -> Path | None:
    """Locate ViPE annotation directories for depth-based scene memory."""
    base_dir = prediction_path.resolve().parent
    values = [
        runtime.get("vipe_annotation_path"),
        runtime.get("prediction_annotation_path"),
        (action_payload or {}).get("vipe_annotation_path"),
        (action_payload or {}).get("prediction_annotation_path"),
    ]
    for value in values:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        if path.exists():
            return path
    for candidate in [
        base_dir / "annotations_vipe" / f"{prediction_path.stem}_vipe_ann",
        base_dir / f"{prediction_path.stem}_vipe_ann",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_depth_mask(path: Path) -> np.ndarray:
    """Load a boolean depth-valid mask from npy or npz archives."""
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "mask" in payload:
            return np.asarray(payload["mask"], dtype=bool)
        first_key = payload.files[0]
        return np.asarray(payload[first_key], dtype=bool)
    return np.asarray(payload, dtype=bool)


def _intrinsics_for_index(intrinsics: np.ndarray, index: int) -> tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from several supported intrinsics layouts."""
    values = np.asarray(intrinsics)
    if values.ndim == 3:
        matrix = values[min(index, len(values) - 1)]
        return float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])
    if values.ndim == 2 and values.shape == (3, 3):
        return float(values[0, 0]), float(values[1, 1]), float(values[0, 2]), float(values[1, 2])
    if values.ndim == 2 and values.shape[1] >= 4:
        vector = values[min(index, len(values) - 1)]
        return float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3])
    if values.ndim == 1 and values.shape[0] >= 4:
        return float(values[0]), float(values[1]), float(values[2]), float(values[3])
    raise ValueError(f"unsupported intrinsics shape: {values.shape}")


def _frame_depth_points(
    depth: np.ndarray,
    valid: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsics: np.ndarray,
    *,
    frame_index: int,
    max_points: int,
) -> np.ndarray:
    """Back-project masked depth pixels into world-space 3D points."""
    mask = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(xs) > max_points:
        selected = np.rint(np.linspace(0, len(xs) - 1, num=max_points)).astype(np.int64)
        ys = ys[selected]
        xs = xs[selected]
    z = np.asarray(depth[ys, xs], dtype=np.float32)
    fx, fy, cx, cy = _intrinsics_for_index(intrinsics, frame_index)
    x = (xs.astype(np.float32) - cx) * z / max(fx, 1e-8)
    y = (ys.astype(np.float32) - cy) * z / max(fy, 1e-8)
    camera_points = np.stack([x, y, z], axis=1).astype(np.float32)
    world_points = (pose_c2w[:3, :3] @ camera_points.T).T + pose_c2w[:3, 3]
    return world_points.astype(np.float32)


def _select_segment_indices(start: int, end: int, count: int) -> list[int]:
    """Pick evenly spaced frame indices inside a segment window."""
    if end <= start:
        return []
    frame_count = end - start
    target_count = min(max(int(count), 1), frame_count)
    return sorted({int(start + round(value)) for value in np.linspace(0, frame_count - 1, num=target_count)})


def _reconstruct_vipe_scene_clouds(
    vipe_dir: Path,
    action_parts: Sequence[frozenset[str]],
    *,
    runtime: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rebuild observation and revisit clouds from ViPE depth and poses."""
    poses = _load_pose_array(vipe_dir)
    intrinsics = np.asarray(np.load(vipe_dir / "intrinsics.npy"), dtype=np.float32)
    depth = np.load(vipe_dir / "depth_metric.npy", mmap_mode="r")
    valid = _load_depth_mask(vipe_dir / "depth_valid_mask.npz")
    frame_total = min(len(poses), int(depth.shape[0]), int(valid.shape[0]))
    if frame_total <= 1:
        raise ValueError("ViPE scene memory sidecar has too few frames")

    segments = _action_segments(action_parts)
    if len(segments) >= 2:
        transition = segments[0][1]
    else:
        transition = max(len(action_parts) // 2, 1)
    transition_index = int(round(transition * (frame_total - 1) / max(len(action_parts), 1)))
    transition_index = min(max(transition_index, 1), frame_total - 1)
    max_frames_per_segment = int(runtime.get("max_frames_per_segment", 8))
    max_points_per_frame = int(runtime.get("max_points_per_frame", 4096))
    obs_indices = _select_segment_indices(0, transition_index, max_frames_per_segment)
    rev_indices = _select_segment_indices(transition_index, frame_total, max_frames_per_segment)

    obs_points = [
        _frame_depth_points(
            np.asarray(depth[index]),
            np.asarray(valid[index]),
            poses[index],
            intrinsics,
            frame_index=index,
            max_points=max_points_per_frame,
        )
        for index in obs_indices
    ]
    rev_points = [
        _frame_depth_points(
            np.asarray(depth[index]),
            np.asarray(valid[index]),
            poses[index],
            intrinsics,
            frame_index=index,
            max_points=max_points_per_frame,
        )
        for index in rev_indices
    ]
    obs = np.concatenate(obs_points, axis=0) if obs_points else np.empty((0, 3), dtype=np.float32)
    rev = np.concatenate(rev_points, axis=0) if rev_points else np.empty((0, 3), dtype=np.float32)
    return obs, rev, {
        "vipe_annotation_path": str(vipe_dir),
        "transition_frame": transition_index,
        "observation_frame_indices": obs_indices,
        "revisit_frame_indices": rev_indices,
    }


def compute_scene_memory_metric(
    prediction_path: Path,
    action_payload: dict[str, Any] | None,
    *,
    frame_count: int,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve scene-memory point clouds and compute retention F1."""
    runtime = dict(runtime or {})
    source = _explicit_scene_memory_clouds(prediction_path, action_payload, runtime)
    details: dict[str, Any] = {}
    if source is not None:
        obs, rev, source_text = source
        details["point_cloud_source"] = source_text
    else:
        if action_payload is None:
            raise FileNotFoundError("scene memory requires action payload or explicit point-cloud sidecar")
        action_parts = action_parts_from_payload(action_payload, frame_count)
        vipe_dir = _resolve_vipe_dir(prediction_path, action_payload, runtime)
        if vipe_dir is None:
            raise FileNotFoundError("scene memory point-cloud or ViPE sidecar is unavailable")
        obs, rev, details = _reconstruct_vipe_scene_clouds(vipe_dir, action_parts, runtime=runtime)
        details["point_cloud_source"] = "vipe_depth_reconstruction"

    voxel_size = float(runtime.get("voxel_size", 0.0))
    obs = _voxel_downsample(obs, voxel_size)
    rev = _voxel_downsample(rev, voxel_size)
    alignment_mode = runtime.get("alignment")
    if alignment_mode == "trimmed_icp" and len(obs) and len(rev):
        rev, alignment_details = trimmed_icp_align(
            rev,
            obs,
            trim_fraction=float(runtime.get("icp_trim_fraction", 0.80)),
            max_iterations=int(runtime.get("icp_max_iterations", 20)),
            tolerance=float(runtime.get("icp_tolerance", 1e-5)),
            max_translation_ratio=float(runtime.get("icp_max_translation_ratio", 0.10)),
            max_rotation_degrees=float(runtime.get("icp_max_rotation_degrees", 10.0)),
        )
        details.update(alignment_details)
    elif bool(runtime.get("centroid_align", True)) and len(obs) and len(rev):
        rev = rev + (np.mean(obs, axis=0) - np.mean(rev, axis=0))
        details["alignment"] = "centroid_translation"
    else:
        details["alignment"] = "identity"

    combined = np.concatenate([obs, rev], axis=0)
    diagonal = float(np.linalg.norm(np.max(combined, axis=0) - np.min(combined, axis=0))) if len(combined) else 0.0
    threshold = runtime.get("distance_threshold")
    if threshold is None:
        threshold = max(diagonal * float(runtime.get("distance_threshold_ratio", 0.02)), 1e-6)
    result = scene_memory_f1(obs, rev, distance_threshold=float(threshold))
    result["observation_point_count"] = float(len(obs))
    result["revisit_point_count"] = float(len(rev))
    result["scene_diagonal"] = diagonal
    return {**result, "details": details}


class KeystrokeActionMetric(Metric):
    """Expose one keystroke diagnostic score from pose and action sidecars."""
    def __init__(
        self,
        *,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Register a keystroke sub-metric with pose-action runtime options."""
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Resolve backend -> str."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="pose_action",
            supported_backends=ACTION_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Score one keystroke metric from poses and action payload sidecars."""
        del reference
        backend = self._resolve_backend()
        prediction_path = Path(prediction.path)
        details: dict[str, Any] = {
            "prediction_path": str(prediction_path),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
            "native_frame_count": prediction.native_frame_count,
        }
        action_payload, action_source = _load_action_payload_for_prediction(sample, prediction_path)
        if action_payload is None:
            return _not_applicable(self.name, backend, "action specification unavailable", **details)
        poses, pose_source = _load_prediction_poses(prediction_path, action_payload)
        if poses is None:
            details["action_source"] = action_source
            return _not_applicable(self.name, backend, "prediction pose sidecar unavailable", **details)
        frame_count = int(prediction.native_frame_count or len(prediction.frames) or max(len(poses) - 1, 1))
        try:
            result = compute_action_payload_metrics(
                poses,
                action_payload,
                frame_count=frame_count,
                runtime=self.runtime,
            )
            details.update(
                {
                    "action_source": action_source,
                    "pose_source": pose_source,
                    "frame_count": frame_count,
                    "gt_frame_count": int(result["gt_frame_count"]),
                    "aligned_frame_count": int(result["aligned_frame_count"]),
                    "selected_move_threshold": result["threshold"],
                    "threshold_sweep": result["threshold_sweep"],
                    "strict_accuracy": round(float(result["strict_accuracy"]), 6),
                    "partial_accuracy": round(float(result["partial_accuracy"]), 6),
                    "move_accuracy": round(float(result["move_accuracy"]), 6),
                    "look_accuracy": round(float(result["look_accuracy"]), 6),
                    "traj_score": round(float(result["traj_score"]), 6),
                    "natet": round(float(result["natet"]), 6),
                    "nater": round(float(result["nater"]), 6),
                    "ate_t": round(float(result["ate_t"]), 6),
                    "ate_r_degrees": round(float(result["ate_r_degrees"]), 6),
                    "path_length": round(float(result["path_length"]), 6),
                    "total_rotation_degrees": round(float(result["total_rotation_degrees"]), 6),
                }
            )
            raw_key = {
                "keystroke_strict_action_accuracy": "strict_accuracy",
                "keystroke_partial_action_accuracy": "partial_accuracy",
                "keystroke_traj_score": "traj_score",
                "keystroke_natet": "natet",
                "keystroke_nater": "nater",
                "keystroke_action_score": "action_score",
            }[self.name]
            raw = float(result[raw_key])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization) if self.normalization else raw,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:
            details["action_source"] = action_source
            details["pose_source"] = pose_source
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


class VisualDriftMetric(Metric):
    """Expose one rollout visual-drift sub-metric from quality scoring."""
    def __init__(
        self,
        *,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Register a visual-drift sub-metric with quality runtime options."""
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Resolve backend -> str."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="quality_drift",
            supported_backends=VISUAL_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Return aesthetic or imaging rollout drift scores."""
        del sample
        del reference
        backend = self._resolve_backend()
        prediction_path = Path(prediction.path)
        details: dict[str, Any] = {
            "prediction_path": str(prediction_path),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }
        if prediction.modality != "video":
            return _not_applicable(self.name, backend, "visual drift requires a video prediction", **details)
        try:
            result = compute_visual_drift_metrics(prediction_path, runtime=self.runtime)
            details.update(
                {
                    "frame_count": result["frame_count"],
                    "sampled_full_video_indices": result["sampled_frame_indices"],
                    "aesthetic_backend": result["aesthetic_backend"],
                    "imaging_backend": result["imaging_backend"],
                    "aesthetic_score": round(float(result["aesthetic_score"]), 6),
                    "imaging_score": round(float(result["imaging_score"]), 6),
                    "visual_score": round(float(result["visual_score"]), 6),
                    "aesthetic_drift": result["aesthetic_drift"],
                    "imaging_drift": result["imaging_drift"],
                }
            )
            raw = {
                "rollout_aesthetic_drift": float(result["aesthetic_drift"]["drift"]),
                "rollout_imaging_drift": float(result["imaging_drift"]["drift"]),
            }[self.name]
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization) if self.normalization else raw,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


class SceneMemoryF1Metric(Metric):
    """Score long-horizon scene memory with point-cloud F1 diagnostics."""
    def __init__(
        self,
        *,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Store scene-memory reconstruction and threshold runtime options."""
        super().__init__("scene_memory_f1", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Resolve backend -> str."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="point_cloud_f1",
            supported_backends=MEMORY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Compute scene-memory F1 from explicit clouds or ViPE reconstruction."""
        del reference
        backend = self._resolve_backend()
        prediction_path = Path(prediction.path)
        details: dict[str, Any] = {
            "prediction_path": str(prediction_path),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }
        action_payload, action_source = _load_action_payload_for_prediction(sample, prediction_path)
        try:
            result = compute_scene_memory_metric(
                prediction_path,
                action_payload,
                frame_count=int(prediction.native_frame_count or len(prediction.frames) or 0),
                runtime=self.runtime,
            )
            metric_details = dict(result.pop("details", {}))
            details.update(metric_details)
            details.update(
                {
                    "action_source": action_source,
                    "retention": round(float(result["retention"]), 6),
                    "hallucination": round(float(result["hallucination"]), 6),
                    "precision": round(float(result["precision"]), 6),
                    "distance_threshold": round(float(result["distance_threshold"]), 6),
                    "observation_point_count": int(result["observation_point_count"]),
                    "revisit_point_count": int(result["revisit_point_count"]),
                    "scene_diagonal": round(float(result["scene_diagonal"]), 6),
                    "metric_uses_vlm": False,
                    "metric_uses_llm_as_judge": False,
                }
            )
            raw = float(result["scene_memory_f1"])
            if bool(self.runtime.get("require_return_gate", False)):
                from worldarena.benchmark.metrics.memory import (
                    compute_revisit_return_gate,
                    resolve_revisit_pose_sidecar,
                )

                poses, pose_source = resolve_revisit_pose_sidecar(prediction_path, self.runtime)
                gate = compute_revisit_return_gate(
                    poses,
                    tail_fraction=float(self.runtime.get("return_tail_fraction", 0.15)),
                    min_departure_translation=float(
                        self.runtime.get("min_departure_translation", 0.10)
                    ),
                    min_departure_rotation_degrees=float(
                        self.runtime.get("min_departure_rotation_degrees", 45.0)
                    ),
                    max_return_translation_ratio=float(
                        self.runtime.get("max_return_translation_ratio", 0.12)
                    ),
                    max_return_rotation_degrees=float(
                        self.runtime.get("max_return_rotation_degrees", 8.0)
                    ),
                )
                details["pose_source"] = pose_source
                details["return_gate"] = gate
                details["conditional_scene_memory_f1"] = round(raw, 6)
                raw = raw if gate["passed"] else 0.0
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization) if self.normalization else raw,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except FileNotFoundError as exc:
            return _not_applicable(self.name, backend, str(exc), **details)
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


__all__ = [
    "LONG_HORIZON_ACTION_METRICS",
    "LONG_HORIZON_DIAGNOSTIC_METRICS",
    "LONG_HORIZON_MEMORY_METRICS",
    "LONG_HORIZON_VISUAL_METRICS",
    "KeystrokeActionMetric",
    "SceneMemoryF1Metric",
    "VisualDriftMetric",
    "action_parts_from_label",
    "action_parts_from_payload",
    "compute_action_payload_metrics",
    "compute_scene_memory_metric",
    "compute_visual_drift_metrics",
    "discretize_pose_actions",
    "rollout_visual_score_from_components",
    "scene_memory_f1",
    "segment_drift",
    "strict_partial_accuracy",
    "trajectory_score",
    "trimmed_icp_align",
]
