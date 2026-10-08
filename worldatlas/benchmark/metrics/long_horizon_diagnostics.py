"""Diagnostic metrics for long-horizon video evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.memory_pairing import (
    MEMORY_SAFETY_REVISION, load_pose_frame_indices, validate_pose_frame_indices,
)
from worldatlas.benchmark.metrics.base import Metric, clamp01, normalize_score, resolve_metric_backend
from worldatlas.benchmark.quality_backends import compute_musiq_scores, compute_quality_component_scores
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput
from worldatlas.benchmark.sidecars import prediction_sidecar_candidates
from worldatlas.benchmark.taxonomy import is_memory_suite
from worldatlas.common.video_io import probe_video_fps, read_video_frames
from worldatlas.benchmark.action_timeline import realize_matrix_game_2_actions

# Same token-to-keystroke map the Matrix-Game adapters use when they execute a
# WorldAtlas camera_path. Memory-loop samples store the closed itinerary as
# camera tokens, not as *_actions.json, so the keystroke metrics fall back to
# this when no generation-record or sidecar payload exists.
_MEMORY_LOOP_CAMERA_ACTIONS = {
    "fixed": "idle",
    "push_in": "forward",
    "pull_out": "back",
    "pan_left": "back_left_camera_l",
    "pan_right": "back_right_camera_r",
    "move_left": "left",
    "move_right": "right",
    "orbit_left": "forward_left_camera_r",
    "orbit_right": "forward_right_camera_l",
    "pedestal_up": "camera_up",
    "pedestal_down": "camera_down",
    "tilt_up": "camera_up",
    "tilt_down": "camera_down",
}


LONG_HORIZON_ACTION_METRICS = {
    "keystroke_strict_action_accuracy",
    "keystroke_partial_action_accuracy",
    "keystroke_traj_score",
    "keystroke_natet",
    "keystroke_nater",
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
# Provisional compatibility rates: old smallest thresholds at a declared 30 FPS.
# These are NOT physically calibrated ViPE thresholds and never depend on test GT.
DEFAULT_MOVE_SPEED_THRESHOLD = 0.002 * 30.0
DEFAULT_ROTATION_SPEED_THRESHOLD = 0.2 * 30.0
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
    raw = str(value or "").strip().lower()
    # InfiniteWorld's native commands are an explicit grammar, not arbitrary
    # words to discard: unknown/uncertain controls must still be rejected.
    native = raw.replace("_", " ")
    if re.fullmatch(r"(?:go (?:forward|back|left|right)|turn (?:up|down|left|right))(?: and (?:go (?:forward|back|left|right)|turn (?:up|down|left|right)))*", native):
        translations = {
            "go forward": "forward", "go back": "back", "go left": "left", "go right": "right",
            "turn up": "camera_up", "turn down": "camera_down",
            "turn left": "camera_l", "turn right": "camera_r",
        }
        return frozenset(translations[part] for part in native.split(" and "))
    # Preserve signs before '+'/'-' are treated as compound-label separators.
    for signed in ("yaw+", "yaw-", "pitch+", "pitch-"):
        raw = re.sub(r"(?<![a-z0-9])" + re.escape(signed) + r"(?![a-z0-9])", ACTION_ALIASES[signed], raw)
    label = _normalize_action_text(raw)
    if label in NOOP_LABELS:
        return frozenset()

    parts: set[str] = set()
    mutable = label
    for raw_token in sorted(ACTION_ALIASES, key=len, reverse=True):
        if len(raw_token) == 1 or raw_token in {"yaw+", "yaw-", "pitch+", "pitch-"}:
            continue
        canonical = ACTION_ALIASES[raw_token]
        token = raw_token.replace("-", "_")
        pattern = r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])"
        if re.search(pattern, mutable):
            parts.add(canonical)
            mutable = re.sub(pattern, "_", mutable)

    for token in mutable.split("_"):
        if not token:
            continue
        canonical = ACTION_ALIASES.get(token)
        if canonical is None:
            raise ValueError(f"unrecognized action label: {value!r}")
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
    if payload.get("native_action_space") == "matrix_game_3_5_camera_poses":
        return _matrix_game_35_camera_parts(payload, frame_count)
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


def _matrix_game_35_camera_parts(payload: dict[str, Any], frame_count: int) -> list[frozenset[str]]:
    """Use the actual input c2w trajectory, not another adapter's keyboard map.

    Matrix 3.5 pan is pure yaw, unlike Matrix 1/2's diagonal-backward pan.
    Read native frame intervals without stretching, re-timing, or consulting
    predicted poses. Tolerances only remove numerical noise in input matrices.
    """
    if not payload.get("camera_npz"):
        raise ValueError("Matrix-Game-3.5 action payload requires its input camera_npz")
    with np.load(Path(payload["camera_npz"]), allow_pickle=False) as archive:
        poses = np.asarray(archive["extrinsics_c2w"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("invalid Matrix-Game-3.5 input camera poses")
    if frame_count < 2:
        return [frozenset()] * max(frame_count, 0)
    if len(poses) < frame_count or int(payload.get("total_frames", len(poses))) < frame_count:
        raise ValueError("input camera poses do not cover the video timeline")
    relative = np.linalg.inv(poses[:frame_count - 1]) @ poses[1:frame_count]
    # Current keystroke diagnostics cannot represent vertical translation/roll.
    if np.any(np.abs(relative[:, 1, 3]) > 1e-6) or np.any(np.abs(relative[:, 1, 0]) > 1e-5):
        raise ValueError("input camera has vertical translation or roll outside the keystroke action space")
    parts = [_pose_parts_from_relative_pose(
        pose, move_threshold=1e-6, axis_threshold=1e-6, rotation_threshold_degrees=1e-4,
    ) for pose in relative]
    return parts + [frozenset()] if frame_count else []


def _candidate_action_payload_paths(prediction_path: Path) -> list[Path]:
    """List likely action sidecar paths beside a prediction video."""
    return prediction_sidecar_candidates(
        prediction_path,
        ("{stem}_actions.json", "actions/{stem}_actions.json", "_actions/{stem}_actions.json"),
    )


def _payload_from_memory_camera_path(sample: BenchmarkSample) -> dict[str, Any] | None:
    """Build a keystroke payload from the closed-loop camera itinerary."""
    if not is_memory_suite(sample.suite):
        return None
    tokens = [str(token).strip() for token in (sample.camera_path or []) if str(token).strip()]
    if not tokens:
        return None
    return {
        "actions": [_MEMORY_LOOP_CAMERA_ACTIONS.get(token, token) for token in tokens],
        "camera_path": tokens,
        "action_source": "memory_loop_camera_path",
    }


def _load_action_payload_for_prediction(
    sample: BenchmarkSample,
    prediction_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load action JSON from generation records, sidecars, or the memory camera path."""
    try:
        from worldatlas.benchmark.generation_records import _find_generation_record, _load_action_payload

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
    payload = _payload_from_memory_camera_path(sample)
    if payload is not None:
        return payload, "memory_loop_camera_path"
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
    keys = (
        "prediction_pose_path",
        "pose_path",
        "poses_path",
        "vipe_annotation_path",
        "prediction_annotation_path",
    )
    return prediction_sidecar_candidates(
        prediction_path,
        (
            "annotations_vipe/{stem}_vipe_ann",
            "{stem}_vipe_ann",
            "{stem}_poses_c2w.npy",
            "{stem}_poses.npy",
        ),
        ((payload or {}).get(key) for key in keys),
    )


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
    """Positive yaw turns +Z toward +X, matching the action decoder's camera_r."""
    radians = math.radians(angle_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
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
    *,
    durations: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Exact agreement and duration-weighted set F1, including extra actions.

    Empty/empty is correct idle, but idle is separated from commanded accuracy.
    A missing timeline cannot be silently truncated or padded into a good score.
    """
    count = len(ground_truth)
    if count <= 0 or len(predicted) != count:
        raise ValueError("action timelines require the same nonzero aligned length")
    if any(parts - ALL_ACTION_PARTS for parts in (*predicted, *ground_truth)):
        raise ValueError("unsupported action parts")
    weights = np.ones(count) if durations is None else np.asarray(durations, dtype=float)
    if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("action durations must be finite, positive, and aligned")
    weights = weights / float(np.max(weights))
    strict = [float(pred == gt) for pred, gt in zip(predicted, ground_truth)]
    partial = [2 * len(pred & gt) / (len(pred) + len(gt)) if pred or gt else 1.0
               for pred, gt in zip(predicted, ground_truth)]
    def conditional(values, mask):
        selected = weights * np.asarray(mask, dtype=bool)
        return float(np.average(values, weights=selected)) if np.sum(selected) else None
    return {
        "strict_accuracy": float(np.average(strict, weights=weights)),
        "partial_accuracy": float(np.average(partial, weights=weights)),
        "partial_definition": "duration_weighted_set_f1",
        "move_accuracy": float(np.average([(p & MOVE_PARTS) == (g & MOVE_PARTS) for p, g in zip(predicted, ground_truth)], weights=weights)),
        "look_accuracy": float(np.average([(p & LOOK_PARTS) == (g & LOOK_PARTS) for p, g in zip(predicted, ground_truth)], weights=weights)),
        "commanded_strict_accuracy": conditional(strict, [bool(g) for g in ground_truth]),
        "commanded_partial_accuracy": conditional(partial, [bool(g) for g in ground_truth]),
        "idle_false_action_rate": conditional([bool(p) for p in predicted], [not g for g in ground_truth]),
        "aligned_frame_count": float(count),
    }


def _pose_action_intervals(poses_c2w, action_count, *, fps, frame_indices=None):
    poses = np.asarray(poses_c2w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("action evaluation needs finite [N,4,4] poses")
    if not math.isfinite(fps) or fps <= 0 or action_count <= 0:
        raise ValueError("action evaluation needs positive FPS and a nonempty timeline")
    indices = validate_pose_frame_indices(np.arange(len(poses)) if frame_indices is None else frame_indices, len(poses))
    # Missing observations are not interpolated, stretched, or called freezes.
    # Direction within a multi-frame gap is not enough to certify its execution.
    valid = np.flatnonzero((np.diff(indices) == 1) & (indices[:-1] < action_count) & (indices[1:] <= action_count))
    if not len(valid):
        raise ValueError("no observed consecutive-frame action intervals")
    return poses, indices, valid


def discretize_pose_actions(
    poses_c2w: np.ndarray,
    ground_truth: Sequence[frozenset[str]],
    *,
    latent_stride: int | None = None,
    move_thresholds: Sequence[float] | None = None,
    rotation_threshold_degrees: float | None = None,
    fps: float = 30.0,
    frame_indices: np.ndarray | None = None,
    move_speed_threshold: float = DEFAULT_MOVE_SPEED_THRESHOLD,
    rotation_speed_threshold: float = DEFAULT_ROTATION_SPEED_THRESHOLD,
) -> dict[str, Any]:
    """Fixed-rate decoding; GT only scores labels after the predictions are fixed.

    ``latent_stride`` is deprecated: frame grouping must not hide partial freezes.
    Single legacy thresholds are convertible at 30 FPS; oracle sweeps are rejected.
    """
    del latent_stride
    if move_thresholds is not None:
        if len(move_thresholds) != 1:
            raise ValueError("test-label threshold sweeps are removed; set one move_speed_threshold")
        move_speed_threshold = float(move_thresholds[0]) * 30.0
    if rotation_threshold_degrees is not None:
        rotation_speed_threshold = float(rotation_threshold_degrees) * 30.0
    if not np.isfinite([move_speed_threshold, rotation_speed_threshold]).all() or min(move_speed_threshold, rotation_speed_threshold) <= 0:
        raise ValueError("action rate thresholds must be finite and positive")
    poses, indices, valid = _pose_action_intervals(poses_c2w, len(ground_truth), fps=fps, frame_indices=frame_indices)
    relative_poses = np.linalg.inv(poses[valid]) @ poses[valid + 1]
    predictions = [_pose_parts_from_relative_pose(
        relative, move_threshold=move_speed_threshold / fps,
        rotation_threshold_degrees=rotation_speed_threshold / fps,
        axis_threshold=0.5 * move_speed_threshold / fps,
    ) for relative in relative_poses]
    aligned_truth = [ground_truth[int(indices[index])] for index in valid]
    return {
        "predicted_parts": predictions,
        "threshold": float(move_speed_threshold),
        "rotation_speed_threshold": float(rotation_speed_threshold),
        "threshold_units": "pose_translation_units_per_second; rotation_degrees_per_second",
        "threshold_selection": "fixed_without_test_labels",
        "threshold_sweep": {},
        "fps": float(fps),
        "observed_interval_count": int(len(valid)),
        "expected_interval_count": len(ground_truth),
        "observation_coverage": len(valid) / len(ground_truth),
        "observed_frame_indices": indices[valid].tolist(),
        **strict_partial_accuracy(predictions, aligned_truth, durations=np.repeat(1 / fps, len(valid))),
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
    fps: float = 30.0,
    frame_indices: np.ndarray | None = None,
    move_speed_threshold: float = DEFAULT_MOVE_SPEED_THRESHOLD,
    rotation_speed_threshold: float = DEFAULT_ROTATION_SPEED_THRESHOLD,
) -> dict[str, Any]:
    """Time-aligned shape diagnostic with per-interval activity penalties.

    Targets still use predicted segment amplitude, so this is explicitly NOT
    physical target-trajectory accuracy. No arc-length or whole-video time warp
    can erase a freeze/delay. Missing pose intervals remain unobserved, not zeros.
    """
    del total_points  # Deprecated; native intervals determine the time weighting.
    if not np.isfinite([min_path_length, min_rotation_degrees, move_speed_threshold, rotation_speed_threshold]).all() or min(min_path_length, min_rotation_degrees, move_speed_threshold, rotation_speed_threshold) <= 0:
        raise ValueError("trajectory normalizers and rate thresholds must be finite and positive")
    poses, indices, valid = _pose_action_intervals(poses_c2w, len(ground_truth), fps=fps, frame_indices=frame_indices)
    if any(parts - ALL_ACTION_PARTS for parts in ground_truth):
        raise ValueError("unsupported trajectory action parts")
    parts_by_interval = [ground_truth[int(indices[index])] for index in valid]
    segments = []
    for index, parts in zip(valid, parts_by_interval):
        if segments and segments[-1][1] == index and segments[-1][2] == parts:
            segments[-1] = (segments[-1][0], int(index + 1), parts)
        else:
            segments.append((int(index), int(index + 1), parts))
    translation_error = rotation_error = 0.0
    for start, end, parts in segments:
        predicted_segment = poses[start:end + 1]
        ideal = _ideal_segment_poses(predicted_segment, parts)
        errors_t = np.linalg.norm(predicted_segment[:, :3, 3] - ideal[:, :3, 3], axis=1)
        errors_r = np.array([_rotation_angle_degrees(p[:3, :3], g[:3, :3]) for p, g in zip(predicted_segment, ideal)])
        translation_error += float(np.sum(0.5 * (errors_t[:-1] + errors_t[1:])))
        rotation_error += float(np.sum(0.5 * (errors_r[:-1] + errors_r[1:])))
    ate_t, ate_r = translation_error / len(valid), rotation_error / len(valid)
    translation_steps = np.linalg.norm(poses[valid + 1, :3, 3] - poses[valid, :3, 3], axis=1)
    rotation_steps = np.array([_rotation_angle_degrees(poses[i, :3, :3], poses[i + 1, :3, :3]) for i in valid])
    path_length, total_rotation = float(np.sum(translation_steps)), float(np.sum(rotation_steps))
    move_commands = np.array([bool(p & MOVE_PARTS) for p in parts_by_interval])
    look_commands = np.array([bool(p & LOOK_PARTS) for p in parts_by_interval])
    moving = translation_steps * fps > move_speed_threshold
    rotating = rotation_steps * fps > rotation_speed_threshold
    commanded_move_frames, commanded_look_frames = int(np.sum(move_commands)), int(np.sum(look_commands))
    stalled_move_frames = int(np.sum(move_commands & ~moving))
    stalled_look_frames = int(np.sum(look_commands & ~rotating))
    missing_translation = stalled_move_frames / commanded_move_frames if commanded_move_frames else 0.0
    missing_rotation = stalled_look_frames / commanded_look_frames if commanded_look_frames else 0.0
    def activity_error(commanded, active):
        evaluated = commanded | active
        return float(np.mean((commanded != active)[evaluated])) if np.any(evaluated) else 0.0
    natet = max(clamp01(ate_t / max(path_length, float(min_path_length))), activity_error(move_commands, moving))
    nater = max(clamp01(ate_r / max(total_rotation, float(min_rotation_degrees))), activity_error(look_commands, rotating))
    translation_evaluated = bool(np.any(move_commands | moving))
    rotation_evaluated = bool(np.any(look_commands | rotating))
    axis_errors = ([natet] if translation_evaluated else []) + ([nater] if rotation_evaluated else [])
    score = 1.0 - float(np.mean(axis_errors)) if axis_errors else 1.0
    return {
        "traj_score": clamp01(score),
        "natet": natet,
        "nater": nater,
        "ate_t": ate_t,
        "ate_r_degrees": ate_r,
        "path_length": path_length,
        "total_rotation_degrees": total_rotation,
        "commanded_move_frames": commanded_move_frames,
        "commanded_look_frames": commanded_look_frames,
        "missing_translation_fraction": missing_translation,
        "missing_rotation_fraction": missing_rotation,
        "translation_evaluated": bool(translation_evaluated),
        "rotation_evaluated": bool(rotation_evaluated),
        "trajectory_definition": "time_aligned_prediction_scaled_shape_with_activity_guard",
        "target_amplitude_source": "prediction_segment_endpoint_not_independent_ground_truth",
        "amplitude_calibrated": False,
        "observation_coverage": len(valid) / len(ground_truth),
        "implementation_revision": MEMORY_SAFETY_REVISION,
    }


def compute_action_payload_metrics(
    poses_c2w: np.ndarray,
    action_payload: dict[str, Any],
    *,
    frame_count: int,
    runtime: dict[str, Any] | None = None,
    fps: float = 30.0,
    frame_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Combine discretized action accuracy and trajectory scores."""
    runtime = dict(runtime or {})
    action_payload = realize_matrix_game_2_actions(action_payload)
    needed_intervals = max(frame_count - 1, 0)
    for field in ("frame_actions", "keyboard_condition", "mouse_condition"):
        rows = action_payload.get(field)
        if rows and len(rows) < needed_intervals:
            raise ValueError(f"{field} does not cover the video timeline; cannot pad unknown actions")
    if action_payload.get("actions") and action_payload.get("frames_per_action") is not None:
        if len(action_payload["actions"]) * int(action_payload["frames_per_action"]) < needed_intervals:
            raise ValueError("action program does not cover the video timeline; cannot pad unknown actions")
    gt_parts = action_parts_from_payload(action_payload, frame_count)
    if not gt_parts or frame_count < 2:
        raise ValueError("action payload does not contain usable frame actions")
    # F video frames contain F-1 observable intervals; the last command has no
    # subsequent image. Never stretch a shorter pose track to fill that interval.
    interval_parts = gt_parts[:-1]
    poses = np.asarray(poses_c2w, dtype=np.float32)
    if runtime.get("move_thresholds") is not None or runtime.get("rotation_threshold_degrees") is not None:
        raise ValueError("configure fixed per-second action thresholds; legacy per-frame/swept runtime removed")
    move_rate = float(runtime.get("move_speed_threshold", DEFAULT_MOVE_SPEED_THRESHOLD))
    rotation_rate = float(runtime.get("rotation_speed_threshold", DEFAULT_ROTATION_SPEED_THRESHOLD))
    discretized = discretize_pose_actions(
        poses,
        interval_parts, fps=fps, frame_indices=frame_indices,
        move_speed_threshold=move_rate, rotation_speed_threshold=rotation_rate,
    )
    trajectory = trajectory_score(
        poses,
        interval_parts, fps=fps, frame_indices=frame_indices,
        move_speed_threshold=move_rate, rotation_speed_threshold=rotation_rate,
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
        "threshold_calibration": str(runtime.get("action_threshold_calibration", "uncalibrated_legacy_30fps_reference")),
        "action_timeline_definition": (
            "provided_input_camera_pose_timeline" if action_payload.get("native_action_space") == "matrix_game_3_5_camera_poses"
            else "provided_frame_timeline" if any(action_payload.get(k) for k in ("frame_actions", "keyboard_condition", "mouse_condition"))
            else "provided_frames_per_action" if action_payload.get("frames_per_action") is not None
            else "uniform_action_token_timing_assumption"
        ),
        "action_timeline_alignment": action_payload.get("action_timeline_alignment"),
        "input_camera_source": action_payload.get("camera_npz"),
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


def segment_drift(
    scores: Sequence[float], *, segment_count: int = 10,
    times_seconds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Maximum chronological relative drawdown of temporal quality segments.

    Improving quality is not degradation. A dip followed by recovery still has
    drawdown; endpoint change and signed slope distinguish it from lasting decay.
    Absolute quality is separate. Missing scores must not collapse the time axis.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("segment drift requires at least two finite nonnegative scores")
    if int(segment_count) < 2:
        raise ValueError("segment drift requires at least two temporal segments")
    times = np.linspace(0, 1, len(values)) if times_seconds is None else np.asarray(times_seconds, dtype=float)
    if times.shape != values.shape or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("quality timestamps must be finite, increasing, and aligned")
    times = times - times[0]
    count = min(int(segment_count), len(values))
    bins = np.minimum(np.floor(times / times[-1] * count).astype(int), count - 1)
    segment_means = [float(np.mean(values[bins == i])) for i in range(count) if np.any(bins == i)]
    best = max(segment_means)
    worst = min(segment_means)
    peaks = np.maximum.accumulate(segment_means)
    drawdowns = (peaks - segment_means) / np.maximum(peaks, 1e-12)
    drift = float(np.max(drawdowns))
    centered = times - np.mean(times)
    slope = float(np.sum(centered * (values - np.mean(values))) / np.sum(centered ** 2))
    return {
        "drift": clamp01(drift),
        "drift_definition": "maximum_chronological_relative_drawdown",
        "mean_quality": float(np.mean(values)),
        "quality_slope_per_second": slope if times_seconds is not None else None,
        "quality_slope_per_rollout": slope * float(times[-1]),
        "signed_endpoint_drop": segment_means[0] - segment_means[-1],
        "quality_range": best - worst,
        "best_segment_mean": best,
        "worst_segment_mean": worst,
        "segment_means": segment_means,
        "segment_count": len(segment_means),
        "requested_segment_count": count,
        "implementation_revision": MEMORY_SAFETY_REVISION,
    }


_VISUAL_CACHE: dict[str, dict[str, Any]] = {}


def _visual_cache_key(prediction_path: Path, runtime: dict[str, Any]) -> str:
    """Key cached visual-drift results by video identity and runtime options."""
    relevant = {
        "max_frames": runtime.get("max_frames"),
        "segment_count": runtime.get("segment_count"),
        "musiq_model_path": runtime.get("musiq_model_path"),
        "imaging_quality_preprocessing_mode": runtime.get("imaging_quality_preprocessing_mode"),
        "fps": runtime.get("fps"),
    }
    from worldatlas.benchmark.integrity import digest_json, path_content
    return digest_json({'video': path_content(prediction_path), 'runtime': relevant})


def _sample_frames_for_visual(frames: list[np.ndarray], *, max_frames: int | None) -> tuple[list[np.ndarray], list[int]]:
    """Uniformly subsample decoded frames for quality drift analysis."""
    if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return frames, list(range(len(frames)))
    indices = np.rint(np.linspace(0, len(frames) - 1, num=max_frames)).astype(np.int64).tolist()
    return [frames[index] for index in indices], [int(index) for index in indices]


def _optional_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _quality_component_or_error(compute) -> tuple[dict[str, Any], Exception | None]:
    try:
        payload = compute()
    except Exception as exc:
        return {"raw": None, "frame_scores": [], "backend": None}, exc
    if not isinstance(payload, dict):
        return {"raw": None, "frame_scores": [], "backend": None}, RuntimeError(
            "quality backend returned a non-dict payload"
        )
    return payload, None


def _component_drift_or_error(
    payload: dict[str, Any],
    *,
    segment_count: int,
    times_seconds: Sequence[float] | None = None,
) -> tuple[dict[str, Any] | None, Exception | None]:
    try:
        return segment_drift(payload.get("frame_scores") or [], segment_count=segment_count, times_seconds=times_seconds), None
    except Exception as exc:
        return None, exc


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
    # Ordering/drawdown remains available without an FPS. Do not invent seconds
    # for a per-second slope if the video header cannot supply them.
    fps = None
    try:
        measured_fps = float(runtime["fps"]) if "fps" in runtime else probe_video_fps(prediction_path, default=float("nan"))
        fps = measured_fps if math.isfinite(measured_fps) and measured_fps > 0 else None
    except (OSError, RuntimeError, ValueError):
        pass
    quality_times = None if fps is None else [index / fps for index in sampled_indices]
    aesthetic, aesthetic_error = _quality_component_or_error(
        lambda: compute_quality_component_scores(frames, component_name="aesthetic")
    )
    imaging, imaging_error = _quality_component_or_error(
        lambda: compute_musiq_scores(
            frames,
            model_path=None if runtime.get("musiq_model_path") is None else str(runtime.get("musiq_model_path")),
            preprocess_mode=str(runtime.get("imaging_quality_preprocessing_mode") or "longer"),
        )
    )
    segment_count = int(runtime.get("segment_count", 10))
    for component_name, component in (('aesthetic', aesthetic), ('imaging', imaging)):
        scores = component.get('frame_scores') or []
        if len(scores) != len(frames) or not np.isfinite(scores).all():
            error = ValueError(f'{component_name} quality must return finite scores for every sampled frame')
            if component_name == 'aesthetic':
                aesthetic_error = aesthetic_error or error
            else:
                imaging_error = imaging_error or error
    aesthetic_drift = None
    if aesthetic_error is None:
        aesthetic_drift, aesthetic_error = _component_drift_or_error(
            aesthetic, segment_count=segment_count, times_seconds=quality_times,
        )
    imaging_drift = None
    if imaging_error is None:
        imaging_drift, imaging_error = _component_drift_or_error(
            imaging, segment_count=segment_count, times_seconds=quality_times,
        )
    aesthetic_raw = None if aesthetic_error is not None else _optional_finite_float(aesthetic.get("raw"))
    imaging_raw = None if imaging_error is not None else _optional_finite_float(imaging.get("raw"))
    if aesthetic_error is None and aesthetic_raw is None:
        aesthetic_error = RuntimeError("aesthetic backend did not return a score")
    if imaging_error is None and imaging_raw is None:
        imaging_error = RuntimeError("imaging backend did not return a score")
    visual_score = None
    if (
        aesthetic_raw is not None
        and imaging_raw is not None
        and aesthetic_drift is not None
        and imaging_drift is not None
    ):
        visual_score = rollout_visual_score_from_components(
            aesthetic_raw,
            imaging_raw,
            float(aesthetic_drift["drift"]),
            float(imaging_drift["drift"]),
        )
    result = {
        "backend": "quality_drift",
        "implementation_revision": MEMORY_SAFETY_REVISION,
        "fps": fps,
        "sampled_frame_indices": sampled_indices,
        "frame_count": len(frames),
        "aesthetic_score": aesthetic_raw,
        "imaging_score": imaging_raw,
        "aesthetic_drift": aesthetic_drift,
        "imaging_drift": imaging_drift,
        "visual_score": visual_score,
        "aesthetic_backend": aesthetic.get("backend"),
        "imaging_backend": imaging.get("backend"),
        "aesthetic_error": None if aesthetic_error is None else str(aesthetic_error),
        "imaging_error": None if imaging_error is None else str(imaging_error),
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
    values = [
        runtime.get("vipe_annotation_path"),
        runtime.get("prediction_annotation_path"),
        (action_payload or {}).get("vipe_annotation_path"),
        (action_payload or {}).get("prediction_annotation_path"),
    ]
    for candidate in prediction_sidecar_candidates(
        prediction_path, ("annotations_vipe/{stem}_vipe_ann", "{stem}_vipe_ann"), values,
    ):
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
    """Back-project masked depth pixels using the supplied signed intrinsics.

    Do not repair an estimator's focal length here. Replacing a negative focal by
    epsilon changes the projection and can inflate overlap scores catastrophically.
    A signed, nonzero focal is used verbatim; a singular/nonfinite projection fails.
    This arithmetic policy does not certify the upstream camera calibration.
    """
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
    if not np.isfinite([fx, fy, cx, cy]).all() or fx == 0 or fy == 0:
        raise ValueError("backprojection requires finite intrinsics and nonzero focal lengths")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        x = (xs.astype(np.float32) - cx) * z / fx
        y = (ys.astype(np.float32) - cy) * z / fy
        camera_points = np.stack([x, y, z], axis=1).astype(np.float32)
        world_points = ((pose_c2w[:3, :3] @ camera_points.T).T + pose_c2w[:3, 3]).astype(np.float32)
    if not np.isfinite(world_points).all():
        raise ValueError("backprojection produced nonfinite world points")
    return world_points


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


def score_point_cloud_memory(
    observation: np.ndarray,
    revisit: np.ndarray,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Downsample, rigidly align, and score two clouds with retention F1.

    Shared by every point-cloud memory metric so they differ only in how the two
    clouds are produced, not in how the comparison is set up. Rigid alignment matters
    because pose drift shifts the whole return-leg cloud, which would otherwise read
    as forgetting everywhere at once.
    """
    runtime = dict(runtime or {})
    details: dict[str, Any] = {}
    voxel_size = float(runtime.get("voxel_size", 0.0))
    observation = _voxel_downsample(observation, voxel_size)
    revisit = _voxel_downsample(revisit, voxel_size)

    alignment_mode = runtime.get("alignment")
    if alignment_mode == "trimmed_icp" and len(observation) and len(revisit):
        revisit, alignment_details = trimmed_icp_align(
            revisit,
            observation,
            trim_fraction=float(runtime.get("icp_trim_fraction", 0.80)),
            max_iterations=int(runtime.get("icp_max_iterations", 20)),
            tolerance=float(runtime.get("icp_tolerance", 1e-5)),
            max_translation_ratio=float(runtime.get("icp_max_translation_ratio", 0.10)),
            max_rotation_degrees=float(runtime.get("icp_max_rotation_degrees", 10.0)),
        )
        details.update(alignment_details)
    elif bool(runtime.get("centroid_align", True)) and len(observation) and len(revisit):
        revisit = revisit + (np.mean(observation, axis=0) - np.mean(revisit, axis=0))
        details["alignment"] = "centroid_translation"
    else:
        details["alignment"] = "identity"

    combined = np.concatenate([observation, revisit], axis=0)
    diagonal = (
        float(np.linalg.norm(np.max(combined, axis=0) - np.min(combined, axis=0)))
        if len(combined)
        else 0.0
    )
    threshold = runtime.get("distance_threshold")
    if threshold is None:
        threshold = max(diagonal * float(runtime.get("distance_threshold_ratio", 0.02)), 1e-6)
    result = scene_memory_f1(observation, revisit, distance_threshold=float(threshold))
    result["observation_point_count"] = float(len(observation))
    result["revisit_point_count"] = float(len(revisit))
    result["scene_diagonal"] = diagonal
    return {**result, "details": details}


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
    reconstruction: dict[str, Any] = {}
    if source is not None:
        obs, rev, source_text = source
        reconstruction["point_cloud_source"] = source_text
    else:
        if action_payload is None:
            raise FileNotFoundError("scene memory requires action payload or explicit point-cloud sidecar")
        action_parts = action_parts_from_payload(action_payload, frame_count)
        vipe_dir = _resolve_vipe_dir(prediction_path, action_payload, runtime)
        if vipe_dir is None:
            raise FileNotFoundError("scene memory point-cloud or ViPE sidecar is unavailable")
        obs, rev, reconstruction = _reconstruct_vipe_scene_clouds(vipe_dir, action_parts, runtime=runtime)
        reconstruction["point_cloud_source"] = "vipe_depth_reconstruction"

    scored = score_point_cloud_memory(obs, rev, runtime=runtime)
    return {**scored, "details": {**reconstruction, **scored["details"]}}


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
            "implementation_revision": MEMORY_SAFETY_REVISION,
        }
        action_payload, action_source = _load_action_payload_for_prediction(sample, prediction_path)
        if action_payload is None:
            return _not_applicable(self.name, backend, "action specification unavailable", **details)
        poses, pose_source = _load_prediction_poses(prediction_path, action_payload)
        if poses is None:
            details["action_source"] = action_source
            details["measurement_status"] = "pose_sidecar_unavailable"
            return _not_applicable(self.name, backend, "prediction pose sidecar unavailable", **details)
        frame_count = int(prediction.native_frame_count or len(prediction.frames) or max(len(poses) - 1, 1))
        try:
            if action_payload.get("camera_npz") and not Path(action_payload["camera_npz"]).is_absolute():
                action_payload = {**action_payload, "camera_npz": str(prediction_path.parent / action_payload["camera_npz"])}
            fps = float(self.runtime["fps"]) if "fps" in self.runtime else probe_video_fps(prediction_path, default=float("nan"))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("action evaluation requires known positive video FPS")
            pose_location = Path(pose_source)
            pose_directory = pose_location if pose_location.is_dir() else pose_location.parent
            frame_indices = load_pose_frame_indices(pose_directory, len(poses))
            result = compute_action_payload_metrics(
                poses,
                action_payload,
                frame_count=frame_count,
                runtime=self.runtime,
                fps=fps,
                frame_indices=frame_indices,
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
                    "commanded_move_frames": result["commanded_move_frames"],
                    "commanded_look_frames": result["commanded_look_frames"],
                    "missing_translation_fraction": result["missing_translation_fraction"],
                    "missing_rotation_fraction": result["missing_rotation_fraction"],
                    "translation_evaluated": result["translation_evaluated"],
                    "rotation_evaluated": result["rotation_evaluated"],
                    "fps": result["fps"],
                    "threshold_units": result["threshold_units"],
                    "rotation_speed_threshold": result["rotation_speed_threshold"],
                    "threshold_selection": result["threshold_selection"],
                    "threshold_calibration": result["threshold_calibration"],
                    "action_timeline_definition": result["action_timeline_definition"],
                    "partial_definition": result["partial_definition"],
                    "commanded_strict_accuracy": result["commanded_strict_accuracy"],
                    "commanded_partial_accuracy": result["commanded_partial_accuracy"],
                    "idle_false_action_rate": result["idle_false_action_rate"],
                    "observation_coverage": result["observation_coverage"],
                    "observed_interval_count": result["observed_interval_count"],
                    "expected_interval_count": result["expected_interval_count"],
                    "trajectory_definition": result["trajectory_definition"],
                    "target_amplitude_source": result["target_amplitude_source"],
                    "amplitude_calibrated": result["amplitude_calibrated"],
                    "measurement_status": "observed",
                }
            )
            evaluated_key = {"keystroke_natet": "translation_evaluated", "keystroke_nater": "rotation_evaluated"}.get(self.name)
            if evaluated_key and not result[evaluated_key]:
                return _not_applicable(self.name, backend, "no commanded or observed motion on this axis", **details)
            raw_key = {
                "keystroke_strict_action_accuracy": "strict_accuracy",
                "keystroke_partial_action_accuracy": "partial_accuracy",
                "keystroke_traj_score": "traj_score",
                "keystroke_natet": "natet",
                "keystroke_nater": "nater",
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
            details["measurement_status"] = "unavailable_or_invalid"
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
            "implementation_revision": MEMORY_SAFETY_REVISION,
        }
        if prediction.modality != "video":
            return _not_applicable(self.name, backend, "visual drift requires a video prediction", **details)
        try:
            result = compute_visual_drift_metrics(prediction_path, runtime=self.runtime)
            details.update(
                {
                    "frame_count": result["frame_count"],
                    "fps": result.get("fps"),
                    "sampled_full_video_indices": result["sampled_frame_indices"],
                    "aesthetic_backend": result["aesthetic_backend"],
                    "imaging_backend": result["imaging_backend"],
                    "aesthetic_score": result["aesthetic_score"],
                    "imaging_score": result["imaging_score"],
                    "visual_score": result["visual_score"],
                    "aesthetic_drift": result["aesthetic_drift"],
                    "imaging_drift": result["imaging_drift"],
                    "aesthetic_error": result.get("aesthetic_error"),
                    "imaging_error": result.get("imaging_error"),
                }
            )
            own_error = {
                "rollout_aesthetic_drift": result.get("aesthetic_error"),
                "rollout_imaging_drift": result.get("imaging_error"),
            }.get(self.name)
            if own_error:
                raise RuntimeError(str(own_error))
            drift = {
                "rollout_aesthetic_drift": result.get("aesthetic_drift"),
                "rollout_imaging_drift": result.get("imaging_drift"),
            }.get(self.name)
            raw = None if not isinstance(drift, dict) else _optional_finite_float(drift.get("drift"))
            if raw is None:
                raise RuntimeError(f"{self.name} did not return a score")
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
                from worldatlas.benchmark.metrics.memory import (
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
    "score_point_cloud_memory",
    "segment_drift",
    "strict_partial_accuracy",
    "trajectory_score",
    "trimmed_icp_align",
]
