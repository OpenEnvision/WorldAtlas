"""Camera pose utilities for native (non-vendored) metric backends."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.benchmark.taxonomy import VIDEO_PREDICTION_SUITES


@dataclass(frozen=True, slots=True)
class NativeMotionIntent:
    label: str
    translation: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None
    perspective: str = "first_person"
    source: str = "camera_path"
    frame_count: int | None = None


def _sanitize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


_IDLE_ACTION_TOKENS = {"fixed", "idle", "stop", "none", "noop", "no_op", "no-op"}
_NOOP_CONTROL_TOKENS = _IDLE_ACTION_TOKENS | {"uncertain"}
_DEFAULT_KEY_ORDER = ("w", "a", "s", "d", "i", "j", "k", "l")


def _is_flag_value(value: Any) -> bool:
    if isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        return _sanitize_token(value) in {"0", "1", "true", "false", "yes", "no", "on", "off"}
    return False


def _flag_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return _sanitize_token(value) in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_mapping_literal(text: str) -> dict[str, Any] | None:
    """Recover a dict from JSON, Python literals, or sanitized `{'w':_true}` strings."""
    candidates = [text, text.replace("_", " ")]
    for candidate in candidates:
        stripped = candidate.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue
        pythonized = re.sub(r"\btrue\b", "True", stripped, flags=re.IGNORECASE)
        pythonized = re.sub(r"\bfalse\b", "False", pythonized, flags=re.IGNORECASE)
        pythonized = re.sub(r"\bnull\b", "None", pythonized, flags=re.IGNORECASE)
        for raw in (stripped, pythonized):
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                return parsed
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _infinite_world_move_components(move: str) -> list[str]:
    token = _sanitize_token(move)
    if not token or token in _NOOP_CONTROL_TOKENS:
        return []
    parts: list[str] = []
    if "forward" in token:
        parts.append("forward")
    if "back" in token:
        parts.append("back")
    if "left" in token:
        parts.append("left")
    if "right" in token:
        parts.append("right")
    return parts


def _infinite_world_view_components(view: str) -> list[str]:
    token = _sanitize_token(view)
    if not token or token in _NOOP_CONTROL_TOKENS:
        return []
    parts: list[str] = []
    # Emit camera_* tokens so "turn left/right" cannot collapse into strafe.
    if "left" in token:
        parts.append("camera_l")
    if "right" in token:
        parts.append("camera_r")
    if "up" in token:
        parts.append("camera_up")
    if "down" in token:
        parts.append("camera_down")
    return parts


def _action_string_from_mapping(mapping: dict[str, Any]) -> str:
    if any(key in mapping for key in ("move", "view")):
        parts = _infinite_world_move_components(str(mapping.get("move") or "no-op"))
        parts.extend(_infinite_world_view_components(str(mapping.get("view") or "no-op")))
        return "+".join(parts) if parts else "idle"

    enabled: list[str] = []
    for key, value in mapping.items():
        token = _sanitize_token(key)
        if not token or not _is_flag_value(value):
            continue
        if _flag_enabled(value):
            enabled.append(token)
    return "+".join(enabled) if enabled else "idle"


def _action_string_from_vector(values: list[Any], key_order: list[str] | None) -> str | None:
    order = [token for token in (key_order or list(_DEFAULT_KEY_ORDER)) if token]
    if not values or not order or len(values) != len(order):
        return None
    if not all(_is_flag_value(item) for item in values):
        return None
    return _action_string_from_mapping(dict(zip(order, values)))


def _coerce_action_item(item: Any, *, key_order: list[str] | None = None) -> str | None:
    """Canonicalize one payload action into a parseable token string.

    Adapters persist several native shapes: ABot WASD dicts, Infinite-World
    `{move, view}` maps, boolean key vectors, camera-path tokens, and
    keyboard chords. Stringifying a dict used to produce labels like
    `{'w':_true,_'a':_true}` that `_intent_from_action` treated as idle.
    """
    if item is None:
        return None
    if isinstance(item, dict):
        return _action_string_from_mapping(item)
    if isinstance(item, (list, tuple)):
        vector = _action_string_from_vector(list(item), key_order)
        if vector is not None:
            return vector
        names = [_sanitize_token(part) for part in item if str(part).strip()]
        return "+".join(names) if names else None

    text = str(item).strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        parsed = _parse_mapping_literal(text)
        if parsed is not None:
            return _action_string_from_mapping(parsed)
    return _sanitize_token(text)


def _payload_key_order(payload: dict[str, Any]) -> list[str] | None:
    raw = [_sanitize_token(item) for item in _as_list(payload.get("key_order")) if str(item).strip()]
    return raw or None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _prediction_root(prediction_path: Path) -> Path:
    if prediction_path.parent.name in VIDEO_PREDICTION_SUITES:
        return prediction_path.parent.parent
    return prediction_path.parent


def _action_sidecar_paths(prediction_path: Path) -> list[Path]:
    stem = prediction_path.stem
    candidates = [
        prediction_path.with_name(f"{stem}_actions.json"),
        prediction_path.with_suffix(".actions.json"),
    ]
    if prediction_path.parent.name in VIDEO_PREDICTION_SUITES:
        candidates.extend(
            [
                prediction_path.parent.parent / f"{stem}_actions.json",
                prediction_path.parent.parent / f"{stem}.actions.json",
            ]
        )
    return candidates


@lru_cache(maxsize=128)
def _generation_record_index(root_raw: str) -> dict[str, dict[str, Any]]:
    root = Path(root_raw)
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("generation_records*.jsonl")):
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                sample_id = str(payload.get("sample_id") or "").strip()
                if sample_id:
                    index.setdefault(sample_id, payload)
                prediction_path = payload.get("prediction_path")
                if prediction_path:
                    index.setdefault(Path(str(prediction_path)).stem, payload)
    return index


def _action_payload_for_prediction(
    sample: BenchmarkSample,
    prediction_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    from worldatlas.benchmark.camera_video_identity import POLICY as CAMERA_POLICY, validate_intent
    if sample.sample_id_policy == CAMERA_POLICY:
        validate_intent(sample)
        # This route scores the pinned camera_path, not mutable sibling actions.
        return None, None
    for path in _action_sidecar_paths(prediction_path):
        if path.exists():
            payload = _read_json(path)
            if payload is not None:
                return payload, str(path)

    records = _generation_record_index(str(_prediction_root(prediction_path)))
    record = records.get(sample.sample_id) or records.get(prediction_path.stem)
    if record is None:
        return None, None

    alignment = record.get("action_alignment")
    if isinstance(alignment, dict):
        payload = dict(alignment)
        for key, value in record.items():
            payload.setdefault(key, value)
        return payload, "generation_records.action_alignment"
    return record, "generation_records"


def _extract_action_values(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    key_order = _payload_key_order(payload)
    for key in ("actions", "action_seq", "native_actions"):
        actions = [
            coerced
            for item in _as_list(payload.get(key))
            for coerced in [_coerce_action_item(item, key_order=key_order)]
            if coerced
        ]
        if actions:
            return actions, key
    return [], None


def _is_continuous_action_vector(payload: dict[str, Any], key: str | None, actions: list[str]) -> bool:
    if key not in {"actions", "action_seq"} or len(actions) != 3:
        return False
    action_space = _sanitize_token(
        payload.get("native_action_space") or payload.get("variant") or payload.get("mode") or payload.get("control_source")
    )
    # NOTE: this used to also trigger on `key == "action_seq"` alone, on the
    # assumption that an "action_seq" key always means "one frame's
    # simultaneous discrete-key combo" (matrix-game-style). But dreamx_world's
    # sidecar payloads *also* use the "action_seq" key -- for a genuinely
    # sequential list of per-segment actions (each with its own
    # `action_speed_list` entry), not a simultaneous combo. That made every
    # 3-action dreamx clip collapse into a single joined "a+b+c" action,
    # silently discarding two of its three intended camera-motion segments.
    # Only fold multiple actions into one when there is a concrete signal
    # that they represent simultaneous keys/mouse/camera axes, not a
    # coincidental length of exactly 3.
    return action_space.startswith("matrix_game") or any(
        payload.get(condition_key) is not None
        for condition_key in ("keyboard_condition", "mouse_condition", "camera_condition")
    )


def _normalize_actions(payload: dict[str, Any], actions: list[str], key: str | None) -> list[str]:
    if _is_continuous_action_vector(payload, key, actions):
        return ["+".join(actions)]
    return actions


def _extract_actions(payload: dict[str, Any]) -> list[str]:
    actions, key = _extract_action_values(payload)
    return _normalize_actions(payload, actions, key)


def _as_positive_ints(value: Any, expected: int) -> list[int] | None:
    values: list[int] = []
    for item in _as_list(value):
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        values.append(parsed)
    return values if len(values) == expected else None


def _condition_rows(payload: dict[str, Any]) -> list[tuple[float, ...]]:
    arrays: list[list[Any]] = []
    for key in ("keyboard_condition", "mouse_condition", "camera_condition"):
        value = payload.get(key)
        if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
            arrays.append(value)
    if not arrays:
        return []

    row_count = min(len(array) for array in arrays)
    rows: list[tuple[float, ...]] = []
    for index in range(row_count):
        parts: list[float] = []
        for array in arrays:
            row = np.asarray(array[index], dtype=np.float32).reshape(-1)
            parts.extend(round(float(item), 6) for item in row)
        rows.append(tuple(parts))
    return rows


def _condition_run_lengths(payload: dict[str, Any]) -> list[int]:
    rows = _condition_rows(payload)
    if not rows:
        return []
    lengths: list[int] = []
    previous = rows[0]
    count = 1
    for row in rows[1:]:
        if row == previous:
            count += 1
            continue
        lengths.append(count)
        previous = row
        count = 1
    lengths.append(count)
    return lengths


def _infer_action_frame_counts(payload: dict[str, Any], actions: list[str]) -> list[int] | None:
    action_count = len(actions)
    if action_count <= 0:
        return None

    for key in ("action_frame_counts", "segment_frame_counts", "native_action_frame_counts"):
        explicit = _as_positive_ints(payload.get(key), action_count)
        if explicit is not None:
            return explicit

    run_lengths = _condition_run_lengths(payload)
    if len(run_lengths) == action_count:
        return run_lengths

    total_frames = None
    for key in ("total_frames", "num_frames", "video_length", "save_num_frames"):
        try:
            parsed = int(payload.get(key))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            total_frames = parsed
            break
    if total_frames is None:
        rows = _condition_rows(payload)
        total_frames = len(rows) if rows else None
    if total_frames is None or total_frames < action_count:
        return None

    base = total_frames // action_count
    remainder = total_frames % action_count
    counts = [base for _ in actions]
    # Matrix-Game payloads pad/trim at the tail, so assign leftover frames to the
    # final action instead of moving the first boundary.
    counts[-1] += remainder
    return counts


def _camera_token_intent(token: str, *, source: str = "camera_path") -> NativeMotionIntent | None:
    token = _sanitize_token(token)
    if token in {"fixed", "idle", "stop", "none", "noop", "no_op", "no-op"}:
        return NativeMotionIntent(label=token or "fixed", source=source)
    if token == "push_in":
        return NativeMotionIntent(label=token, translation=(0.0, 0.0, 1.0), source=source)
    if token == "pull_out":
        return NativeMotionIntent(label=token, translation=(0.0, 0.0, -1.0), source=source)
    if token == "move_left":
        return NativeMotionIntent(label=token, translation=(-1.0, 0.0, 0.0), source=source)
    if token == "move_right":
        return NativeMotionIntent(label=token, translation=(1.0, 0.0, 0.0), source=source)
    if token == "pan_left":
        return NativeMotionIntent(label=token, rotation=(-1.0, 0.0, 0.0), source=source)
    if token == "pan_right":
        return NativeMotionIntent(label=token, rotation=(1.0, 0.0, 0.0), source=source)
    if token == "tilt_up":
        return NativeMotionIntent(label=token, rotation=(0.0, 0.0, 1.0), source=source)
    if token == "tilt_down":
        return NativeMotionIntent(label=token, rotation=(0.0, 0.0, -1.0), source=source)
    if token == "orbit_left":
        return NativeMotionIntent(
            label=token,
            rotation=(-1.0, 0.0, 0.0),
            perspective="third_person",
            source=source,
        )
    if token == "orbit_right":
        return NativeMotionIntent(
            label=token,
            rotation=(1.0, 0.0, 0.0),
            perspective="third_person",
            source=source,
        )
    return None


def _action_components(action: str) -> set[str]:
    normalized = _sanitize_token(action)
    replacements = {
        "camera_left": "camera_l",
        "camera_right": "camera_r",
        "left_rot": "camera_l",
        "right_rot": "camera_r",
        "up_rot": "camera_up",
        "down_rot": "camera_down",
        "arrow_left": "camera_l",
        "arrow_right": "camera_r",
        "arrow_up": "camera_up",
        "arrow_down": "camera_down",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    components: set[str] = set()
    for camera_token in ("camera_l", "camera_r", "camera_up", "camera_down"):
        if camera_token in normalized:
            components.add(camera_token)
            normalized = normalized.replace(camera_token, "")

    for piece in re.split(r"[_+,/]+", normalized):
        if piece:
            components.add(piece)

    compact = normalized.replace("_", "")
    compact_keys = {"w", "s", "a", "d", "j", "l", "i", "k"}
    if compact and len(compact) <= 4 and set(compact) <= compact_keys:
        for char in compact:
            components.add(char)
    return components


def _intent_from_action(action: str) -> NativeMotionIntent | None:
    # Camera-path tokens written into `actions` (Matrix-Game 3.5, and any
    # adapter that records the WorldAtlas token rather than a key chord)
    # must keep third-person orbit / push-in semantics. The WASD parser
    # below would otherwise turn `push_in` into idle and `orbit_left` into
    # a first-person strafe.
    camera_intent = _camera_token_intent(action, source="native_action")
    if camera_intent is not None:
        return camera_intent

    components = _action_components(action)
    if not components:
        return None
    if components <= _IDLE_ACTION_TOKENS:
        return NativeMotionIntent(label=action, source="native_action")

    x = 0.0
    z = 0.0
    yaw = 0.0
    pitch = 0.0
    if components & {"forward", "w", "go_forward"}:
        z += 1.0
    if components & {"back", "backward", "s", "go_back"}:
        z -= 1.0
    if components & {"left", "a", "go_left"}:
        x -= 1.0
    if components & {"right", "d", "go_right"}:
        x += 1.0
    if components & {"camera_l", "turn_left", "j"}:
        yaw -= 1.0
    if components & {"camera_r", "turn_right", "l"}:
        yaw += 1.0
    if components & {"camera_up", "look_up", "i"}:
        pitch += 1.0
    if components & {"camera_down", "look_down", "k"}:
        pitch -= 1.0

    translation = (x, 0.0, z) if x or z else None
    rotation = (yaw, 0.0, pitch) if yaw or pitch else None
    if translation is None and rotation is None:
        # Unrecognized tokens (stringified junk, "official_bench", ...)
        # must not become idle -- that silently zeros a moving camera.
        return None
    return NativeMotionIntent(label=action, translation=translation, rotation=rotation, source="native_action")


def _rot_x(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rot_y(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rot_angle_deg(rotation: np.ndarray) -> float:
    cos_val = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _path_length(poses: np.ndarray) -> float:
    if len(poses) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())


def _total_rotation_deg(poses: np.ndarray) -> float:
    if len(poses) < 2:
        return 0.0
    return float(
        sum(
            _rot_angle_deg(poses[index, :3, :3].T @ poses[index + 1, :3, :3])
            for index in range(len(poses) - 1)
        )
    )


def _estimate_orbit_radius(pred_turn: np.ndarray) -> float | None:
    chord = float(np.linalg.norm(pred_turn[-1, :3, 3] - pred_turn[0, :3, 3]))
    theta = np.deg2rad(_total_rotation_deg(pred_turn))
    if theta < np.deg2rad(5.0) or np.sin(theta / 2.0) < 1e-6:
        return None
    return float(chord / (2.0 * np.sin(theta / 2.0)))


def _look_at_pose(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right.astype(np.float32)
    pose[:3, 1] = up.astype(np.float32)
    pose[:3, 2] = forward.astype(np.float32)
    pose[:3, 3] = position.astype(np.float32)
    return pose


def _reference_trajectory(
    intent: NativeMotionIntent,
    n_frames: int,
    *,
    rotation_angle_degrees: float,
    step_size: float,
    subject_depth: float,
) -> np.ndarray:
    frame_count = max(int(n_frames), 1)
    if intent.translation is None and intent.rotation is None:
        return np.repeat(np.eye(4, dtype=np.float32)[None], repeats=frame_count, axis=0)

    n = max(frame_count - 1, 1)
    translation = (
        np.asarray(intent.translation, dtype=np.float64) * float(step_size)
        if intent.translation is not None
        else np.zeros(3, dtype=np.float64)
    )
    yaw_dir = float(intent.rotation[0]) if intent.rotation is not None else 0.0
    pitch_dir = float(intent.rotation[2]) if intent.rotation is not None else 0.0

    if intent.perspective != "first_person" and intent.rotation is not None:
        subject_base = np.asarray([0.0, 0.0, subject_depth], dtype=np.float64)
        offset = np.asarray([0.0, 0.0, -subject_depth], dtype=np.float64)
        poses: list[np.ndarray] = []
        for index in range(frame_count):
            t = index / n
            subject_pos = subject_base + t * translation
            rotation = np.eye(3, dtype=np.float32)
            if pitch_dir:
                rotation = rotation @ _rot_x(pitch_dir * rotation_angle_degrees * t)
            if yaw_dir:
                rotation = rotation @ _rot_y(yaw_dir * rotation_angle_degrees * t)
            camera_pos = subject_pos + rotation.astype(np.float64) @ offset
            poses.append(_look_at_pose(camera_pos, subject_pos))
        return np.asarray(poses, dtype=np.float32)

    poses: list[np.ndarray] = []
    current_pose = np.eye(4, dtype=np.float32)
    translation_step = translation / n
    for index in range(frame_count):
        poses.append(current_pose.copy())
        if index >= frame_count - 1:
            continue
        if pitch_dir:
            current_pose[:3, :3] = current_pose[:3, :3] @ _rot_x(pitch_dir * rotation_angle_degrees / n)
        if yaw_dir:
            current_pose[:3, :3] = current_pose[:3, :3] @ _rot_y(yaw_dir * rotation_angle_degrees / n)
        if intent.translation is not None:
            current_pose[:3, 3] += (current_pose[:3, :3].astype(np.float64) @ translation_step).astype(np.float32)
    return np.asarray(poses, dtype=np.float32)


def _build_adaptive_gt_segment(
    pred_turn: np.ndarray,
    intent: NativeMotionIntent,
    *,
    current_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    min_disp_threshold = 0.1
    min_rot_threshold = 3.0
    fallback_disp = 1.0
    fallback_rot = 30.0
    fallback_depth = 1.0

    if intent.translation is not None and intent.rotation is None:
        direction = np.asarray(intent.translation, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        world_dir = current_rotation.astype(np.float64) @ direction
        pred_disp = float(np.linalg.norm(pred_turn[-1, :3, 3] - pred_turn[0, :3, 3]))
        gt_disp = pred_disp if pred_disp > min_disp_threshold else fallback_disp
        gt = np.zeros_like(pred_turn, dtype=np.float32)
        start_pos = pred_turn[0, :3, 3].astype(np.float64)
        for index in range(len(pred_turn)):
            t = index / max(len(pred_turn) - 1, 1)
            gt[index, :3, :3] = current_rotation.astype(np.float32)
            gt[index, :3, 3] = (start_pos + t * gt_disp * world_dir).astype(np.float32)
            gt[index, 3, 3] = 1.0
        return gt, current_rotation

    pred_rot = _total_rotation_deg(pred_turn)
    pred_range = float(np.linalg.norm(pred_turn[:, :3, 3] - pred_turn[0, :3, 3], axis=1).mean())
    rotation_angle = max(pred_rot, fallback_rot) if pred_rot >= min_rot_threshold else fallback_rot
    subject_depth = max(_estimate_orbit_radius(pred_turn) or fallback_depth, fallback_depth)
    step_size = pred_range if pred_range > min_disp_threshold else fallback_disp
    reference = _reference_trajectory(
        intent,
        len(pred_turn),
        rotation_angle_degrees=rotation_angle,
        step_size=step_size,
        subject_depth=subject_depth,
    )

    rotation_align = pred_turn[0, :3, :3] @ reference[0, :3, :3].T
    ref_pos = reference[:, :3, 3] - reference[0, :3, 3]
    ref_range = float(np.linalg.norm(ref_pos, axis=1).mean())
    scale = pred_range / (ref_range + 1e-12) if pred_range > min_disp_threshold and ref_range > 1e-12 else 1.0
    start_pos = pred_turn[0, :3, 3]
    gt = np.zeros_like(reference, dtype=np.float32)
    for index in range(len(reference)):
        gt[index, :3, :3] = (rotation_align @ reference[index, :3, :3]).astype(np.float32)
        gt[index, :3, 3] = (start_pos + scale * (rotation_align @ ref_pos[index])).astype(np.float32)
        gt[index, 3, 3] = 1.0
    return gt, gt[-1, :3, :3].copy()


def _compute_ate(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    count = min(len(gt), len(pred))
    if count <= 0:
        return 0.0, 0.0
    gt = gt[:count]
    pred = pred[:count]
    translation_error = float(np.linalg.norm(gt[:, :3, 3] - pred[:, :3, 3], axis=1).mean())
    rotation_error = float(
        np.asarray(
            [_rot_angle_deg(gt[index, :3, :3].T @ pred[index, :3, :3]) for index in range(count)],
            dtype=np.float64,
        ).mean()
    )
    return translation_error, rotation_error


# --- Direction/confidence based per-segment scoring -------------------------------
#
# The plain ATE/NATE formulation (`ate / max(achieved_motion, floor)`, see the
# legacy fields below) is a *relative* error: an absolute trajectory error is
# divided by however much motion the model actually produced. When the real
# motion is small -- routine for compact, per-keypress camera control
# (discrete WASD-style actions) as opposed to models given one large explicit
# camera trajectory -- a roughly-constant absolute error (monocular
# pose-estimation noise, plus the "assumed" GT magnitude used whenever the
# model's own motion is too small to trust, see `_build_adaptive_gt_segment`'s
# `fallback_disp`/`fallback_rot`) gets divided by a shrinking, floor-clamped
# denominator and blows up toward the 1.0 cap. That is indistinguishable, from
# the score alone, from "ignored the instruction" even when the camera moved
# in exactly the right direction, just by less than the hard-coded 1.0 unit /
# 30 degree reference the fallback assumes.
#
# The fix scores each segment as (a) whether the achieved motion is large
# enough, relative to typical monocular pose-estimation noise, to be judged
# at all, blended with (b) whether that motion points in the intended
# direction -- instead of how closely its *magnitude* matches an assumed
# reference. This is scale-invariant by construction, so "correct but modest"
# camera moves are no longer penalized the same as "barely moved at all".
TRANSLATION_NOISE_FLOOR = 0.03
TRANSLATION_CONFIDENT = 0.15
ROTATION_NOISE_FLOOR_DEG = 1.5
ROTATION_CONFIDENT_DEG = 8.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = _clamp01((value - edge0) / (edge1 - edge0))
    return x * x * (3.0 - 2.0 * x)


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def _direction_cosine(actual: np.ndarray, intended: np.ndarray | None) -> float | None:
    if intended is None:
        return None
    actual_unit = _unit_vector(actual)
    intended_unit = _unit_vector(intended)
    if actual_unit is None or intended_unit is None:
        return None
    return float(np.clip(np.dot(actual_unit, intended_unit), -1.0, 1.0))


def _direction_component(cos_sim: float | None) -> float:
    """Map a [-1, 1] cosine similarity to a [0, 1] score; unmeasurable -> neutral."""
    return 0.5 if cos_sim is None else _clamp01((cos_sim + 1.0) / 2.0)


def _rotation_axis(rotation: np.ndarray, angle_deg: float) -> np.ndarray | None:
    if angle_deg < 1e-6:
        return None
    sin_val = float(np.sin(np.deg2rad(angle_deg)))
    if abs(sin_val) < 1e-9:
        return None
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sin_val)
    return _unit_vector(axis)


def _score_segment(
    pred_turn: np.ndarray,
    intent: NativeMotionIntent,
    *,
    segment_start_rotation: np.ndarray,
) -> dict[str, Any]:
    """Direction+confidence score for one first-person (or idle) segment.

    Scale-invariant: a segment scores well once the achieved motion is
    unambiguously deliberate (well above typical pose-estimation noise) and
    points the right way -- it does not need to match any assumed reference
    magnitude. When the achieved motion is too small to judge reliably, the
    direction term blends toward a neutral 0.5 instead of being scored as
    outright wrong, since "can't tell" is not the same as "wrong direction".
    """
    components: list[float] = []
    diagnostics: dict[str, Any] = {}

    if intent.translation is not None:
        local_dir = _unit_vector(np.asarray(intent.translation, dtype=np.float64))
        actual_disp = (pred_turn[-1, :3, 3] - pred_turn[0, :3, 3]).astype(np.float64)
        pred_disp = float(np.linalg.norm(actual_disp))
        world_dir = segment_start_rotation.astype(np.float64) @ local_dir if local_dir is not None else None
        cos_sim = _direction_cosine(actual_disp, world_dir)
        confidence = _smoothstep(TRANSLATION_NOISE_FLOOR, TRANSLATION_CONFIDENT, pred_disp)
        score = confidence * _direction_component(cos_sim) + (1.0 - confidence) * 0.5
        components.append(score)
        diagnostics.update(
            {
                "translation_pred_disp": pred_disp,
                "translation_direction_cos": cos_sim,
                "translation_confidence": confidence,
                "translation_component": score,
            }
        )

    if intent.rotation is not None:
        local_axis = _unit_vector(np.asarray([intent.rotation[2], intent.rotation[0], 0.0], dtype=np.float64))
        relative_rotation = (pred_turn[0, :3, :3].T @ pred_turn[-1, :3, :3]).astype(np.float64)
        angle_deg = _rot_angle_deg(relative_rotation)
        axis = _rotation_axis(relative_rotation, angle_deg)
        cos_sim = _direction_cosine(axis, local_axis) if axis is not None else None
        confidence = _smoothstep(ROTATION_NOISE_FLOOR_DEG, ROTATION_CONFIDENT_DEG, angle_deg)
        score = confidence * _direction_component(cos_sim) + (1.0 - confidence) * 0.5
        components.append(score)
        diagnostics.update(
            {
                "rotation_pred_deg": angle_deg,
                "rotation_direction_cos": cos_sim,
                "rotation_confidence": confidence,
                "rotation_component": score,
            }
        )

    if not components:
        # "fixed"/idle intent: any drift away from the starting pose is a
        # compliance failure, scored by how much the camera moved relative to
        # typical monocular pose-estimation noise -- no assumed reference
        # magnitude applies here since none was intended.
        disp = float(np.linalg.norm((pred_turn[-1, :3, 3] - pred_turn[0, :3, 3]).astype(np.float64)))
        relative_rotation = (pred_turn[0, :3, :3].T @ pred_turn[-1, :3, :3]).astype(np.float64)
        rot_deg = _rot_angle_deg(relative_rotation)
        drift = max(
            _smoothstep(TRANSLATION_NOISE_FLOOR, TRANSLATION_CONFIDENT, disp),
            _smoothstep(ROTATION_NOISE_FLOOR_DEG, ROTATION_CONFIDENT_DEG, rot_deg),
        )
        components.append(1.0 - drift)
        diagnostics.update({"idle_translation_drift": disp, "idle_rotation_drift_deg": rot_deg})

    diagnostics["segment_score"] = float(sum(components) / len(components))
    return diagnostics


def _score_orbit_segment(gt_turn: np.ndarray, pred_turn: np.ndarray) -> dict[str, Any]:
    """Confidence-softened relative trajectory error for third-person orbit segments.

    Orbiting is a trajectory-*shape* check (position and facing over time),
    not a single direction, so it keeps comparing against the adaptively
    built reference via ATE. The fix here is only to stop the same
    small-motion blow-up as the legacy formula: blend the relative error
    toward a neutral 0.5 instead of a hard 1.0 cap when the achieved motion
    is too small to trust, rather than dividing by a near-zero denominator.
    """
    ate_t, ate_r = _compute_ate(gt_turn, pred_turn)
    path_length = _path_length(pred_turn)
    rotation_total = _total_rotation_deg(pred_turn)
    t_confidence = _smoothstep(TRANSLATION_NOISE_FLOOR, TRANSLATION_CONFIDENT, path_length)
    r_confidence = _smoothstep(ROTATION_NOISE_FLOOR_DEG, ROTATION_CONFIDENT_DEG, rotation_total)
    relative_error_t = _clamp01(ate_t / max(path_length, TRANSLATION_CONFIDENT))
    relative_error_r = _clamp01(ate_r / max(rotation_total, ROTATION_CONFIDENT_DEG))
    score_t = t_confidence * (1.0 - relative_error_t) + (1.0 - t_confidence) * 0.5
    score_r = r_confidence * (1.0 - relative_error_r) + (1.0 - r_confidence) * 0.5
    return {
        "orbit_ate_translation": float(ate_t),
        "orbit_ate_rotation": float(ate_r),
        "orbit_path_length": float(path_length),
        "orbit_rotation_deg": float(rotation_total),
        "segment_score": float((score_t + score_r) / 2.0),
    }


def _native_intents_for_sample(
    sample: BenchmarkSample,
    prediction_path: Path,
) -> tuple[list[NativeMotionIntent], dict[str, Any]]:
    payload, source_path = _action_payload_for_prediction(sample, prediction_path)
    if payload is not None:
        source_actions, action_key = _extract_action_values(payload)
        actions = _normalize_actions(payload, source_actions, action_key)
        frame_counts = _infer_action_frame_counts(payload, actions)
        intents: list[NativeMotionIntent] = []
        for index, action in enumerate(actions):
            intent = _intent_from_action(action)
            if intent is None:
                continue
            if frame_counts is not None and index < len(frame_counts):
                intent = replace(intent, frame_count=int(frame_counts[index]))
            intents.append(intent)
        if intents:
            return intents, {
                "native_camera_target_policy": "native_intended",
                "native_camera_target_type": "discrete_action",
                "native_camera_source_path": source_path,
                "native_action_space": payload.get("native_action_space") or payload.get("variant") or payload.get("mode"),
                "native_actions": actions,
                "native_source_actions": source_actions if source_actions != actions else None,
                "native_action_source_key": action_key,
                "native_control_source": payload.get("control_source"),
                "native_action_source": payload.get("action_source"),
                "native_action_frame_counts": frame_counts,
                "native_action_speeds": payload.get("action_speed_list"),
                "native_condition_frame_count": sum(frame_counts) if frame_counts is not None else None,
                "native_camera_tokens": payload.get("camera_tokens"),
                "native_camera_path": payload.get("camera_path"),
                "native_approximations": payload.get("approximations"),
                "native_unsupported_camera_tokens": payload.get("unsupported_camera_tokens"),
            }

    intents = [
        intent
        for token in sample.camera_path
        for intent in [_camera_token_intent(token, source="camera_path")]
        if intent is not None
    ]
    if intents:
        return intents, {
            "native_camera_target_policy": "native_intended",
            "native_camera_target_type": "camera_path_intent",
            "native_camera_path": list(sample.camera_path),
        }
    return [], {
        "native_camera_target_policy": "native_intended",
        "native_camera_target_type": "unavailable",
    }


def score_native_intended_cameras(
    sample: BenchmarkSample,
    prediction_path: str | Path,
    cameras_pred: np.ndarray,
) -> dict[str, Any]:
    path = Path(prediction_path)
    poses = np.asarray(cameras_pred, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"predicted cameras must have shape (N, 4, 4), got {poses.shape}")
    if len(poses) == 0:
        raise ValueError("predicted cameras are empty")

    intents, details = _native_intents_for_sample(sample, path)
    if not intents:
        raise FileNotFoundError("native intended camera target is unavailable for this sample")

    frame_count = len(poses)
    segment_count = len(intents)
    frame_weights = [intent.frame_count for intent in intents]
    use_weighted_boundaries = all(weight is not None and weight > 0 for weight in frame_weights)
    boundaries: list[int] | None = None
    if use_weighted_boundaries and frame_count > 1:
        total_weight = float(sum(int(weight or 0) for weight in frame_weights))
        cumulative = 0.0
        boundaries = [0]
        for weight in frame_weights[:-1]:
            cumulative += float(weight or 0)
            boundary = int(round(cumulative / max(total_weight, 1.0) * float(frame_count)))
            boundary = min(max(boundary, boundaries[-1] + 1), frame_count - 1)
            boundaries.append(boundary)
        boundaries.append(frame_count)
    else:
        remaining_intervals = max(frame_count - 1, 0)
        base_intervals = remaining_intervals // max(segment_count, 1)
        remainder = remaining_intervals % max(segment_count, 1)

    start = 0
    current_rotation = poses[0, :3, :3].copy()
    gt_segments: list[np.ndarray] = []
    pred_segments: list[np.ndarray] = []
    segment_details: list[dict[str, Any]] = []
    weighted_scores: list[tuple[float, float]] = []
    for index, intent in enumerate(intents):
        if boundaries is not None:
            start = int(boundaries[index])
            end = int(boundaries[index + 1])
            if index:
                start = max(start - 1, 0)
        else:
            intervals = base_intervals + (1 if index < remainder else 0)
            end = min(frame_count, start + intervals + 1)
            if index == segment_count - 1:
                end = frame_count
        pred_turn = poses[start:end]
        if len(pred_turn) <= 0:
            continue
        segment_start_rotation = current_rotation.copy()
        gt_turn, current_rotation = _build_adaptive_gt_segment(
            pred_turn,
            intent,
            current_rotation=current_rotation,
        )
        if intent.perspective == "third_person" and intent.rotation is not None:
            segment_diagnostics = _score_orbit_segment(gt_turn, pred_turn)
        else:
            segment_diagnostics = _score_segment(
                pred_turn,
                intent,
                segment_start_rotation=segment_start_rotation,
            )
        weighted_scores.append((segment_diagnostics["segment_score"], float(len(pred_turn))))
        if index:
            gt_turn = gt_turn[1:]
            pred_turn = pred_turn[1:]
        gt_segments.append(gt_turn)
        pred_segments.append(pred_turn)
        segment_details.append(
            {
                "label": intent.label,
                "translation": list(intent.translation) if intent.translation is not None else None,
                "rotation": list(intent.rotation) if intent.rotation is not None else None,
                "perspective": intent.perspective,
                "source": intent.source,
                "action_frame_count": intent.frame_count,
                "start_frame": int(start),
                "end_frame": int(end),
                **segment_diagnostics,
            }
        )
        if boundaries is None:
            start = max(end - 1, start)

    if not gt_segments or not pred_segments:
        raise ValueError("native intended camera target produced no evaluable segments")

    gt = np.concatenate(gt_segments, axis=0).astype(np.float32)
    pred = np.concatenate(pred_segments, axis=0).astype(np.float32)
    ate_t, ate_r = _compute_ate(gt, pred)
    path_length = _path_length(pred)
    rotation_total = _total_rotation_deg(pred)
    # Legacy relative-error fields, kept for backward-compatible diagnostics
    # (dashboards / scripts already key off `raw.nate_translation` etc.) --
    # this is the formula with the small-motion floor bug; it is no longer
    # what determines `normalized`, see `accuracy` below.
    legacy_nate_t = min(ate_t / max(path_length, 0.5), 1.0)
    legacy_nate_r = min(ate_r / max(rotation_total, 10.0), 1.0)
    legacy_accuracy = max(0.0, 1.0 - (legacy_nate_t + legacy_nate_r) / 2.0)

    total_weight = sum(weight for _, weight in weighted_scores)
    accuracy = (
        sum(score * weight for score, weight in weighted_scores) / total_weight
        if total_weight > 0
        else legacy_accuracy
    )

    details = {key: value for key, value in details.items() if value is not None}
    details.update(
        {
            "native_intent_count": len(intents),
            "native_intent_segments": segment_details,
            "native_intent_boundary_policy": "action_frame_counts" if boundaries is not None else "equal_intervals",
            "native_intent_gt_frame_count": int(len(gt)),
            "native_intent_pred_frame_count": int(len(pred)),
            "native_intent_path_length": float(path_length),
            "native_intent_rotation_deg": float(rotation_total),
            "native_intent_nate_translation": float(legacy_nate_t),
            "native_intent_nate_rotation": float(legacy_nate_r),
            "native_intent_accuracy": float(accuracy),
            "native_intent_legacy_accuracy": float(legacy_accuracy),
            "native_intent_scoring_version": "v2_direction_confidence",
        }
    )
    return {
        "raw": {
            "ate_translation": round(float(ate_t), 4),
            "ate_rotation": round(float(ate_r), 4),
            "nate_translation": round(float(legacy_nate_t), 4),
            "nate_rotation": round(float(legacy_nate_r), 4),
            "legacy_accuracy": round(float(legacy_accuracy), 4),
        },
        "normalized": float(accuracy),
        "details": details,
    }


__all__ = ["NativeMotionIntent", "score_native_intended_cameras"]
