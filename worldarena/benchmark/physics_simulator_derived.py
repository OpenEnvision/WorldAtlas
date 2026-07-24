"""Derived physics metrics computed from simulator outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


FINAL_STATE_METRIC = "final_state_accuracy"
TRAJECTORY_METRICS: tuple[str, ...] = (
    "trajectory_rmse",
    "final_position_error",
    "speed_similarity",
    "acceleration_similarity",
    "directional_consistency",
)
DERIVED_SIMULATOR_METRICS: tuple[str, ...] = (FINAL_STATE_METRIC, *TRAJECTORY_METRICS)
_EPS = 1e-8


def _payload(sample: Any) -> Mapping[str, Any]:
    if hasattr(sample, "to_dict"):
        return sample.to_dict()
    if isinstance(sample, Mapping):
        return sample
    return {}


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    return {key: np.asarray(payload[key]) for key in payload.files}


def _load_worldlines(path: Path) -> dict[str, np.ndarray]:
    payload = _load_npz_dict(path)
    if "xyz" not in payload:
        raise KeyError(f"worldline artifact {path} must contain xyz")
    xyz = np.asarray(payload["xyz"], dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"expected xyz shape [N,T,3], got {xyz.shape}")
    valid = np.asarray(
        payload.get("valid", np.ones(xyz.shape[:2], dtype=bool)),
        dtype=bool,
    )
    if valid.shape != xyz.shape[:2]:
        raise ValueError(f"expected valid shape {xyz.shape[:2]}, got {valid.shape}")
    output: dict[str, np.ndarray] = {"xyz": xyz, "valid": valid}
    if "pixels" in payload:
        pixels = np.asarray(payload["pixels"], dtype=np.float64)
        if pixels.ndim != 3 or pixels.shape[-1] < 2:
            raise ValueError(f"expected pixels shape [N,T,2], got {pixels.shape}")
        output["pixels"] = pixels[..., :2]
    return output


def _load_first_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    payload = _load_npz_dict(path)
    for key in preferred_keys:
        if key in payload:
            return np.asarray(payload[key])
    if len(payload) == 1:
        return np.asarray(next(iter(payload.values())))
    raise KeyError(f"could not infer array from {path}; found keys {sorted(payload)}")


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 2:
        depth = depth[None]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"expected depth shape [T,H,W], got {depth.shape}")
    return depth


def _normalize_c2w(c2w: np.ndarray) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.ndim == 2:
        c2w = c2w[None]
    if c2w.ndim != 3:
        raise ValueError(f"expected camera pose sequence, got {c2w.shape}")
    if c2w.shape[-2:] == (4, 4):
        return c2w
    if c2w.shape[-2:] == (3, 4):
        bottom = np.repeat(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)[None, None, :],
            c2w.shape[0],
            axis=0,
        )
        return np.concatenate([c2w, bottom], axis=1)
    raise ValueError(f"unsupported camera pose layout: {c2w.shape}")


def _intrinsics_to_vec(intrinsics: np.ndarray, frames: int) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim == 1 and intrinsics.shape[0] == 4:
        intrinsics = intrinsics[None]
    elif intrinsics.ndim == 2 and intrinsics.shape == (3, 3):
        intrinsics = np.asarray(
            [[intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]],
            dtype=np.float64,
        )
    elif intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3):
        intrinsics = np.stack(
            [
                np.asarray([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]], dtype=np.float64)
                for matrix in intrinsics
            ],
            axis=0,
        )
    elif not (intrinsics.ndim == 2 and intrinsics.shape[-1] == 4):
        raise ValueError(f"unsupported intrinsics layout: {intrinsics.shape}")

    if intrinsics.shape[0] == 1 and frames > 1:
        intrinsics = np.repeat(intrinsics, frames, axis=0)
    if intrinsics.shape[0] < frames:
        raise ValueError(f"intrinsics has {intrinsics.shape[0]} frames, expected {frames}")
    return intrinsics[:frames]


def _load_cameras(path: Path, frames: int) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_npz_dict(path)
    c2w = None
    for key in ("c2w", "poses", "pose", "camera", "extrinsics"):
        if key in payload:
            c2w = _normalize_c2w(payload[key])
            break
    if c2w is None:
        raise KeyError(f"could not find c2w camera poses in {path}; found keys {sorted(payload)}")
    intrinsics = None
    for key in ("K", "intrinsics", "camera_intrinsics"):
        if key in payload:
            intrinsics = _intrinsics_to_vec(payload[key], frames)
            break
    if intrinsics is None:
        raise KeyError(f"could not find camera intrinsics in {path}; found keys {sorted(payload)}")
    if c2w.shape[0] < frames:
        raise ValueError(f"camera poses has {c2w.shape[0]} frames, expected {frames}")
    return c2w[:frames], intrinsics


def _sample_depth_nearest(depth_hw: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    height, width = depth_hw.shape
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    x = np.clip(np.rint(pixels[:, 0]).astype(int), 0, max(width - 1, 0))
    y = np.clip(np.rint(pixels[:, 1]).astype(int), 0, max(height - 1, 0))
    return np.asarray(depth_hw[y, x], dtype=np.float64)


def ensure_pred_worldlines(
    *,
    world_gt_root: Path,
    pred_world_root: Path,
) -> dict[str, Any]:
    output_path = pred_world_root / "worldlines.npz"
    if output_path.exists():
        return {"created": False, "path": str(output_path.resolve()), "reason": "already_exists"}

    gt_worldlines = _load_worldlines(world_gt_root / "worldlines.npz")
    if "pixels" not in gt_worldlines:
        raise KeyError("world_gt/worldlines.npz must contain pixels to reconstruct pred_world/worldlines.npz")
    pred_depth = _normalize_depth(_load_first_array(pred_world_root / "depth.npz", ("depth", "data")))
    frames = min(pred_depth.shape[0], gt_worldlines["xyz"].shape[1])
    c2w, intrinsics = _load_cameras(pred_world_root / "cameras.npz", frames)
    pixels = np.asarray(gt_worldlines["pixels"][:, :frames], dtype=np.float64)
    valid = np.asarray(gt_worldlines["valid"][:, :frames], dtype=bool)
    xyz = np.full((pixels.shape[0], frames, 3), np.nan, dtype=np.float32)
    for frame_idx in range(frames):
        sampled_depth = _sample_depth_nearest(pred_depth[frame_idx], pixels[:, frame_idx])
        fx, fy, cx, cy = intrinsics[frame_idx].tolist()
        x = (pixels[:, frame_idx, 0] - cx) / max(fx, _EPS) * sampled_depth
        y = (pixels[:, frame_idx, 1] - cy) / max(fy, _EPS) * sampled_depth
        cam_points = np.stack([x, y, sampled_depth, np.ones_like(sampled_depth)], axis=-1)
        xyz[:, frame_idx] = (c2w[frame_idx] @ cam_points.T).T[:, :3].astype(np.float32)
    np.savez_compressed(
        output_path,
        xyz=xyz,
        pixels=pixels.astype(np.float32),
        valid=valid,
    )
    return {
        "created": True,
        "path": str(output_path.resolve()),
        "frames": int(frames),
        "tracks": int(xyz.shape[0]),
    }


def _trajectory_spec(sample: Any) -> Mapping[str, Any]:
    sample_payload = _payload(sample)
    physics_spec = dict(sample_payload.get("physics_spec") or {})
    spec = physics_spec.get("trajectory_metric")
    return spec if isinstance(spec, Mapping) else {}


def _resolution(sample: Any) -> tuple[float, float]:
    sample_payload = _payload(sample)
    physics_spec = dict(sample_payload.get("physics_spec") or {})
    resolution = physics_spec.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        width, height = float(resolution[0]), float(resolution[1])
        if width > 0 and height > 0:
            return width, height
    width = float(sample_payload.get("width") or 0)
    height = float(sample_payload.get("height") or 0)
    if width > 0 and height > 0:
        return width, height
    raise ValueError("normalized pixel trajectory metrics require sample width and height")


def _select_positions(
    gt_worldlines: Mapping[str, np.ndarray],
    pred_worldlines: Mapping[str, np.ndarray],
    *,
    sample: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    spec = _trajectory_spec(sample)
    coordinate_space = str(spec.get("coordinate_space") or "world_xyz").strip()
    if coordinate_space == "normalized_pixels":
        if "pixels" not in gt_worldlines or "pixels" not in pred_worldlines:
            raise KeyError("normalized_pixels trajectory metrics require pixels in both worldline artifacts")
        width, height = _resolution(sample)
        scale = np.asarray([width, height], dtype=np.float64)
        gt = np.asarray(gt_worldlines["pixels"], dtype=np.float64) / scale
        pred = np.asarray(pred_worldlines["pixels"], dtype=np.float64) / scale
    elif coordinate_space == "world_xyz":
        gt = np.asarray(gt_worldlines["xyz"], dtype=np.float64)
        pred = np.asarray(pred_worldlines["xyz"], dtype=np.float64)
    else:
        raise ValueError(f"unsupported trajectory coordinate_space: {coordinate_space}")

    if gt.shape != pred.shape:
        raise ValueError(f"mismatched trajectory shapes: {gt.shape} vs {pred.shape}")
    valid = (
        np.asarray(gt_worldlines["valid"], dtype=bool)
        & np.asarray(pred_worldlines["valid"], dtype=bool)
        & np.isfinite(gt).all(axis=-1)
        & np.isfinite(pred).all(axis=-1)
    )
    if gt.shape[:2] != valid.shape:
        raise ValueError(f"trajectory valid mask shape mismatch: {valid.shape} for {gt.shape}")
    if not np.any(valid):
        raise ValueError("no valid trajectory points")
    return gt, pred, valid, coordinate_space


def _cosine_similarity_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    both_zero = (left_norm <= _EPS) & (right_norm <= _EPS)
    one_zero = ((left_norm <= _EPS) ^ (right_norm <= _EPS))
    valid = ~(both_zero | one_zero)
    values = np.empty(left_norm.shape, dtype=np.float64)
    values[both_zero] = 1.0
    values[one_zero] = 0.0
    values[valid] = np.sum(left[valid] * right[valid], axis=-1) / np.maximum(
        left_norm[valid] * right_norm[valid],
        _EPS,
    )
    return np.clip(values, -1.0, 1.0)


def compute_trajectory_dynamics_metrics(
    *,
    gt_worldlines_path: Path,
    pred_worldlines_path: Path,
    sample: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    gt_worldlines = _load_worldlines(gt_worldlines_path)
    pred_worldlines = _load_worldlines(pred_worldlines_path)
    gt, pred, valid, coordinate_space = _select_positions(
        gt_worldlines,
        pred_worldlines,
        sample=sample,
    )

    delta = pred - gt
    squared_distance = np.sum(delta * delta, axis=-1)
    trajectory_rmse = float(np.sqrt(np.mean(squared_distance[valid])))

    final_errors = []
    for track_idx in range(gt.shape[0]):
        if not valid[track_idx, -1]:
            continue
        gt_track = gt[track_idx]
        pred_track = pred[track_idx]
        gt_valid = valid[track_idx]
        adjacent_valid = gt_valid[:-1] & gt_valid[1:]
        path_length = float(np.sum(np.linalg.norm(np.diff(gt_track, axis=0)[adjacent_valid], axis=-1)))
        if path_length <= _EPS:
            continue
        final_errors.append(float(np.linalg.norm(gt_track[-1] - pred_track[-1]) / path_length))
    if not final_errors:
        raise ValueError("final_position_error requires at least one valid non-stationary trajectory")
    final_position_error = float(np.mean(final_errors))

    velocity_valid = valid[:, 1:] & valid[:, :-1]
    gt_velocity = np.diff(gt, axis=1)
    pred_velocity = np.diff(pred, axis=1)
    if not np.any(velocity_valid):
        raise ValueError("speed_similarity requires at least one valid adjacent trajectory pair")
    speed_similarity = float(
        np.mean(_cosine_similarity_rows(gt_velocity[velocity_valid], pred_velocity[velocity_valid]))
    )

    acceleration_valid = velocity_valid[:, 1:] & velocity_valid[:, :-1]
    gt_acceleration = np.diff(gt_velocity, axis=1)
    pred_acceleration = np.diff(pred_velocity, axis=1)
    if not np.any(acceleration_valid):
        raise ValueError("acceleration_similarity requires at least one valid acceleration pair")
    acceleration_similarity = float(
        np.mean(
            _cosine_similarity_rows(
                gt_acceleration[acceleration_valid],
                pred_acceleration[acceleration_valid],
            )
        )
    )

    direction_scores = []
    for track_idx in range(gt.shape[0]):
        if not (valid[track_idx, 0] and valid[track_idx, -1]):
            continue
        gt_direction = gt[track_idx, -1] - gt[track_idx, 0]
        pred_direction = pred[track_idx, -1] - pred[track_idx, 0]
        gt_norm = float(np.linalg.norm(gt_direction))
        pred_norm = float(np.linalg.norm(pred_direction))
        if gt_norm <= _EPS and pred_norm <= _EPS:
            direction_scores.append(1.0)
            continue
        if gt_norm <= _EPS or pred_norm <= _EPS:
            direction_scores.append(0.0)
            continue
        cosine = float(np.dot(gt_direction, pred_direction) / max(gt_norm * pred_norm, _EPS))
        theta = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        direction_scores.append(float(np.clip((180.0 - theta) / 180.0, 0.0, 1.0)))
    if not direction_scores:
        raise ValueError("directional_consistency requires valid first and final trajectory points")

    metrics = {
        "trajectory_rmse": trajectory_rmse,
        "final_position_error": final_position_error,
        "speed_similarity": speed_similarity,
        "acceleration_similarity": acceleration_similarity,
        "directional_consistency": float(np.mean(direction_scores)),
    }
    details = {
        "coordinate_space": coordinate_space,
        "valid_points": int(np.count_nonzero(valid)),
        "tracks": int(gt.shape[0]),
        "frames": int(gt.shape[1]),
        "fpe_tracks": int(len(final_errors)),
        "direction_tracks": int(len(direction_scores)),
    }
    return metrics, details


def _normalize_label(label: Any) -> str:
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def _final_state_spec(sample: Any) -> Mapping[str, Any]:
    sample_payload = _payload(sample)
    physics_spec = dict(sample_payload.get("physics_spec") or {})
    metadata_template = physics_spec.get("metadata_template")
    candidates = [
        physics_spec.get("final_state_metric"),
        physics_spec.get("final_state"),
        metadata_template.get("final_state_metric") if isinstance(metadata_template, Mapping) else None,
        metadata_template.get("final_state") if isinstance(metadata_template, Mapping) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate:
            return candidate
    return {}


def _label_from_spec(spec: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in spec and str(spec[key]).strip():
            return _normalize_label(spec[key])
    return None


def _axis_index(axis: Any) -> int:
    axis_name = str(axis or "y").strip().lower()
    if axis_name in {"x", "0"}:
        return 0
    if axis_name in {"y", "1"}:
        return 1
    if axis_name in {"z", "2"}:
        return 2
    raise ValueError(f"unsupported final-state axis: {axis}")


def _lever_track_indices(spec: Mapping[str, Any]) -> tuple[int, int]:
    track_indices = spec.get("track_indices")
    if isinstance(track_indices, Mapping):
        if "left" in track_indices and "right" in track_indices:
            return int(track_indices["left"]), int(track_indices["right"])
    if "left_track_index" in spec and "right_track_index" in spec:
        return int(spec["left_track_index"]), int(spec["right_track_index"])
    raise ValueError("lever final-state metric requires left/right track indices")


def _classify_lever_state(worldlines: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    left_idx, right_idx = _lever_track_indices(spec)
    xyz = np.asarray(worldlines["xyz"], dtype=np.float64)
    valid = np.asarray(worldlines["valid"], dtype=bool)
    if left_idx >= xyz.shape[0] or right_idx >= xyz.shape[0]:
        raise ValueError("lever final-state track index is outside the worldline array")
    joint_valid = valid[left_idx] & valid[right_idx] & np.isfinite(xyz[[left_idx, right_idx]]).all(axis=-1).all(axis=0)
    valid_frames = np.where(joint_valid)[0]
    if len(valid_frames) == 0:
        raise ValueError("lever final-state metric has no frame where both lever tracks are valid")
    frame_idx = int(valid_frames[-1])
    axis_idx = _axis_index(spec.get("axis"))
    tolerance = float(spec.get("balanced_tolerance", spec.get("tolerance", 0.0)))
    lower_is_down = bool(spec.get("lower_is_down", True))
    left_value = float(xyz[left_idx, frame_idx, axis_idx])
    right_value = float(xyz[right_idx, frame_idx, axis_idx])
    signed_delta = left_value - right_value
    labels = dict(spec.get("labels") or {})
    left_label = _normalize_label(labels.get("left_down", "left_down"))
    right_label = _normalize_label(labels.get("right_down", "right_down"))
    balanced_label = _normalize_label(labels.get("balanced", "balanced"))
    if abs(signed_delta) <= tolerance:
        label = balanced_label
    elif (signed_delta < 0.0) == lower_is_down:
        label = left_label
    else:
        label = right_label
    return label, {
        "left_track_index": left_idx,
        "right_track_index": right_idx,
        "frame_index": frame_idx,
        "axis": str(spec.get("axis") or "y"),
        "left_value": left_value,
        "right_value": right_value,
        "balanced_tolerance": tolerance,
        "lower_is_down": lower_is_down,
    }


def compute_final_state_accuracy(
    *,
    gt_worldlines_path: Path,
    pred_worldlines_path: Path,
    sample: Any,
) -> tuple[float | None, dict[str, Any], str | None]:
    spec = _final_state_spec(sample)
    if not spec:
        return None, {}, "physics sample does not define final_state_metric"

    true_label = _label_from_spec(
        spec,
        "true_label",
        "gt_label",
        "target_label",
        "expected_label",
        "label",
    )
    pred_label = _label_from_spec(
        spec,
        "predicted_label",
        "prediction_label",
        "pred_label",
    )
    details: dict[str, Any] = {}
    if true_label is None or pred_label is None:
        if str(spec.get("type", "lever")).strip().lower() != "lever":
            raise ValueError("final_state_metric without explicit labels currently supports only type=lever")
        gt_worldlines = _load_worldlines(gt_worldlines_path)
        pred_worldlines = _load_worldlines(pred_worldlines_path)
        if true_label is None:
            true_label, gt_details = _classify_lever_state(gt_worldlines, spec)
            details["gt_rule"] = gt_details
        if pred_label is None:
            pred_label, pred_details = _classify_lever_state(pred_worldlines, spec)
            details["pred_rule"] = pred_details

    assert true_label is not None
    assert pred_label is not None
    score = 1.0 if pred_label == true_label else 0.0
    details.update(
        {
            "true_label": true_label,
            "predicted_label": pred_label,
            "matched": bool(score == 1.0),
        }
    )
    return score, details, None


def append_derived_simulator_metrics(
    result: dict[str, Any],
    *,
    sample: Any,
    world_gt_root: Path,
    pred_world_root: Path | None,
) -> None:
    metric_details = result.setdefault("simulator_metric_details", {})
    metric_errors = result.setdefault("simulator_metric_errors", {})
    not_applicable = result.setdefault("not_applicable_metrics", {})
    if pred_world_root is None:
        message = "pred_world artifacts are unavailable"
        for metric_name in TRAJECTORY_METRICS:
            metric_errors[metric_name] = message
        if _final_state_spec(sample):
            metric_errors[FINAL_STATE_METRIC] = message
        else:
            not_applicable[FINAL_STATE_METRIC] = "physics sample does not define final_state_metric"
        return

    gt_worldlines_path = world_gt_root / "worldlines.npz"
    pred_worldlines_path = pred_world_root / "worldlines.npz"

    try:
        trajectory_metrics, trajectory_details = compute_trajectory_dynamics_metrics(
            gt_worldlines_path=gt_worldlines_path,
            pred_worldlines_path=pred_worldlines_path,
            sample=sample,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for metric_name in TRAJECTORY_METRICS:
            metric_errors[metric_name] = message
    else:
        for metric_name, value in trajectory_metrics.items():
            result[metric_name] = value
            metric_details[metric_name] = trajectory_details

    try:
        score, details, reason = compute_final_state_accuracy(
            gt_worldlines_path=gt_worldlines_path,
            pred_worldlines_path=pred_worldlines_path,
            sample=sample,
        )
    except Exception as exc:
        metric_errors[FINAL_STATE_METRIC] = f"{type(exc).__name__}: {exc}"
    else:
        if reason is not None:
            not_applicable[FINAL_STATE_METRIC] = reason
        elif score is not None:
            result[FINAL_STATE_METRIC] = score
            metric_details[FINAL_STATE_METRIC] = details


__all__ = [
    "DERIVED_SIMULATOR_METRICS",
    "FINAL_STATE_METRIC",
    "TRAJECTORY_METRICS",
    "append_derived_simulator_metrics",
    "compute_final_state_accuracy",
    "compute_trajectory_dynamics_metrics",
]
