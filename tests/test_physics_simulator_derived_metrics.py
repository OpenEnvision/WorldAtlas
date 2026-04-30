from __future__ import annotations

from pathlib import Path

import numpy as np

from worldarena.benchmark.config import load_config
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.registry import build_metrics
from worldarena.benchmark.physics_simulator_derived import (
    compute_final_state_accuracy,
    compute_trajectory_dynamics_metrics,
    ensure_pred_worldlines,
)


def _write_worldlines(path: Path, xyz: np.ndarray, *, pixels: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.ones(xyz.shape[:2], dtype=bool)
    if pixels is None:
        np.savez_compressed(path, xyz=xyz.astype(np.float32), valid=valid)
    else:
        np.savez_compressed(path, xyz=xyz.astype(np.float32), pixels=pixels.astype(np.float32), valid=valid)


def test_trajectory_dynamics_match_identical_worldlines(tmp_path: Path) -> None:
    xyz = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    gt_path = tmp_path / "world_gt" / "worldlines.npz"
    pred_path = tmp_path / "pred_world" / "worldlines.npz"
    _write_worldlines(gt_path, xyz)
    _write_worldlines(pred_path, xyz)

    metrics, details = compute_trajectory_dynamics_metrics(
        gt_worldlines_path=gt_path,
        pred_worldlines_path=pred_path,
        sample={"physics_spec": {"trajectory_metric": {"coordinate_space": "world_xyz"}}},
    )

    assert metrics == {
        "trajectory_rmse": 0.0,
        "final_position_error": 0.0,
        "speed_similarity": 1.0,
        "acceleration_similarity": 1.0,
        "directional_consistency": 1.0,
    }
    assert details["coordinate_space"] == "world_xyz"
    assert details["valid_points"] == 4


def test_trajectory_dynamics_penalize_reverse_motion(tmp_path: Path) -> None:
    gt = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)
    pred = np.asarray([[[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]], dtype=np.float32)
    gt_path = tmp_path / "world_gt" / "worldlines.npz"
    pred_path = tmp_path / "pred_world" / "worldlines.npz"
    _write_worldlines(gt_path, gt)
    _write_worldlines(pred_path, pred)

    metrics, _ = compute_trajectory_dynamics_metrics(
        gt_worldlines_path=gt_path,
        pred_worldlines_path=pred_path,
        sample={"physics_spec": {"trajectory_metric": {"coordinate_space": "world_xyz"}}},
    )

    assert metrics["trajectory_rmse"] > 0.0
    assert metrics["final_position_error"] == 2.0
    assert metrics["speed_similarity"] == -1.0
    assert metrics["directional_consistency"] == 0.0


def test_final_state_accuracy_uses_lever_rule(tmp_path: Path) -> None:
    gt = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    pred = gt.copy()
    gt_path = tmp_path / "world_gt" / "worldlines.npz"
    pred_path = tmp_path / "pred_world" / "worldlines.npz"
    _write_worldlines(gt_path, gt)
    _write_worldlines(pred_path, pred)

    score, details, reason = compute_final_state_accuracy(
        gt_worldlines_path=gt_path,
        pred_worldlines_path=pred_path,
        sample={
            "physics_spec": {
                "final_state_metric": {
                    "type": "lever",
                    "left_track_index": 0,
                    "right_track_index": 1,
                    "axis": "y",
                    "balanced_tolerance": 0.05,
                }
            }
        },
    )

    assert reason is None
    assert score == 1.0
    assert details["true_label"] == "left_down"
    assert details["predicted_label"] == "left_down"


def test_final_state_accuracy_is_not_applicable_without_spec(tmp_path: Path) -> None:
    score, details, reason = compute_final_state_accuracy(
        gt_worldlines_path=tmp_path / "missing_gt.npz",
        pred_worldlines_path=tmp_path / "missing_pred.npz",
        sample={"physics_spec": {}},
    )

    assert score is None
    assert details == {}
    assert reason == "physics sample does not define final_state_metric"


def test_ensure_pred_worldlines_reconstructs_from_depth_and_cameras(tmp_path: Path) -> None:
    world_gt_root = tmp_path / "world_gt"
    pred_world_root = tmp_path / "pred_world"
    pixels = np.asarray([[[1.0, 1.0], [2.0, 1.0]]], dtype=np.float32)
    gt_xyz = np.zeros((1, 2, 3), dtype=np.float32)
    _write_worldlines(world_gt_root / "worldlines.npz", gt_xyz, pixels=pixels)
    pred_world_root.mkdir(parents=True)
    np.savez_compressed(pred_world_root / "depth.npz", depth=np.ones((2, 4, 4), dtype=np.float32))
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    intrinsics = np.asarray([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    np.savez_compressed(pred_world_root / "cameras.npz", c2w=c2w, K=intrinsics)

    info = ensure_pred_worldlines(world_gt_root=world_gt_root, pred_world_root=pred_world_root)
    payload = np.load(pred_world_root / "worldlines.npz")

    assert info["created"] is True
    assert payload["xyz"].shape == (1, 2, 3)
    assert np.allclose(payload["xyz"][0, 0], [1.0, 1.0, 1.0])


def test_physics_config_registers_derived_simulator_metrics() -> None:
    config = load_config(Path("config/benchmark_physics.yaml"))
    names = config.metrics_by_suite["experimental"]
    expected = {
        "trajectory_rmse",
        "final_position_error",
        "speed_similarity",
        "acceleration_similarity",
        "directional_consistency",
        "final_state_accuracy",
    }

    assert expected.issubset(names)
    assert all(METRIC_DIMENSIONS[name] == "physics" for name in expected)
    built = {metric.name for metric in build_metrics(names, config)}
    assert expected.issubset(built)
