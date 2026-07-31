from __future__ import annotations

import numpy as np
import pytest

from worldarena.benchmark.camera_alignment import score_camera_trajectories
from worldarena.benchmark.official_backends.base import mean_squared_pixel_error


def _rotation_x(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _trajectory(frame_count: int) -> np.ndarray:
    times = np.linspace(0.0, 1.0, num=frame_count, dtype=np.float64)
    cameras = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    cameras[:, :3, 3] = np.stack(
        [times, times**2, 0.25 * np.sin(np.pi * times)],
        axis=1,
    )
    for index, time in enumerate(times):
        cameras[index, :3, :3] = _rotation_y(20.0 * time) @ _rotation_x(-8.0 * time)
    return cameras


def _inverse_similarity(
    cameras_gt: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    cameras_pred = np.asarray(cameras_gt, dtype=np.float64).copy()
    cameras_pred[:, :3, 3] = (
        (cameras_gt[:, :3, 3] - translation) @ rotation / scale
    )
    cameras_pred[:, :3, :3] = np.einsum(
        "ij,njk->nik",
        rotation.T,
        cameras_gt[:, :3, :3],
    )
    return cameras_pred


def test_camera_alignment_recovers_full_similarity_transform() -> None:
    cameras_gt = _trajectory(9)
    global_rotation = _rotation_y(37.0) @ _rotation_x(-13.0)
    cameras_pred = _inverse_similarity(
        cameras_gt,
        scale=2.75,
        rotation=global_rotation,
        translation=np.asarray([3.0, -2.0, 4.5], dtype=np.float64),
    )

    (rotation_error, translation_error), details = score_camera_trajectories(
        cameras_pred,
        cameras_gt,
    )

    assert rotation_error < 1e-5
    assert translation_error < 1e-6
    assert details["similarity_alignment"] == "sim3"
    assert details["camera_alignment_version"] == "sim3_v1"
    assert details["similarity_alignment_method"] == "umeyama"
    assert details["similarity_scale"] == pytest.approx(2.75)
    assert details["camera_frame_alignment"] == "matched"


def test_camera_alignment_resamples_gt_when_slam_drops_frames() -> None:
    cameras_gt = _trajectory(9)
    global_rotation = _rotation_y(-24.0) @ _rotation_x(9.0)
    cameras_pred = _inverse_similarity(
        cameras_gt[::2],
        scale=1.8,
        rotation=global_rotation,
        translation=np.asarray([-1.0, 0.5, 2.0], dtype=np.float64),
    )

    (rotation_error, translation_error), details = score_camera_trajectories(
        cameras_pred,
        cameras_gt,
    )

    assert rotation_error < 5e-3
    assert translation_error < 1e-5
    assert details["camera_pred_frame_count"] == 5
    assert details["camera_gt_frame_count"] == 9
    assert details["camera_frame_alignment"] == "gt_resampled_to_prediction"


@pytest.mark.parametrize("stationary", [False, True])
def test_camera_alignment_handles_degenerate_static_paths(stationary: bool) -> None:
    frame_count = 7
    times = np.linspace(0.0, 1.0, num=frame_count, dtype=np.float64)
    cameras_gt = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    if not stationary:
        cameras_gt[:, :3, 3] = np.stack(
            [np.zeros_like(times), np.zeros_like(times), 2.0 * times],
            axis=1,
        )
    for index, time in enumerate(times):
        cameras_gt[index, :3, :3] = _rotation_y(-25.0 * time)

    global_rotation = _rotation_y(31.0) @ _rotation_x(-11.0)
    cameras_pred = _inverse_similarity(
        cameras_gt,
        scale=2.2,
        rotation=global_rotation,
        translation=np.asarray([1.5, -0.75, 3.0], dtype=np.float64),
    )

    (rotation_error, translation_error), details = score_camera_trajectories(
        cameras_pred,
        cameras_gt,
    )

    assert rotation_error < 1e-5
    assert translation_error < 1e-6
    assert details["similarity_alignment_method"] == "orientation_fallback"
    if not stationary:
        assert details["similarity_scale"] == pytest.approx(2.2)


def test_pixel_mse_uses_float_arithmetic_for_uint8_inputs() -> None:
    left = np.asarray([0, 20, 255], dtype=np.uint8)
    right = np.asarray([255, 0, 0], dtype=np.uint8)

    score = mean_squared_pixel_error(left, right)

    assert score == pytest.approx((255.0**2 + 20.0**2 + 255.0**2) / 3.0)
