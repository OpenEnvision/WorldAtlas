"""Similarity alignment and error computation for camera trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np


_EPSILON = 1e-12


def _as_camera_matrices(cameras: np.ndarray, *, name: str) -> np.ndarray:
    matrices = np.asarray(cameras, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (N, 4, 4), got {matrices.shape}")
    if len(matrices) == 0:
        raise ValueError(f"{name} must contain at least one pose")
    if not np.all(np.isfinite(matrices)):
        raise ValueError(f"{name} contains non-finite values")
    return matrices


def resample_camera_matrices(cameras: np.ndarray, target_frames: int) -> np.ndarray:
    """Uniformly resample camera centers and orientations to ``target_frames``."""
    matrices = _as_camera_matrices(cameras, name="camera matrices")
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if len(matrices) == target_frames:
        return matrices.astype(np.float32)

    from worldarena.benchmark.annotations import (  # noqa: PLC0415
        _matrix_to_pose_vector,
        _pose_vector_to_matrix,
        _resample_pose_vectors,
    )

    pose_vectors = np.stack(
        [_matrix_to_pose_vector(matrix.astype(np.float32)) for matrix in matrices],
        axis=0,
    )
    pose_indices = np.linspace(
        0.0,
        float(len(pose_vectors) - 1),
        num=len(pose_vectors),
        dtype=np.float32,
    )
    interpolated = _resample_pose_vectors(pose_vectors, pose_indices, target_frames)
    return np.stack(
        [_pose_vector_to_matrix(pose_row) for pose_row in interpolated],
        axis=0,
    ).astype(np.float32)


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest proper rotation matrix to ``matrix``."""
    left, _, right_t = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0.0:
        correction[-1, -1] = -1.0
    return left @ correction @ right_t


def _orientation_alignment(pred_rotations: np.ndarray, gt_rotations: np.ndarray) -> np.ndarray:
    cross_rotation = np.zeros((3, 3), dtype=np.float64)
    for pred_rotation, gt_rotation in zip(pred_rotations, gt_rotations):
        cross_rotation += gt_rotation @ pred_rotation.T
    return _project_to_rotation(cross_rotation)


def _estimate_similarity_transform(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    pred_rotations: np.ndarray,
    gt_rotations: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate the positive-scale Sim(3) mapping prediction coordinates to GT."""
    pred_mean = pred_positions.mean(axis=0)
    gt_mean = gt_positions.mean(axis=0)
    pred_centered = pred_positions - pred_mean
    gt_centered = gt_positions - gt_mean
    pred_variance = float(np.sum(pred_centered**2) / len(pred_positions))
    covariance = gt_centered.T @ pred_centered / float(len(pred_positions))
    covariance_rank = int(np.linalg.matrix_rank(covariance))

    method = "umeyama"
    if pred_variance > _EPSILON and covariance_rank >= 2:
        left, singular_values, right_t = np.linalg.svd(covariance)
        correction = np.ones(3, dtype=np.float64)
        if np.linalg.det(left @ right_t) < 0.0:
            correction[-1] = -1.0
        rotation = left @ np.diag(correction) @ right_t
        similarity_scale = float(np.dot(singular_values, correction) / pred_variance)
    else:
        # A stationary or collinear camera path cannot determine all three axes
        # from camera centers alone. Use the corresponding camera orientations to
        # resolve that gauge, then solve the remaining positive scale in closed form.
        method = "orientation_fallback"
        rotation = _orientation_alignment(pred_rotations, gt_rotations)
        rotated_pred = pred_centered @ rotation.T
        denominator = float(np.sum(rotated_pred**2))
        numerator = float(np.sum(rotated_pred * gt_centered))
        if denominator <= _EPSILON:
            similarity_scale = 1.0
        elif numerator > _EPSILON:
            similarity_scale = numerator / denominator
        else:
            gt_energy = float(np.sum(gt_centered**2))
            similarity_scale = np.sqrt(gt_energy / denominator) if gt_energy > _EPSILON else 1.0

    if not np.isfinite(similarity_scale) or similarity_scale <= _EPSILON:
        raise ValueError(f"invalid Sim(3) scale: {similarity_scale}")
    translation = gt_mean - similarity_scale * (rotation @ pred_mean)
    return (
        similarity_scale,
        rotation,
        translation,
        {
            "similarity_alignment": "sim3",
            "camera_alignment_version": "sim3_v1",
            "similarity_alignment_method": method,
            "similarity_covariance_rank": covariance_rank,
            "similarity_scale": similarity_scale,
            "similarity_rotation": rotation.tolist(),
            "similarity_translation": translation.tolist(),
            # Preserve the legacy detail keys for report consumers.
            "translation_alignment": f"sim3_{method}",
            "translation_alignment_scale": similarity_scale,
        },
    )


def score_camera_trajectories(
    cameras_pred: np.ndarray,
    cameras_gt: np.ndarray,
    *,
    gt_scale: float = 1.0,
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Align camera-to-world trajectories with Sim(3) and return rotation/ATE errors."""
    pred = _as_camera_matrices(cameras_pred, name="predicted cameras")
    gt = _as_camera_matrices(cameras_gt, name="ground-truth cameras")
    if not np.isfinite(gt_scale) or gt_scale <= 0.0:
        raise ValueError(f"gt_scale must be positive and finite, got {gt_scale}")

    pred_frame_count = len(pred)
    gt_frame_count = len(gt)
    frame_alignment = "matched"
    if gt_frame_count != pred_frame_count:
        gt = resample_camera_matrices(gt, pred_frame_count).astype(np.float64)
        frame_alignment = "gt_resampled_to_prediction"

    gt = gt.copy()
    gt[:, :3, 3] /= float(gt_scale)
    similarity_scale, similarity_rotation, similarity_translation, details = (
        _estimate_similarity_transform(
            pred[:, :3, 3],
            gt[:, :3, 3],
            pred[:, :3, :3],
            gt[:, :3, :3],
        )
    )

    aligned_positions = (
        similarity_scale * (pred[:, :3, 3] @ similarity_rotation.T)
        + similarity_translation
    )
    aligned_rotations = np.einsum(
        "ij,njk->nik",
        similarity_rotation,
        pred[:, :3, :3],
    )
    translation_error = float(
        np.linalg.norm(aligned_positions - gt[:, :3, 3], axis=1).mean()
    )
    rotation_product = aligned_rotations @ np.swapaxes(gt[:, :3, :3], 1, 2)
    cosine = (np.trace(rotation_product, axis1=1, axis2=2) - 1.0) / 2.0
    rotation_error = float(
        np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean()
    )
    details.update(
        {
            "camera_pred_frame_count": pred_frame_count,
            "camera_gt_frame_count": gt_frame_count,
            "camera_aligned_frame_count": pred_frame_count,
            "camera_frame_alignment": frame_alignment,
        }
    )
    return (rotation_error, translation_error), details


__all__ = ["resample_camera_matrices", "score_camera_trajectories"]
