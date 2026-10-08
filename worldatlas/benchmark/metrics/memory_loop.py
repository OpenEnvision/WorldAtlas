"""Closed-loop memory metrics scored on pose-aligned revisit pairs.

Every metric here is self-referential: the reference for a revisit frame is an earlier
frame the model itself generated at the same estimated viewpoint. No ground-truth long
video is needed, which matters because no dataset ships minute-long closed-loop
footage. All five share one expensive pass over the rollout, cached per prediction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from worldatlas.benchmark.runtime_health import is_cuda_failure, raise_if_cuda_failure

from worldatlas.benchmark.consistency_backends import dinov2_image_features
from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.memory_pairing import (
    DEFAULT_MAX_PAIR_DISTANCE,
    DEFAULT_MIN_DEPARTURE_FACTOR,
    DEFAULT_MIN_EXCURSION_FACTOR,
    DEFAULT_MIN_GAP_SECONDS,
    DEFAULT_ROTATION_WEIGHT,
    MEMORY_SAFETY_REVISION,
    PoseTrack,
    RevisitPair,
    RevisitStructure,
    find_revisit_pairs,
    fit_exponential_half_life,
    observation_revisit_split,
    resolve_pose_track,
    select_diverse_pairs,
)
from worldatlas.benchmark.metrics.base import Metric, clamp01, normalize_score, resolve_metric_backend
from worldatlas.benchmark.metrics.legacy import _resize_like, _ssim_rgb
from worldatlas.benchmark.metrics.long_horizon_diagnostics import (
    _frame_depth_points,
    _load_depth_mask,
    _select_segment_indices,
    score_point_cloud_memory,
)
from worldatlas.benchmark.metrics.style import _gram_distance
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput
from worldatlas.common.video_io import probe_video_fps, read_video_frames_at


MEMORY_LOOP_METRICS = (
    "memory_return_gate",
    "memory_scene_f1",
    "memory_revisit_fidelity",
    "memory_geometric_closure",
    "memory_rendering_recovery",
    "memory_half_life",
)
MEMORY_LOOP_BACKENDS = {
    "memory_return_gate": ("pose_loop_gate", {"pose_loop_gate"}),
    "memory_scene_f1": ("point_cloud_f1", {"point_cloud_f1"}),
    "memory_revisit_fidelity": ("pose_aligned_dino", {"pose_aligned_dino"}),
    "memory_geometric_closure": ("sift_epipolar", {"sift_epipolar"}),
    "memory_rendering_recovery": ("cielab_gram", {"cielab_gram"}),
    "memory_half_life": ("exponential_fit", {"exponential_fit"}),
}
MEMORY_LOOP_SCORING_REVISION = f"{MEMORY_SAFETY_REVISION}:ungated_memory_20260919"

DEFAULT_MAX_SCORED_PAIRS = 24
DEFAULT_MIN_SIFT_MATCHES = 12
# Epipolar and reprojection residuals are in pixels; the exponentials below map a few
# pixels of error onto the upper half of [0, 1] for a 1280-wide frame.
DEFAULT_EPIPOLAR_TAU_PIXELS = 4.0
DEFAULT_REPROJECTION_TAU_PIXELS = 8.0
# CIELAB lightness spans 0-100 and chroma roughly -128..127, so these taus treat a few
# units of drift as a meaningful change in how the scene is lit.
DEFAULT_LIGHTNESS_TAU = 6.0
DEFAULT_CHROMA_TAU = 8.0
DEFAULT_GRAM_TAU = 0.002
DEFAULT_HALF_LIFE_REFERENCE_SECONDS = 60.0


def _skip(metric_name: str, backend: str, reason: str, details: dict[str, Any]) -> MetricOutput:
    """Build a not-applicable output for a rollout that cannot be scored."""
    del metric_name
    if is_cuda_failure(reason):
        return MetricOutput(raw=None, normalized=None, backend=backend,
                            details={**details, "measurement_status": "backend_error"}, error=reason)
    return MetricOutput(
        raw=None,
        normalized=None,
        backend=backend,
        details=details,
        eligibility_status="not_applicable",
        error=reason,
    )


def _relative_pose(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotation and translation taking camera A's frame into camera B's."""
    world_to_b = np.linalg.inv(pose_b)
    relative = world_to_b @ pose_a
    return relative[:3, :3], relative[:3, 3]


def _skew(vector: np.ndarray) -> np.ndarray:
    """Cross-product matrix of a 3-vector."""
    x, y, z = (float(value) for value in vector)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _sift_correspondences(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    *,
    ratio: float = 0.75,
    max_matches: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Match SIFT keypoints between two frames with Lowe's ratio test."""
    detector = cv2.SIFT_create()
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    keypoints_a, descriptors_a = detector.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(gray_b, None)
    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 2 or len(keypoints_b) < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs: list[tuple[float, int, int]] = []
    for match in matcher.knnMatch(descriptors_a, descriptors_b, k=2):
        if len(match) < 2:
            continue
        best, second = match
        if best.distance < float(ratio) * second.distance:
            pairs.append((best.distance, best.queryIdx, best.trainIdx))
    pairs.sort(key=lambda item: item[0])
    pairs = pairs[: int(max_matches)]
    if not pairs:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    points_a = np.asarray([keypoints_a[query].pt for _, query, _ in pairs], dtype=np.float64)
    points_b = np.asarray([keypoints_b[train].pt for _, _, train in pairs], dtype=np.float64)
    return points_a, points_b


def geometric_closure_for_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    intrinsics_a: np.ndarray,
    intrinsics_b: np.ndarray,
    *,
    min_matches: int = DEFAULT_MIN_SIFT_MATCHES,
    epipolar_tau: float = DEFAULT_EPIPOLAR_TAU_PIXELS,
    reprojection_tau: float = DEFAULT_REPROJECTION_TAU_PIXELS,
) -> dict[str, Any] | None:
    """Test whether two revisit frames obey the geometry their estimated poses imply.

    The fundamental matrix is derived from the poses rather than fitted to the matches,
    so the residual measures whether the generated pixels are consistent with that
    viewpoint change. A model that redraws the scene with a different layout produces
    matches that violate the epipolar constraint even when the frames look plausible.
    """
    points_a, points_b = _sift_correspondences(frame_a, frame_b)
    if len(points_a) < int(min_matches):
        return None

    if not np.isfinite([epipolar_tau, reprojection_tau]).all() or min(epipolar_tau, reprojection_tau) <= 0:
        raise ValueError("geometry pixel scales must be finite and positive")
    if not all(np.isfinite(value).all() for value in (points_a, points_b, pose_a, pose_b, intrinsics_a, intrinsics_b)):
        return None
    try:
        inverse_a = np.linalg.inv(intrinsics_a)
        inverse_b = np.linalg.inv(intrinsics_b)
        rotation, translation = _relative_pose(pose_a, pose_b)
    except np.linalg.LinAlgError:
        return None
    baseline = float(np.linalg.norm(translation))
    if baseline <= 1e-6:
        # E = [t]x R is zero here, not evidence of perfect epipolar agreement.
        # Use the pose-induced rotation homography (never fit H to the matches).
        homography = intrinsics_b @ rotation @ inverse_a
        homogeneous_a = np.hstack([points_a, np.ones((len(points_a), 1))])
        homogeneous_b = np.hstack([points_b, np.ones((len(points_b), 1))])
        predicted_b = homogeneous_a @ homography.T
        predicted_a = homogeneous_b @ np.linalg.inv(homography).T
        valid = (
            np.isfinite(predicted_a).all(axis=1)
            & np.isfinite(predicted_b).all(axis=1)
            & (predicted_a[:, 2] > 1e-9)
            & (predicted_b[:, 2] > 1e-9)
        )
        if int(np.count_nonzero(valid)) < int(min_matches):
            return None
        errors_a = np.linalg.norm(
            predicted_a[valid, :2] / predicted_a[valid, 2:3] - points_a[valid], axis=1,
        )
        errors_b = np.linalg.norm(
            predicted_b[valid, :2] / predicted_b[valid, 2:3] - points_b[valid], axis=1,
        )
        transfer_error = float(np.median(0.5 * (errors_a + errors_b)))
        if not math.isfinite(transfer_error):
            return None
        return {
            "score": float(np.exp(-transfer_error / max(float(reprojection_tau), 1e-6))),
            "geometry_mode": "rotation_homography",
            "epipolar_error_pixels": None,
            "epipolar_score": None,
            "reprojection_error_pixels": transfer_error,
            "match_count": int(np.count_nonzero(valid)),
            "baseline": baseline,
            "implementation_revision": MEMORY_SAFETY_REVISION,
        }
    # Epipolar distance must not depend on arbitrary monocular translation scale.
    essential = _skew(translation / baseline) @ rotation
    fundamental = inverse_b.T @ essential @ inverse_a

    homogeneous_a = np.hstack([points_a, np.ones((len(points_a), 1))])
    homogeneous_b = np.hstack([points_b, np.ones((len(points_b), 1))])
    lines_in_b = homogeneous_a @ fundamental.T
    lines_in_a = homogeneous_b @ fundamental
    numerator = np.abs(np.sum(homogeneous_b * lines_in_b, axis=1))
    distance_b = numerator / np.maximum(np.linalg.norm(lines_in_b[:, :2], axis=1), 1e-9)
    distance_a = numerator / np.maximum(np.linalg.norm(lines_in_a[:, :2], axis=1), 1e-9)
    epipolar_error = float(np.median(0.5 * (distance_a + distance_b)))
    if not math.isfinite(epipolar_error):
        return None

    reprojection_error: float | None = None
    positive_depth_fraction = 0.0
    # Triangulation is ill-conditioned when the two views sit on top of each other, which
    # is exactly the case for a rotation-only revisit, so it is reported only when the
    # pair actually has a baseline.
    if baseline > 1e-6:
        projection_a = intrinsics_a @ np.linalg.inv(pose_a)[:3, :4]
        projection_b = intrinsics_b @ np.linalg.inv(pose_b)[:3, :4]
        points_4d = cv2.triangulatePoints(projection_a, projection_b, points_a.T, points_b.T)
        depths = points_4d[3]
        valid = np.isfinite(points_4d).all(axis=0) & (np.abs(depths) > 1e-9)
        if int(np.count_nonzero(valid)) >= int(min_matches):
            points_3d = (points_4d[:3, valid] / depths[valid]).T
            errors: list[float] = []
            positive = np.ones(len(points_3d), dtype=bool)
            for projection, observed in ((projection_a, points_a[valid]), (projection_b, points_b[valid])):
                projected = np.hstack([points_3d, np.ones((len(points_3d), 1))]) @ projection.T
                positive &= np.isfinite(projected).all(axis=1) & (projected[:, 2] > 1e-9)
                behind = np.abs(projected[:, 2]) <= 1e-9
                projected[behind, 2] = 1e-9
                pixels = projected[:, :2] / projected[:, 2:3]
                errors.append(float(np.median(np.linalg.norm(pixels - observed, axis=1))))
            reprojection_error = float(np.mean(errors))
            positive_depth_fraction = float(np.count_nonzero(positive)) / len(points_a)
        else:
            return None

    epipolar_score = float(np.exp(-epipolar_error / max(float(epipolar_tau), 1e-6)))
    if reprojection_error is None:
        score = epipolar_score
    else:
        reprojection_score = float(np.exp(-reprojection_error / max(float(reprojection_tau), 1e-6)))
        score = 0.5 * (epipolar_score + reprojection_score)
    # A perfect reprojection behind the cameras is not physically valid closure.
    score *= positive_depth_fraction
    if not math.isfinite(score) or (reprojection_error is not None and not math.isfinite(reprojection_error)):
        return None
    return {
        "score": clamp01(score),
        "geometry_mode": "epipolar_triangulation",
        "epipolar_error_pixels": epipolar_error,
        "epipolar_score": clamp01(epipolar_score),
        "reprojection_error_pixels": reprojection_error,
        "match_count": int(len(points_a)),
        "baseline": baseline,
        "positive_depth_fraction": positive_depth_fraction,
        "implementation_revision": MEMORY_SAFETY_REVISION,
    }


def rendering_recovery_for_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    *,
    lightness_tau: float = DEFAULT_LIGHTNESS_TAU,
    chroma_tau: float = DEFAULT_CHROMA_TAU,
    gram_tau: float = DEFAULT_GRAM_TAU,
) -> dict[str, Any]:
    """Score whether a revisited viewpoint is lit and textured the way it first was.

    Lightness and chroma come from CIELAB spatial means, which discard high-frequency
    detail so the comparison reflects global illumination rather than content. Texture
    is a VGG Gram distance, the same descriptor the style metric uses elsewhere.
    """
    aligned = _resize_like(frame_b, frame_a)
    lab_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab_b = cv2.cvtColor(aligned, cv2.COLOR_RGB2LAB).astype(np.float64)
    # OpenCV packs 8-bit LAB as L in [0,255] and a/b offset by 128.
    lightness_delta = abs(float(np.mean(lab_a[..., 0]) - np.mean(lab_b[..., 0])) * 100.0 / 255.0)
    chroma_a = np.asarray([np.mean(lab_a[..., 1]), np.mean(lab_a[..., 2])]) - 128.0
    chroma_b = np.asarray([np.mean(lab_b[..., 1]), np.mean(lab_b[..., 2])]) - 128.0
    chroma_delta = float(np.linalg.norm(chroma_a - chroma_b))

    lighting_score = float(np.exp(-lightness_delta / max(float(lightness_tau), 1e-6)))
    chroma_score = float(np.exp(-chroma_delta / max(float(chroma_tau), 1e-6)))
    result: dict[str, Any] = {
        "lightness_delta": lightness_delta,
        "chroma_delta": chroma_delta,
        "lighting_score": clamp01(lighting_score),
        "chroma_score": clamp01(chroma_score),
    }
    try:
        gram_distance, _ = _gram_distance(frame_a, aligned)
        texture_score = float(np.exp(-float(gram_distance) / max(float(gram_tau), 1e-9)))
        result["gram_distance"] = float(gram_distance)
        result["texture_score"] = clamp01(texture_score)
        result["score"] = clamp01((lighting_score + chroma_score + texture_score) / 3.0)
    except (ImportError, ModuleNotFoundError):
        result["gram_distance"] = None
        result["texture_score"] = None
        result["score"] = clamp01(0.5 * (lighting_score + chroma_score))
    return result


def _fidelity_scores(frames_a: list[np.ndarray], frames_b: list[np.ndarray]) -> list[dict[str, float]]:
    """Blend DINOv2 cosine similarity with SSIM over every revisit pair."""
    aligned_b = [_resize_like(right, left) for left, right in zip(frames_a, frames_b)]
    features = dinov2_image_features([*frames_a, *aligned_b])
    left_features = features[: len(frames_a)]
    right_features = features[len(frames_a) :]

    scores: list[dict[str, float]] = []
    for index, (left, right) in enumerate(zip(frames_a, aligned_b)):
        cosine = float(np.dot(left_features[index], right_features[index]))
        semantic = clamp01((cosine + 1.0) / 2.0)
        structural = _ssim_rgb(left, right)
        scores.append(
            {
                "dino_similarity": semantic,
                "ssim": structural,
                "score": clamp01(0.7 * semantic + 0.3 * structural),
            }
        )
    return scores


def _segment_point_cloud(
    track: PoseTrack,
    segment: range,
    *,
    depth: np.ndarray,
    valid: np.ndarray,
    max_frames: int,
    max_points_per_frame: int,
) -> np.ndarray:
    """Back-project one trajectory segment's depth into a world-space point cloud."""
    indices = _select_segment_indices(segment.start, segment.stop, max_frames)
    clouds = [
        _frame_depth_points(
            np.asarray(depth[index]),
            np.asarray(valid[index]),
            track.poses_c2w[index],
            track.intrinsics,
            frame_index=index,
            max_points=max_points_per_frame,
        )
        for index in indices
        if index < len(depth) and index < len(valid)
    ]
    if not clouds:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(clouds, axis=0)


MEMORY_SCENE_CALCULATION_REVISION = "revisit_episode_signed_intrinsics_20260911"


def scene_memory_for_rollout(
    track: PoseTrack,
    structure: RevisitStructure,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score how much observed 3D geometry survives into the return leg.

    Unlike the long-horizon diagnostic, the segments come from the longest supported
    completed pose-revisit episode, not a ground-truth action boundary or the global
    farthest-from-start frame. A later departure cannot erase an earlier return.
    If the return gate fails and no supported episode exists, compare the two halves
    of the pose track and identify that fallback in the result.
    """
    runtime = dict(runtime or {})
    depth_path = track.sidecar_dir / "depth_metric.npy"
    mask_path = track.sidecar_dir / "depth_valid_mask.npz"
    if not depth_path.exists() or not mask_path.exists():
        raise FileNotFoundError(
            f"scene memory needs ViPE depth exports under {track.sidecar_dir}"
        )
    if track.intrinsics is None:
        raise FileNotFoundError(f"scene memory needs intrinsics under {track.sidecar_dir}")

    depth = np.load(depth_path, mmap_mode="r")
    valid = _load_depth_mask(mask_path)
    split_source = "longest_supported_revisit_episode"
    try:
        observation_range, revisit_range = observation_revisit_split(structure)
    except ValueError:
        if structure.departed and structure.returned:
            raise
        if structure.frame_count < 4:
            raise ValueError("scene memory needs at least four pose frames for a midpoint split")
        midpoint = structure.frame_count // 2
        observation_range = range(0, midpoint)
        revisit_range = range(midpoint, structure.frame_count)
        split_source = "temporal_midpoint_fallback"
    max_frames = int(runtime.get("max_frames_per_segment", 8))
    max_points = int(runtime.get("max_points_per_frame", 4096))
    observation = _segment_point_cloud(
        track,
        observation_range,
        depth=depth,
        valid=valid,
        max_frames=max_frames,
        max_points_per_frame=max_points,
    )
    revisit = _segment_point_cloud(
        track,
        revisit_range,
        depth=depth,
        valid=valid,
        max_frames=max_frames,
        max_points_per_frame=max_points,
    )
    if not len(observation) or not len(revisit):
        raise ValueError("one of the trajectory segments produced no valid depth points")

    scored = score_point_cloud_memory(observation, revisit, runtime=runtime)
    return {
        **scored,
        "details": {
            **scored["details"],
            "point_cloud_source": "vipe_depth_reconstruction",
            "split_source": split_source,
            "scene_calculation_revision": MEMORY_SCENE_CALCULATION_REVISION,
            "backprojection_intrinsics_policy": "preserve_signed_nonzero_focal",
            "scene_episode_first_index": observation_range.start,
            "scene_episode_turnaround_index": revisit_range.start,
            "scene_episode_revisit_index": revisit_range.stop - 1,
            "observation_frames": [observation_range.start, observation_range.stop],
            "revisit_frames": [revisit_range.start, revisit_range.stop],
        },
    }


def _video_frame_count(path: Path) -> int:
    """Read a video's frame count without decoding it."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count <= 0:
        raise RuntimeError(f"video reports no decodable frames: {path}")
    return count


_RESULT_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _cache_key(prediction_path: Path, runtime: dict[str, Any]) -> tuple[Any, ...]:
    """Keep entry-specific sidecars distinct even when videos share a target."""
    stat = prediction_path.stat()
    return (
        str(prediction_path.absolute()),
        stat.st_mtime_ns,
        stat.st_size,
        MEMORY_LOOP_SCORING_REVISION,
        json.dumps(runtime, sort_keys=True, default=str, allow_nan=False),
    )


def analyze_memory_rollout(
    prediction_path: Path,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect revisits in a rollout and score every closed-loop memory component.

    Cached per prediction so the five metrics that read this share a single pass over
    pose estimation, frame decoding, and feature extraction.
    """
    runtime = dict(runtime or {})
    prediction_path = Path(prediction_path)
    # Partial evaluations must not invoke unrequested feature backends.
    supported_components = {"scene", "fidelity", "geometric", "rendering", "half_life"}
    selected_components = runtime.get("analysis_components")
    if selected_components is not None and (
        not isinstance(selected_components, (list, tuple))
        or not set(selected_components) <= supported_components
    ):
        raise ValueError("analysis_components must be a list of supported memory components")
    components = supported_components if selected_components is None else set(selected_components)
    key = _cache_key(prediction_path, runtime)
    cached = _RESULT_CACHE.get(key)
    if cached is not None:
        return cached

    track: PoseTrack = resolve_pose_track(prediction_path, runtime)
    fps = float(runtime["fps"]) if "fps" in runtime else probe_video_fps(prediction_path, default=float("nan"))
    structure: RevisitStructure = find_revisit_pairs(
        track.poses_c2w,
        fps=fps,
        frame_indices=track.frame_indices,
        max_pair_distance=float(runtime.get("max_pair_distance", DEFAULT_MAX_PAIR_DISTANCE)),
        min_gap_seconds=float(runtime.get("min_gap_seconds", DEFAULT_MIN_GAP_SECONDS)),
        rotation_weight=float(runtime.get("rotation_weight", DEFAULT_ROTATION_WEIGHT)),
        min_departure_factor=float(runtime.get("min_departure_factor", DEFAULT_MIN_DEPARTURE_FACTOR)),
        min_excursion_factor=float(runtime.get("min_excursion_factor", DEFAULT_MIN_EXCURSION_FACTOR)),
    )

    result: dict[str, Any] = {
        "pose_source": track.source,
        "fps": fps,
        "structure": structure,
        "gate_passed": bool(structure.departed and structure.returned),
        "pair_source": "pose_revisit",
        "pairs": (),
        "fidelity": None,
        "geometric": None,
        "rendering": None,
        "half_life": None,
        "half_life_error": None,
        "scene": None,
        "scene_error": None,
        "fidelity_error": None,
        "geometric_error": None,
        "rendering_error": None,
    }
    # Components are isolated so one unavailable backend degrades a single metric
    # instead of taking the whole memory dimension down with it.
    if "scene" in components:
        try:
            result["scene"] = scene_memory_for_rollout(track, structure, runtime=runtime)
        except Exception as exc:
            raise_if_cuda_failure(exc)
            result["scene_error"] = str(exc)

    scoring_pairs = structure.pairs
    if not scoring_pairs:
        scoring_pairs = find_revisit_pairs(
            track.poses_c2w,
            fps=fps,
            frame_indices=track.frame_indices,
            min_gap_seconds=float(runtime.get("min_gap_seconds", DEFAULT_MIN_GAP_SECONDS)),
            rotation_weight=float(runtime.get("rotation_weight", DEFAULT_ROTATION_WEIGHT)),
            require_revisit=False,
        ).pairs
        result["pair_source"] = "nearest_pose_fallback"
    selected: tuple[RevisitPair, ...] = select_diverse_pairs(
        scoring_pairs,
        max_pairs=int(runtime.get("max_scored_pairs", DEFAULT_MAX_SCORED_PAIRS)),
    )
    if not selected:
        result["pair_error"] = "no frame pair satisfies the minimum time gap"
        _RESULT_CACHE[key] = result
        return result
    # Only the paired frames are decoded: a minute-long rollout holds gigabytes of
    # pixels if read whole, and at most a few dozen frames are ever scored.
    frame_total = _video_frame_count(prediction_path)
    usable: list[RevisitPair] = []
    wanted: list[int] = []
    for pair in selected:
        index_a = track.video_frame(pair.first_index)
        index_b = track.video_frame(pair.revisit_index)
        if index_a >= frame_total or index_b >= frame_total:
            continue
        usable.append(pair)
        wanted.extend([index_a, index_b])

    if not usable:
        result["pair_error"] = "no scoring pair maps onto a decoded frame"
        _RESULT_CACHE[key] = result
        return result

    unique_indices = sorted(set(wanted))
    decoded = dict(zip(unique_indices, read_video_frames_at(prediction_path, unique_indices)))
    frames_a = [decoded[wanted[position]] for position in range(0, len(wanted), 2)]
    frames_b = [decoded[wanted[position]] for position in range(1, len(wanted), 2)]

    result["pairs"] = tuple(usable)
    gaps = np.asarray([pair.gap_seconds for pair in usable])
    result["gap_support"] = {
        "pair_count": len(usable),
        "gap_min_seconds": float(np.min(gaps)),
        "gap_max_seconds": float(np.max(gaps)),
        "gap_span_seconds": float(np.ptp(gaps)),
        "unique_gap_count": int(len(np.unique(gaps))),
        "distinct_first_frame_count": len({pair.first_index for pair in usable}),
        "distinct_revisit_frame_count": len({pair.revisit_index for pair in usable}),
        "independent_event_count": None,
        "event_independence_verified": False,
    }

    fidelity_scores: list[dict[str, float]] | None = None
    if components & {"fidelity", "half_life"}:
        try:
            fidelity_scores = _fidelity_scores(frames_a, frames_b)
            result["fidelity"] = {
                "score": float(np.mean([item["score"] for item in fidelity_scores])),
                "per_pair": fidelity_scores,
            }
        except Exception as exc:
            raise_if_cuda_failure(exc)
            result["fidelity_error"] = str(exc)

    if "rendering" in components:
        try:
            rendering_scores = [
                rendering_recovery_for_pair(
                    left,
                    right,
                    lightness_tau=float(runtime.get("lightness_tau", DEFAULT_LIGHTNESS_TAU)),
                    chroma_tau=float(runtime.get("chroma_tau", DEFAULT_CHROMA_TAU)),
                    gram_tau=float(runtime.get("gram_tau", DEFAULT_GRAM_TAU)),
                )
                for left, right in zip(frames_a, frames_b)
            ]
            result["rendering"] = {
                "score": float(np.mean([item["score"] for item in rendering_scores])),
                "per_pair": rendering_scores,
            }
        except Exception as exc:
            raise_if_cuda_failure(exc)
            result["rendering_error"] = str(exc)

    if "geometric" in components:
        try:
            geometric_scores: list[dict[str, Any] | None] = []
            for pair, left, right in zip(usable, frames_a, frames_b):
                if track.intrinsics is None:
                    geometric_scores.append(None)
                    continue
                geometric_scores.append(
                    geometric_closure_for_pair(
                        left,
                        right,
                        track.poses_c2w[pair.first_index],
                        track.poses_c2w[pair.revisit_index],
                        track.intrinsics_at(pair.first_index),
                        track.intrinsics_at(pair.revisit_index),
                        min_matches=int(runtime.get("min_sift_matches", DEFAULT_MIN_SIFT_MATCHES)),
                        epipolar_tau=float(
                            runtime.get("epipolar_tau_pixels", DEFAULT_EPIPOLAR_TAU_PIXELS)
                        ),
                        reprojection_tau=float(
                            runtime.get("reprojection_tau_pixels", DEFAULT_REPROJECTION_TAU_PIXELS)
                        ),
                    )
                )
            valid_geometric = [item["score"] for item in geometric_scores if item is not None]
            result["geometric"] = {
                "score": float(np.mean(valid_geometric)) if valid_geometric else None,
                "scored_pair_count": len(valid_geometric),
                "per_pair": geometric_scores,
            }
        except Exception as exc:
            raise_if_cuda_failure(exc)
            result["geometric_error"] = str(exc)

    if "half_life" in components:
        if fidelity_scores is None:
            result["half_life_error"] = (
                f"revisit fidelity is unavailable, so its decay cannot be fitted: "
                f"{result['fidelity_error']}"
            )
        else:
            try:
                result["half_life"] = fit_exponential_half_life(
                    [pair.gap_seconds for pair in usable],
                    [item["score"] for item in fidelity_scores],
                    min_points=int(runtime.get("half_life_min_points", 5)),
                    min_r_squared=float(runtime.get("half_life_min_r_squared", 0.3)),
                )
            except ValueError as exc:
                result["half_life_error"] = str(exc)

    _RESULT_CACHE[key] = result
    return result


class _MemoryLoopMetric(Metric):
    """Shared plumbing for metrics read off one cached closed-loop rollout analysis."""

    requires_reference = False

    def __init__(
        self,
        *,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        auto_backend, supported = MEMORY_LOOP_BACKENDS[self.name]
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend=auto_backend,
            supported_backends=supported,
        )

    def _base_details(self, prediction: LoadedMedia) -> dict[str, Any]:
        return {
            "implementation_revision": (
                MEMORY_SAFETY_REVISION if self.name == "memory_return_gate"
                else MEMORY_LOOP_SCORING_REVISION
            ),
            "prediction_path": str(prediction.path),
            "metric_uses_vlm": False,
            "metric_uses_llm_as_judge": False,
        }

    def _emit(self, raw: float, backend: str, details: dict[str, Any]) -> MetricOutput:
        if not math.isfinite(float(raw)):
            details["measurement_status"] = "invalid_component_score"
            return MetricOutput(raw=None, normalized=None, backend=backend, details=details, error="non-finite memory component score")
        return MetricOutput(
            raw=float(raw),
            normalized=normalize_score(float(raw), self.normalization)
            if self.normalization
            else float(raw),
            backend=backend,
            details=details,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia | None,
    ) -> MetricOutput:
        del sample, reference
        backend = self._resolve_backend()
        details = self._base_details(prediction)
        try:
            analysis = analyze_memory_rollout(Path(prediction.path), runtime=self.runtime)
        except FileNotFoundError as exc:
            details["measurement_status"] = "sidecar_unavailable"
            return _skip(self.name, backend, str(exc), details)
        except Exception as exc:
            details["measurement_status"] = "unavailable_or_invalid"
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )
        details.update(analysis["structure"].as_details())
        details["pose_source"] = analysis["pose_source"]
        details["gap_support"] = analysis.get("gap_support")
        details["gate_passed"] = bool(analysis["gate_passed"])
        details["gate_failure_reason"] = None if analysis["gate_passed"] else _gate_reason(analysis)
        details["pair_source"] = analysis["pair_source"]
        details["measurement_status"] = "observed_pose_track"
        return self._score(analysis, backend=backend, details=details)

    def _score(self, analysis: dict[str, Any], *, backend: str, details: dict[str, Any]) -> MetricOutput:
        raise NotImplementedError


class MemoryReturnGateMetric(_MemoryLoopMetric):
    """Diagnose departure and return without blocking other memory metrics."""

    def __init__(self, backend: str, normalization: dict[str, Any], runtime: dict[str, Any] | None = None) -> None:
        super().__init__(
            metric_name="memory_return_gate",
            backend=backend,
            normalization=normalization,
            runtime=runtime,
        )

    def _score(self, analysis: dict[str, Any], *, backend: str, details: dict[str, Any]) -> MetricOutput:
        structure: RevisitStructure = analysis["structure"]
        details["gate_passed"] = bool(analysis["gate_passed"])
        return MetricOutput(
            raw=1.0 if analysis["gate_passed"] else 0.0,
            normalized=1.0 if analysis["gate_passed"] else 0.0,
            backend=backend,
            details=details,
            # The gate reports how often a model actually departed and returned; averaging it into
            # the memory dimension would double-count control ability as memory.
            eligibility_status="diagnostic",
            error=None if structure.returned else "rollout never revisited an earlier viewpoint",
        )


class MemorySceneF1Metric(_MemoryLoopMetric):
    """Score retained versus hallucinated 3D geometry across the return leg."""

    def __init__(self, backend: str, normalization: dict[str, Any], runtime: dict[str, Any] | None = None) -> None:
        super().__init__(
            metric_name="memory_scene_f1",
            backend=backend,
            normalization=normalization,
            runtime=runtime,
        )

    def _base_details(self, prediction: LoadedMedia) -> dict[str, Any]:
        # Stamp successes and unavailable measurements alike, so the existing
        # aggregate revision guard cannot silently mix pre-fix and corrected F1.
        return {
            **super()._base_details(prediction),
            "implementation_revision": f"{MEMORY_LOOP_SCORING_REVISION}:{MEMORY_SCENE_CALCULATION_REVISION}",
            "scene_calculation_revision": MEMORY_SCENE_CALCULATION_REVISION,
        }

    def _score(self, analysis: dict[str, Any], *, backend: str, details: dict[str, Any]) -> MetricOutput:
        if analysis["scene"] is None:
            return _skip(
                self.name,
                backend,
                str(analysis["scene_error"] or "scene memory could not be reconstructed"),
                details,
            )
        scene = analysis["scene"]
        details.update(dict(scene["details"]))
        details["measurement_scope"] = (
            "temporal_halves_point_cloud_overlap"
            if details.get("split_source") == "temporal_midpoint_fallback"
            else "outbound_return_point_cloud_overlap"
        )
        details["common_visibility_conditioned"] = False
        details["estimator_independence_verified"] = False
        details.update(
            {
                "retention": round(float(scene["retention"]), 6),
                "hallucination": round(float(scene["hallucination"]), 6),
                "precision": round(float(scene["precision"]), 6),
                "distance_threshold": round(float(scene["distance_threshold"]), 6),
            }
        )
        return self._emit(float(scene["scene_memory_f1"]), backend, details)


class MemoryRevisitFidelityMetric(_MemoryLoopMetric):
    """Score how faithfully a revisited viewpoint reproduces its first appearance."""

    def __init__(self, backend: str, normalization: dict[str, Any], runtime: dict[str, Any] | None = None) -> None:
        super().__init__(
            metric_name="memory_revisit_fidelity",
            backend=backend,
            normalization=normalization,
            runtime=runtime,
        )

    def _score(self, analysis: dict[str, Any], *, backend: str, details: dict[str, Any]) -> MetricOutput:
        if analysis["fidelity"] is None:
            return _skip(self.name, backend, _component_reason(analysis, "fidelity"), details)
        fidelity = analysis["fidelity"]
        details.update(
            {
                "scored_pair_count": len(analysis["pairs"]),
                "per_pair": [
                    {**pair.as_dict(), **score}
                    for pair, score in zip(analysis["pairs"], fidelity["per_pair"])
                ],
            }
        )
        return self._emit(fidelity["score"], backend, details)


class MemoryGeometricClosureMetric(_MemoryLoopMetric):
    """Score whether revisited frames obey the geometry their poses imply."""

    def __init__(self, backend: str, normalization: dict[str, Any], runtime: dict[str, Any] | None = None) -> None:
        super().__init__(
            metric_name="memory_geometric_closure",
            backend=backend,
            normalization=normalization,
            runtime=runtime,
        )

    def _score(self, analysis: dict[str, Any], *, backend: str, details: dict[str, Any]) -> MetricOutput:
        if analysis["geometric"] is None:
            return _skip(self.name, backend, _component_reason(analysis, "geometric"), details)
        geometric = analysis["geometric"]
        if geometric["score"] is None:
            return _skip(
                self.name,
                backend,
                "no revisit pair produced enough SIFT correspondences, or intrinsics are missing",
                details,
            )
        details.update(
            {
                "scored_pair_count": geometric["scored_pair_count"],
                "per_pair": [
                    {**pair.as_dict(), **(score or {"score": None})}
                    for pair, score in zip(analysis["pairs"], geometric["per_pair"])
                ],
            }
        )
        return self._emit(geometric["score"], backend, details)


class MemoryRenderingRecoveryMetric(_MemoryLoopMetric):
    """Score whether lighting, chroma, and texture recover on revisit."""

    def __init__(self, backend: str, normalization: dict[str, Any], runtime: dict[str, Any] | None = None) -> None:
        super().__init__(
            metric_name="memory_rendering_recovery",
            backend=backend,
            normalization=normalization,
            runtime=runtime,
        )

    def _score(self, analysis: dict[str, Any], *, backend: str, details: dict[str, Any]) -> MetricOutput:
        if analysis["rendering"] is None:
            return _skip(self.name, backend, _component_reason(analysis, "rendering"), details)
        rendering = analysis["rendering"]
        details.update(
            {
                "scored_pair_count": len(analysis["pairs"]),
                "per_pair": [
                    {**pair.as_dict(), **score}
                    for pair, score in zip(analysis["pairs"], rendering["per_pair"])
                ],
            }
        )
        return self._emit(rendering["score"], backend, details)


class MemoryHalfLifeMetric(_MemoryLoopMetric):
    """Exploratory decay fit, never a certified memory lifetime or headline score.

    Pose-selected neighboring pairs do not establish independent revisit events.
    Report their time support and retain only finite reliable fits diagnostically.
    """

    def __init__(self, backend: str, normalization: dict[str, Any], runtime: dict[str, Any] | None = None) -> None:
        super().__init__(
            metric_name="memory_half_life",
            backend=backend,
            normalization=normalization,
            runtime=runtime,
        )

    def _score(self, analysis: dict[str, Any], *, backend: str, details: dict[str, Any]) -> MetricOutput:
        details["metric_interpretation"] = "exploratory_decay_fit_not_validated_memory_lifetime"
        details["event_independence_verified"] = False
        details["gap_support"] = analysis.get("gap_support")
        if analysis["half_life"] is None:
            return _skip(
                self.name,
                backend,
                _component_reason(analysis, "half_life"),
                details,
            )
        fit = analysis["half_life"]
        quality_floor = float(self.runtime.get("half_life_min_r_squared", 0.3))
        if (
            not np.isfinite([fit["tau_seconds"], fit["half_life_seconds"], fit["r_squared"]]).all()
            or fit["tau_seconds"] <= 0.0
            or fit["half_life_seconds"] <= 0.0
            or not math.isfinite(quality_floor) or not 0 <= quality_floor <= 1
            or not quality_floor <= fit["r_squared"] <= 1
        ):
            return _skip(self.name, backend, "half-life fit is non-finite or unreliable", details)
        reference_seconds = float(
            self.runtime.get("half_life_reference_seconds", DEFAULT_HALF_LIFE_REFERENCE_SECONDS)
        )
        if not math.isfinite(reference_seconds) or reference_seconds <= 0.0:
            return _skip(self.name, backend, "half-life reference window must be finite and positive", details)
        details.update(
            {
                "half_life_seconds": round(float(fit["half_life_seconds"]), 4),
                "tau_seconds": round(float(fit["tau_seconds"]), 4),
                "fit_r_squared": round(float(fit["r_squared"]), 4),
                "fit_point_count": int(fit["point_count"]),
                "half_life_reference_seconds": reference_seconds,
            }
        )
        output = self._emit(
            clamp01(float(fit["half_life_seconds"]) / max(reference_seconds, 1e-6)),
            backend,
            details,
        )
        output.eligibility_status = "diagnostic"
        return output


def _gate_reason(analysis: dict[str, Any]) -> str:
    """Explain a diagnostic gate failure without controlling metric eligibility."""
    structure: RevisitStructure = analysis["structure"]
    if not structure.departed:
        return (
            "camera never departed its anchor viewpoint, so returning to it does not test memory"
        )
    if not structure.returned:
        return "rollout never revisited an earlier viewpoint"
    return str(analysis.get("pair_error") or "no scorable revisit pair")


def _component_reason(analysis: dict[str, Any], component: str) -> str:
    """Report the actual scoring prerequisite or backend failure, independently of the gate."""
    return str(analysis.get(f"{component}_error") or analysis.get("pair_error") or f"{component} could not be scored")


def build_memory_loop_metric(
    metric_name: str,
    *,
    backend: str,
    normalization: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> Metric:
    """Instantiate one closed-loop memory metric by name."""
    classes = {
        "memory_return_gate": MemoryReturnGateMetric,
        "memory_scene_f1": MemorySceneF1Metric,
        "memory_revisit_fidelity": MemoryRevisitFidelityMetric,
        "memory_geometric_closure": MemoryGeometricClosureMetric,
        "memory_rendering_recovery": MemoryRenderingRecoveryMetric,
        "memory_half_life": MemoryHalfLifeMetric,
    }
    return classes[metric_name](backend, normalization, runtime)


__all__ = [
    "MEMORY_LOOP_BACKENDS",
    "MEMORY_LOOP_METRICS",
    "MemoryGeometricClosureMetric",
    "MemoryHalfLifeMetric",
    "MemoryRenderingRecoveryMetric",
    "MemoryReturnGateMetric",
    "MemoryRevisitFidelityMetric",
    "MemorySceneF1Metric",
    "analyze_memory_rollout",
    "build_memory_loop_metric",
    "geometric_closure_for_pair",
    "rendering_recovery_for_pair",
    "scene_memory_for_rollout",
]
