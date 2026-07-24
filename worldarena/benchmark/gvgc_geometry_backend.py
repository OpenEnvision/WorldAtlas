"""Geometry backend for GVGc-related reconstruction metrics."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


DEFAULT_BACKEND = "opencv_epipolar"


def _to_gray(frame: np.ndarray, max_width: int) -> np.ndarray:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame.astype(np.uint8, copy=False)
    height, width = gray.shape[:2]
    if width > max_width:
        scale = float(max_width) / float(width)
        gray = cv2.resize(gray, (max_width, max(int(round(height * scale)), 1)))
    return gray


def _create_detector(method: str, max_features: int):
    method = method.lower()
    if method == "shi_tomasi_lk":
        return None, None, "shi_tomasi_lk"
    if method == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2, "sift"
    if method not in {"orb", "sift", "shi_tomasi_lk"}:
        raise ValueError(f"unsupported feature detector: {method}")
    return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING, "orb"


def _detect_and_match(
    left: np.ndarray,
    right: np.ndarray,
    *,
    detector: Any,
    norm_type: int,
    ratio: float,
) -> dict[str, Any]:
    keypoints_left, descriptors_left = detector.detectAndCompute(left, None)
    keypoints_right, descriptors_right = detector.detectAndCompute(right, None)
    keypoints_left = keypoints_left or []
    keypoints_right = keypoints_right or []
    if descriptors_left is None or descriptors_right is None:
        return {
            "keypoints_left": len(keypoints_left),
            "keypoints_right": len(keypoints_right),
            "matches": [],
        }

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)
    matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            matches.append(best)
    return {
        "keypoints_left": len(keypoints_left),
        "keypoints_right": len(keypoints_right),
        "matches": matches,
        "points_left": np.asarray([keypoints_left[item.queryIdx].pt for item in matches], dtype=np.float32),
        "points_right": np.asarray([keypoints_right[item.trainIdx].pt for item in matches], dtype=np.float32),
    }


def _detect_and_track_lk(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_features: int,
    quality_level: float,
    min_distance: float,
    block_size: int,
    window_size: int,
    max_level: int,
    max_iterations: int,
    epsilon: float,
    forward_backward_max_error: float,
) -> dict[str, Any]:
    """Track deterministic Shi-Tomasi corners with pyramidal Lucas-Kanade flow."""
    try:
        corners = cv2.goodFeaturesToTrack(
            left,
            maxCorners=max(max_features, 1),
            qualityLevel=max(quality_level, 1e-9),
            minDistance=max(min_distance, 0.0),
            blockSize=max(block_size, 2),
            useHarrisDetector=False,
        )
    except cv2.error as exc:
        return {
            "keypoints_left": 0,
            "keypoints_right": 0,
            "matches": [],
            "tracking_error": f"corner detection error: {exc}",
        }
    if corners is None or not len(corners):
        return {
            "keypoints_left": 0,
            "keypoints_right": 0,
            "matches": [],
        }

    corner_count = int(len(corners))
    lk_parameters = {
        "winSize": (max(window_size, 3), max(window_size, 3)),
        "maxLevel": max(max_level, 0),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max(max_iterations, 1),
            max(epsilon, 1e-9),
        ),
    }
    try:
        tracked, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            left,
            right,
            corners,
            None,
            **lk_parameters,
        )
    except cv2.error as exc:
        return {
            "keypoints_left": corner_count,
            "keypoints_right": corner_count,
            "matches": [],
            "tracking_error": f"forward optical-flow error: {exc}",
        }
    if tracked is None or forward_status is None:
        return {
            "keypoints_left": corner_count,
            "keypoints_right": corner_count,
            "matches": [],
        }

    points_left = corners.reshape(-1, 2).astype(np.float32, copy=False)
    points_right = tracked.reshape(-1, 2).astype(np.float32, copy=False)
    valid = forward_status.reshape(-1).astype(bool)
    valid &= np.isfinite(points_left).all(axis=1)
    valid &= np.isfinite(points_right).all(axis=1)
    forward_backward_errors: np.ndarray | None = None
    if forward_backward_max_error > 0.0 and valid.any():
        try:
            returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                right,
                left,
                tracked,
                None,
                **lk_parameters,
            )
        except cv2.error as exc:
            return {
                "keypoints_left": corner_count,
                "keypoints_right": corner_count,
                "matches": [],
                "tracking_error": f"backward optical-flow error: {exc}",
            }
        if returned is None or backward_status is None:
            valid[:] = False
        else:
            returned_points = returned.reshape(-1, 2).astype(np.float32, copy=False)
            backward_valid = backward_status.reshape(-1).astype(bool)
            backward_valid &= np.isfinite(returned_points).all(axis=1)
            forward_backward_errors = np.linalg.norm(returned_points - points_left, axis=1)
            valid &= backward_valid
            valid &= forward_backward_errors <= forward_backward_max_error

    points_left = points_left[valid]
    points_right = points_right[valid]
    tracked_count = int(points_left.shape[0])
    payload: dict[str, Any] = {
        # Keep both counts at the number of detected source corners so coverage
        # remains inliers / detected features, rather than inliers / valid tracks.
        "keypoints_left": corner_count,
        "keypoints_right": corner_count,
        "matches": [None] * tracked_count,
        "points_left": points_left,
        "points_right": points_right,
        "tracked_points": tracked_count,
    }
    if forward_backward_errors is not None and valid.any():
        payload["median_forward_backward_error"] = float(np.median(forward_backward_errors[valid]))
    return payload


def _ransac_method() -> int:
    """Ransac method -> int."""
    return int(getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC))


def _sampson_errors(points_left: np.ndarray, points_right: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    ones = np.ones((points_left.shape[0], 1), dtype=np.float64)
    left_h = np.concatenate([points_left.astype(np.float64), ones], axis=1)
    right_h = np.concatenate([points_right.astype(np.float64), ones], axis=1)
    f_left = left_h @ fundamental.T
    ft_right = right_h @ fundamental
    numerator = np.sum(right_h * f_left, axis=1) ** 2
    denominator = f_left[:, 0] ** 2 + f_left[:, 1] ** 2 + ft_right[:, 0] ** 2 + ft_right[:, 1] ** 2
    denominator = np.maximum(denominator, 1e-12)
    return np.sqrt(numerator / denominator)


def _frame_pairs(frame_count: int, *, pair_stride: int, max_pairs: int) -> list[tuple[int, int]]:
    stride = max(int(pair_stride), 1)
    pairs = [(index, index + stride) for index in range(0, frame_count - stride, stride)]
    if max_pairs > 0 and len(pairs) > max_pairs:
        indices = np.linspace(0, len(pairs) - 1, num=max_pairs)
        return [pairs[int(round(index))] for index in indices]
    return pairs


def compute_gvgc_geometry(frames: list[np.ndarray], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime = dict(runtime or {})
    opencv_num_threads = max(int(runtime.get("opencv_num_threads", 1)), 1)
    opencv_rng_seed = int(runtime.get("opencv_rng_seed", 17))
    cv2.setNumThreads(opencv_num_threads)
    cv2.setRNGSeed(opencv_rng_seed)
    if len(frames) < 2:
        return {
            "backend": DEFAULT_BACKEND,
            "raw": {},
            "details": {"frame_count": len(frames), "pair_count": 0},
            "error": "GVGC geometry metrics require at least two frames",
        }

    max_features = int(runtime.get("max_features", 2000))
    detector_method = str(runtime.get("feature_detector", "sift")).lower()
    ratio = float(runtime.get("ratio_test", 0.75))
    min_matches = int(runtime.get("min_matches", 12))
    max_width = int(runtime.get("max_analysis_width", 640))
    ransac_threshold = float(runtime.get("ransac_reproj_threshold", 1.0))
    confidence = float(runtime.get("ransac_confidence", 0.999))
    max_iters = int(runtime.get("ransac_max_iters", 10000))
    pair_stride = int(runtime.get("pair_stride", 1))
    max_pairs = int(runtime.get("max_pairs", 8))
    fallback_enabled = bool(runtime.get("fallback_enabled", True))
    tracking_fallback_enabled = bool(runtime.get("tracking_fallback_enabled", True))
    tracking_quality_level = float(runtime.get("tracking_quality_level", 0.001))
    tracking_min_distance = float(runtime.get("tracking_min_distance", 3.0))
    tracking_block_size = int(runtime.get("tracking_block_size", 3))
    tracking_window_size = int(runtime.get("tracking_window_size", 21))
    tracking_max_level = int(runtime.get("tracking_max_level", 3))
    tracking_max_iterations = int(runtime.get("tracking_max_iterations", 30))
    tracking_epsilon = float(runtime.get("tracking_epsilon", 0.01))
    tracking_forward_backward_max_error = float(
        runtime.get("tracking_forward_backward_max_error", 1.5)
    )

    detector, norm_type, detector_name = _create_detector(detector_method, max_features)
    gray_frames = [_to_gray(frame, max_width=max_width) for frame in frames]

    pair_results: list[dict[str, Any]] = []
    epipolar_errors: list[float] = []
    inlier_rates: list[float] = []
    coverage_values: list[float] = []
    failed_pairs = 0

    for left_index, right_index in _frame_pairs(len(gray_frames), pair_stride=pair_stride, max_pairs=max_pairs):
        if detector_name == "shi_tomasi_lk":
            match_payload = _detect_and_track_lk(
                gray_frames[left_index],
                gray_frames[right_index],
                max_features=max_features,
                quality_level=tracking_quality_level,
                min_distance=tracking_min_distance,
                block_size=tracking_block_size,
                window_size=tracking_window_size,
                max_level=tracking_max_level,
                max_iterations=tracking_max_iterations,
                epsilon=tracking_epsilon,
                forward_backward_max_error=tracking_forward_backward_max_error,
            )
        else:
            match_payload = _detect_and_match(
                gray_frames[left_index],
                gray_frames[right_index],
                detector=detector,
                norm_type=norm_type,
                ratio=ratio,
            )
        matches = list(match_payload.get("matches") or [])
        keypoint_floor = max(
            min(int(match_payload["keypoints_left"]), int(match_payload["keypoints_right"])),
            1,
        )
        pair_details: dict[str, Any] = {
            "left_index": left_index,
            "right_index": right_index,
            "keypoints_left": int(match_payload["keypoints_left"]),
            "keypoints_right": int(match_payload["keypoints_right"]),
            "good_matches": len(matches),
        }
        if "tracked_points" in match_payload:
            pair_details["tracked_points"] = int(match_payload["tracked_points"])
        if "median_forward_backward_error" in match_payload:
            pair_details["median_forward_backward_error"] = round(
                float(match_payload["median_forward_backward_error"]),
                6,
            )
        if len(matches) < min_matches:
            pair_details["error"] = str(
                match_payload.get("tracking_error") or "not enough feature matches"
            )
            pair_results.append(pair_details)
            failed_pairs += 1
            continue

        points_left = np.asarray(match_payload["points_left"], dtype=np.float32)
        points_right = np.asarray(match_payload["points_right"], dtype=np.float32)
        try:
            fundamental, mask = cv2.findFundamentalMat(
                points_left,
                points_right,
                method=_ransac_method(),
                ransacReprojThreshold=ransac_threshold,
                confidence=confidence,
                maxIters=max_iters,
            )
        except cv2.error as exc:
            pair_details["error"] = f"fundamental matrix estimation error: {exc}"
            pair_results.append(pair_details)
            failed_pairs += 1
            continue
        if fundamental is None or mask is None:
            pair_details["error"] = "fundamental matrix estimation failed"
            pair_results.append(pair_details)
            failed_pairs += 1
            continue
        if fundamental.shape != (3, 3):
            fundamental = fundamental[:3, :3]

        inlier_mask = mask.reshape(-1).astype(bool)
        inlier_count = int(inlier_mask.sum())
        if inlier_count < 8:
            pair_details["error"] = "not enough epipolar inliers"
            pair_details["inliers"] = inlier_count
            pair_results.append(pair_details)
            failed_pairs += 1
            continue

        errors = _sampson_errors(points_left[inlier_mask], points_right[inlier_mask], fundamental)
        median_error = float(np.median(errors))
        mean_error = float(np.mean(errors))
        inlier_rate = inlier_count / float(len(matches))
        coverage = min(inlier_count / float(keypoint_floor), 1.0)
        epipolar_errors.append(median_error)
        inlier_rates.append(inlier_rate)
        coverage_values.append(coverage)
        pair_details.update(
            {
                "inliers": inlier_count,
                "inlier_rate": round(inlier_rate, 6),
                "feature_match_coverage": round(coverage, 6),
                "median_epipolar_error": round(median_error, 6),
                "mean_epipolar_error": round(mean_error, 6),
            }
        )
        pair_results.append(pair_details)

    if not epipolar_errors:
        primary_details = {
            "frame_count": len(frames),
            "pair_count": len(pair_results),
            "failed_pair_count": failed_pairs,
            "feature_detector": detector_name,
            "max_features": max_features,
            "ratio_test": ratio,
            "min_matches": min_matches,
            "ransac_reproj_threshold": ransac_threshold,
            "opencv_num_threads": opencv_num_threads,
            "opencv_rng_seed": opencv_rng_seed,
            "pairs": pair_results,
        }
        if detector_name == "shi_tomasi_lk":
            primary_details["tracking_configuration"] = {
                "quality_level": tracking_quality_level,
                "min_distance": tracking_min_distance,
                "block_size": tracking_block_size,
                "window_size": tracking_window_size,
                "max_level": tracking_max_level,
                "max_iterations": tracking_max_iterations,
                "epsilon": tracking_epsilon,
                "forward_backward_max_error": tracking_forward_backward_max_error,
            }
        if fallback_enabled:
            fallback_detector = str(runtime.get("fallback_feature_detector", detector_method)).lower()
            fallback_ratio = float(runtime.get("fallback_ratio_test", max(ratio, 0.85)))
            fallback_min_matches = int(runtime.get("fallback_min_matches", 8))
            fallback_changes_settings = (
                fallback_detector != detector_method
                or fallback_ratio != ratio
                or fallback_min_matches != min_matches
            )
            if fallback_changes_settings:
                fallback_runtime = dict(runtime)
                fallback_runtime.update(
                    {
                        "feature_detector": fallback_detector,
                        "ratio_test": fallback_ratio,
                        "min_matches": fallback_min_matches,
                        "fallback_enabled": False,
                        "tracking_fallback_enabled": False,
                    }
                )
                fallback_result = compute_gvgc_geometry(frames, fallback_runtime)
                fallback_details = dict(fallback_result.get("details") or {})
                fallback_details.update(
                    {
                        "fallback_used": fallback_result.get("error") is None,
                        "fallback_method": "relaxed_descriptor_matching",
                        "fallback_reason": "no valid epipolar frame pairs with primary settings",
                        "fallback_configuration": {
                            "feature_detector": fallback_detector,
                            "ratio_test": fallback_ratio,
                            "min_matches": fallback_min_matches,
                        },
                        "primary_attempt": primary_details,
                    }
                )
                fallback_result["details"] = fallback_details
                if fallback_result.get("error") is None:
                    return fallback_result
                fallback_details.pop("primary_attempt", None)
                primary_details["fallback_attempt"] = fallback_details

        if tracking_fallback_enabled and detector_name != "shi_tomasi_lk":
            tracking_min_matches = int(runtime.get("tracking_min_matches", 8))
            tracking_runtime = dict(runtime)
            tracking_runtime.update(
                {
                    "feature_detector": "shi_tomasi_lk",
                    "min_matches": tracking_min_matches,
                    "fallback_enabled": False,
                    "tracking_fallback_enabled": False,
                }
            )
            tracking_result = compute_gvgc_geometry(frames, tracking_runtime)
            tracking_details = dict(tracking_result.get("details") or {})
            tracking_details.update(
                {
                    "fallback_used": tracking_result.get("error") is None,
                    "fallback_method": "shi_tomasi_lk_tracking",
                    "fallback_reason": (
                        "no valid epipolar frame pairs with descriptor matching settings"
                    ),
                    "fallback_configuration": {
                        "feature_detector": "shi_tomasi_lk",
                        "min_matches": tracking_min_matches,
                        "quality_level": tracking_quality_level,
                        "min_distance": tracking_min_distance,
                        "block_size": tracking_block_size,
                        "window_size": tracking_window_size,
                        "max_level": tracking_max_level,
                        "max_iterations": tracking_max_iterations,
                        "epsilon": tracking_epsilon,
                        "forward_backward_max_error": tracking_forward_backward_max_error,
                    },
                    "primary_attempt": primary_details,
                }
            )
            tracking_result["details"] = tracking_details
            if tracking_result.get("error") is None:
                return tracking_result
            tracking_details.pop("primary_attempt", None)
            primary_details["tracking_fallback_attempt"] = tracking_details

        return {
            "backend": DEFAULT_BACKEND,
            "raw": {},
            "details": primary_details,
            "error": "no valid epipolar frame pairs",
        }

    raw = {
        "epipolar_error": float(np.median(epipolar_errors)),
        "epipolar_inlier_rate": float(np.mean(inlier_rates)),
        "feature_match_coverage": float(np.mean(coverage_values)),
    }
    return {
        "backend": DEFAULT_BACKEND,
        "raw": raw,
        "details": {
            "frame_count": len(frames),
            "pair_count": len(pair_results),
            "valid_pair_count": len(epipolar_errors),
            "failed_pair_count": failed_pairs,
            "feature_detector": detector_name,
            "max_features": max_features,
            "ratio_test": ratio,
            "min_matches": min_matches,
            "ransac_reproj_threshold": ransac_threshold,
            "opencv_num_threads": opencv_num_threads,
            "opencv_rng_seed": opencv_rng_seed,
            "fallback_used": False,
            **(
                {
                    "tracking_configuration": {
                        "quality_level": tracking_quality_level,
                        "min_distance": tracking_min_distance,
                        "block_size": tracking_block_size,
                        "window_size": tracking_window_size,
                        "max_level": tracking_max_level,
                        "max_iterations": tracking_max_iterations,
                        "epsilon": tracking_epsilon,
                        "forward_backward_max_error": tracking_forward_backward_max_error,
                    }
                }
                if detector_name == "shi_tomasi_lk"
                else {}
            ),
            "pairs": pair_results,
        },
        "error": None,
    }


__all__ = ["DEFAULT_BACKEND", "compute_gvgc_geometry"]
