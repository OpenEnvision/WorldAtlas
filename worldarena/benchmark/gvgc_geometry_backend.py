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
    if method == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2, "sift"
    if method not in {"orb", "sift"}:
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


def _ransac_method() -> int:
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

    detector, norm_type, detector_name = _create_detector(detector_method, max_features)
    gray_frames = [_to_gray(frame, max_width=max_width) for frame in frames]

    pair_results: list[dict[str, Any]] = []
    epipolar_errors: list[float] = []
    inlier_rates: list[float] = []
    coverage_values: list[float] = []
    failed_pairs = 0

    for left_index, right_index in _frame_pairs(len(gray_frames), pair_stride=pair_stride, max_pairs=max_pairs):
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
        if len(matches) < min_matches:
            pair_details["error"] = "not enough feature matches"
            pair_results.append(pair_details)
            failed_pairs += 1
            continue

        points_left = np.asarray(match_payload["points_left"], dtype=np.float32)
        points_right = np.asarray(match_payload["points_right"], dtype=np.float32)
        fundamental, mask = cv2.findFundamentalMat(
            points_left,
            points_right,
            method=_ransac_method(),
            ransacReprojThreshold=ransac_threshold,
            confidence=confidence,
            maxIters=max_iters,
        )
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
        return {
            "backend": DEFAULT_BACKEND,
            "raw": {},
            "details": {
                "frame_count": len(frames),
                "pair_count": len(pair_results),
                "failed_pair_count": failed_pairs,
                "feature_detector": detector_name,
                "pairs": pair_results,
            },
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
            "ransac_reproj_threshold": ransac_threshold,
            "pairs": pair_results,
        },
        "error": None,
    }


__all__ = ["DEFAULT_BACKEND", "compute_gvgc_geometry"]
