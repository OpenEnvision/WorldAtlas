"""Conservative CPU source-image/first-frame matching for legacy audits.

Visual evidence identifies image content, not the historical generation prompt.
Candidates from this module are never silently made evaluation bindings.
"""
from __future__ import annotations

import cv2
import numpy as np


def features(image: np.ndarray, max_side: int = 640):
    height, width = image.shape[:2]
    scale = min(1., max_side / max(height, width))
    small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    points, descriptors = cv2.SIFT_create(nfeatures=1000).detectAndCompute(gray, None)
    coordinates = np.asarray([p.pt for p in points], dtype=np.float32).reshape(-1, 2)
    return coordinates, descriptors, (width, height), (small.shape[1], small.shape[0])


def compare(source, prediction) -> dict:
    source_points, source_desc, _, source_size = source
    video_points, video_desc, _, video_size = prediction
    empty = dict(inliers=0, matches=0, inlier_ratio=0., source_coverage=0., median_error=None)
    if source_desc is None or video_desc is None or min(len(source_desc), len(video_desc)) < 4:
        return empty
    pairs = cv2.BFMatcher().knnMatch(source_desc, video_desc, k=2)
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .7 * b.distance]
    if len(good) < 4:
        return {**empty, 'matches': len(good)}
    src = np.float32([source_points[m.queryIdx] for m in good])
    dst = np.float32([video_points[m.trainIdx] for m in good])
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3., maxIters=2000, confidence=.995)
    if homography is None or mask is None or not np.isfinite(homography).all():
        return {**empty, 'matches': len(good)}
    valid = mask.ravel().astype(bool)
    if valid.sum() < 4:
        return {**empty, 'matches': len(good)}
    projected = cv2.perspectiveTransform(src[:, None, :], homography)[:, 0]
    hull = cv2.convexHull(src[valid])
    return dict(inliers=int(valid.sum()), matches=len(good), inlier_ratio=float(valid.mean()),
                source_coverage=float(cv2.contourArea(hull) / (source_size[0] * source_size[1])),
                median_error=float(np.median(np.linalg.norm(projected[valid] - dst[valid], axis=1))),
                homography=homography.tolist(), source_feature_size=list(source_size), video_feature_size=list(video_size))


def confident_match(best: dict, runner_up: dict | None) -> bool:
    second = 0 if runner_up is None else runner_up['inliers']
    return (best['inliers'] >= 30 and best['inlier_ratio'] >= .55
            and best['source_coverage'] >= .08 and best['median_error'] is not None
            and best['median_error'] <= 2. and best['inliers'] >= 3 * max(1, second)
            and best['inliers'] - second >= 20)
