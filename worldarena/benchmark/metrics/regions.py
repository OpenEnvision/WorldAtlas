"""Seen/novel region metric implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, clamp01, cosine_similarity, resolve_metric_backend
from worldarena.benchmark.metrics.legacy import (
    _dinov2_image_features,
    _normalize_or_identity,
    _protocol_details,
    _resize_like,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.common.dyn_masks import load_dyn_mask_sequence
from worldarena.common.video_io import read_video_frame


SEEN_REGION_BACKENDS = {"flow_dino"}
OBJECT_PERMANENCE_BACKENDS = {"mask_dino"}
_FLOW_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 5,
    "poly_n": 7,
    "poly_sigma": 1.5,
    "flags": 0,
}
_OPEN_KERNEL = np.ones((3, 3), dtype=np.uint8)
_DILATE_KERNEL = np.ones((5, 5), dtype=np.uint8)
_MIN_MASK_PIXELS = 64


def _resolve_static_mask_source(sample: BenchmarkSample) -> Path | None:
    """Locate dynamic-mask archives used to ignore moving regions in flow masks."""
    if sample.mask_path:
        candidate = Path(sample.mask_path)
        if candidate.exists():
            return candidate
    if sample.annotation_path:
        archive = Path(sample.annotation_path) / "dyn_masks.npz"
        if archive.exists():
            return archive
    return None


def _load_ignore_masks(sample: BenchmarkSample, reference: LoadedMedia) -> np.ndarray | None:
    """Load per-frame ignore masks aligned to sampled reference frames."""
    source = _resolve_static_mask_source(sample)
    if source is None or not reference.frames:
        return None
    try:
        return load_dyn_mask_sequence(
            source,
            frame_indices=reference.sampled_frame_indices,
            target_shape=reference.frames[0].shape[:2],
        )
    except Exception:
        return None


def _frame_coverage(mask: np.ndarray) -> float:
    """Return the fraction of pixels covered by a boolean region mask."""
    return float(np.mean(mask.astype(np.float32)))


def _masked_psnr(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
    """Compute mask-weighted PSNR mapped into [0, 1] for region scoring."""
    valid = bool(np.any(mask))
    if not valid:
        return None
    left_rgb = left.astype(np.float32)
    right_rgb = right.astype(np.float32)
    mask_rgb = mask.astype(np.float32)[..., None]
    mse = float(np.sum(((left_rgb - right_rgb) ** 2) * mask_rgb) / max(np.sum(mask_rgb), 1.0))
    if mse <= 1e-12:
        return 1.0
    psnr = 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)
    return clamp01((float(psnr) - 10.0) / 30.0)


def _masked_crop(frame: np.ndarray, mask: np.ndarray, *, margin: int = 4) -> np.ndarray | None:
    """Crop a tight bounding box around a mask and fill exterior pixels with mean color."""
    coords = np.argwhere(mask)
    if coords.shape[0] < _MIN_MASK_PIXELS:
        return None
    y0 = max(int(coords[:, 0].min()) - margin, 0)
    y1 = min(int(coords[:, 0].max()) + margin + 1, frame.shape[0])
    x0 = max(int(coords[:, 1].min()) - margin, 0)
    x1 = min(int(coords[:, 1].max()) + margin + 1, frame.shape[1])
    crop = frame[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    if not np.any(crop_mask):
        return None
    fill = np.round(crop[crop_mask].mean(axis=0)).astype(np.uint8)
    crop[~crop_mask] = fill
    return crop


def _masked_dino_similarity(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
    """Compare DINO features on masked crops when enough valid pixels exist."""
    left_crop = _masked_crop(left, mask)
    right_crop = _masked_crop(right, mask)
    if left_crop is None or right_crop is None:
        return None
    features = _dinov2_image_features([left_crop, right_crop])
    return clamp01((cosine_similarity(features[0], features[1]) + 1.0) / 2.0)


def _flow(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """Estimate dense Farneback optical flow between two RGB frames."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, **_FLOW_PARAMS)


def _seen_novel_masks(
    source_frame: np.ndarray,
    target_frames: list[np.ndarray],
    *,
    ignore_masks: np.ndarray | None = None,
    fb_threshold: float = 1.5,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split each target frame into forward-warped seen and complementary novel regions."""
    height, width = source_frame.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    seen_masks = [np.ones((height, width), dtype=bool)]
    novel_masks = [np.zeros((height, width), dtype=bool)]
    for index, target in enumerate(target_frames[1:], start=1):
        forward = _flow(source_frame, target)
        backward = _flow(target, source_frame)
        src_x = grid_x + backward[..., 0]
        src_y = grid_y + backward[..., 1]
        inside = (src_x >= 0.0) & (src_x <= width - 1.0) & (src_y >= 0.0) & (src_y <= height - 1.0)
        sampled_forward = cv2.remap(
            forward,
            src_x.astype(np.float32),
            src_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        fb_error = np.linalg.norm(sampled_forward + backward, axis=-1)
        seen = inside & (fb_error <= float(fb_threshold))
        seen = cv2.morphologyEx(seen.astype(np.uint8), cv2.MORPH_OPEN, _OPEN_KERNEL).astype(bool)
        if ignore_masks is not None and index < len(ignore_masks):
            seen &= ~ignore_masks[index]
        novel = ~cv2.dilate(seen.astype(np.uint8), _DILATE_KERNEL).astype(bool)
        if ignore_masks is not None and index < len(ignore_masks):
            novel &= ~ignore_masks[index]
        seen_masks.append(seen)
        novel_masks.append(novel)
    return seen_masks, novel_masks


def _composite_mask_score(
    prediction_frame: np.ndarray,
    reference_frame: np.ndarray,
    mask: np.ndarray,
) -> tuple[float | None, dict[str, float | None]]:
    """Blend masked PSNR and DINO similarity into one region fidelity score."""
    prediction_aligned = _resize_like(prediction_frame, reference_frame)
    psnr_component = _masked_psnr(prediction_aligned, reference_frame, mask)
    dino_component = _masked_dino_similarity(prediction_aligned, reference_frame, mask)
    components = [value for value in (psnr_component, dino_component) if value is not None]
    if not components:
        return None, {"psnr_component": psnr_component, "dino_component": dino_component}
    return (
        float(np.mean(components)),
        {"psnr_component": psnr_component, "dino_component": dino_component},
    )


def _weighted_average(values: list[float], weights: list[float]) -> float | None:
    """Average frame scores with coverage weights, falling back to a plain mean."""
    if not values:
        return None
    weights_array = np.asarray(weights, dtype=np.float32)
    if weights_array.size != len(values) or float(weights_array.sum()) <= 0.0:
        return float(np.mean(values))
    return float(np.average(np.asarray(values, dtype=np.float32), weights=weights_array))


class _RegionMetric(Metric):
    """Shared flow-based seen/novel mask extraction for region metrics."""

    backend_name: str

    def _resolve_backend(self, supported_backends: set[str], auto_backend: str) -> str:
        """Validate backend selection for flow-and-DINO region scoring."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend=auto_backend,
            supported_backends=supported_backends,
        )

    def _region_masks(
        self,
        sample: BenchmarkSample,
        reference: LoadedMedia,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build seen and novel masks from the first reference frame via optical flow."""
        if reference.modality != "video" or len(reference.frames) < 2:
            raise NotImplementedError("region metrics require at least two reference video frames")
        ignore_masks = _load_ignore_masks(sample, reference)
        return _seen_novel_masks(reference.frames[0], reference.frames, ignore_masks=ignore_masks)


class SeenRegionPreservationMetric(_RegionMetric):
    """Score how well previously seen reference regions are preserved in the prediction."""

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        """Register seen-region preservation with flow-DINO backend validation."""
        super().__init__("seen_region_preservation", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Average masked fidelity over forward-warped seen regions in later frames."""
        backend = self._resolve_backend(SEEN_REGION_BACKENDS, "flow_dino")
        details = _protocol_details(prediction)
        try:
            seen_masks, _ = self._region_masks(sample, reference)
            scores: list[float] = []
            weights: list[float] = []
            frame_details: list[dict[str, float | None]] = []
            for index, (prediction_frame, reference_frame, mask) in enumerate(
                zip(prediction.frames[1:], reference.frames[1:], seen_masks[1:]),
                start=1,
            ):
                coverage = _frame_coverage(mask)
                score, components = _composite_mask_score(prediction_frame, reference_frame, mask)
                frame_details.append(
                    {
                        "frame_index": float(index),
                        "coverage": coverage,
                        **components,
                        "score": score,
                    }
                )
                if score is None or int(mask.sum()) < _MIN_MASK_PIXELS:
                    continue
                scores.append(score)
                weights.append(max(coverage, 1e-6))
            raw = _weighted_average(scores, weights)
            if raw is None:
                raise ValueError("no valid seen-region frame scores were available")
            details["frame_scores"] = frame_details
            return MetricOutput(
                raw=raw,
                normalized=_normalize_or_identity(raw, self.normalization),
                backend=backend,
                details=details,
            )
        except NotImplementedError as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="not_applicable",
                error=str(exc),
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


def _resolve_dynamic_mask_source(sample: BenchmarkSample) -> Path:
    """Find segmentation mask archives required for object-permanence scoring."""
    if sample.mask_path:
        candidate = Path(sample.mask_path)
        if candidate.exists():
            return candidate
    if sample.annotation_path:
        archive = Path(sample.annotation_path) / "dyn_masks.npz"
        if archive.exists():
            return archive
    raise FileNotFoundError("segmentation masks are unavailable for this sample")


class ObjectPermanenceMetric(Metric):
    """Track whether masked objects stay as stable as in the reference over time."""

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        """Register object permanence with mask-DINO backend validation."""
        super().__init__("object_permanence", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Compare prediction object appearance drift against reference using masked DINO features."""
        backend = resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="mask_dino",
            supported_backends=OBJECT_PERMANENCE_BACKENDS,
        )
        details = _protocol_details(prediction)
        if reference.modality != "video" or not reference.frames:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="not_applicable",
                error="object permanence requires a video reference",
            )
        try:
            mask_source = _resolve_dynamic_mask_source(sample)
            masks = load_dyn_mask_sequence(
                mask_source,
                frame_indices=reference.sampled_frame_indices,
                target_shape=reference.frames[0].shape[:2],
            )
            if masks.shape[0] != len(reference.frames):
                raise RuntimeError("mask sequence is not aligned with sampled reference frames")
            anchor_index = max(int(reference.sampled_frame_indices[0]) - 1, 0)
            anchor_frame = read_video_frame(Path(sample.reference_path), frame_index=anchor_index)
            anchor_mask = load_dyn_mask_sequence(
                mask_source,
                frame_indices=[anchor_index],
                target_shape=anchor_frame.shape[:2],
            )[0]
            if int(anchor_mask.sum()) < _MIN_MASK_PIXELS:
                anchor_mask = masks[0]
                anchor_frame = reference.frames[0]

            anchor_crop = _masked_crop(anchor_frame, anchor_mask)
            if anchor_crop is None:
                raise ValueError("anchor dynamic mask is too small for permanence scoring")

            scores: list[float] = []
            weights: list[float] = []
            gt_curve: list[float] = []
            pred_curve: list[float] = []
            frame_details: list[dict[str, float | None]] = []
            for index, (prediction_frame, reference_frame, mask) in enumerate(
                zip(prediction.frames, reference.frames, masks),
                start=0,
            ):
                if int(mask.sum()) < _MIN_MASK_PIXELS:
                    frame_details.append(
                        {
                            "frame_index": float(index),
                            "coverage": _frame_coverage(mask),
                            "gt_similarity": None,
                            "prediction_similarity": None,
                            "score": None,
                        }
                    )
                    continue
                reference_crop = _masked_crop(reference_frame, mask)
                prediction_aligned = _resize_like(prediction_frame, reference_frame)
                prediction_crop = _masked_crop(prediction_aligned, mask)
                if reference_crop is None or prediction_crop is None:
                    frame_details.append(
                        {
                            "frame_index": float(index),
                            "coverage": _frame_coverage(mask),
                            "gt_similarity": None,
                            "prediction_similarity": None,
                            "score": None,
                        }
                    )
                    continue
                features = _dinov2_image_features([anchor_crop, reference_crop, prediction_crop])
                gt_similarity = clamp01((cosine_similarity(features[0], features[1]) + 1.0) / 2.0)
                prediction_similarity = clamp01((cosine_similarity(features[0], features[2]) + 1.0) / 2.0)
                score = clamp01(1.0 - abs(prediction_similarity - gt_similarity))
                coverage = _frame_coverage(mask)
                gt_curve.append(gt_similarity)
                pred_curve.append(prediction_similarity)
                scores.append(score)
                weights.append(max(coverage, 1e-6))
                frame_details.append(
                    {
                        "frame_index": float(index),
                        "coverage": coverage,
                        "gt_similarity": gt_similarity,
                        "prediction_similarity": prediction_similarity,
                        "score": score,
                    }
                )
            raw = _weighted_average(scores, weights)
            if raw is None:
                raise ValueError("no valid masked object permanence scores were available")
            details.update(
                {
                    "anchor_frame_index": anchor_index,
                    "frame_scores": frame_details,
                    "gt_similarity_curve": gt_curve,
                    "prediction_similarity_curve": pred_curve,
                }
            )
            return MetricOutput(
                raw=raw,
                normalized=_normalize_or_identity(raw, self.normalization),
                backend=backend,
                details=details,
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


__all__ = [
    "ObjectPermanenceMetric",
    "SeenRegionPreservationMetric",
]
