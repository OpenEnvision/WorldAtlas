"""Metric implementations for camera control and trajectory accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldatlas.benchmark.annotations import load_camera_matrices
from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.benchmark.metrics.official import OfficialMetric
from worldatlas.benchmark.official_runtime import (
    compute_camera_error,
    compute_camera_error_megasam,
    compute_camera_error_megasam_native_intended,
    compute_motion_accuracy,
    materialize_frame_paths,
)
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput
from worldatlas.benchmark.synthetic_camera import synthetic_camera_matrices_for_sample
from worldatlas.common.progress import log_exception


CAMERA_ERROR_BACKENDS = {"droid_slam", "megasam"}
MOTION_ACCURACY_BACKENDS = {"sam2_sea_raft"}


def _manually_emptied_mask(sample: BenchmarkSample) -> bool:
    """Recognize an explicit deletion of all instances, never missing mask files."""
    if not sample.mask_path or sample.has_mask or sample.mask_count != 0:
        return False
    directory = Path(sample.mask_path)
    try:
        meta = json.loads((directory / "meta.json").read_text())
        relative = str(meta.get("relative_image_path") or "")
        if not relative.startswith("dynamic/"):
            relative = "dynamic/" + relative
        return bool(
            sample.suite == "image_dynamic" and relative == sample.relative_path
            and meta.get("num_saved_masks") == 0 and meta.get("saved_masks") == []
            and (meta.get("last_manual_deletion") or meta.get("manual_deletions"))
            and not any(p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                        for p in directory.iterdir())
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


class CameraErrorMetric(OfficialMetric):
    """Compare estimated camera trajectory against annotation or synthetic ground truth."""

    supported_backends = CAMERA_ERROR_BACKENDS
    auto_backend = "megasam"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Attach camera-error runtime options such as GT policy and SLAM settings."""
        super().__init__("camera_error", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Score camera pose error using annotated, synthetic, or native-intended GT."""
        del reference

        backend = self._resolve_backend()
        details = self._base_details(prediction)
        try:
            camera_gt_policy = str(self.runtime.get("camera_gt_policy", "")).strip().lower()
            if camera_gt_policy in {"native_intended", "native_intent", "intended"}:
                if backend != "megasam":
                    raise ValueError("native_intended camera_error currently requires the megasam backend")
                result = compute_camera_error_megasam_native_intended(
                    prediction.path,
                    sample=sample,
                    normalization=self.normalization,
                    runtime=self.runtime,
                )
                details.update(result["details"])
                details["camera_pose_source"] = "native_intended"
                return MetricOutput(
                    raw=result["raw"],
                    normalized=result["normalized"],
                    backend=str(result["backend"]),
                    details=details,
                )

            frame_paths = materialize_frame_paths(prediction.path)
            cameras_gt = None
            camera_scale = 1.0
            camera_pose_source = None
            if sample.annotation_path and sample.has_pose:
                cameras_gt = load_camera_matrices(
                    sample.annotation_path,
                    target_frames=len(frame_paths),
                )
                if cameras_gt is not None:
                    camera_pose_source = "annotation"
            if cameras_gt is None:
                synthetic = synthetic_camera_matrices_for_sample(
                    sample,
                    target_frames=len(frame_paths),
                    runtime=self.runtime,
                )
                if synthetic is not None:
                    cameras_gt = synthetic.matrices
                    camera_scale = synthetic.scale
                    camera_pose_source = synthetic.source
                    details["camera_path"] = list(synthetic.camera_path)
            if cameras_gt is None:
                raise FileNotFoundError("camera poses are unavailable for this sample")
            if backend == "megasam":
                result = compute_camera_error_megasam(
                    prediction.path,
                    cameras_gt=cameras_gt,
                    scale=camera_scale,
                    normalization=self.normalization,
                    frame_paths=frame_paths,
                    runtime=self.runtime,
                )
            else:
                result = compute_camera_error(
                    prediction.path,
                    cameras_gt=cameras_gt,
                    scale=camera_scale,
                    normalization=self.normalization,
                    frame_paths=frame_paths,
                    runtime=self.runtime,
                )
            details.update(result["details"])
            if camera_pose_source is not None:
                details["camera_pose_source"] = camera_pose_source
            return MetricOutput(
                raw=result["raw"],
                normalized=result["normalized"],
                backend=str(result["backend"]),
                details=details,
            )
        except MetricLoadError as exc:
            log_exception(
                "metric",
                exc,
                sample_id=sample.sample_id,
                metric=self.name,
                status="fail",
            )
            raise
        except Exception as exc:  # pragma: no cover - heavy runtime failures
            log_exception(
                "metric",
                exc,
                sample_id=sample.sample_id,
                metric=self.name,
                status="fail",
            )
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=f"{type(exc).__name__}: {exc}",
            )

class MotionAccuracyMetric(OfficialMetric):
    """Measure masked object motion fidelity with SAM2 and SEA-RAFT."""

    supported_backends = MOTION_ACCURACY_BACKENDS
    auto_backend = "sam2_sea_raft"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Attach SAM2/SEA-RAFT motion-accuracy runtime overrides."""
        super().__init__("motion_accuracy", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Score segmentation-mask motion accuracy when per-frame masks are available."""
        del reference

        backend = self._resolve_backend()
        details = self._base_details(prediction)
        if not sample.mask_path or not sample.has_mask:
            if _manually_emptied_mask(sample):
                reason = "mask annotation contains no foreground instances after manual deletion"
                details.update(mask_dir=sample.mask_path, not_applicable_reason=reason)
                details['implementation_revision'] = 'aligned_mask_contrast_v2'
                return MetricOutput(
                    raw=None, normalized=None, backend=backend, details=details,
                    eligibility_status="not_applicable", error=reason,
                )
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error="segmentation masks are unavailable for this sample",
            )
        try:
            from worldatlas.benchmark.mask_provenance import validate_mask_metadata
            details['implementation_revision'] = 'aligned_mask_contrast_v2'
            validate_mask_metadata(Path(sample.mask_path), sample.relative_path)
            geometry = (sample.prediction_binding or {}).get("conditioning_geometry")
            if not geometry:
                raise ValueError("Motion accuracy requires verified source-to-video conditioning_geometry; resize cannot be assumed")
            from worldatlas.benchmark.motion_mask_support import conditioning_mask_support
            support = conditioning_mask_support(sample.mask_path, source_path=sample.path,
                relative_path=sample.relative_path, geometry=geometry)
            details['conditioning_mask_support'] = support
            if support['not_applicable_reason']:
                details.update(conditioning_geometry=geometry,
                               not_applicable_reason=support['not_applicable_reason'])
                return MetricOutput(raw=None, normalized=None, backend=backend, details=details,
                                    eligibility_status='not_applicable', error=support['not_applicable_reason'])
            result = compute_motion_accuracy(
                prediction.path,
                mask_dir=sample.mask_path,
                normalization=self.normalization,
                source_path=sample.path,
                relative_path=sample.relative_path,
                conditioning_geometry=geometry,
                generate_type=(
                    "i2v"
                    if sample.conditioning_strategy
                    in {"first_frame", "reference_image", "reference_video", "external_video"}
                    else "t2v"
                ),
                runtime=self.runtime,
            )
            details.update(result["details"])
            if result.get("eligibility_status") == "not_applicable":
                return MetricOutput(
                    raw=None,
                    normalized=None,
                    backend=str(result["backend"]),
                    details=details,
                    eligibility_status="not_applicable",
                    error=result.get("error") or details.get("not_applicable_reason"),
                )
            return MetricOutput(
                raw=result["raw"],
                normalized=result["normalized"],
                backend=str(result["backend"]),
                details=details,
            )
        except Exception as exc:  # pragma: no cover - heavy runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


__all__ = [
    "CameraErrorMetric",
    "MotionAccuracyMetric",
]
