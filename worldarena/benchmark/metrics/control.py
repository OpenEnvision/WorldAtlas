"""Metric implementations for camera control and trajectory accuracy."""

from __future__ import annotations

from typing import Any

from worldarena.benchmark.annotations import derive_object_prompts, load_camera_matrices
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.official import OfficialMetric
from worldarena.benchmark.official_runtime import (
    compute_camera_error,
    compute_camera_error_megasam,
    compute_camera_error_megasam_native_intended,
    compute_motion_accuracy,
    compute_object_detection,
    materialize_frame_paths,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices_for_sample


CAMERA_ERROR_BACKENDS = {"droid_slam", "megasam"}
OBJECT_DETECTION_BACKENDS = {"grounding_dino"}
MOTION_ACCURACY_BACKENDS = {"sam2_sea_raft"}


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
            frame_paths = materialize_frame_paths(prediction.path)
            camera_gt_policy = str(self.runtime.get("camera_gt_policy", "")).strip().lower()
            if camera_gt_policy in {"native_intended", "native_intent", "intended"}:
                if backend != "megasam":
                    raise ValueError("native_intended camera_error currently requires the megasam backend")
                result = compute_camera_error_megasam_native_intended(
                    prediction.path,
                    sample=sample,
                    normalization=self.normalization,
                    frame_paths=frame_paths,
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
        except Exception as exc:  # pragma: no cover - heavy runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


class ObjectDetectionMetric(OfficialMetric):
    """Verify explicitly annotated objects appear via Grounding DINO detections."""

    supported_backends = OBJECT_DETECTION_BACKENDS
    auto_backend = "grounding_dino"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Attach Grounding DINO detection runtime overrides."""
        super().__init__("object_detection", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Run object detection only when annotations supply explicit object prompts."""
        del reference

        backend = self._resolve_backend()
        if not prompt_candidates:
            details["prompt_source"] = prompt_source
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="not_applicable",
                error="object_detection requires explicit object annotations",
            )
        if prompt_source != "annotation_explicit":
            details["prompt_source"] = prompt_source
            details["prompt_candidates"] = prompt_candidates
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="not_applicable",
                error="caption-derived object prompts are not scored",
            )
        prompt_string = ", ".join(["scene", *prompt_candidates])
        details["prompt_source"] = prompt_source
        details["prompt_candidates"] = prompt_candidates
        try:
            result = compute_object_detection(
                prediction.path,
                prompt_string=prompt_string,
                normalization=self.normalization,
                runtime=self.runtime,
            )
            details.update(result["details"])
            return MetricOutput(
                raw=result["raw"],
                normalized=result["normalized"],
                backend=str(result["backend"]),
                details=details,
                eligibility_status="official",
            )
        except Exception as exc:  # pragma: no cover - heavy runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="official",
                error=str(exc),
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
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error="segmentation masks are unavailable for this sample",
            )
        try:
            result = compute_motion_accuracy(
                prediction.path,
                mask_dir=sample.mask_path,
                normalization=self.normalization,
                generate_type=(
                    "i2v"
                    if sample.conditioning_strategy
                    in {"first_frame", "reference_image", "reference_video", "external_video"}
                    else "t2v"
                ),
                runtime=self.runtime,
            )
            details.update(result["details"])
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
    "ObjectDetectionMetric",
]
