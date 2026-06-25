from __future__ import annotations

from typing import Any

from worldarena.benchmark.annotations import derive_object_prompts, load_camera_matrices
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.official import OfficialMetric
from worldarena.benchmark.official_runtime import (
    compute_camera_error,
    compute_motion_accuracy,
    compute_object_detection,
    materialize_frame_paths,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices_for_sample


CAMERA_ERROR_BACKENDS = {"droid_slam"}
OBJECT_DETECTION_BACKENDS = {"grounding_dino"}
MOTION_ACCURACY_BACKENDS = {"sam2_sea_raft"}


class CameraErrorMetric(OfficialMetric):
    supported_backends = CAMERA_ERROR_BACKENDS
    auto_backend = "droid_slam"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("camera_error", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del reference

        backend = self._resolve_backend()
        details = self._base_details(prediction)
        try:
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
    supported_backends = OBJECT_DETECTION_BACKENDS
    auto_backend = "grounding_dino"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("object_detection", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del reference

        backend = self._resolve_backend()
        details = self._base_details(prediction)
        prompt_candidates, prompt_source = derive_object_prompts(sample.annotation_path)
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
    supported_backends = MOTION_ACCURACY_BACKENDS
    auto_backend = "sam2_sea_raft"

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("motion_accuracy", backend, normalization)
        self.runtime = dict(runtime or {})

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
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
