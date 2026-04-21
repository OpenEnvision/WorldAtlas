from __future__ import annotations

from typing import Any

from worldarena.benchmark.annotations import derive_object_prompts, load_camera_matrices
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.official import OfficialMetric
from worldarena.benchmark.official_runtime import (
    compute_camera_error,
    compute_motion_accuracy,
    compute_object_detection,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


CAMERA_ERROR_BACKENDS = {"droid_slam"}
OBJECT_DETECTION_BACKENDS = {"grounding_dino"}
MOTION_ACCURACY_BACKENDS = {"sam2_sea_raft"}


class CameraErrorMetric(OfficialMetric):
    supported_backends = CAMERA_ERROR_BACKENDS
    auto_backend = "droid_slam"

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("camera_error", backend, normalization)

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del reference

        backend = self._resolve_backend()
        details = self._base_details(prediction)
        if not sample.annotation_path or not sample.has_pose:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error="camera poses are unavailable for this sample",
            )
        try:
            cameras_gt = load_camera_matrices(
                sample.annotation_path,
                target_frames=len(prediction.frames),
            )
            if cameras_gt is None:
                raise FileNotFoundError("camera poses are unavailable for this sample")
            result = compute_camera_error(
                prediction.path,
                cameras_gt=cameras_gt,
                scale=1.0,
                normalization=self.normalization,
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


class ObjectDetectionMetric(OfficialMetric):
    supported_backends = OBJECT_DETECTION_BACKENDS
    auto_backend = "grounding_dino"

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("object_detection", backend, normalization)

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
                error="object prompts are unavailable for this sample",
            )
        prompt_string = ", ".join(["scene", *prompt_candidates])
        eligibility_status = "official" if prompt_source == "annotation_explicit" else "weak_supervision"
        if eligibility_status != "official":
            details["supervision_warning"] = (
                "object prompts were derived from caption text instead of explicit object annotations"
            )
        details["prompt_source"] = prompt_source
        details["prompt_candidates"] = prompt_candidates
        try:
            result = compute_object_detection(
                prediction.path,
                prompt_string=prompt_string,
                normalization=self.normalization,
            )
            details.update(result["details"])
            return MetricOutput(
                raw=result["raw"],
                normalized=result["normalized"],
                backend=str(result["backend"]),
                details=details,
                eligibility_status=eligibility_status,
            )
        except Exception as exc:  # pragma: no cover - heavy runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status=eligibility_status,
                error=str(exc),
            )


class MotionAccuracyMetric(OfficialMetric):
    supported_backends = MOTION_ACCURACY_BACKENDS
    auto_backend = "sam2_sea_raft"

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("motion_accuracy", backend, normalization)

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
                generate_type="i2v" if sample.conditioning_strategy == "first_frame" else "t2v",
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
