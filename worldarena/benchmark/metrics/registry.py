from __future__ import annotations

from worldarena.benchmark.config import BenchmarkConfig
from worldarena.benchmark.metrics.consistency import (
    OpticalFlowAepeMetric,
    ReprojectionErrorMetric,
)
from worldarena.benchmark.metrics.control import (
    CameraErrorMetric,
    MotionAccuracyMetric,
    ObjectDetectionMetric,
)
from worldarena.benchmark.metrics.alignment import (
    AnchorConsistencyMetric,
    PromptAlignmentMetric,
    ReferenceAppearanceAlignmentMetric,
)
from worldarena.benchmark.metrics.base import Metric
from worldarena.benchmark.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark.metrics.physics import PhysicsMetric, is_physics_metric
from worldarena.benchmark.metrics.quality import PerceptualQualityMetric
from worldarena.benchmark.metrics.style import StyleConsistencyMetric
from worldarena.benchmark.metrics.temporal import TemporalCalibrationMetric


METRIC_ALIASES: dict[str, str] = {
    "scene_alignment": "prompt_alignment",
    "dynamic_alignment": "prompt_alignment",
    "subjective_quality": "perceptual_quality",
    "rollout_quality": "anchor_consistency",
    "camera_control": "camera_error",
    "object_control": "object_detection",
    "content_alignment": "prompt_alignment",
    "clip_score": "prompt_alignment",
    "3d_consistency": "reprojection_error",
    "photometric_consistency": "optical_flow_aepe",
    "gram_matrix": "style_consistency",
    "optical_flow": "motion_magnitude",
}
TEMPORAL_CALIBRATION_METRICS = {
    "physical_fps_error",
    "physical_fps_pct_error",
    "physical_fps_intra_video_cv",
}


def canonical_metric_name(metric_name: str) -> str:
    return METRIC_ALIASES.get(metric_name, metric_name)


def build_metrics(metric_names: list[str], config: BenchmarkConfig) -> list[Metric]:
    metrics: list[Metric] = []
    seen: set[str] = set()
    shared_official_runtime = dict(config.metric_runtime.get("official_source", {}))
    shared_temporal_runtime = dict(config.metric_runtime.get("temporal_calibration", {}))
    for raw_metric_name in metric_names:
        metric_name = canonical_metric_name(raw_metric_name)
        if metric_name in seen:
            continue
        seen.add(metric_name)
        backend = config.metric_backends.get(metric_name, config.metric_backends.get(raw_metric_name, "auto"))
        normalization = config.normalization.get(metric_name, config.normalization.get(raw_metric_name, {}))
        runtime = dict(config.metric_runtime.get(metric_name, config.metric_runtime.get(raw_metric_name, {})))
        official_runtime = {**shared_official_runtime, **runtime}
        if metric_name == "prompt_alignment":
            metrics.append(PromptAlignmentMetric(backend=backend, normalization=normalization))
        elif metric_name == "reference_appearance_alignment":
            metrics.append(ReferenceAppearanceAlignmentMetric(backend=backend, normalization=normalization))
        elif metric_name == "style_consistency":
            metrics.append(StyleConsistencyMetric(backend=backend, normalization=normalization))
        elif metric_name == "perceptual_quality":
            metrics.append(PerceptualQualityMetric(backend=backend, normalization=normalization))
        elif metric_name == "motion_magnitude":
            metrics.append(
                MotionMagnitudeMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=official_runtime,
                )
            )
        elif metric_name == "motion_smoothness":
            metrics.append(
                MotionSmoothnessMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=official_runtime,
                )
            )
        elif metric_name == "camera_error":
            metrics.append(
                CameraErrorMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=official_runtime,
                )
            )
        elif metric_name == "object_detection":
            metrics.append(
                ObjectDetectionMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=official_runtime,
                )
            )
        elif metric_name == "reprojection_error":
            metrics.append(
                ReprojectionErrorMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=official_runtime,
                )
            )
        elif metric_name == "optical_flow_aepe":
            metrics.append(
                OpticalFlowAepeMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=official_runtime,
                )
            )
        elif metric_name == "motion_accuracy":
            metrics.append(
                MotionAccuracyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=official_runtime,
                )
            )
        elif metric_name == "anchor_consistency":
            metrics.append(AnchorConsistencyMetric(backend=backend, normalization=normalization))
        elif metric_name in TEMPORAL_CALIBRATION_METRICS:
            metrics.append(
                TemporalCalibrationMetric(
                    metric_name=metric_name,
                    backend=backend,
                    normalization=normalization,
                    runtime={**shared_temporal_runtime, **runtime},
                )
            )
        elif is_physics_metric(metric_name):
            metrics.append(PhysicsMetric(metric_name=metric_name, backend=backend, normalization=normalization))
        else:
            raise KeyError(f"unsupported metric: {raw_metric_name}")
    return metrics
