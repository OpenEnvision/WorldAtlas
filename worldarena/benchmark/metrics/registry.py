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
from worldarena.benchmark.metrics.depth_collision import DepthCollisionMetric
from worldarena.benchmark.metrics.alignment import (
    AnchorConsistencyMetric,
    PromptAlignmentMetric,
)
from worldarena.benchmark.metrics.base import Metric
from worldarena.benchmark.metrics.legacy import (
    BackgroundConsistencyMetric,
    DepthAccuracyMetric,
    PsnrMetric,
    SemanticAlignmentMetric,
    SsimMetric,
    SubjectConsistencyMetric,
)
from worldarena.benchmark.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark.metrics.met3r import Met3RConsistencyMetric
from worldarena.benchmark.metrics.physics import PhysicsMetric, is_physics_metric
from worldarena.benchmark.metrics.physics_judge import PhysicsJudgeMetric
from worldarena.benchmark.metrics.pmf import PhysicalMotionFidelityMetric
from worldarena.benchmark.metrics.quality import PerceptualQualityMetric
from worldarena.benchmark.metrics.reconstruction import ReconstructionConsistencyMetric
from worldarena.benchmark.metrics.regions import (
    NovelRegionFidelityMetric,
    ObjectPermanenceMetric,
    SeenRegionPreservationMetric,
)
from worldarena.benchmark.metrics.style import StyleConsistencyMetric
from worldarena.benchmark.metrics.temporal import TemporalCalibrationMetric
from worldarena.benchmark.metrics.long_sequence import LongSequenceMetric
from worldarena.benchmark.metrics.memory import MemoryRevisitConsistencyMetric


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
    "physical_motion_fidelity": "pmf",
    "S_recon": "reconstruction_consistency",
    "s_recon": "reconstruction_consistency",
    "score_reconstruction": "reconstruction_consistency",
}
TEMPORAL_CALIBRATION_METRICS = {
    "physical_fps_error",
    "physical_fps_pct_error",
    "physical_fps_intra_video_cv",
}
LONG_SEQUENCE_METRICS = {
    "long_sequence_subject_consistency",
    "long_sequence_background_consistency",
    "long_sequence_motion_smoothness",
    "long_sequence_dynamic_degree",
    "long_sequence_aesthetic_quality",
    "long_sequence_imaging_quality",
}
PHYSICS_JUDGE_METRICS = {
    "physics_judge_overall",
    "physics_judge_phenomenon",
    "physics_judge_causality",
    "physics_judge_material",
    "physics_judge_temporal",
    "physics_judge_prompt",
}


def canonical_metric_name(metric_name: str) -> str:
    return METRIC_ALIASES.get(metric_name, metric_name)


def build_metrics(metric_names: list[str], config: BenchmarkConfig) -> list[Metric]:
    metrics: list[Metric] = []
    seen: set[str] = set()
    shared_official_runtime = dict(config.metric_runtime.get("official_source", {}))
    shared_temporal_runtime = dict(config.metric_runtime.get("temporal_calibration", {}))
    shared_long_sequence_runtime = dict(config.metric_runtime.get("long_sequence", {}))
    shared_physics_judge_runtime = dict(config.metric_runtime.get("physics_judge", {}))
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
        elif metric_name == "semantic_alignment":
            metrics.append(SemanticAlignmentMetric(backend=backend, normalization=normalization))
        elif metric_name == "style_consistency":
            metrics.append(StyleConsistencyMetric(backend=backend, normalization=normalization))
        elif metric_name == "perceptual_quality":
            metrics.append(PerceptualQualityMetric(backend=backend, normalization=normalization))
        elif metric_name == "subject_consistency":
            metrics.append(SubjectConsistencyMetric(backend=backend, normalization=normalization))
        elif metric_name == "background_consistency":
            metrics.append(BackgroundConsistencyMetric(backend=backend, normalization=normalization))
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
        elif metric_name == "seen_region_preservation":
            metrics.append(
                SeenRegionPreservationMetric(
                    backend=backend,
                    normalization=normalization,
                )
            )
        elif metric_name == "novel_region_fidelity":
            metrics.append(
                NovelRegionFidelityMetric(
                    backend=backend,
                    normalization=normalization,
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
        elif metric_name == "object_permanence":
            metrics.append(
                ObjectPermanenceMetric(
                    backend=backend,
                    normalization=normalization,
                )
            )
        elif metric_name == "pmf":
            metrics.append(
                PhysicalMotionFidelityMetric(
                    backend=backend,
                    normalization=normalization,
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
        elif metric_name == "depth_accuracy":
            metrics.append(
                DepthAccuracyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "depth_collision":
            metrics.append(
                DepthCollisionMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "psnr":
            metrics.append(PsnrMetric(backend=backend, normalization=normalization))
        elif metric_name == "ssim":
            metrics.append(SsimMetric(backend=backend, normalization=normalization))
        elif metric_name == "anchor_consistency":
            metrics.append(AnchorConsistencyMetric(backend=backend, normalization=normalization))
        elif metric_name == "met3r_consistency":
            metrics.append(
                Met3RConsistencyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "reconstruction_consistency":
            metrics.append(
                ReconstructionConsistencyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name in TEMPORAL_CALIBRATION_METRICS:
            metrics.append(
                TemporalCalibrationMetric(
                    metric_name=metric_name,
                    backend=backend,
                    normalization=normalization,
                    runtime={**shared_temporal_runtime, **runtime},
                )
            )
        elif metric_name in LONG_SEQUENCE_METRICS:
            metrics.append(
                LongSequenceMetric(
                    metric_name=metric_name,
                    backend=backend,
                    normalization=normalization,
                    runtime={**shared_long_sequence_runtime, **runtime},
                )
            )
        elif metric_name == "memory_revisit_consistency":
            metrics.append(
                MemoryRevisitConsistencyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name in PHYSICS_JUDGE_METRICS:
            metrics.append(
                PhysicsJudgeMetric(
                    metric_name=metric_name,
                    backend=backend,
                    normalization=normalization,
                    runtime={**shared_physics_judge_runtime, **runtime},
                )
            )
        elif is_physics_metric(metric_name):
            metrics.append(PhysicsMetric(metric_name=metric_name, backend=backend, normalization=normalization))
        else:
            raise KeyError(f"unsupported metric: {raw_metric_name}")
    return metrics
