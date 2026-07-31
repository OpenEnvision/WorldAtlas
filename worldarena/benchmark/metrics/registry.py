"""Metric registry: build and configure metric instances from benchmark config."""

from __future__ import annotations

import json

from worldarena.benchmark.config import BenchmarkConfig
from worldarena.benchmark.metrics.appearance import (
    BrightnessConsistencyMetric,
    BrightnessDistributionConsistencyMetric,
    ColorTemperatureConstraintMetric,
    ColorTemperatureConsistencyMetric,
    SharpnessRetentionMetric,
    SharpnessVectorRetentionMetric,
)
from worldarena.benchmark.metrics.background_consistency import BackgroundConsistencyMetric
from worldarena.benchmark.metrics.segment_continuity import SegmentContinuityMetric
from worldarena.benchmark.metrics.consistency import (
    GVGC_GEOMETRY_METRICS,
    GvgcGeometryMetric,
    OpticalFlowAepeMetric,
    ReprojectionErrorMetric,
)
from worldarena.benchmark.metrics.control import (
    CameraErrorMetric,
    MotionAccuracyMetric,
)
from worldarena.benchmark.metrics.depth_collision import DepthCollisionMetric
from worldarena.benchmark.metrics.alignment import PromptAlignmentMetric
from worldarena.benchmark.metrics.base import Metric
from worldarena.benchmark.metrics.legacy import (
    DepthAccuracyMetric,
    SemanticAlignmentMetric,
)
from worldarena.benchmark.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark.metrics.met3r import Met3RConsistencyMetric
from worldarena.benchmark.metrics.physics import PhysicsMetric, is_physics_metric
from worldarena.benchmark.metrics.physics_judge import PhysicsJudgeMetric
from worldarena.benchmark.metrics.pmf import PhysicalMotionFidelityMetric
from worldarena.benchmark.metrics.quality import HPSv3NormMetric, ImageQualityMetric, PerceptualQualityMetric
from worldarena.benchmark.metrics.reconstruction import (
    RECONSTRUCTION_CONSISTENCY_METRICS,
    ReconstructionConsistencyEvaluator,
    ReconstructionConsistencyMetric,
)
from worldarena.benchmark.metrics.realtime import REALTIME_METRICS, build_realtime_metric
from worldarena.benchmark.metrics.regions import (
    ObjectPermanenceMetric,
    SeenRegionPreservationMetric,
)
from worldarena.benchmark.metrics.style import StyleConsistencyMetric
from worldarena.benchmark.metrics.temporal import TemporalCalibrationMetric
from worldarena.benchmark.metrics.temporal_flickering import TemporalFlickeringMetric
from worldarena.benchmark.metrics.long_sequence import LongSequenceMetric
from worldarena.benchmark.metrics.memory import (
    EntityReappearanceConsistencyMetric,
    MemoryRevisitConsistencyMetric,
    MemorySymmetryMetric,
    RevisitReturnGateMetric,
)
from worldarena.benchmark.metrics.long_horizon_diagnostics import (
    LONG_HORIZON_ACTION_METRICS,
    LONG_HORIZON_MEMORY_METRICS,
    LONG_HORIZON_VISUAL_METRICS,
    KeystrokeActionMetric,
    SceneMemoryF1Metric,
    VisualDriftMetric,
)


METRIC_ALIASES: dict[str, str] = {
    "scene_alignment": "prompt_alignment",
    "dynamic_alignment": "prompt_alignment",
    "subjective_quality": "perceptual_quality",
    "camera_control": "camera_error",
    "content_alignment": "prompt_alignment",
    "clip_score": "prompt_alignment",
    "3d_consistency": "reprojection_error",
    "gram_matrix": "style_consistency",
    "optical_flow": "motion_magnitude",
    "physical_motion_fidelity": "pmf",
    "S_recon": "geometric_consistency",
    "s_recon": "geometric_consistency",
    "score_reconstruction": "geometric_consistency",
    "frechet_video_motion_distance": "fvmd",
    "frechet_motion_distance": "fvmd",
    "jepa_embedding_distance": "jedi",
    "jedi_metric": "jedi",
    "trajectory_accuracy": "camera_error",
    "trajectory_tolerance": "camera_error",
    "trajectory_alignment": "camera_error",
    "action_strict_accuracy": "keystroke_strict_action_accuracy",
    "action_partial_accuracy": "keystroke_partial_action_accuracy",
    "traj_score": "keystroke_traj_score",
    "nATEt": "keystroke_natet",
    "nATEr": "keystroke_nater",
    "drift_aesthetic": "rollout_aesthetic_drift",
    "drift_imaging": "rollout_imaging_drift",
    "color_temperature": "color_temperature_constraint",
    "sharpness_vector_retention": "sharpness_vector_retention",
    "hpsv3": "hpsv3_norm",
    "hpsv3_quality": "hpsv3_norm",
    "HPSv3-Norm": "hpsv3_norm",
    "Sflick": "temporal_flickering",
    "s_flick": "temporal_flickering",
    "temporal_flicker": "temporal_flickering",
    "S_bg": "background_consistency",
    "background_consistency_score": "background_consistency",
    "S_seg": "segment_continuity",
    "segment_continuity_score": "segment_continuity",
}
SUITE_LEVEL_METRICS = {"fvmd", "jedi"}
TEMPORAL_CALIBRATION_METRICS = {
    "physical_fps_error",
    "physical_fps_pct_error",
    "physical_fps_intra_video_cv",
}
LONG_SEQUENCE_METRICS = {
    "long_sequence_motion_smoothness",
    "long_sequence_dynamic_degree",
    "long_sequence_aesthetic_quality",
    "long_sequence_imaging_quality",
}
LONG_HORIZON_DIAGNOSTIC_METRICS = (
    LONG_HORIZON_ACTION_METRICS | LONG_HORIZON_VISUAL_METRICS | LONG_HORIZON_MEMORY_METRICS
)
PHYSICS_JUDGE_METRICS = {
    "physics_judge_overall",
    "physics_judge_phenomenon",
    "physics_judge_causality",
    "physics_judge_material",
    "physics_judge_temporal",
    "physics_judge_prompt",
}


def canonical_metric_name(metric_name: str) -> str:
    """Map legacy or alias metric names to their canonical registry key."""
    return METRIC_ALIASES.get(metric_name, metric_name)


def is_suite_level_metric(metric_name: str) -> bool:
    """Return whether a metric is aggregated once per suite rather than per sample."""
    return canonical_metric_name(metric_name) in SUITE_LEVEL_METRICS


def split_suite_level_metrics(metric_names: list[str]) -> tuple[list[str], list[str]]:
    """Partition requested metrics into per-sample and suite-level groups."""
    per_sample_metrics: list[str] = []
    suite_level_metrics: list[str] = []
    seen_suite_metrics: set[str] = set()
    for raw_metric_name in metric_names:
        metric_name = canonical_metric_name(raw_metric_name)
        if metric_name in SUITE_LEVEL_METRICS:
            if metric_name not in seen_suite_metrics:
                suite_level_metrics.append(metric_name)
                seen_suite_metrics.add(metric_name)
            continue
        per_sample_metrics.append(raw_metric_name)
    return per_sample_metrics, suite_level_metrics


def build_metrics(metric_names: list[str], config: BenchmarkConfig) -> list[Metric]:
    """Instantiate configured Metric objects, deduplicating aliases and suite-level names."""
    metrics: list[Metric] = []
    seen: set[str] = set()
    shared_official_runtime = dict(config.metric_runtime.get("official_source", {}))
    shared_gvgc_geometry_runtime = dict(config.metric_runtime.get("gvgc_geometry", {}))
    shared_temporal_runtime = dict(config.metric_runtime.get("temporal_calibration", {}))
    shared_realtime_runtime = {
        "target_fps": config.target_fps,
        **dict(config.metric_runtime.get("real_time", {})),
    }
    shared_long_sequence_runtime = dict(config.metric_runtime.get("long_sequence", {}))
    shared_long_horizon_runtime = dict(config.metric_runtime.get("long_horizon_diagnostics", {}))
    shared_physics_judge_runtime = dict(config.metric_runtime.get("physics_judge", {}))
    shared_reconstruction_runtime = dict(
        config.metric_runtime.get("reconstruction_consistency", {})
    )
    reconstruction_evaluators: dict[str, ReconstructionConsistencyEvaluator] = {}
    for raw_metric_name in metric_names:
        metric_name = canonical_metric_name(raw_metric_name)
        if metric_name in seen:
            continue
        seen.add(metric_name)
        if metric_name in SUITE_LEVEL_METRICS:
            continue
        backend = config.metric_backends.get(metric_name, config.metric_backends.get(raw_metric_name, "auto"))
        normalization = config.normalization.get(metric_name, config.normalization.get(raw_metric_name, {}))
        runtime = dict(config.metric_runtime.get(metric_name, config.metric_runtime.get(raw_metric_name, {})))
        if metric_name in RECONSTRUCTION_CONSISTENCY_METRICS:
            backend = config.metric_backends.get(
                metric_name,
                config.metric_backends.get("reconstruction_consistency", backend),
            )
            normalization = config.normalization.get(
                metric_name,
                config.normalization.get("reconstruction_consistency", normalization),
            )
            runtime = {**shared_reconstruction_runtime, **runtime}
        official_runtime = {**shared_official_runtime, **runtime}
        if metric_name == "prompt_alignment":
            metrics.append(PromptAlignmentMetric(backend=backend, normalization=normalization))
        elif metric_name == "semantic_alignment":
            metrics.append(SemanticAlignmentMetric(backend=backend, normalization=normalization))
        elif metric_name == "style_consistency":
            metrics.append(StyleConsistencyMetric(backend=backend, normalization=normalization))
        elif metric_name == "perceptual_quality":
            metrics.append(PerceptualQualityMetric(backend=backend, normalization=normalization))
        elif metric_name == "image_quality":
            metrics.append(
                ImageQualityMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "hpsv3_norm":
            metrics.append(
                HPSv3NormMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "brightness_consistency":
            metrics.append(
                BrightnessConsistencyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "brightness_distribution_consistency":
            metrics.append(
                BrightnessDistributionConsistencyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "color_temperature_consistency":
            metrics.append(
                ColorTemperatureConsistencyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "color_temperature_constraint":
            metrics.append(
                ColorTemperatureConstraintMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "sharpness_retention":
            metrics.append(
                SharpnessRetentionMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "sharpness_vector_retention":
            metrics.append(
                SharpnessVectorRetentionMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "temporal_flickering":
            metrics.append(
                TemporalFlickeringMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "background_consistency":
            metrics.append(
                BackgroundConsistencyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "segment_continuity":
            metrics.append(
                SegmentContinuityMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
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
        elif metric_name in GVGC_GEOMETRY_METRICS:
            metrics.append(
                GvgcGeometryMetric(
                    metric_name=metric_name,
                    backend=backend,
                    normalization=normalization,
                    runtime={**shared_gvgc_geometry_runtime, **runtime},
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
        elif metric_name == "met3r_consistency":
            metrics.append(
                Met3RConsistencyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name in RECONSTRUCTION_CONSISTENCY_METRICS:
            evaluator_key = json.dumps(runtime, sort_keys=True, default=str)
            evaluator = reconstruction_evaluators.setdefault(
                evaluator_key,
                ReconstructionConsistencyEvaluator(runtime),
            )
            metrics.append(
                ReconstructionConsistencyMetric(
                    metric_name=metric_name,
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                    evaluator=evaluator,
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
        elif metric_name in REALTIME_METRICS:
            metrics.append(
                build_realtime_metric(
                    metric_name=metric_name,
                    backend=backend,
                    normalization=normalization,
                    runtime={**shared_realtime_runtime, **runtime},
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
        elif metric_name == "revisit_return_gate":
            metrics.append(
                RevisitReturnGateMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "entity_reappearance_consistency":
            metrics.append(
                EntityReappearanceConsistencyMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name == "memory_symmetry":
            metrics.append(
                MemorySymmetryMetric(
                    backend=backend,
                    normalization=normalization,
                    runtime=runtime,
                )
            )
        elif metric_name in LONG_HORIZON_ACTION_METRICS:
            metrics.append(
                KeystrokeActionMetric(
                    metric_name=metric_name,
                    backend=backend,
                    normalization=normalization,
                    runtime={**shared_long_horizon_runtime, **runtime},
                )
            )
        elif metric_name in LONG_HORIZON_VISUAL_METRICS:
            metrics.append(
                VisualDriftMetric(
                    metric_name=metric_name,
                    backend=backend,
                    normalization=normalization,
                    runtime={**shared_long_horizon_runtime, **runtime},
                )
            )
        elif metric_name in LONG_HORIZON_MEMORY_METRICS:
            metrics.append(
                SceneMemoryF1Metric(
                    backend=backend,
                    normalization=normalization,
                    runtime={**shared_long_horizon_runtime, **runtime},
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
