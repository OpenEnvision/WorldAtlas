from __future__ import annotations

from worldarena.benchmark_core.config import BenchmarkConfig
from worldarena.benchmark_core.metrics.alignment import (
    AnchorConsistencyMetric,
    PromptAlignmentMetric,
    ReferenceAppearanceAlignmentMetric,
)
from worldarena.benchmark_core.metrics.base import Metric
from worldarena.benchmark_core.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark_core.metrics.physics import PhysicsMetric, is_physics_metric
from worldarena.benchmark_core.metrics.quality import PerceptualQualityMetric
from worldarena.benchmark_core.metrics.style import StyleConsistencyMetric


METRIC_ALIASES: dict[str, str] = {
    "scene_alignment": "prompt_alignment",
    "dynamic_alignment": "prompt_alignment",
    "subjective_quality": "perceptual_quality",
    "rollout_quality": "anchor_consistency",
}


def canonical_metric_name(metric_name: str) -> str:
    return METRIC_ALIASES.get(metric_name, metric_name)


def build_metrics(metric_names: list[str], config: BenchmarkConfig) -> list[Metric]:
    metrics: list[Metric] = []
    seen: set[str] = set()
    for raw_metric_name in metric_names:
        metric_name = canonical_metric_name(raw_metric_name)
        if metric_name in seen:
            continue
        seen.add(metric_name)
        backend = config.metric_backends.get(metric_name, config.metric_backends.get(raw_metric_name, "auto"))
        normalization = config.normalization.get(metric_name, config.normalization.get(raw_metric_name, {}))
        if metric_name == "prompt_alignment":
            metrics.append(PromptAlignmentMetric(backend=backend, normalization=normalization))
        elif metric_name == "reference_appearance_alignment":
            metrics.append(ReferenceAppearanceAlignmentMetric(backend=backend, normalization=normalization))
        elif metric_name == "style_consistency":
            metrics.append(StyleConsistencyMetric(backend=backend, normalization=normalization))
        elif metric_name == "perceptual_quality":
            metrics.append(PerceptualQualityMetric(backend=backend, normalization=normalization))
        elif metric_name == "motion_magnitude":
            metrics.append(MotionMagnitudeMetric(backend=backend, normalization=normalization))
        elif metric_name == "motion_smoothness":
            metrics.append(MotionSmoothnessMetric(backend=backend, normalization=normalization))
        elif metric_name == "anchor_consistency":
            metrics.append(AnchorConsistencyMetric(backend=backend, normalization=normalization))
        elif is_physics_metric(metric_name):
            metrics.append(PhysicsMetric(metric_name=metric_name, backend=backend, normalization=normalization))
        else:
            raise KeyError(f"unsupported metric: {raw_metric_name}")
    return metrics
