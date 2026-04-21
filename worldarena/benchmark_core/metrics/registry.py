from __future__ import annotations

from worldarena.benchmark_core.config import BenchmarkConfig
from worldarena.benchmark_core.metrics.alignment import DynamicAlignmentMetric, RolloutQualityMetric, SceneAlignmentMetric
from worldarena.benchmark_core.metrics.base import Metric
from worldarena.benchmark_core.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark_core.metrics.quality import SubjectiveQualityMetric
from worldarena.benchmark_core.metrics.style import StyleConsistencyMetric


def build_metrics(metric_names: list[str], config: BenchmarkConfig) -> list[Metric]:
    metrics: list[Metric] = []
    for metric_name in metric_names:
        backend = config.metric_backends.get(metric_name, "auto")
        normalization = config.normalization.get(metric_name, {})
        if metric_name == "scene_alignment":
            metrics.append(SceneAlignmentMetric(backend=backend, normalization=normalization))
        elif metric_name == "dynamic_alignment":
            metrics.append(DynamicAlignmentMetric(backend=backend, normalization=normalization))
        elif metric_name == "style_consistency":
            metrics.append(StyleConsistencyMetric(backend=backend, normalization=normalization))
        elif metric_name == "subjective_quality":
            metrics.append(SubjectiveQualityMetric(backend=backend, normalization=normalization))
        elif metric_name == "motion_magnitude":
            metrics.append(MotionMagnitudeMetric(backend=backend, normalization=normalization))
        elif metric_name == "motion_smoothness":
            metrics.append(MotionSmoothnessMetric(backend=backend, normalization=normalization))
        elif metric_name == "rollout_quality":
            metrics.append(RolloutQualityMetric(backend=backend, normalization=normalization))
        else:
            raise KeyError(f"unsupported metric: {metric_name}")
    return metrics
