from __future__ import annotations

from collections import defaultdict
from typing import Any

from worldarena.benchmark_core.config import BenchmarkAggregationConfig
from worldarena.benchmark_core.schemas import SampleScore
from worldarena.benchmark_core.suites import OFFICIAL_SUITES


OFFICIAL_SCORE_NAMES: dict[str, str] = {
    "image_static": "S_image_static",
    "image_dynamic": "S_image_dynamic",
    "video_static": "S_video_static",
    "video_dynamic": "S_video_dynamic",
}
DERIVED_WORLD_SCORE_NAME = "S_worldarena_derived"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _derived_worldarena_score(
    suite_scores: dict[str, float | None],
    aggregation: BenchmarkAggregationConfig,
) -> float | None:
    if not aggregation.enabled:
        return None

    weighted_sum = 0.0
    present_weight = 0.0
    for suite in OFFICIAL_SUITES:
        suite_score = suite_scores.get(suite)
        if suite_score is None:
            continue
        weight = aggregation.suite_weights.get(suite, 0.0)
        weighted_sum += suite_score * weight
        present_weight += weight
    return round(weighted_sum / present_weight, 4) if present_weight else None


def aggregate_results(
    sample_scores: list[SampleScore],
    aggregation: BenchmarkAggregationConfig | None = None,
) -> dict[str, Any]:
    aggregation = aggregation or BenchmarkAggregationConfig()
    suite_values: dict[str, list[float]] = defaultdict(list)
    metric_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    domain_values: dict[str, dict[str, list[float]]] = {
        "style": defaultdict(list),
        "environment": defaultdict(list),
        "scene": defaultdict(list),
        "motion_category": defaultdict(list),
    }

    for sample in sample_scores:
        suite_values[sample.suite].append(sample.score)
        for metric_name, output in sample.metrics.items():
            if output.normalized is not None:
                metric_values[sample.suite][metric_name].append(output.normalized)
        if sample.style:
            domain_values["style"][sample.style].append(sample.score)
        if sample.environment:
            domain_values["environment"][sample.environment].append(sample.score)
        if sample.scene:
            domain_values["scene"][sample.scene].append(sample.score)
        if sample.motion_category:
            domain_values["motion_category"][sample.motion_category].append(sample.score)

    suite_scores = {suite: _mean(values) for suite, values in suite_values.items()}
    overall_scores = {
        score_name: suite_scores.get(suite_name)
        for suite_name, score_name in OFFICIAL_SCORE_NAMES.items()
    }
    if aggregation.enabled:
        overall_scores[DERIVED_WORLD_SCORE_NAME] = _derived_worldarena_score(
            suite_scores=suite_scores,
            aggregation=aggregation,
        )

    per_metric_summary = {
        suite: {
            metric_name: _mean(values)
            for metric_name, values in metric_map.items()
        }
        for suite, metric_map in metric_values.items()
    }
    domain_breakdown = {
        dimension: {
            label: _mean(values)
            for label, values in buckets.items()
        }
        for dimension, buckets in domain_values.items()
    }

    leaderboard_row = dict(overall_scores)

    return {
        "suite_scores": suite_scores,
        "overall_scores": overall_scores,
        "per_metric_summary": per_metric_summary,
        "domain_breakdown": domain_breakdown,
        "leaderboard_row": leaderboard_row,
    }
