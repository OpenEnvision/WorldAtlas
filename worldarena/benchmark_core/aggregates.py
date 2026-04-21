from __future__ import annotations

from collections import defaultdict
from typing import Any

from worldarena.benchmark_core.schemas import SampleScore
from worldarena.benchmark_core.suites import OFFICIAL_SUITES


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def aggregate_results(
    sample_scores: list[SampleScore],
    aggregate_weights: dict[str, float],
) -> dict[str, Any]:
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
        "S_image_static": suite_scores.get("image_static"),
        "S_image_dynamic": suite_scores.get("image_dynamic"),
        "S_video_static": suite_scores.get("video_static"),
        "S_video_dynamic": suite_scores.get("video_dynamic"),
        "S_auxiliary": suite_scores.get("auxiliary"),
        "S_experimental": suite_scores.get("experimental"),
    }

    weighted_sum = 0.0
    present_weight = 0.0
    for suite in OFFICIAL_SUITES:
        suite_score = suite_scores.get(suite)
        if suite_score is None:
            continue
        weight = aggregate_weights.get(suite, 0.0)
        weighted_sum += suite_score * weight
        present_weight += weight
    overall_scores["S_worldarena"] = round(weighted_sum / present_weight, 4) if present_weight else None

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

    leaderboard_row = {
        "S_image_static": overall_scores["S_image_static"],
        "S_image_dynamic": overall_scores["S_image_dynamic"],
        "S_video_static": overall_scores["S_video_static"],
        "S_video_dynamic": overall_scores["S_video_dynamic"],
        "S_worldarena": overall_scores["S_worldarena"],
        "S_auxiliary": overall_scores["S_auxiliary"],
        "S_experimental": overall_scores["S_experimental"],
    }

    return {
        "suite_scores": suite_scores,
        "overall_scores": overall_scores,
        "per_metric_summary": per_metric_summary,
        "domain_breakdown": domain_breakdown,
        "leaderboard_row": leaderboard_row,
    }
