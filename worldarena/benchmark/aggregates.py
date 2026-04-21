from __future__ import annotations

from collections import defaultdict
from typing import Any

from worldarena.benchmark.schemas import SampleScore


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _flatten_metric_summary(
    per_metric_summary: dict[str, dict[str, float | None]],
) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for suite_name in sorted(per_metric_summary):
        for metric_name in sorted(per_metric_summary[suite_name]):
            row[f"{suite_name}__{metric_name}"] = per_metric_summary[suite_name][metric_name]
    return row


def _summarize_metric_values(
    metric_values: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float | None]]:
    return {
        suite: {
            metric_name: _mean(values)
            for metric_name, values in metric_map.items()
        }
        for suite, metric_map in metric_values.items()
        if metric_map
    }


def aggregate_results(sample_scores: list[SampleScore]) -> dict[str, Any]:
    official_metric_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    diagnostic_metric_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    domain_values = {
        "style": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "environment": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "scene": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "motion_category": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }

    for sample in sample_scores:
        for metric_name, output in sample.metrics.items():
            if output.normalized is None:
                continue
            if output.eligibility_status == "official":
                official_metric_values[sample.suite][metric_name].append(output.normalized)
                if sample.style:
                    domain_values["style"][sample.style][sample.suite][metric_name].append(
                        output.normalized
                    )
                if sample.environment:
                    domain_values["environment"][sample.environment][sample.suite][
                        metric_name
                    ].append(output.normalized)
                if sample.scene:
                    domain_values["scene"][sample.scene][sample.suite][metric_name].append(
                        output.normalized
                    )
                if sample.motion_category:
                    domain_values["motion_category"][sample.motion_category][sample.suite][
                        metric_name
                    ].append(output.normalized)
            else:
                diagnostic_metric_values[sample.suite][metric_name].append(output.normalized)

    per_metric_summary = _summarize_metric_values(official_metric_values)
    diagnostic_metric_summary = _summarize_metric_values(diagnostic_metric_values)
    domain_breakdown = {
        dimension: {
            label: {
                suite: {
                    metric_name: _mean(values)
                    for metric_name, values in metric_map.items()
                }
                for suite, metric_map in suite_map.items()
                if metric_map
            }
            for label, suite_map in buckets.items()
            if suite_map
        }
        for dimension, buckets in domain_values.items()
    }

    return {
        "per_metric_summary": per_metric_summary,
        "diagnostic_metric_summary": diagnostic_metric_summary,
        "domain_breakdown": domain_breakdown,
        "leaderboard_row": _flatten_metric_summary(per_metric_summary),
    }
