"""Aggregate per-sample metric scores into suite and dimension summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
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


def _flatten_dimension_summary(
    per_dimension_summary: dict[str, dict[str, float | None]],
) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for suite_name in sorted(per_dimension_summary):
        for dimension_name in sorted(per_dimension_summary[suite_name]):
            row[f"{suite_name}__{dimension_name}"] = per_dimension_summary[suite_name][dimension_name]
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


def _summarize_dimensions(
    per_metric_summary: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    per_dimension_summary: dict[str, dict[str, float | None]] = {}
    for suite, metric_map in per_metric_summary.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for metric_name, value in metric_map.items():
            if value is None:
                continue
            dimension = METRIC_DIMENSIONS.get(metric_name)
            if dimension is not None:
                grouped[dimension].append(float(value))

        summary = {
            dimension: _mean(values)
            for dimension, values in grouped.items()
            if values
        }
        if summary:
            per_dimension_summary[suite] = {
                key: summary[key]
                for key in sorted(summary)
            }
    return per_dimension_summary


def aggregate_results(sample_scores: list[SampleScore]) -> dict[str, Any]:
    official_metric_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    diagnostic_metric_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failure_values: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "failed_samples": 0,
                "errors": defaultdict(int),
                "eligibility_statuses": set(),
                "sample_ids": [],
            }
        )
    )
    not_applicable_values: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "samples": 0,
                "reasons": defaultdict(int),
                "sample_ids": [],
            }
        )
    )
    domain_values = {
        "style": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "environment": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "scene": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "motion_category": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }

    for sample in sample_scores:
        for metric_name, output in sample.metrics.items():
            if output.eligibility_status == "not_applicable":
                entry = not_applicable_values[sample.suite][metric_name]
                entry["samples"] += 1
                reason = output.error or "not applicable"
                entry["reasons"][reason] += 1
                if len(entry["sample_ids"]) < 5:
                    entry["sample_ids"].append(sample.sample_id)
                continue
            if output.normalized is None:
                failure_entry = failure_values[sample.suite][metric_name]
                failure_entry["failed_samples"] += 1
                failure_entry["eligibility_statuses"].add(output.eligibility_status)
                error_key = output.error or "score unavailable"
                failure_entry["errors"][error_key] += 1
                if len(failure_entry["sample_ids"]) < 5:
                    failure_entry["sample_ids"].append(sample.sample_id)
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
    per_dimension_summary = _summarize_dimensions(per_metric_summary)
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
    failure_summary = {
        suite: {
            metric_name: {
                "failed_samples": int(entry["failed_samples"]),
                "eligibility_statuses": sorted(str(status) for status in entry["eligibility_statuses"]),
                "sample_ids": list(entry["sample_ids"]),
                "errors": [
                    {"message": message, "count": count}
                    for message, count in sorted(
                        entry["errors"].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
            }
            for metric_name, entry in metric_map.items()
        }
        for suite, metric_map in failure_values.items()
        if metric_map
    }
    not_applicable_summary = {
        suite: {
            metric_name: {
                "samples": int(entry["samples"]),
                "sample_ids": list(entry["sample_ids"]),
                "reasons": [
                    {"message": message, "count": count}
                    for message, count in sorted(
                        entry["reasons"].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
            }
            for metric_name, entry in metric_map.items()
        }
        for suite, metric_map in not_applicable_values.items()
        if metric_map
    }

    return {
        "per_metric_summary": per_metric_summary,
        "per_dimension_summary": per_dimension_summary,
        "diagnostic_metric_summary": diagnostic_metric_summary,
        "failure_summary": failure_summary,
        "not_applicable_summary": not_applicable_summary,
        "domain_breakdown": domain_breakdown,
        "leaderboard_row": _flatten_metric_summary(per_metric_summary),
        "leaderboard_dimension_row": _flatten_dimension_summary(per_dimension_summary),
    }


def refresh_aggregate_indexes(aggregate_payload: dict[str, Any]) -> dict[str, Any]:
    per_metric_summary = aggregate_payload.get("per_metric_summary", {})
    per_dimension_summary = _summarize_dimensions(per_metric_summary)
    aggregate_payload["per_dimension_summary"] = per_dimension_summary
    aggregate_payload["leaderboard_row"] = _flatten_metric_summary(per_metric_summary)
    aggregate_payload["leaderboard_dimension_row"] = _flatten_dimension_summary(per_dimension_summary)
    return aggregate_payload
