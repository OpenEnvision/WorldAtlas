"""Write benchmark run artifacts (JSON summaries, CSV exports)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from worldatlas.common.serialization import ensure_dir, write_json
from worldatlas.benchmark.schemas import SampleScore


def _distribution_group_breakdown(distribution_details: dict[str, Any]) -> dict[str, Any]:
    """Flatten the distribution metric strata into a readable per-group table.

    ``distribution_metric_details.json`` carries full backend payloads, which is
    what you want when debugging a single score but not when comparing strata.
    """
    breakdown: dict[str, Any] = {}
    for suite, metric_map in distribution_details.items():
        for metric_name, payload in metric_map.items():
            if not isinstance(payload, dict):
                continue
            details = payload.get("details", {})
            rows: dict[str, Any] = {
                "overall": {
                    "raw": payload.get("raw"),
                    "normalized": payload.get("normalized"),
                    "prediction_clip_count": details.get("prediction", {}).get("clip_count"),
                    "reference_clip_count": details.get("reference", {}).get("clip_count"),
                }
            }
            for label, group in (payload.get("groups") or {}).items():
                if group.get("status") != "ok":
                    rows[label] = {
                        "status": group.get("status"),
                        "reason": group.get("reason") or group.get("error"),
                    }
                    continue
                group_details = group.get("details", {})
                rows[label] = {
                    "raw": group.get("raw"),
                    "normalized": group.get("normalized"),
                    "prediction_clip_count": group_details.get("prediction", {}).get("clip_count"),
                    "reference_clip_count": group_details.get("reference", {}).get("clip_count"),
                    "prediction_sample_count": group.get("prediction_sample_count"),
                    "reference_video_count": group.get("reference_video_count"),
                }
            breakdown.setdefault(suite, {})[metric_name] = rows
    return breakdown


def write_run_outputs(
    output_dir: Path,
    run_manifest: dict[str, Any],
    sample_scores: list[SampleScore],
    aggregate_payload: dict[str, Any],
    model_name: str,
) -> None:
    ensure_dir(output_dir)
    # Config dataclasses contain Paths; persist exactly the JSON-normalized
    # context used by integrity.digest_json, not unserializable Python objects.
    run_manifest = json.loads(json.dumps(run_manifest, default=str))
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "per_metric_summary.json", aggregate_payload["per_metric_summary"])
    write_json(output_dir / "per_dimension_summary.json", aggregate_payload.get("per_dimension_summary", {}))

    diagnostic_summary = aggregate_payload.get("diagnostic_metric_summary", {})
    diagnostic_path = output_dir / "diagnostic_metric_summary.json"
    if diagnostic_summary:
        write_json(diagnostic_path, diagnostic_summary)
    elif diagnostic_path.exists():
        diagnostic_path.unlink()

    failure_summary = aggregate_payload.get("failure_summary", {})
    failure_path = output_dir / "metric_failures.json"
    if failure_summary:
        write_json(failure_path, failure_summary)
    elif failure_path.exists():
        failure_path.unlink()

    not_applicable_summary = aggregate_payload.get("not_applicable_summary", {})
    not_applicable_path = output_dir / "not_applicable_summary.json"
    if not_applicable_summary:
        write_json(not_applicable_path, not_applicable_summary)
    elif not_applicable_path.exists():
        not_applicable_path.unlink()

    distribution_details = aggregate_payload.get("distribution_metric_details", {})
    distribution_path = output_dir / "distribution_metric_details.json"
    group_path = output_dir / "distribution_group_breakdown.json"
    if distribution_details:
        write_json(distribution_path, distribution_details)
        write_json(group_path, _distribution_group_breakdown(distribution_details))
    else:
        if distribution_path.exists():
            distribution_path.unlink()
        if group_path.exists():
            group_path.unlink()

    write_json(output_dir / "domain_breakdown.json", aggregate_payload["domain_breakdown"])

    per_sample_path = output_dir / "per_sample_scores.jsonl"
    lines = [json.dumps(sample.to_dict(), ensure_ascii=False) for sample in sample_scores]
    per_sample_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    leaderboard_path = output_dir / "leaderboard.csv"
    metric_keys = sorted(aggregate_payload["leaderboard_row"].keys())
    fieldnames = ["model_name", *metric_keys]
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {"model_name": model_name}
        row.update({key: aggregate_payload["leaderboard_row"][key] for key in metric_keys})
        writer.writerow(row)
