from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from worldarena.common.serialization import ensure_dir, write_json
from worldarena.benchmark.schemas import SampleScore


def write_run_outputs(
    output_dir: Path,
    run_manifest: dict[str, Any],
    sample_scores: list[SampleScore],
    aggregate_payload: dict[str, Any],
    model_name: str,
) -> None:
    ensure_dir(output_dir)
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "per_metric_summary.json", aggregate_payload["per_metric_summary"])

    diagnostic_summary = aggregate_payload.get("diagnostic_metric_summary", {})
    diagnostic_path = output_dir / "diagnostic_metric_summary.json"
    if diagnostic_summary:
        write_json(diagnostic_path, diagnostic_summary)
    elif diagnostic_path.exists():
        diagnostic_path.unlink()

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
