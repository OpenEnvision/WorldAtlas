#!/usr/bin/env python3
"""Build a benchmark manifest containing cases without valid metric results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--metrics", required=True, help="Comma-separated required metrics.")
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        type=Path,
        help="Report directory or per_sample_scores.jsonl; may be repeated.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _metric_names(raw: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not names:
        raise ValueError("at least one metric is required")
    return names


def _score_path(report: Path) -> Path:
    return report if report.is_file() else report / "per_sample_scores.jsonl"


def _valid_metric(payload: dict[str, Any] | None) -> bool:
    return bool(
        payload
        and payload.get("normalized") is not None
        and not payload.get("error")
        and payload.get("eligibility_status") != "not_applicable"
    )


def main() -> int:
    args = parse_args()
    metrics = _metric_names(args.metrics)
    valid: dict[str, set[str]] = {metric: set() for metric in metrics}

    for report in args.report:
        score_path = _score_path(report)
        if not score_path.is_file():
            raise FileNotFoundError(f"report scores not found: {score_path}")
        with score_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("suite") != args.suite:
                    continue
                sample_id = str(row["sample_id"])
                outputs = row.get("metrics") or {}
                for metric in metrics:
                    if _valid_metric(outputs.get(metric)):
                        valid[metric].add(sample_id)

    selected: list[dict[str, Any]] = []
    suite_total = 0
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("suite") != args.suite:
                continue
            suite_total += 1
            sample_id = str(row["sample_id"])
            if any(sample_id not in valid[metric] for metric in metrics):
                selected.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "suite": args.suite,
                "suite_total": suite_total,
                "metrics": list(metrics),
                "valid_by_metric": {metric: len(ids) for metric, ids in valid.items()},
                "gap_count": len(selected),
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
