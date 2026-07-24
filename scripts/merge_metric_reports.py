#!/usr/bin/env python3
"""Merge complete per-metric reports into one coverage-audited benchmark report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
        required=True,
        type=Path,
        help="Source report directory or per_sample_scores.jsonl; later sources win.",
    )
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=None)
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
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(project_root))

    from worldarena.benchmark.aggregates import aggregate_results
    from worldarena.benchmark.report import write_run_outputs
    from worldarena.benchmark.schemas import MetricOutput, SampleScore

    metrics = _metric_names(args.metrics)
    manifest_rows: list[dict[str, Any]] = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("suite") == args.suite:
                manifest_rows.append(row)

    expected = args.expected if args.expected is not None else len(manifest_rows)
    if len(manifest_rows) != expected:
        raise ValueError(
            f"manifest suite count mismatch: suite={args.suite} rows={len(manifest_rows)} expected={expected}"
        )

    expected_ids = {str(row["sample_id"]) for row in manifest_rows}
    merged: dict[str, dict[str, dict[str, Any]]] = {sample_id: {} for sample_id in expected_ids}
    prediction_paths: dict[str, str] = {}
    source_records: list[dict[str, Any]] = []

    for report in args.report:
        score_path = _score_path(report)
        if not score_path.is_file():
            raise FileNotFoundError(f"report scores not found: {score_path}")
        accepted = 0
        with score_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id") or "")
                if row.get("suite") != args.suite or sample_id not in expected_ids:
                    continue
                if row.get("prediction_path"):
                    prediction_paths[sample_id] = str(row["prediction_path"])
                outputs = row.get("metrics") or {}
                for metric in metrics:
                    payload = outputs.get(metric)
                    if _valid_metric(payload):
                        merged[sample_id][metric] = payload
                        accepted += 1
        source_records.append(
            {
                "report": str(report.resolve()),
                "scores": str(score_path.resolve()),
                "accepted_metric_values": accepted,
            }
        )

    missing_by_metric = {
        metric: sorted(sample_id for sample_id in expected_ids if metric not in merged[sample_id])
        for metric in metrics
    }
    incomplete = {metric: ids for metric, ids in missing_by_metric.items() if ids}
    if incomplete:
        summary = {metric: len(ids) for metric, ids in incomplete.items()}
        examples = {metric: ids[:5] for metric, ids in incomplete.items()}
        raise RuntimeError(f"metric coverage incomplete: counts={summary} examples={examples}")

    predictions_root = args.predictions_root.resolve()
    scores: list[SampleScore] = []
    for row in manifest_rows:
        sample_id = str(row["sample_id"])
        prediction_path = prediction_paths.get(sample_id)
        if prediction_path is None:
            prediction_path = str(predictions_root / f"{row.get('prediction_stem', sample_id)}.mp4")
        scores.append(
            SampleScore(
                sample_id=sample_id,
                suite=args.suite,
                metrics={
                    metric: MetricOutput(**merged[sample_id][metric])
                    for metric in metrics
                },
                path=str(row.get("path") or ""),
                prediction_path=prediction_path,
                style=row.get("style"),
                environment=row.get("environment"),
                scene=row.get("scene"),
                motion_category=row.get("motion_category"),
                error=None,
            )
        )

    aggregate_payload = aggregate_results(scores)
    official = sorted((aggregate_payload.get("per_metric_summary") or {}).get(args.suite, {}))
    diagnostics = sorted(
        (aggregate_payload.get("diagnostic_metric_summary") or {}).get(args.suite, {})
    )
    failures = aggregate_payload.get("failure_summary") or {}
    if failures:
        raise RuntimeError(f"merged report unexpectedly contains failures: {failures}")

    generated_at = datetime.now(timezone.utc).isoformat()
    coverage = {
        metric: {"expected": expected, "present": expected, "nonnull": expected}
        for metric in metrics
    }
    run_manifest = {
        "generated_at": generated_at,
        "model_name": args.model_name,
        "predictions_root": str(predictions_root),
        "output_dir": str(args.output_dir.resolve()),
        "items": expected,
        "rows": expected,
        "expected_rows": expected,
        "row_errors": 0,
        "missing_predictions": 0,
        "suites": [args.suite],
        "reported_metrics": {args.suite: official} if official else {},
        "reported_dimensions": {
            args.suite: sorted(
                (aggregate_payload.get("per_dimension_summary") or {}).get(args.suite, {})
            )
        },
        "reported_diagnostics": {args.suite: diagnostics} if diagnostics else {},
        "reported_failures": {},
        "reported_not_applicable": {},
        "metric_coverage": coverage,
        "required_metrics": list(metrics),
        "source_reports": source_records,
        "reproducibility": {
            "tool": str(Path(__file__).resolve()),
            "manifest": str(args.manifest.resolve()),
            "sources": source_records,
        },
    }
    write_run_outputs(
        output_dir=args.output_dir.resolve(),
        run_manifest=run_manifest,
        sample_scores=scores,
        aggregate_payload=aggregate_payload,
        model_name=args.model_name,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "items": expected,
                "official_metrics": official,
                "diagnostic_metrics": diagnostics,
                "metric_coverage": coverage,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
