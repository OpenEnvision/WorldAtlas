#!/usr/bin/env python3
"""Audit top-level WorldArena report directories for paper-ready coverage.

The script intentionally avoids walking shard subdirectories. It reads only
top-level aggregate files so the audit is fast enough to rerun on CPFS and
does not confuse shard intermediates with complete reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRIMARY_METRICS: dict[str, list[str]] = {
    "dynamic_motion3": ["motion_accuracy", "motion_magnitude", "motion_smoothness"],
    "static7": [
        "style_consistency",
        "image_quality",
        "brightness_distribution_consistency",
        "color_temperature_constraint",
        "sharpness_retention",
        "psnr",
        "ssim",
    ],
    "static_noslam19": [
        "prompt_alignment",
        "style_consistency",
        "image_quality",
        "brightness_distribution_consistency",
        "color_temperature_constraint",
        "sharpness_retention",
        "psnr",
        "ssim",
        "optical_flow_aepe",
        "depth_accuracy",
        "depth_collision",
    ],
    "static_quality_missing": ["perceptual_quality", "style_consistency"],
}


CSV_FIELDS = [
    "report",
    "suite",
    "type",
    "status",
    "rows",
    "expected",
    "row_errors",
    "metric_rows",
    "missing_predictions",
    "primary_coverage",
    "metrics",
    "notes",
]


@dataclass
class ReportAudit:
    report: str
    suite: str = ""
    type: str = ""
    status: str = ""
    rows: int = 0
    expected: int | str = ""
    row_errors: int | str = ""
    metric_rows: int | str = ""
    missing_predictions: int | str = ""
    primary_coverage: str = ""
    metrics: str = ""
    notes: str = ""

    def to_row(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in CSV_FIELDS}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Audit WorldArena report directories for reproducible result readiness."
    )
    parser.add_argument("--reports-root", type=Path, default=project_root / "artifacts" / "reports")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "artifacts" / "benchmark" / "manifest" / "worldarena_manifest.jsonl",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=project_root / "artifacts" / "reports" / "report_readiness_audit",
        help="Output prefix. The script writes <prefix>.csv, <prefix>.md, and <prefix>.manifest.json.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def load_expected_counts(manifest_path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            suite = str(payload.get("suite") or "")
            if suite:
                counts[suite] += 1
    return dict(counts)


def iter_report_dirs(reports_root: Path) -> list[Path]:
    with os.scandir(reports_root) as entries:
        return sorted(
            (Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False)),
            key=lambda path: path.name,
        )


def count_direct_children(report_dir: Path) -> tuple[int, list[str]]:
    subdirs = 0
    files: list[str] = []
    with os.scandir(report_dir) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subdirs += 1
            elif entry.is_file(follow_symlinks=False):
                files.append(entry.name)
    return subdirs, sorted(files)


def metric_items(metrics: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(metrics, dict):
        return [(str(name), value if isinstance(value, dict) else {}) for name, value in metrics.items()]
    if not isinstance(metrics, list):
        return []
    items: list[tuple[str, dict[str, Any]]] = []
    for metric in metrics:
        if isinstance(metric, str):
            items.append((metric, {}))
        elif isinstance(metric, dict):
            name = metric.get("name") or metric.get("metric")
            if name:
                items.append((str(name), metric))
    return items


def detect_report_type(report_name: str, suite: str, metric_names: set[str]) -> str:
    if suite == "image_dynamic" and set(PRIMARY_METRICS["dynamic_motion3"]) <= metric_names:
        return "dynamic_motion3"
    if suite == "image_static" and "static7" in report_name and set(PRIMARY_METRICS["static7"]) <= metric_names:
        return "static7"
    if suite == "image_static" and "noslam19" in report_name:
        return "static_noslam19"
    if suite == "image_static" and "quality_missing" in report_name:
        return "static_quality_missing"
    return "other"


def classify_report(
    *,
    expected: int | None,
    rows: int,
    row_errors: int,
    metric_names: set[str],
    metric_rows: int,
    report_type: str,
    metric_present: Counter[str],
    metric_nonnull: Counter[str],
) -> str:
    full_rows = expected is not None and rows == expected
    primary_metrics = PRIMARY_METRICS.get(report_type, [])
    full_primary = bool(primary_metrics) and all(metric_nonnull.get(metric, 0) == rows for metric in primary_metrics)
    attempted_primary = bool(primary_metrics) and all(metric_present.get(metric, 0) == rows for metric in primary_metrics)

    if full_rows and row_errors == 0 and full_primary:
        return "PAPER_READY_PRIMARY_METRICS"
    if full_rows and row_errors == 0 and attempted_primary:
        return "FULL_ATTEMPT_METRIC_GAPS"
    if full_rows and metric_names and metric_rows == rows:
        return "FULL_ROWS_METRICS_PRESENT_CAVEAT"
    if full_rows and metric_names:
        return "FULL_ROWS_PARTIAL_METRICS"
    if full_rows:
        return "FULL_ROWS_EMPTY_OR_FAILED"
    return "SUBSET_OR_PATCH"


def audit_report(report_dir: Path, expected_counts: dict[str, int]) -> ReportAudit:
    per_sample = report_dir / "per_sample_scores.jsonl"
    run_manifest = report_dir / "run_manifest.json"
    if not per_sample.exists():
        subdirs, files = count_direct_children(report_dir)
        return ReportAudit(
            report=report_dir.name,
            status="NO_TOPLEVEL_AGGREGATE",
            notes=f"No top-level per_sample_scores.jsonl; direct_subdirs={subdirs}; files={'|'.join(files[:8])}",
        )

    run_payload: dict[str, Any] = {}
    if run_manifest.exists():
        try:
            run_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            run_payload = {}

    suites: Counter[str] = Counter()
    metric_present: Counter[str] = Counter()
    metric_nonnull: Counter[str] = Counter()
    rows = 0
    row_errors = 0
    metric_rows = 0

    with per_sample.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows += 1
            suite = str(payload.get("suite") or "")
            if suite:
                suites[suite] += 1
            if payload.get("error"):
                row_errors += 1
            metrics = payload.get("metrics") or {}
            if metrics:
                metric_rows += 1
            for metric_name, metric_payload in metric_items(metrics):
                metric_present[metric_name] += 1
                if metric_payload.get("normalized") is not None:
                    metric_nonnull[metric_name] += 1

    suite = suites.most_common(1)[0][0] if suites else ""
    expected = expected_counts.get(suite)
    metric_names = set(metric_present)
    report_type = detect_report_type(report_dir.name, suite, metric_names)
    status = classify_report(
        expected=expected,
        rows=rows,
        row_errors=row_errors,
        metric_names=metric_names,
        metric_rows=metric_rows,
        report_type=report_type,
        metric_present=metric_present,
        metric_nonnull=metric_nonnull,
    )

    primary_coverage = ";".join(
        f"{metric}:{metric_nonnull.get(metric, 0)}/{rows}"
        for metric in PRIMARY_METRICS.get(report_type, [])
    )
    notes: list[str] = []
    if row_errors:
        notes.append(f"row_errors={row_errors}")
    missing_predictions = run_payload.get("missing_predictions", "")
    if missing_predictions not in ("", None, 0):
        notes.append(f"missing_predictions={missing_predictions}")
    reconstruction_metrics = (
        "geometric_consistency",
        "photometric_consistency",
        "reconstruction_consistency",
    )
    if (
        report_type == "static_noslam19"
        and any(metric_present.get(name, 0) for name in reconstruction_metrics)
        and all(metric_nonnull.get(name, 0) == 0 for name in reconstruction_metrics)
    ):
        notes.append("reconstruction_consistency_all_null")
    if report_type == "static_quality_missing":
        notes.append("quality_missing rerun; many are failed-to-open or subset-only")

    return ReportAudit(
        report=report_dir.name,
        suite=suite,
        type=report_type,
        status=status,
        rows=rows,
        expected=expected if expected is not None else "",
        row_errors=row_errors,
        metric_rows=metric_rows,
        missing_predictions=missing_predictions,
        primary_coverage=primary_coverage,
        metrics=";".join(sorted(metric_names)),
        notes=";".join(notes),
    )


def write_csv(path: Path, rows: list[ReportAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(row.to_row() for row in rows)


def write_markdown(path: Path, rows: list[ReportAudit], expected_counts: dict[str, int], csv_path: Path) -> None:
    counts = Counter(row.status for row in rows)
    paper_ready = [row for row in rows if row.status == "PAPER_READY_PRIMARY_METRICS"]
    groups = [
        ("Image Dynamic Motion3", [row for row in paper_ready if row.type == "dynamic_motion3"]),
        ("Image Static Static7", [row for row in paper_ready if row.type == "static7"]),
        ("Image Static Noslam19", [row for row in paper_ready if row.type == "static_noslam19"]),
    ]

    lines: list[str] = [
        "# Report Readiness Audit",
        "",
        f"Audited `{path.parent}` top-level report directories.",
        "Official manifest counts: "
        + ", ".join(f"`{suite}={count}`" for suite, count in sorted(expected_counts.items()))
        + ".",
        "",
        "`PAPER_READY_PRIMARY_METRICS` means official row count, no row-level errors, "
        "and all primary metrics for that report type have non-null values for every row. "
        "It does not mean every diagnostic metric in the file is valid.",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Paper-Ready Primary Results", ""])

    for title, group_rows in groups:
        lines.extend([f"### {title}", "", "| report | rows | primary coverage | notes |", "| --- | ---: | --- | --- |"])
        for row in group_rows:
            lines.append(f"| `{row.report}` | {row.rows} | `{row.primary_coverage}` | {row.notes} |")
        if not group_rows:
            lines.append("| _none_ | 0 |  |  |")
        lines.append("")

    lines.extend(
        [
            "## Non-Ready / Partial",
            "",
            f"See `{csv_path.name}` for the complete directory-level audit. Main reasons are subset row counts, "
            "top-level aggregate missing, row-level `failed to open video`, or primary metric coverage below the official row count.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_repro_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    expected_counts: dict[str, int],
    rows: list[ReportAudit],
    csv_path: Path,
    md_path: Path,
) -> None:
    payload = {
        "generated_at": utc_now(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "reports_root": str(args.reports_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "expected_counts": expected_counts,
        "status_counts": dict(Counter(row.status for row in rows)),
        "outputs": {
            "csv": str(csv_path.resolve()),
            "md": str(md_path.resolve()),
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.reports_root = resolve_path(args.reports_root)
    args.manifest = resolve_path(args.manifest)
    args.output_prefix = resolve_path(args.output_prefix)

    expected_counts = load_expected_counts(args.manifest)
    rows = [audit_report(report_dir, expected_counts) for report_dir in iter_report_dirs(args.reports_root)]

    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    manifest_path = args.output_prefix.with_suffix(".manifest.json")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, expected_counts, csv_path)
    write_repro_manifest(
        manifest_path,
        args=args,
        expected_counts=expected_counts,
        rows=rows,
        csv_path=csv_path,
        md_path=md_path,
    )
    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
