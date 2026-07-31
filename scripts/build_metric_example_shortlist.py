#!/usr/bin/env python3
"""Build same-case metric example candidates from retained formal reports.

The shortlist is intentionally not a publish step. It narrows complete report
rows to numerically representative same-case pairs so a human can review frame
strips before any pair is shown in the documentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TARGETS: tuple[tuple[str, str], ...] = (
    ("image_static", "background_consistency"),
    ("image_static", "brightness_distribution_consistency"),
    ("image_static", "color_temperature_constraint"),
    ("image_static", "hpsv3_norm"),
    ("image_static", "image_quality"),
    ("image_static", "perceptual_quality"),
    ("image_static", "sharpness_retention"),
    ("image_static", "style_consistency"),
    ("image_static", "segment_continuity"),
    ("image_static", "prompt_alignment"),
    ("image_static", "psnr"),
    ("image_static", "ssim"),
    ("image_static", "epipolar_error"),
    ("image_static", "epipolar_inlier_rate"),
    ("image_static", "feature_match_coverage"),
    ("image_static", "optical_flow_aepe"),
    ("image_static", "physical_fps_error"),
    ("image_static", "physical_fps_pct_error"),
    ("image_static", "physical_fps_intra_video_cv"),
    ("image_dynamic", "motion_accuracy"),
    ("image_dynamic", "motion_magnitude"),
    ("image_dynamic", "motion_smoothness"),
)

PHYSICAL_FPS_METRICS = {
    "physical_fps_error",
    "physical_fps_pct_error",
    "physical_fps_intra_video_cv",
}

FRESH_REPORT_METRICS = {
    "background_consistency",
    "segment_continuity",
    *PHYSICAL_FPS_METRICS,
}

PREFERRED_MODEL_COUNT = 10
SHORTLIST_PER_METRIC = 12


@dataclass(frozen=True)
class ReportMetric:
    suite: str
    metric: str
    model: str
    report_dir: Path
    generated_at: str
    aggregate_score: float


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def complete_manifest_entry(entry: dict[str, Any]) -> bool:
    return (
        int(entry.get("rows") or 0) > 0
        and int(entry.get("rows") or 0) == int(entry.get("expected_rows") or 0)
        and int(entry.get("row_errors") or 0) == 0
        and int(entry.get("missing_predictions") or 0) == 0
    )


def coverage_complete(entry: dict[str, Any], metric: str) -> bool:
    coverage = (entry.get("metric_coverage") or {}).get(metric) or {}
    expected = int(coverage.get("expected") or 0)
    return (
        expected > 0
        and int(coverage.get("present") or 0) == expected
        and int(coverage.get("nonnull") or 0) == expected
    )


def aggregate_score(entry: dict[str, Any], suite: str, metric: str) -> float | None:
    return finite_number(((entry.get("summary") or {}).get(suite) or {}).get(metric))


def retained_report_metrics(project_root: Path) -> list[ReportMetric]:
    manifest_path = project_root / "artifacts/reports/FINAL_RESULTS_MANIFEST.json"
    manifest = read_json(manifest_path)
    targets = set(TARGETS)
    latest: dict[tuple[str, str, str], ReportMetric] = {}

    for entry in manifest.get("reports", []):
        suite = str(entry.get("suite") or "")
        if not complete_manifest_entry(entry):
            continue
        configured_dir = Path(str(entry.get("report_dir") or ""))
        report_dir = configured_dir if configured_dir.is_absolute() else project_root / configured_dir
        try:
            report_dir.relative_to(project_root / "artifacts/reports")
        except ValueError:
            # Promoted runtime reports are exposed under artifacts/reports as
            # symlinks while FINAL_RESULTS_MANIFEST retains their runtime path.
            promoted = project_root / "artifacts/reports" / report_dir.name
            if not promoted.is_dir():
                continue
            report_dir = promoted
        if not (report_dir / "per_sample_scores.jsonl").is_file():
            continue

        for metric in entry.get("metrics", []):
            metric = str(metric)
            if (suite, metric) not in targets or not coverage_complete(entry, metric):
                continue
            score = aggregate_score(entry, suite, metric)
            if score is None:
                continue
            candidate = ReportMetric(
                suite=suite,
                metric=metric,
                model=str(entry.get("model_name") or ""),
                report_dir=report_dir,
                generated_at=str(entry.get("generated_at") or ""),
                aggregate_score=score,
            )
            key = (suite, candidate.model, metric)
            current = latest.get(key)
            if current is None or (candidate.generated_at, str(candidate.report_dir)) > (
                current.generated_at,
                str(current.report_dir),
            ):
                latest[key] = candidate

    add_physical_fps_reports(project_root, latest)
    return list(latest.values())


def add_physical_fps_reports(
    project_root: Path,
    latest: dict[tuple[str, str, str], ReportMetric],
) -> None:
    reports_root = project_root / "artifacts/reports"
    patterns = (
        "*temporal_calibration_4gpu2m_validref_20260722",
        "*physical_fps_completion_14shard_7gpu2w_20260728",
        "*physical_fps_audited_20260728",
    )
    report_dirs = {path for pattern in patterns for path in reports_root.glob(pattern)}

    for report_dir in sorted(report_dirs):
        manifest_path = report_dir / "run_manifest.json"
        summary_path = report_dir / "per_metric_summary.json"
        scores_path = report_dir / "per_sample_scores.jsonl"
        if not (manifest_path.is_file() and summary_path.is_file() and scores_path.is_file()):
            continue
        manifest = read_json(manifest_path)
        failures = manifest.get("reported_failures") or {}
        if (
            int(manifest.get("items") or 0) != 3372
            or int(manifest.get("missing_predictions") or 0) != 0
            or failures
        ):
            continue
        summary = (read_json(summary_path).get("image_static") or {})
        model = str(manifest.get("model_name") or "")
        generated_at = str(manifest.get("generated_at") or "")
        for metric in PHYSICAL_FPS_METRICS:
            score = finite_number(summary.get(metric))
            if score is None:
                continue
            candidate = ReportMetric(
                suite="image_static",
                metric=metric,
                model=model,
                report_dir=report_dir,
                generated_at=generated_at,
                aggregate_score=score,
            )
            key = (candidate.suite, model, metric)
            current = latest.get(key)
            if current is None or (candidate.generated_at, str(candidate.report_dir)) > (
                current.generated_at,
                str(current.report_dir),
            ):
                latest[key] = candidate


def preferred_reports(reports: list[ReportMetric]) -> list[ReportMetric]:
    by_metric: dict[tuple[str, str], list[ReportMetric]] = defaultdict(list)
    for report in reports:
        by_metric[(report.suite, report.metric)].append(report)

    selected: dict[tuple[Path, str], ReportMetric] = {}
    for metric_reports in by_metric.values():
        ordered = sorted(metric_reports, key=lambda item: (item.aggregate_score, item.model))
        if len(ordered) <= PREFERRED_MODEL_COUNT:
            picks = ordered
        else:
            indices = {
                round(index * (len(ordered) - 1) / (PREFERRED_MODEL_COUNT - 1))
                for index in range(PREFERRED_MODEL_COUNT)
            }
            picks = [ordered[index] for index in sorted(indices)]
        for report in picks:
            selected[(report.report_dir, report.metric)] = report
    return list(selected.values())


def load_samples(reports: list[ReportMetric]) -> list[dict[str, Any]]:
    metrics_by_report: dict[Path, dict[str, ReportMetric]] = defaultdict(dict)
    for report in reports:
        metrics_by_report[report.report_dir][report.metric] = report

    samples: list[dict[str, Any]] = []
    for report_dir, report_metrics in sorted(metrics_by_report.items(), key=lambda item: str(item[0])):
        for row in read_jsonl(report_dir / "per_sample_scores.jsonl"):
            if row.get("error"):
                continue
            suite = str(row.get("suite") or "")
            for metric, report in report_metrics.items():
                if suite != report.suite:
                    continue
                payload = (row.get("metrics") or {}).get(metric) or {}
                normalized = finite_number(payload.get("normalized"))
                prediction_path = remap_workspace_path(
                    Path(str(row.get("prediction_path") or "")), report_dir.parents[2]
                )
                source_path = remap_workspace_path(
                    Path(str(row.get("path") or "")), report_dir.parents[2]
                )
                if (
                    normalized is None
                    or payload.get("error")
                ):
                    continue
                samples.append(
                    {
                        "suite": suite,
                        "metric": metric,
                        "model": report.model,
                        "sample_id": str(row.get("sample_id") or ""),
                        "normalized": normalized,
                        "raw": payload.get("raw"),
                        "details": payload.get("details") or {},
                        "backend": str(payload.get("backend") or ""),
                        "prediction_path": str(prediction_path),
                        "source_path": str(source_path),
                        "style": row.get("style"),
                        "environment": row.get("environment"),
                        "scene": row.get("scene"),
                        "motion_category": row.get("motion_category"),
                        "report_dir": str(report_dir),
                        "report_generated_at": report.generated_at,
                        "model_aggregate_score": report.aggregate_score,
                    }
                )
    return samples


def remap_workspace_path(path: Path, _fallback_root: Path) -> Path:
    value = str(path)
    if value.startswith("/mnt/cpfs/"):
        return Path("/mnt/cpfsB") / value.removeprefix("/mnt/cpfs/")
    return path


def load_benchmark_cases(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "artifacts/benchmark/manifest/worldarena_manifest.jsonl"
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def load_samples_from_compare_cache(
    project_root: Path,
    benchmark_cases: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the report-derived comparison cache used by the frontend.

    The cache is produced by ``site/build-compare-index.cjs`` from
    ``artifacts/reports`` and preserves the exact sample/model score join while
    avoiding repeated cold reads of every report JSONL on CPFS.
    """

    cache_root = project_root / "site/.cache/compare"
    meta = read_json(cache_root / "meta.json")
    if meta.get("reportsRoot") != "artifacts/reports":
        raise ValueError("comparison cache was not built from artifacts/reports")

    targets = set(TARGETS)
    samples: list[dict[str, Any]] = []
    for suite in sorted({suite for suite, _metric in targets}):
        data_path = cache_root / f"data-{suite}.json"
        if not data_path.is_file():
            continue
        data = read_json(data_path)
        for sample_id, case_entry in data.items():
            case = benchmark_cases.get(sample_id)
            if not case:
                continue
            source_path = Path(str(case.get("path") or case.get("reference_path") or ""))
            if not str(source_path):
                continue
            for model in case_entry.get("models", []):
                media_url = str(model.get("mediaUrl") or "")
                prefix = "/prediction-media/"
                if not media_url.startswith(prefix):
                    continue
                prediction_path = project_root / "artifacts/predictions" / media_url.removeprefix(prefix)
                scores = model.get("scores") or {}
                for metric, score_value in scores.items():
                    if (suite, metric) not in targets:
                        continue
                    normalized = finite_number(score_value)
                    if normalized is None:
                        continue
                    samples.append(
                        {
                            "suite": suite,
                            "metric": metric,
                            "model": str(model.get("runName") or model.get("runId") or ""),
                            "sample_id": sample_id,
                            "normalized": normalized,
                            "raw": None,
                            "details": {},
                            "backend": "",
                            "prediction_path": str(prediction_path),
                            "source_path": str(source_path),
                            "style": case.get("style") or case_entry.get("style"),
                            "environment": case.get("environment") or case_entry.get("environment"),
                            "scene": case.get("scene") or case_entry.get("scene"),
                            "motion_category": case.get("motion_category")
                            or case_entry.get("motionCategory"),
                            "report_dir": "artifacts/reports (frontend comparison cache)",
                            "report_generated_at": str(meta.get("generatedAt") or ""),
                            "model_aggregate_score": None,
                        }
                    )
    return samples, meta


def pair_candidates(
    samples: list[dict[str, Any]],
    benchmark_cases: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    distributions: dict[tuple[str, str], list[float]] = defaultdict(list)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        metric_key = (sample["suite"], sample["metric"])
        distributions[metric_key].append(sample["normalized"])
        group_key = (
            sample["suite"],
            sample["metric"],
            sample["sample_id"],
            sample["source_path"],
        )
        groups[group_key].append(sample)

    distribution_quantiles = {
        key: tuple(
            percentile(values, quantile)
            for quantile in (0.10, 0.20, 0.35, 0.65, 0.80, 0.90)
        )
        for key, values in distributions.items()
    }

    candidates: list[dict[str, Any]] = []
    for (suite, metric, sample_id, _source_path), rows in groups.items():
        by_model = {row["model"]: row for row in rows}
        if len(by_model) < 2 or sample_id not in benchmark_cases:
            continue
        rows = sorted(by_model.values(), key=lambda row: row["normalized"])
        low, high = rows[0], rows[-1]
        if low["prediction_path"] == high["prediction_path"]:
            continue
        q10, q20, q35, q65, q80, q90 = distribution_quantiles[(suite, metric)]
        gap = high["normalized"] - low["normalized"]
        spread = max(q90 - q10, 1e-6)
        if high["normalized"] < q65 or low["normalized"] > q35 or gap < spread * 0.45:
            continue
        typicality = (
            abs(high["normalized"] - q80) + abs(low["normalized"] - q20)
        ) / spread
        saturation_penalty = (
            (0.35 if high["normalized"] >= 0.9999 else 0.0)
            + (0.35 if low["normalized"] <= 0.0001 else 0.0)
        )
        rank_score = gap / spread - 0.35 * typicality - saturation_penalty
        case = benchmark_cases[sample_id]
        candidates.append(
            {
                "suite": suite,
                "metric": metric,
                "sample_id": sample_id,
                "prompt": str(case.get("prompt_target") or case.get("prompt_current") or ""),
                "prompt_source": "prompt_target" if case.get("prompt_target") else "prompt_current",
                "source_path": high["source_path"],
                "style": high.get("style"),
                "environment": high.get("environment"),
                "scene": high.get("scene"),
                "motion_category": high.get("motion_category"),
                "gap": round(gap, 8),
                "rank_score": round(rank_score, 8),
                "distribution": {
                    "q10": q10,
                    "q20": q20,
                    "q35": q35,
                    "q65": q65,
                    "q80": q80,
                    "q90": q90,
                },
                "high": high,
                "low": low,
            }
        )

    shortlist: list[dict[str, Any]] = []
    by_metric_candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_metric_candidates[(candidate["suite"], candidate["metric"])].append(candidate)
    for metric_candidates in by_metric_candidates.values():
        ordered = sorted(
            metric_candidates,
            key=lambda row: (-row["rank_score"], -row["gap"], row["sample_id"]),
        )
        shortlist.extend(ordered[:SHORTLIST_PER_METRIC])
    return sorted(shortlist, key=lambda row: (row["suite"], row["metric"], -row["rank_score"]))


def csv_scalar(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def write_outputs(shortlist: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "shortlist.json"
    csv_path = output_dir / "shortlist.csv"
    json_path.write_text(json.dumps(shortlist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "suite",
        "metric",
        "sample_id",
        "style",
        "environment",
        "scene",
        "motion_category",
        "gap",
        "rank_score",
        "prompt",
        "source_path",
        "high_model",
        "high_normalized",
        "high_raw",
        "high_prediction_path",
        "high_report_dir",
        "low_model",
        "low_normalized",
        "low_raw",
        "low_prediction_path",
        "low_report_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in shortlist:
            writer.writerow(
                {
                    "suite": row["suite"],
                    "metric": row["metric"],
                    "sample_id": row["sample_id"],
                    "style": row.get("style"),
                    "environment": row.get("environment"),
                    "scene": row.get("scene"),
                    "motion_category": row.get("motion_category"),
                    "gap": row["gap"],
                    "rank_score": row["rank_score"],
                    "prompt": row["prompt"],
                    "source_path": row["source_path"],
                    "high_model": row["high"]["model"],
                    "high_normalized": row["high"]["normalized"],
                    "high_raw": csv_scalar(row["high"]["raw"]),
                    "high_prediction_path": row["high"]["prediction_path"],
                    "high_report_dir": row["high"]["report_dir"],
                    "low_model": row["low"]["model"],
                    "low_normalized": row["low"]["normalized"],
                    "low_raw": csv_scalar(row["low"]["raw"]),
                    "low_prediction_path": row["low"]["prediction_path"],
                    "low_report_dir": row["low"]["report_dir"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cases = load_benchmark_cases(project_root)
    samples, cache_meta = load_samples_from_compare_cache(project_root, cases)
    fresh_reports = [
        report for report in retained_report_metrics(project_root)
        if report.metric in FRESH_REPORT_METRICS
    ]
    selected_fresh_reports = preferred_reports(fresh_reports)
    samples.extend(load_samples(selected_fresh_reports))
    shortlist = pair_candidates(samples, cases)
    write_outputs(shortlist, args.output_dir.resolve())

    summary = {
        "source": "site/.cache/compare",
        "source_reports_root": cache_meta.get("reportsRoot"),
        "source_generated_at": cache_meta.get("generatedAt"),
        "sample_metric_rows": len(samples),
        "fresh_report_metrics": len(fresh_reports),
        "selected_fresh_report_metrics": len(selected_fresh_reports),
        "shortlisted_pairs": len(shortlist),
        "by_metric": {
            f"{suite}:{metric}": sum(
                1 for row in shortlist if row["suite"] == suite and row["metric"] == metric
            )
            for suite, metric in TARGETS
        },
        "output_dir": str(args.output_dir.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
