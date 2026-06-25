from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "benchmark.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "artifacts" / "benchmark" / "manifest" / "worldarena_manifest.jsonl"
DEFAULT_PREDICTIONS_ROOT = PROJECT_ROOT / "artifacts" / "predictions"
DEFAULT_PLAN_DIR = PROJECT_ROOT / "artifacts" / "benchmark" / "eval_plan" / "existing_artifacts"
DEFAULT_REPORTS_ROOT = PROJECT_ROOT / "artifacts" / "benchmark" / "reports" / "existing_artifacts"

REFERENCE_VIDEO_METRICS = {
    "seen_region_preservation",
    "novel_region_fidelity",
    "object_permanence",
    "pmf",
}
ACTION_SPEC_METRICS = {
    "action_response_latency",
    "latency_budget_pass_rate",
}
IMAGE_STATIC_UNAVAILABLE_METRICS = {
    "motion_accuracy",
}
PREDICTION_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare evaluation manifests and commands for already-generated "
            "WorldArena artifacts. This script never generates videos."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--predictions-root", default=str(DEFAULT_PREDICTIONS_ROOT))
    parser.add_argument("--plan-dir", default=str(DEFAULT_PLAN_DIR))
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--suites", default="image_static,image_dynamic")
    parser.add_argument("--include", action="append", default=[], help="Prediction directory substring to include.")
    parser.add_argument("--exclude", action="append", default=[], help="Prediction directory substring to exclude.")
    parser.add_argument("--include-smoke", action="store_true", help="Include prediction directories containing 'smoke'.")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum matched prediction count.")
    parser.add_argument(
        "--raw-config",
        action="store_true",
        help="Do not filter metrics that are impossible for the current image artifacts.",
    )
    parser.add_argument(
        "--include-action-metrics",
        action="store_true",
        help="Keep action latency metrics even when action specs are usually absent.",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--run", action="store_true", help="Run the prepared sharded eval commands sequentially.")
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _selected_suites(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _load_manifest(path: Path):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from worldarena.benchmark import load_manifest

    return load_manifest(path)


def _valid_prediction_file(path: Path) -> bool:
    return path.suffix.lower() in PREDICTION_EXTENSIONS


def _index_runner_visible_stems(predictions_root: Path, suites: set[str]) -> set[str]:
    stems: set[str] = set()
    for path in predictions_root.iterdir():
        if _valid_prediction_file(path):
            stems.add(path.stem)
    for suite in suites:
        suite_dir = predictions_root / suite
        if not suite_dir.is_dir():
            continue
        for path in suite_dir.iterdir():
            if _valid_prediction_file(path):
                stems.add(path.stem)
    return stems


def _prediction_dirs(predictions_root: Path, args: argparse.Namespace) -> list[Path]:
    dirs = sorted(path for path in predictions_root.iterdir() if path.is_dir())
    selected: list[Path] = []
    for path in dirs:
        name = path.name
        if not args.include_smoke and "smoke" in name:
            continue
        if args.include and not any(token in name for token in args.include):
            continue
        if args.exclude and any(token in name for token in args.exclude):
            continue
        selected.append(path)
    return selected


def _write_manifest(samples: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(sample.to_dict(), ensure_ascii=False) for sample in samples)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def _filtered_config_payload(
    config_path: Path,
    *,
    raw_config: bool,
    include_action_metrics: bool,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    patched = deepcopy(payload)
    removed: dict[str, list[str]] = {}
    if raw_config:
        return patched, removed

    metrics_by_suite = patched.get("benchmark", {}).get("metrics_by_suite", {})
    for suite in ("image_static", "image_dynamic"):
        original = list(metrics_by_suite.get(suite, []))
        suite_removed: list[str] = []
        filtered: list[str] = []
        for metric in original:
            remove = False
            if metric in REFERENCE_VIDEO_METRICS:
                remove = True
            if metric in ACTION_SPEC_METRICS and not include_action_metrics:
                remove = True
            if suite == "image_dynamic" and metric == "camera_error":
                remove = True
            if suite == "image_static" and metric in IMAGE_STATIC_UNAVAILABLE_METRICS:
                remove = True
            if remove:
                suite_removed.append(metric)
            else:
                filtered.append(metric)
        metrics_by_suite[suite] = filtered
        removed[suite] = suite_removed
    return patched, removed


def _write_profile_config(args: argparse.Namespace, plan_dir: Path) -> tuple[Path, dict[str, list[str]]]:
    config_path = _project_path(args.config)
    payload, removed = _filtered_config_payload(
        config_path,
        raw_config=args.raw_config,
        include_action_metrics=args.include_action_metrics,
    )
    output = plan_dir / ("benchmark_raw.yaml" if args.raw_config else "benchmark_existing_artifacts.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output, removed


def _coverage(counts: Counter[str], manifest_counts: Counter[str]) -> dict[str, dict[str, float | int]]:
    payload: dict[str, dict[str, float | int]] = {}
    for suite, matched in sorted(counts.items()):
        total = int(manifest_counts.get(suite, 0))
        payload[suite] = {
            "matched": int(matched),
            "manifest": total,
            "ratio": round(float(matched) / float(total), 6) if total else 0.0,
        }
    return payload


def _command_for_item(
    *,
    config_path: Path,
    manifest_path: Path,
    predictions_root: Path,
    output_dir: Path,
    suites: list[str],
    model_name: str,
    gpus: str,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_eval_shards.py"),
        "--config",
        str(config_path),
        "--model-config",
        "",
        "--manifest",
        str(manifest_path),
        "--predictions-root",
        str(predictions_root),
        "--output-dir",
        str(output_dir),
        "--suites",
        ",".join(suites),
        "--model-name",
        model_name,
        "--gpus",
        gpus,
    ]


def _shell_command(cmd: list[str]) -> str:
    prefix = f"PYTHONPATH={shlex.quote(str(PROJECT_ROOT))}"
    return " ".join([prefix, *[shlex.quote(item) for item in cmd]])


def _scan_root(
    root: Path,
    *,
    manifest: list[Any],
    suites: set[str],
) -> tuple[list[Any], Counter[str]]:
    visible_stems = _index_runner_visible_stems(root, suites)
    matched: list[Any] = []
    counts: Counter[str] = Counter()
    for sample in manifest:
        if suites and sample.suite not in suites:
            continue
        if sample.prediction_stem not in visible_stems:
            continue
        matched.append(sample)
        counts[sample.suite] += 1
    return matched, counts


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _project_path(args.manifest)
    predictions_root = _project_path(args.predictions_root)
    plan_dir = _project_path(args.plan_dir)
    reports_root = _project_path(args.reports_root)
    suites = _selected_suites(args.suites)

    manifest = _load_manifest(manifest_path)
    manifest_counts = Counter(sample.suite for sample in manifest)
    config_path, removed_metrics = _write_profile_config(args, plan_dir)

    manifests_dir = plan_dir / "manifests"
    plan_items: list[dict[str, Any]] = []
    for root in _prediction_dirs(predictions_root, args):
        matched, counts = _scan_root(root, manifest=manifest, suites=suites)
        if len(matched) < args.min_count:
            continue
        active_suites = sorted(counts)
        filtered_manifest_path = manifests_dir / f"{root.name}.jsonl"
        _write_manifest(matched, filtered_manifest_path)

        output_dir = reports_root / root.name
        cmd = _command_for_item(
            config_path=config_path,
            manifest_path=filtered_manifest_path,
            predictions_root=root.resolve(),
            output_dir=output_dir.resolve(),
            suites=active_suites,
            model_name=root.name,
            gpus=args.gpus,
        )
        plan_items.append(
            {
                "name": root.name,
                "predictions_root": str(root.resolve()),
                "manifest": str(filtered_manifest_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "suites": active_suites,
                "items": len(matched),
                "counts": dict(sorted(counts.items())),
                "coverage": _coverage(counts, manifest_counts),
                "command": _shell_command(cmd),
            }
        )

    plan = {
        "config": str(config_path.resolve()),
        "source_config": str(_project_path(args.config)),
        "source_manifest": str(manifest_path),
        "predictions_root": str(predictions_root),
        "reports_root": str(reports_root),
        "metric_profile": {
            "raw_config": bool(args.raw_config),
            "include_action_metrics": bool(args.include_action_metrics),
            "removed_metrics": removed_metrics,
            "notes": [
                "image_dynamic does not run camera_error",
                "image_static camera_error uses camera_path-derived synthetic GT when pose annotations are absent",
                "reference-video metrics are removed from the default profile because current image artifacts have no reference_video",
                "filtered per-root manifests include only samples with existing prediction files",
            ],
        },
        "items": plan_items,
    }
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (plan_dir / "commands.sh").write_text(
        "\n".join(item["command"] for item in plan_items) + ("\n" if plan_items else ""),
        encoding="utf-8",
    )
    return plan


def run_plan(plan: dict[str, Any]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(PROJECT_ROOT)]
    )
    for item in plan["items"]:
        cmd = shlex.split(item["command"].split(" ", 1)[1])
        print(f"[prepare_existing_artifacts_eval] running {item['name']} ({item['items']} items)", flush=True)
        completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main() -> int:
    args = parse_args()
    plan = prepare(args)
    print(json.dumps({
        "plan": str((_project_path(args.plan_dir) / "plan.json").resolve()),
        "commands": str((_project_path(args.plan_dir) / "commands.sh").resolve()),
        "config": plan["config"],
        "items": len(plan["items"]),
        "total_predictions": sum(item["items"] for item in plan["items"]),
    }, indent=2, ensure_ascii=False))
    if args.run:
        return run_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
