from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sharded WorldArena evaluation across GPUs.")
    parser.add_argument("--config", default="config/benchmark.yaml")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--manifest", default="artifacts/benchmark/manifest/worldarena_manifest.jsonl")
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suites", default="image_dynamic")
    parser.add_argument(
        "--metrics",
        default="",
        help="Comma-separated metric override applied to each selected suite.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-launch sample limit.")
    parser.add_argument(
        "--skip-existing-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip shard directories that already have per_sample_scores.jsonl and run_manifest.json.",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Seconds between launch heartbeat messages while shards are running.",
    )
    parser.add_argument(
        "--max-concurrent-shards",
        type=int,
        default=0,
        help="Maximum shard processes to run at once. 0 means launch all shards.",
    )
    parser.add_argument(
        "--max-concurrent-per-gpu",
        type=int,
        default=0,
        help="Maximum simultaneous shard processes assigned to one GPU. 0 means unlimited.",
    )
    parser.add_argument(
        "--motion-frame-max-side",
        type=int,
        default=None,
        help="Optional max image side for motion metric frame caches.",
    )
    parser.add_argument(
        "--motion-smoothness-frame-max-side",
        type=int,
        default=None,
        help="Optional max image side for motion_smoothness frame caches.",
    )
    # internal per-shard mode
    parser.add_argument("--internal-shard", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_project_path(path: str | None, project_root: Path) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_source_hashes(project_root: Path) -> dict[str, str]:
    relative_paths = (
        "worldarena/benchmark/gvgc_geometry_backend.py",
        "worldarena/benchmark/transnet_backend.py",
        "worldarena/benchmark/metrics/consistency.py",
        "worldarena/benchmark/metrics/segment_continuity.py",
        "worldarena/benchmark/metrics/registry.py",
        "worldarena/benchmark/runner.py",
        "worldarena/benchmark/media.py",
        "worldarena/benchmark/aggregates.py",
        "worldarena/benchmark/report.py",
        "worldarena/benchmark/config.py",
    )
    return {
        relative_path: digest
        for relative_path in relative_paths
        if (digest := _sha256_file(project_root / relative_path)) is not None
    }


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import cv2

        versions["opencv"] = str(cv2.__version__)
    except Exception:
        pass
    try:
        import numpy

        versions["numpy"] = str(numpy.__version__)
    except Exception:
        pass
    return versions


def _git_head(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _suites_set(raw: str) -> set[str] | None:
    items = {item.strip() for item in raw.split(",") if item.strip()}
    return items or None


def _metric_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _apply_metric_override(config, suites: set[str] | None, metrics: Iterable[str]) -> None:
    metric_names = list(metrics)
    if not metric_names:
        return
    config.metrics_by_suite = dict(config.metrics_by_suite)
    target_suites = suites or set(config.metrics_by_suite)
    for suite in target_suites:
        config.metrics_by_suite[suite] = list(metric_names)


def _bind_runtime_to_visible_gpu(config, metrics: Iterable[str], args: argparse.Namespace) -> None:
    metric_names = set(metrics)
    cacheable_metrics = {
        "motion_accuracy",
        "motion_magnitude",
        "motion_smoothness",
        "optical_flow_aepe",
    }
    selected_cacheable_metrics = metric_names & cacheable_metrics
    if not selected_cacheable_metrics:
        return
    config.metric_runtime = dict(config.metric_runtime)
    for metric_name in selected_cacheable_metrics:
        runtime = dict(config.metric_runtime.get(metric_name, {}))
        runtime.setdefault("cache_instance", True)
        if args.motion_frame_max_side:
            runtime["frame_max_side"] = args.motion_frame_max_side
        if metric_name == "motion_smoothness" and args.motion_smoothness_frame_max_side:
            runtime["frame_max_side"] = args.motion_smoothness_frame_max_side
        if metric_name == "motion_accuracy":
            runtime.setdefault("sam2_device", "cuda:0")
            runtime.setdefault("flow_device", "cuda:0")
            runtime.setdefault("grounding_dino_device", "cuda:0")
        config.metric_runtime[metric_name] = runtime


def run_shard(args: argparse.Namespace) -> int:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.benchmark import load_config, load_manifest, run_benchmark
    from worldarena.benchmark.config import apply_model_metric_overrides
    from worldarena.models import load_model_config

    config = load_config(Path(args.config))
    model_config = load_model_config(Path(args.model_config)) if args.model_config else None
    config = apply_model_metric_overrides(config, model_config)
    manifest = load_manifest(Path(args.manifest).resolve())
    suites = _suites_set(args.suites)
    metrics = _metric_list(args.metrics)
    _apply_metric_override(config, suites, metrics)
    _bind_runtime_to_visible_gpu(config, metrics, args)

    filtered = [s for s in manifest if not suites or s.suite in suites]
    shard = filtered[args.shard_index :: args.num_shards]
    if args.limit is not None:
        shard = shard[: args.limit]

    predictions_root = Path(args.predictions_root).resolve()
    out_dir = Path(args.output_dir).resolve() / f"shard{args.shard_index}"
    model_name = args.model_name or (model_config.name if model_config else predictions_root.name)

    run_benchmark(
        config=config,
        manifest=shard,
        predictions_root=predictions_root,
        output_dir=out_dir,
        model_name=model_name,
        suites=suites,
        limit=None,
    )
    return 0


def _load_scores(path: Path):
    from worldarena.benchmark.schemas import MetricOutput, SampleScore

    scores = []
    if not path.exists():
        return scores
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        metrics = {name: MetricOutput(**mo) for name, mo in d.get("metrics", {}).items()}
        scores.append(
            SampleScore(
                sample_id=d["sample_id"],
                suite=d["suite"],
                metrics=metrics,
                path=d.get("path", ""),
                prediction_path=d.get("prediction_path"),
                style=d.get("style"),
                environment=d.get("environment"),
                scene=d.get("scene"),
                motion_category=d.get("motion_category"),
                error=d.get("error"),
            )
        )
    return scores


def merge(args: argparse.Namespace, num_shards: int) -> dict:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from datetime import datetime, timezone

    from worldarena.benchmark.aggregates import aggregate_results
    from worldarena.benchmark.report import write_run_outputs
    from worldarena.models import load_model_config

    base = Path(args.output_dir).resolve()
    all_scores = []
    for i in range(num_shards):
        all_scores.extend(_load_scores(base / f"shard{i}" / "per_sample_scores.jsonl"))

    aggregate_payload = aggregate_results(all_scores)
    model_config = load_model_config(Path(args.model_config)) if args.model_config else None
    model_name = args.model_name or (model_config.name if model_config else Path(args.predictions_root).name)
    suites = sorted({s.suite for s in all_scores})
    config_path = _resolve_project_path(args.config, project_root)
    model_config_path = _resolve_project_path(args.model_config, project_root)
    manifest_path = _resolve_project_path(args.manifest, project_root)
    predictions_root = Path(args.predictions_root).resolve()

    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "predictions_root": str(predictions_root),
        "output_dir": str(base),
        "items": len(all_scores),
        "missing_predictions": sum(1 for s in all_scores if s.error == "missing prediction"),
        "suites": suites,
        "reported_metrics": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload["per_metric_summary"].items()
        },
        "reported_dimensions": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload.get("per_dimension_summary", {}).items()
            if m
        },
        "reported_diagnostics": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload.get("diagnostic_metric_summary", {}).items()
            if m
        },
        "reported_failures": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload.get("failure_summary", {}).items()
            if m
        },
        "reported_not_applicable": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload.get("not_applicable_summary", {}).items()
            if m
        },
        "num_shards": num_shards,
        "metric_override": _metric_list(args.metrics),
        "limit": args.limit,
        "reproducibility": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "git_head": _git_head(project_root),
            "python": sys.executable,
            "python_version": sys.version,
            "runtime_versions": _runtime_versions(),
            "evaluation_source_sha256": _evaluation_source_hashes(project_root),
            "cwd": str(project_root),
            "inputs": {
                "config": str(config_path) if config_path is not None else None,
                "config_sha256": _sha256_file(config_path),
                "model_config": str(model_config_path) if model_config_path is not None else None,
                "model_config_sha256": _sha256_file(model_config_path),
                "manifest": str(manifest_path) if manifest_path is not None else None,
                "manifest_sha256": _sha256_file(manifest_path),
                "predictions_root": str(predictions_root),
            },
            "launch": {
                "gpus": args.gpus,
                "max_concurrent_shards": args.max_concurrent_shards,
                "max_concurrent_per_gpu": args.max_concurrent_per_gpu,
                "skip_existing_shards": args.skip_existing_shards,
                "suites": args.suites,
                "metrics": args.metrics,
                "motion_frame_max_side": args.motion_frame_max_side,
                "motion_smoothness_frame_max_side": args.motion_smoothness_frame_max_side,
            },
        },
    }
    write_run_outputs(
        output_dir=base,
        run_manifest=run_manifest,
        sample_scores=all_scores,
        aggregate_payload=aggregate_payload,
        model_name=model_name,
    )
    return {
        "run_manifest": run_manifest,
        "per_metric_summary": aggregate_payload["per_metric_summary"],
        "diagnostic_metric_summary": aggregate_payload.get("diagnostic_metric_summary", {}),
    }


def launch(args: argparse.Namespace) -> int:
    project_root = _project_root()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    num_shards = len(gpus)
    base = Path(args.output_dir).resolve()
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = 0
    for shard_index, gpu_id in enumerate(gpus):
        shard_dir = base / f"shard{shard_index}"
        if (
            args.skip_existing_shards
            and (shard_dir / "per_sample_scores.jsonl").exists()
            and (shard_dir / "run_manifest.json").exists()
        ):
            print(f"[run_eval_shards] shard={shard_index} gpu={gpu_id} skip existing {shard_dir}", flush=True)
            skipped += 1
            continue
        pending.append((shard_index, gpu_id))

    max_concurrent = args.max_concurrent_shards if args.max_concurrent_shards and args.max_concurrent_shards > 0 else len(pending)
    max_concurrent = max(max_concurrent, 1)
    max_concurrent_per_gpu = max(args.max_concurrent_per_gpu, 0)

    def pop_launchable():
        if not pending:
            return None
        if max_concurrent_per_gpu == 0:
            return pending.pop(0)
        running_per_gpu: dict[str, int] = {}
        for _, gpu_id, _, _ in remaining:
            running_per_gpu[gpu_id] = running_per_gpu.get(gpu_id, 0) + 1
        for index, (_, gpu_id) in enumerate(pending):
            if running_per_gpu.get(gpu_id, 0) < max_concurrent_per_gpu:
                return pending.pop(index)
        return None

    def launch_one(shard_index: int, gpu_id: str):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("OPENCV_FFMPEG_THREADS", "1")
        if "segment_continuity" in _metric_list(args.metrics):
            env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
            env.setdefault("TF_NUM_INTEROP_THREADS", "1")
            env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env["WORLDARENA_CHECKPOINT_ROOT"] = str(project_root / "ckpt")
        pythonpath = [str(project_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-shard",
            "--config", args.config,
            "--manifest", args.manifest,
            "--predictions-root", args.predictions_root,
            "--output-dir", str(base),
            "--suites", args.suites,
            "--num-shards", str(num_shards),
            "--shard-index", str(shard_index),
        ]
        if args.model_config:
            cmd.extend(["--model-config", args.model_config])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.metrics:
            cmd.extend(["--metrics", args.metrics])
        if args.motion_frame_max_side:
            cmd.extend(["--motion-frame-max-side", str(args.motion_frame_max_side)])
        if args.motion_smoothness_frame_max_side:
            cmd.extend(["--motion-smoothness-frame-max-side", str(args.motion_smoothness_frame_max_side)])
        if args.model_name:
            cmd.extend(["--model-name", args.model_name])
        log_path = log_dir / f"eval_shard{shard_index}_gpu{gpu_id}.log"
        log_file = log_path.open("w", encoding="utf-8")
        print(f"[run_eval_shards] shard={shard_index} gpu={gpu_id} log={log_path}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        return (shard_index, gpu_id, proc, log_file)

    failed = False
    remaining = []
    last_heartbeat = 0.0
    finished = skipped
    while pending or remaining:
        while pending and len(remaining) < max_concurrent:
            launchable = pop_launchable()
            if launchable is None:
                break
            remaining.append(launch_one(*launchable))
        still = []
        for shard_index, gpu_id, proc, log_file in remaining:
            code = proc.poll()
            if code is None:
                still.append((shard_index, gpu_id, proc, log_file))
                continue
            log_file.close()
            print(f"[run_eval_shards] shard={shard_index} gpu={gpu_id} exited code={code}", flush=True)
            failed = failed or code != 0
            finished += 1
        remaining = still
        if pending or remaining:
            now = time.time()
            if now - last_heartbeat >= max(args.heartbeat_interval, 1):
                active = [
                    f"{shard_index}:gpu{gpu_id}:pid{proc.pid}"
                    for shard_index, gpu_id, proc, _ in remaining[:8]
                ]
                suffix = " ..." if len(remaining) > 8 else ""
                print(
                    "[run_eval_shards] HEARTBEAT "
                    f"finished_or_skipped={finished}/{num_shards} running={len(remaining)} pending={len(pending)} "
                    f"active={','.join(active)}{suffix}",
                    flush=True,
                )
                last_heartbeat = now
            time.sleep(5)

    result = merge(args, num_shards)
    print("[run_eval_shards] MERGED per_metric_summary:")
    print(json.dumps(result["per_metric_summary"], indent=2, ensure_ascii=False))
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    if args.internal_shard:
        return run_shard(args)
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
