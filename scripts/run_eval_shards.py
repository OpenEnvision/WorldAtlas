from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sharded WorldArena evaluation across GPUs.")
    parser.add_argument("--config", default="config/benchmark.yaml")
    parser.add_argument("--model-config", default="config/models/matrix_game_3_5s.yaml")
    parser.add_argument("--manifest", default="artifacts/benchmark/manifest/worldarena_manifest.jsonl")
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suites", default="image_dynamic")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    # internal per-shard mode
    parser.add_argument("--internal-shard", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _suites_set(raw: str) -> set[str] | None:
    items = {item.strip() for item in raw.split(",") if item.strip()}
    return items or None


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

    filtered = [s for s in manifest if not suites or s.suite in suites]
    shard = filtered[args.shard_index :: args.num_shards]

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

    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "predictions_root": str(Path(args.predictions_root).resolve()),
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

    procs = []
    for shard_index, gpu_id in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
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
            "--model-config", args.model_config,
            "--manifest", args.manifest,
            "--predictions-root", args.predictions_root,
            "--output-dir", str(base),
            "--suites", args.suites,
            "--num-shards", str(num_shards),
            "--shard-index", str(shard_index),
        ]
        if args.model_name:
            cmd.extend(["--model-name", args.model_name])
        log_path = log_dir / f"eval_shard{shard_index}_gpu{gpu_id}.log"
        log_file = log_path.open("w", encoding="utf-8")
        print(f"[run_eval_shards] shard={shard_index} gpu={gpu_id} log={log_path}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        procs.append((shard_index, gpu_id, proc, log_file))

    failed = False
    remaining = list(procs)
    while remaining:
        still = []
        for shard_index, gpu_id, proc, log_file in remaining:
            code = proc.poll()
            if code is None:
                still.append((shard_index, gpu_id, proc, log_file))
                continue
            log_file.close()
            print(f"[run_eval_shards] shard={shard_index} gpu={gpu_id} exited code={code}", flush=True)
            failed = failed or code != 0
        remaining = still
        if remaining:
            time.sleep(20)

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
