from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch WorldFM generation shards across GPUs.")
    parser.add_argument("--config", default="config/benchmark.yaml")
    parser.add_argument("--model-config", default="config/models/worldfm.yaml")
    parser.add_argument("--manifest", default="artifacts/benchmark/manifest/worldarena_manifest.jsonl")
    parser.add_argument("--output-dir", default="/bench-workspace/yesliu/hjd/model/WorldArena/bench_output/worldfm")
    parser.add_argument("--suites", default="image_static")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _gpu_list(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def _build_command(args: argparse.Namespace, *, num_shards: int, shard_index: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "worldarena.cli",
        "generate",
        "--config",
        str(args.config),
        "--model-config",
        str(args.model_config),
        "--manifest",
        str(args.manifest),
        "--suites",
        str(args.suites),
        "--output-dir",
        str(args.output_dir),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = parse_args()
    gpus = _gpu_list(args.gpus)
    threads_per_worker = int(args.threads_per_worker or max(1, (os.cpu_count() or len(gpus)) // len(gpus)))
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, str, subprocess.Popen, object]] = []
    for shard_index, gpu_id in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("OMP_NUM_THREADS", str(threads_per_worker))
        env.setdefault("OPENBLAS_NUM_THREADS", str(threads_per_worker))
        env.setdefault("MKL_NUM_THREADS", str(threads_per_worker))
        env.setdefault("NUMEXPR_NUM_THREADS", str(threads_per_worker))
        env.setdefault("VECLIB_MAXIMUM_THREADS", str(threads_per_worker))
        command = _build_command(args, num_shards=len(gpus), shard_index=shard_index)
        log_path = log_dir / f"worldfm_shard{shard_index}_gpu{gpu_id}.log"
        log_file = log_path.open("w", encoding="utf-8")
        print(f"[launch_worldfm_shards] shard={shard_index} gpu={gpu_id} log={log_path}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard_index, gpu_id, process, log_file))

    failed = False
    try:
        while processes:
            remaining: list[tuple[int, str, subprocess.Popen, object]] = []
            for shard_index, gpu_id, process, log_file in processes:
                code = process.poll()
                if code is None:
                    remaining.append((shard_index, gpu_id, process, log_file))
                    continue
                log_file.close()
                print(
                    f"[launch_worldfm_shards] shard={shard_index} gpu={gpu_id} exited code={code}",
                    flush=True,
                )
                failed = failed or code != 0
            processes = remaining
            if processes:
                time.sleep(15)
    finally:
        for _shard_index, _gpu_id, process, log_file in processes:
            if process.poll() is None:
                process.terminate()
            log_file.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
