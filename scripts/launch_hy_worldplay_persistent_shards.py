from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch persistent HY-WorldPlay shards across GPUs.")
    parser.add_argument("--model-config", default="config/models/hy_worldplay.yaml")
    parser.add_argument("--manifest", default="artifacts/benchmark/manifest/worldarena_manifest.jsonl")
    parser.add_argument("--output-dir", default="/bench_output/hy_worldplay_image")
    parser.add_argument("--suites", default="image_static,image_dynamic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--max-shard-restarts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _gpu_list(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def _build_command(
    args: argparse.Namespace,
    *,
    python_bin: str,
    num_shards: int,
    shard_index: int,
    gpu_id: str,
) -> list[str]:
    command = [
        python_bin,
        "scripts/run_hy_worldplay_persistent.py",
        "--model-config",
        str(args.model_config),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--suites",
        str(args.suites),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
        "--visible-devices",
        str(gpu_id),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = parse_args()
    if args.workers_per_gpu <= 0:
        raise ValueError(f"--workers-per-gpu must be positive, got {args.workers_per_gpu}")
    if args.max_shard_restarts < 0:
        raise ValueError(f"--max-shard-restarts must be non-negative, got {args.max_shard_restarts}")
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.models.config import load_model_config

    model_config = load_model_config(Path(args.model_config))
    gpus = _gpu_list(args.gpus)
    total_shards = len(gpus) * args.workers_per_gpu
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def start_process(
        *,
        shard_index: int,
        gpu_id: str,
        local_worker_index: int,
        restart_count: int,
    ) -> tuple[int, str, int, int, subprocess.Popen, object]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("LOCAL_RANK", "0")
        env.setdefault("RANK", "0")
        env.setdefault("WORLD_SIZE", "1")
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env["MASTER_PORT"] = str(23456 + shard_index)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        env.setdefault("OPENBLAS_NUM_THREADS", "4")
        env.setdefault("NUMEXPR_NUM_THREADS", "4")
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
        pythonpath = [str(project_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        command = _build_command(
            args,
            python_bin=model_config.python_bin,
            num_shards=total_shards,
            shard_index=shard_index,
            gpu_id=gpu_id,
        )
        restart_suffix = "" if restart_count == 0 else f".restart{restart_count}"
        log_path = log_dir / (
            f"hy_worldplay_shard{shard_index}_gpu{gpu_id}_worker{local_worker_index}"
            f"{restart_suffix}.log"
        )
        log_file = log_path.open("w", encoding="utf-8")
        print(
            f"[launch_hy_worldplay] shard={shard_index} gpu={gpu_id} "
            f"worker={local_worker_index} restart={restart_count} log={log_path}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return shard_index, gpu_id, local_worker_index, restart_count, process, log_file

    processes: list[tuple[int, str, int, int, subprocess.Popen, object]] = []
    shard_index = 0
    for gpu_id in gpus:
        for local_worker_index in range(args.workers_per_gpu):
            processes.append(
                start_process(
                    shard_index=shard_index,
                    gpu_id=gpu_id,
                    local_worker_index=local_worker_index,
                    restart_count=0,
                )
            )
            shard_index += 1

    failed = False
    try:
        while processes:
            remaining: list[tuple[int, str, int, int, subprocess.Popen, object]] = []
            for shard_index, gpu_id, local_worker_index, restart_count, process, log_file in processes:
                code = process.poll()
                if code is None:
                    remaining.append(
                        (shard_index, gpu_id, local_worker_index, restart_count, process, log_file)
                    )
                    continue
                log_file.close()
                print(
                    f"[launch_hy_worldplay] shard={shard_index} gpu={gpu_id} exited code={code}",
                    flush=True,
                )
                if code != 0 and restart_count < args.max_shard_restarts:
                    next_restart = restart_count + 1
                    print(
                        f"[launch_hy_worldplay] restarting shard={shard_index} "
                        f"gpu={gpu_id} restart={next_restart}/{args.max_shard_restarts}",
                        flush=True,
                    )
                    remaining.append(
                        start_process(
                            shard_index=shard_index,
                            gpu_id=gpu_id,
                            local_worker_index=local_worker_index,
                            restart_count=next_restart,
                        )
                    )
                    continue
                failed = failed or code != 0
            processes = remaining
            if processes:
                time.sleep(15)
    finally:
        for _shard_index, _gpu_id, _local_worker_index, _restart_count, process, log_file in processes:
            if process.poll() is None:
                process.terminate()
            log_file.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
