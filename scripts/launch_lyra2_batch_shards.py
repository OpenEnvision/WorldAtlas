from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Lyra-2 image-static batch shards across GPUs.")
    parser.add_argument("--config", default="config/benchmark.yaml")
    parser.add_argument("--model-config", default="config/models/lyra2.yaml")
    parser.add_argument("--manifest", default="artifacts/benchmark/manifest/worldarena_manifest.jsonl")
    parser.add_argument("--output-dir", default="artifacts/predictions/lyra-2-image-static")
    parser.add_argument("--suites", default="image_static")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.0)
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
) -> list[str]:
    command = [
        python_bin,
        "-m",
        "worldarena.cli",
        "generate",
        "--config",
        str(args.config),
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
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def _generation_manifest_path(output_dir: Path, *, num_shards: int, shard_index: int) -> Path:
    suffix = f".shard{shard_index}" if num_shards > 1 else ""
    return output_dir / f"generation_manifest{suffix}.json"


def _shard_failed_count(output_dir: Path, *, num_shards: int, shard_index: int) -> int | None:
    manifest_path = _generation_manifest_path(
        output_dir,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return int(payload.get("failed", 0))


def main() -> int:
    args = parse_args()
    if args.workers_per_gpu <= 0:
        raise ValueError(f"--workers-per-gpu must be positive, got {args.workers_per_gpu}")

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

    processes: list[tuple[int, str, int, subprocess.Popen, object]] = []
    shard_index = 0
    for gpu_id in gpus:
        for local_worker_index in range(args.workers_per_gpu):
            env = os.environ.copy()
            env.update(model_config.env)
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            pythonpath = [str(project_root)]
            if env.get("PYTHONPATH"):
                pythonpath.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)

            command = _build_command(
                args,
                python_bin=model_config.python_bin,
                num_shards=total_shards,
                shard_index=shard_index,
            )
            log_path = log_dir / f"lyra2_batch_shard{shard_index}_gpu{gpu_id}_worker{local_worker_index}.log"
            log_file = log_path.open("w", encoding="utf-8")
            print(
                f"[launch_lyra2_batch] shard={shard_index} gpu={gpu_id} "
                f"worker={local_worker_index} log={log_path}",
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
            processes.append((shard_index, gpu_id, local_worker_index, process, log_file))
            shard_index += 1
            if args.launch_stagger_seconds > 0 and shard_index < total_shards:
                time.sleep(args.launch_stagger_seconds)

    failed = False
    try:
        while processes:
            remaining: list[tuple[int, str, int, subprocess.Popen, object]] = []
            for shard_index, gpu_id, local_worker_index, process, log_file in processes:
                code = process.poll()
                if code is None:
                    remaining.append((shard_index, gpu_id, local_worker_index, process, log_file))
                    continue
                log_file.close()
                failed_count = _shard_failed_count(
                    output_dir,
                    num_shards=total_shards,
                    shard_index=shard_index,
                )
                if failed_count is None:
                    print(
                        f"[launch_lyra2_batch] shard={shard_index} gpu={gpu_id} "
                        f"worker={local_worker_index} generation manifest missing",
                        flush=True,
                    )
                    failed = True
                elif failed_count > 0:
                    print(
                        f"[launch_lyra2_batch] shard={shard_index} gpu={gpu_id} "
                        f"worker={local_worker_index} failed_records={failed_count}",
                        flush=True,
                    )
                    failed = True
                print(
                    f"[launch_lyra2_batch] shard={shard_index} gpu={gpu_id} "
                    f"worker={local_worker_index} exited code={code}",
                    flush=True,
                )
                failed = failed or code != 0
            processes = remaining
            if processes:
                time.sleep(15)
    finally:
        for _shard_index, _gpu_id, _local_worker_index, process, log_file in processes:
            if process.poll() is None:
                process.terminate()
            log_file.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
