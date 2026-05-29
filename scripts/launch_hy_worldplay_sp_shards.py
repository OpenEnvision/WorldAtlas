from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch HY-WorldPlay persistent shards with torchrun SP groups.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suites", default="image_static,image_dynamic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--gpu-groups",
        default="0,1,2,3;4,5,6,7",
        help="Semicolon-separated torchrun groups. Each group runs one benchmark shard.",
    )
    parser.add_argument("--max-shard-restarts", type=int, default=3)
    parser.add_argument("--master-port-base", type=int, default=29700)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_gpu_groups(raw: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for group in raw.split(";"):
        gpus = [item.strip() for item in group.split(",") if item.strip()]
        if gpus:
            groups.append(gpus)
    if not groups:
        raise ValueError("--gpu-groups must contain at least one GPU group")
    return groups


def _build_command(
    args: argparse.Namespace,
    *,
    python_bin: str,
    nproc_per_node: int,
    master_port: int,
    num_shards: int,
    shard_index: int,
) -> list[str]:
    command = [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--master_port",
        str(master_port),
        "--nproc_per_node",
        str(nproc_per_node),
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
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = parse_args()
    if args.max_shard_restarts < 0:
        raise ValueError(f"--max-shard-restarts must be non-negative, got {args.max_shard_restarts}")

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.models.config import load_model_config

    model_config = load_model_config(Path(args.model_config))
    gpu_groups = _parse_gpu_groups(args.gpu_groups)
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def start_process(
        *,
        shard_index: int,
        gpu_group: list[str],
        restart_count: int,
    ) -> tuple[int, list[str], int, subprocess.Popen, object]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        pythonpath = [str(project_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        master_port = args.master_port_base + shard_index
        command = _build_command(
            args,
            python_bin=model_config.python_bin,
            nproc_per_node=len(gpu_group),
            master_port=master_port,
            num_shards=len(gpu_groups),
            shard_index=shard_index,
        )
        restart_suffix = "" if restart_count == 0 else f".restart{restart_count}"
        gpu_label = "-".join(gpu_group)
        log_path = log_dir / f"hy_worldplay_sp_shard{shard_index}_gpus{gpu_label}{restart_suffix}.log"
        log_file = log_path.open("w", encoding="utf-8")
        print(
            f"[launch_hy_worldplay_sp] shard={shard_index} gpus={','.join(gpu_group)} "
            f"restart={restart_count} log={log_path}",
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
        return shard_index, gpu_group, restart_count, process, log_file

    processes: list[tuple[int, list[str], int, subprocess.Popen, object]] = []
    for shard_index, gpu_group in enumerate(gpu_groups):
        processes.append(start_process(shard_index=shard_index, gpu_group=gpu_group, restart_count=0))

    failed = False
    try:
        while processes:
            remaining: list[tuple[int, list[str], int, subprocess.Popen, object]] = []
            for shard_index, gpu_group, restart_count, process, log_file in processes:
                code = process.poll()
                if code is None:
                    remaining.append((shard_index, gpu_group, restart_count, process, log_file))
                    continue
                log_file.close()
                print(
                    f"[launch_hy_worldplay_sp] shard={shard_index} gpus={','.join(gpu_group)} exited code={code}",
                    flush=True,
                )
                if code != 0 and restart_count < args.max_shard_restarts:
                    next_restart = restart_count + 1
                    print(
                        f"[launch_hy_worldplay_sp] restarting shard={shard_index} "
                        f"restart={next_restart}/{args.max_shard_restarts}",
                        flush=True,
                    )
                    remaining.append(
                        start_process(
                            shard_index=shard_index,
                            gpu_group=gpu_group,
                            restart_count=next_restart,
                        )
                    )
                    continue
                failed = failed or code != 0
            processes = remaining
            if processes:
                time.sleep(15)
    finally:
        for _shard_index, _gpu_group, _restart_count, process, log_file in processes:
            if process.poll() is None:
                process.terminate()
            log_file.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
