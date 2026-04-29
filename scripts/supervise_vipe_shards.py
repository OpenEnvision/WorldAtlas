from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldarena.common.vipe_sidecars import default_sidecar_dir, discover_videos, prioritize_videos, sidecar_complete


LOG = logging.getLogger("supervise_vipe_shards")
DEFAULT_INPUT_DIR = Path("/share/project/yesliu/hjd/data/videos_test_split_4971")
DEFAULT_VIPE_ROOT = Path("/share/project/yesliu/hjd/model/WorldEvals/thirdparty/vipe")
DEFAULT_VIPE_PYTHON = Path("/share/project/yesliu/miniconda3/envs/vipe/bin/python")
DEFAULT_STAGE_ROOT = Path("/tmp/worldarena_vipe_stage")
DEFAULT_STATUS_PATH = REPO_ROOT / ".cache" / "vipe_supervisor_status.json"
DEFAULT_LOG_DIR = REPO_ROOT / ".cache" / "vipe_logs"
DEFAULT_SUMMARY_DIR = REPO_ROOT / ".cache"
THREAD_ENV = {
    "OMP_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    "OPENBLAS_NUM_THREADS": "6",
    "NUMEXPR_MAX_THREADS": "6",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}
STOP_REQUESTED = False


@dataclass(slots=True)
class ShardState:
    shard_index: int
    gpu: int
    videos: list[Path]
    log_path: Path
    summary_path: Path
    process: subprocess.Popen[bytes] | None = None
    restarts: int = 0
    last_launch_ts: float | None = None
    last_exit_ts: float | None = None
    last_exit_code: int | None = None

    @property
    def total(self) -> int:
        return len(self.videos)

    @property
    def pid(self) -> int | None:
        if self.process is None:
            return None
        return self.process.pid


def parse_shard_list(text: str) -> list[int]:
    values: set[int] = set()
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.add(int(stripped))
    if not values:
        raise ValueError("empty shard list")
    return sorted(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise shard-based ViPE annotation workers.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pipeline", default="default")
    parser.add_argument("--num-shards", type=int, default=40)
    parser.add_argument("--shard-start", type=int, default=16)
    parser.add_argument("--shard-end", type=int, default=39)
    parser.add_argument(
        "--shards",
        help="Optional comma-separated shard list. Overrides --shard-start/--shard-end.",
    )
    parser.add_argument(
        "--gpu-base-shard",
        type=int,
        default=16,
        help="Base shard index for gpu = (shard - gpu_base_shard) %% gpu_count.",
    )
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--max-active-workers", type=int, default=15)
    parser.add_argument("--max-workers-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=20.0)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--vipe-root", type=Path, default=DEFAULT_VIPE_ROOT)
    parser.add_argument("--vipe-python", type=Path, default=DEFAULT_VIPE_PYTHON)
    parser.add_argument(
        "--worker-log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if args.pipeline != "default":
        parser.error("this supervisor is restricted to pipeline=default")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.gpu_count <= 0:
        parser.error("--gpu-count must be positive")
    if args.max_active_workers <= 0:
        parser.error("--max-active-workers must be positive")
    if args.max_workers_per_gpu <= 0:
        parser.error("--max-workers-per-gpu must be positive")
    if args.shards is not None:
        shard_indices = parse_shard_list(args.shards)
    else:
        shard_indices = list(range(args.shard_start, args.shard_end + 1))
    if any(index < 0 or index >= args.num_shards for index in shard_indices):
        parser.error("all shard indices must be in [0, num_shards)")
    args.shard_indices = shard_indices
    return args


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def mem_available_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                return int(parts[1]) / 1024.0 / 1024.0
    except OSError:
        return -1.0
    return -1.0


def shard_gpu(shard_index: int, *, gpu_base_shard: int, gpu_count: int) -> int:
    return (shard_index - gpu_base_shard) % gpu_count


def build_states(args: argparse.Namespace) -> dict[int, ShardState]:
    videos = prioritize_videos(discover_videos(args.input_dir.resolve()))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    states: dict[int, ShardState] = {}
    for shard_index in args.shard_indices:
        shard_videos = videos[shard_index :: args.num_shards]
        states[shard_index] = ShardState(
            shard_index=shard_index,
            gpu=shard_gpu(
                shard_index,
                gpu_base_shard=args.gpu_base_shard,
                gpu_count=args.gpu_count,
            ),
            videos=shard_videos,
            log_path=args.log_dir / f"vipe_shard_{shard_index:02d}_of_{args.num_shards:02d}.log",
            summary_path=args.summary_dir
            / f"vipe_test_split_4971_summary_shard_{shard_index:02d}_of_{args.num_shards:02d}.json",
        )
    return states


def shard_progress(state: ShardState) -> tuple[int, int]:
    done = sum(1 for video_path in state.videos if sidecar_complete(default_sidecar_dir(video_path)))
    return done, state.total - done


def managed_running_counts(states: dict[int, ShardState]) -> tuple[int, dict[int, int], set[int]]:
    total = 0
    per_gpu = {state.gpu: 0 for state in states.values()}
    managed_pids: set[int] = set()
    for state in states.values():
        if state.process is None:
            continue
        if state.process.poll() is not None:
            continue
        total += 1
        per_gpu[state.gpu] = per_gpu.get(state.gpu, 0) + 1
        managed_pids.add(state.process.pid)
    return total, per_gpu, managed_pids


def parse_gpu_from_args(arg_string: str) -> int | None:
    parts = arg_string.split()
    for index, token in enumerate(parts):
        if token == "--gpu" and index + 1 < len(parts):
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
        if token.startswith("--gpu="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return None
    return None


def external_worker_counts(managed_pids: set[int]) -> tuple[int, dict[int, int]]:
    total = 0
    per_gpu: dict[int, int] = {}
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            pid_text, args_text = stripped.split(None, 1)
        except ValueError:
            continue
        pid = int(pid_text)
        if pid in managed_pids:
            continue
        if "scripts/build_vipe_annotations.py" not in args_text:
            continue
        gpu = parse_gpu_from_args(args_text)
        if gpu is None:
            continue
        total += 1
        per_gpu[gpu] = per_gpu.get(gpu, 0) + 1
    return total, per_gpu


def worker_command(state: ShardState, args: argparse.Namespace) -> list[str]:
    return [
        str(args.vipe_python.resolve()),
        str((REPO_ROOT / "scripts" / "build_vipe_annotations.py").resolve()),
        "--input-dir",
        str(args.input_dir.resolve()),
        "--pipeline",
        args.pipeline,
        "--gpu",
        str(state.gpu),
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(state.shard_index),
        "--summary-path",
        str(state.summary_path.resolve()),
        "--staging-root",
        str(args.staging_root.resolve()),
        "--vipe-root",
        str(args.vipe_root.resolve()),
        "--vipe-python",
        str(args.vipe_python.resolve()),
        "--log-level",
        args.worker_log_level,
    ]


def launch_worker(state: ShardState, args: argparse.Namespace) -> None:
    state.log_path.parent.mkdir(parents=True, exist_ok=True)
    state.summary_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(THREAD_ENV)
    command = worker_command(state, args)
    launch_ts = time.time()
    with state.log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} shard={state.shard_index} gpu={state.gpu} ===\n"
        )
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state.process = process
    state.last_launch_ts = launch_ts
    LOG.info(
        "Started shard %d on gpu %d with pid %d",
        state.shard_index,
        state.gpu,
        process.pid,
    )


def write_status(
    *,
    states: dict[int, ShardState],
    args: argparse.Namespace,
    external_total: int,
    external_per_gpu: dict[int, int],
) -> None:
    per_shard = []
    total = done = pending = 0
    active = 0
    for shard_index in sorted(states):
        state = states[shard_index]
        shard_done, shard_pending = shard_progress(state)
        total += state.total
        done += shard_done
        pending += shard_pending
        running = state.process is not None and state.process.poll() is None
        if running:
            active += 1
        per_shard.append(
            {
                "shard_index": state.shard_index,
                "gpu": state.gpu,
                "total": state.total,
                "done": shard_done,
                "pending": shard_pending,
                "pid": state.pid if running else None,
                "restarts": state.restarts,
                "last_exit_code": state.last_exit_code,
                "last_launch_ts": state.last_launch_ts,
                "last_exit_ts": state.last_exit_ts,
                "log_path": str(state.log_path),
                "summary_path": str(state.summary_path),
            }
        )
    payload = {
        "generated_at_epoch": time.time(),
        "input_dir": str(args.input_dir.resolve()),
        "pipeline": args.pipeline,
        "num_shards": args.num_shards,
        "shard_indices": args.shard_indices,
        "max_active_workers": args.max_active_workers,
        "max_workers_per_gpu": args.max_workers_per_gpu,
        "mem_available_gib": mem_available_gib(),
        "active_workers": active,
        "external_workers": external_total,
        "external_workers_per_gpu": external_per_gpu,
        "total": total,
        "done": done,
        "pending": pending,
        "shards": per_shard,
    }
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    args.status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def handle_signal(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOG.warning("Received signal %s; stopping after current poll cycle.", signum)


def terminate_running_children(states: dict[int, ShardState]) -> None:
    for state in states.values():
        if state.process is None or state.process.poll() is not None:
            continue
        try:
            os.killpg(state.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            state.process.terminate()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    states = build_states(args)
    LOG.info(
        "Supervising shards %s with max_active_workers=%d max_workers_per_gpu=%d",
        args.shard_indices,
        args.max_active_workers,
        args.max_workers_per_gpu,
    )

    while True:
        managed_total, managed_per_gpu, managed_pids = managed_running_counts(states)
        external_total, external_per_gpu = external_worker_counts(managed_pids)
        progress_by_shard = {shard_index: shard_progress(state) for shard_index, state in states.items()}

        for state in states.values():
            if state.process is None:
                continue
            exit_code = state.process.poll()
            if exit_code is None:
                continue
            state.last_exit_code = exit_code
            state.last_exit_ts = time.time()
            LOG.info(
                "Shard %d exited with code %s (pending=%d)",
                state.shard_index,
                exit_code,
                progress_by_shard[state.shard_index][1],
            )
            state.process = None

        write_status(
            states=states,
            args=args,
            external_total=external_total,
            external_per_gpu=external_per_gpu,
        )

        remaining = sum(pending for _, pending in progress_by_shard.values())
        if remaining == 0:
            LOG.info("All supervised shards completed.")
            return

        if STOP_REQUESTED:
            terminate_running_children(states)
            return

        managed_total, managed_per_gpu, managed_pids = managed_running_counts(states)
        external_total, external_per_gpu = external_worker_counts(managed_pids)
        total_used = managed_total + external_total
        per_gpu_used = dict(managed_per_gpu)
        for gpu, count in external_per_gpu.items():
            per_gpu_used[gpu] = per_gpu_used.get(gpu, 0) + count

        now = time.time()
        candidates = []
        for shard_index, state in states.items():
            _done, shard_pending = progress_by_shard[shard_index]
            if shard_pending == 0:
                continue
            if state.process is not None and state.process.poll() is None:
                continue
            if state.last_exit_ts is not None and now - state.last_exit_ts < args.restart_delay_seconds:
                continue
            candidates.append((shard_pending, shard_index))
        candidates.sort(key=lambda item: (-item[0], item[1]))

        for shard_pending, shard_index in candidates:
            if total_used >= args.max_active_workers:
                break
            state = states[shard_index]
            gpu_used = per_gpu_used.get(state.gpu, 0)
            if gpu_used >= args.max_workers_per_gpu:
                continue
            launch_worker(state, args)
            if state.last_exit_ts is not None:
                state.restarts += 1
            total_used += 1
            per_gpu_used[state.gpu] = gpu_used + 1
            LOG.info(
                "Launched shard %d with %d pending videos | total_used=%d/%d gpu%d=%d/%d",
                shard_index,
                shard_pending,
                total_used,
                args.max_active_workers,
                state.gpu,
                per_gpu_used[state.gpu],
                args.max_workers_per_gpu,
            )

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
