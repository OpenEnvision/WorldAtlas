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

from worldarena.common.vipe_sidecars import default_sidecar_dir, discover_videos, prioritize_videos, probe_video, sidecar_complete


LOG = logging.getLogger("supervise_vipe_tail")
DEFAULT_INPUT_DIR = Path("/share/project/yesliu/hjd/data/videos_test_split_4971")
DEFAULT_VIPE_ROOT = Path("/share/project/yesliu/hjd/model/WorldEvals/thirdparty/vipe")
DEFAULT_VIPE_PYTHON = Path("/share/project/yesliu/miniconda3/envs/vipe/bin/python")
DEFAULT_STAGE_ROOT = Path("/tmp/worldarena_vipe_stage")
DEFAULT_CLAIM_ROOT = Path("/tmp/worldarena_vipe_claims")
DEFAULT_LOG_DIR = Path("/tmp/vipe_tail_logs")
DEFAULT_SUMMARY_DIR = Path("/tmp/vipe_tail_summaries")
DEFAULT_STATUS_PATH = Path("/tmp/vipe_tail_status.json")
STOP_REQUESTED = False


@dataclass(slots=True)
class VideoJob:
    video_path: Path
    output_dir: Path
    shard_index: int
    frame_count: int
    fps: float
    width: int
    height: int
    est_cost: float
    attempts: int = 0
    process: subprocess.Popen[bytes] | None = None
    gpu: int | None = None
    last_launch_ts: float | None = None
    last_exit_ts: float | None = None
    last_exit_code: int | None = None
    log_path: Path | None = None
    summary_path: Path | None = None

    @property
    def name(self) -> str:
        return self.video_path.name


def parse_gpus(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("empty gpu list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-video tail supervisor for the remaining ViPE sidecars.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pipeline", default="default")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-workers-per-gpu", type=int, default=1)
    parser.add_argument("--max-active-workers", type=int)
    parser.add_argument("--restart-delay-seconds", type=float, default=5.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--claim-root", type=Path, default=DEFAULT_CLAIM_ROOT)
    parser.add_argument("--vipe-root", type=Path, default=DEFAULT_VIPE_ROOT)
    parser.add_argument("--vipe-python", type=Path, default=DEFAULT_VIPE_PYTHON)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_VIPE_PYTHON)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--min-est-cost", type=float)
    parser.add_argument("--max-est-cost", type=float)
    parser.add_argument("--cost-order", choices=("desc", "asc"), default="desc")
    parser.add_argument("--prefer-cached-stream", action="store_true")
    parser.add_argument("--strict-low-thread", action="store_true")
    parser.add_argument("--worker-log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    if args.pipeline != "default":
        parser.error("this supervisor is restricted to pipeline=default")
    args.gpu_list = parse_gpus(args.gpus)
    if args.max_workers_per_gpu <= 0:
        parser.error("--max-workers-per-gpu must be positive")
    if args.thread_count <= 0:
        parser.error("--thread-count must be positive")
    if args.min_est_cost is not None and args.min_est_cost < 0:
        parser.error("--min-est-cost must be non-negative")
    if args.max_est_cost is not None and args.max_est_cost < 0:
        parser.error("--max-est-cost must be non-negative")
    if (
        args.min_est_cost is not None
        and args.max_est_cost is not None
        and args.min_est_cost > args.max_est_cost
    ):
        parser.error("--min-est-cost must be <= --max-est-cost")
    if args.max_active_workers is None:
        args.max_active_workers = len(args.gpu_list) * args.max_workers_per_gpu
    if args.max_active_workers <= 0:
        parser.error("--max-active-workers must be positive")
    return args


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def handle_signal(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOG.warning("Received signal %s; stopping after current poll cycle.", signum)


def thread_env(thread_count: int) -> dict[str, str]:
    value = str(thread_count)
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_MAX_THREADS": value,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }


def probe_cost(video_path: Path) -> tuple[int, float, int, int, float]:
    probe = probe_video(video_path)
    cost = float(probe.frame_count) * float(probe.width) * float(probe.height) * max(probe.fps, 1.0)
    return probe.frame_count, probe.fps, probe.width, probe.height, cost


def build_jobs(args: argparse.Namespace) -> list[VideoJob]:
    videos = prioritize_videos(discover_videos(args.input_dir.resolve()))
    jobs: list[VideoJob] = []
    for index, video_path in enumerate(videos):
        output_dir = default_sidecar_dir(video_path)
        if sidecar_complete(output_dir):
            continue
        frame_count, fps, width, height, est_cost = probe_cost(video_path)
        if args.min_est_cost is not None and est_cost < args.min_est_cost:
            continue
        if args.max_est_cost is not None and est_cost > args.max_est_cost:
            continue
        jobs.append(
            VideoJob(
                video_path=video_path,
                output_dir=output_dir,
                shard_index=index % 40,
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                est_cost=est_cost,
            )
        )
    if args.cost_order == "asc":
        jobs.sort(key=lambda job: (job.est_cost, job.frame_count, str(job.video_path)))
    else:
        jobs.sort(key=lambda job: (-job.est_cost, -job.frame_count, str(job.video_path)))
    return jobs


def worker_command(job: VideoJob, gpu: int, args: argparse.Namespace) -> list[str]:
    command = [
        str(args.worker_python.resolve()),
        str((REPO_ROOT / "scripts" / "build_vipe_annotations.py").resolve()),
        "--video",
        str(job.video_path),
        "--output-dir",
        str(job.output_dir),
        "--pipeline",
        args.pipeline,
        "--gpu",
        str(gpu),
        "--summary-path",
        str(job.summary_path.resolve()),
        "--staging-root",
        str(args.staging_root.resolve()),
        "--claim-root",
        str(args.claim_root.resolve()),
        "--vipe-root",
        str(args.vipe_root.resolve()),
        "--vipe-python",
        str(args.vipe_python.resolve()),
        "--log-level",
        args.worker_log_level,
    ]
    if args.prefer_cached_stream:
        command.append("--prefer-cached-stream")
    if args.strict_low_thread:
        command.append("--strict-low-thread")
    return command


def launch_job(job: VideoJob, gpu: int, args: argparse.Namespace) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(thread_env(args.thread_count))
    job.gpu = gpu
    job.log_path = args.log_dir / f"{job.video_path.stem}__gpu{gpu}.log"
    job.summary_path = args.summary_dir / f"{job.video_path.stem}__gpu{gpu}.json"
    command = worker_command(job, gpu, args)
    with job.log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} shard={job.shard_index} attempts={job.attempts + 1} ===\n"
        )
        handle.flush()
        job.process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    job.attempts += 1
    job.last_launch_ts = time.time()
    LOG.info(
        "Started %s on gpu %d | attempts=%d | shard=%d | %dx%d @ %.2ffps | frames=%d",
        job.name,
        gpu,
        job.attempts,
        job.shard_index,
        job.width,
        job.height,
        job.fps,
        job.frame_count,
    )


def job_running(job: VideoJob) -> bool:
    return job.process is not None and job.process.poll() is None


def mem_available_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        return -1.0
    return -1.0


def write_status(
    *,
    args: argparse.Namespace,
    jobs: list[VideoJob],
    done_count: int,
) -> None:
    active = [job for job in jobs if job_running(job)]
    pending = [job for job in jobs if not sidecar_complete(job.output_dir)]
    payload = {
        "generated_at_epoch": time.time(),
        "input_dir": str(args.input_dir.resolve()),
        "pipeline": args.pipeline,
        "gpus": args.gpu_list,
        "max_active_workers": args.max_active_workers,
        "max_workers_per_gpu": args.max_workers_per_gpu,
        "thread_count": args.thread_count,
        "min_est_cost": args.min_est_cost,
        "max_est_cost": args.max_est_cost,
        "cost_order": args.cost_order,
        "mem_available_gib": mem_available_gib(),
        "done": done_count,
        "pending": len(pending),
        "active_workers": len(active),
        "videos": [
            {
                "video_path": str(job.video_path),
                "output_dir": str(job.output_dir),
                "shard_index": job.shard_index,
                "frame_count": job.frame_count,
                "fps": job.fps,
                "width": job.width,
                "height": job.height,
                "attempts": job.attempts,
                "gpu": job.gpu,
                "pid": job.process.pid if job_running(job) else None,
                "last_launch_ts": job.last_launch_ts,
                "last_exit_ts": job.last_exit_ts,
                "last_exit_code": job.last_exit_code,
                "log_path": str(job.log_path) if job.log_path is not None else None,
                "summary_path": str(job.summary_path) if job.summary_path is not None else None,
            }
            for job in pending
        ],
    }
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    args.status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def terminate_children(jobs: list[VideoJob]) -> None:
    for job in jobs:
        if not job_running(job):
            continue
        try:
            os.killpg(job.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    jobs = build_jobs(args)
    if not jobs:
        LOG.info("No pending ViPE sidecars found.")
        write_status(args=args, jobs=[], done_count=0)
        return

    total_count = len(jobs)
    done_count = 0
    LOG.info(
        "Supervising %d pending videos on gpus=%s with max_active_workers=%d max_workers_per_gpu=%d thread_count=%d",
        total_count,
        args.gpu_list,
        args.max_active_workers,
        args.max_workers_per_gpu,
        args.thread_count,
    )

    while True:
        for job in jobs:
            if job.process is None:
                continue
            exit_code = job.process.poll()
            if exit_code is None:
                continue
            job.last_exit_code = exit_code
            job.last_exit_ts = time.time()
            job.process = None
            if sidecar_complete(job.output_dir):
                LOG.info("Completed %s after %d attempt(s).", job.name, job.attempts)
            else:
                LOG.warning(
                    "Job exited without a complete sidecar: %s | exit_code=%s | attempts=%d",
                    job.name,
                    exit_code,
                    job.attempts,
                )

        done_count = sum(1 for job in jobs if sidecar_complete(job.output_dir))
        write_status(args=args, jobs=jobs, done_count=done_count)
        if done_count >= total_count:
            LOG.info("All pending tail videos completed.")
            return

        if STOP_REQUESTED:
            terminate_children(jobs)
            return

        active_total = 0
        active_per_gpu = {gpu: 0 for gpu in args.gpu_list}
        for job in jobs:
            if not job_running(job):
                continue
            active_total += 1
            active_per_gpu[job.gpu] = active_per_gpu.get(job.gpu, 0) + 1

        now = time.time()
        for job in jobs:
            if sidecar_complete(job.output_dir):
                continue
            if job_running(job):
                continue
            if job.last_exit_ts is not None and now - job.last_exit_ts < args.restart_delay_seconds:
                continue
            if active_total >= args.max_active_workers:
                break
            chosen_gpu = None
            chosen_load = None
            for gpu in args.gpu_list:
                gpu_load = active_per_gpu.get(gpu, 0)
                if gpu_load >= args.max_workers_per_gpu:
                    continue
                if chosen_gpu is None or gpu_load < chosen_load:
                    chosen_gpu = gpu
                    chosen_load = gpu_load
            if chosen_gpu is None:
                break
            launch_job(job, chosen_gpu, args)
            active_total += 1
            active_per_gpu[chosen_gpu] = active_per_gpu.get(chosen_gpu, 0) + 1

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
