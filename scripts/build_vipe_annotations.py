from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldarena.common.vipe_sidecars import (
    default_sidecar_dir,
    discover_videos,
    export_vipe_sidecar,
    mirrored_sidecar_dir,
    prioritize_videos,
    sidecar_complete,
)


LOG = logging.getLogger("build_vipe_annotations")
BOOTSTRAP_DIR = Path(__file__).resolve().parents[1] / "runtimes" / "vipe-runner" / "bootstrap"
SPECIAL_CACHED_VIDEOS = {
    "44c0e20531b5__pexels_30016241-hd_4096_2160_120fps.mp4",
    "a873483377ef__pixabay_213026-hd_1920_1080_30fps.mp4",
}
SPECIAL_LOW_THREAD_VIDEOS = SPECIAL_CACHED_VIDEOS
LOW_THREAD_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_MAX_THREADS": "2",
}
STRICT_LOW_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
}


@dataclass(slots=True)
class VideoResult:
    video_path: str
    output_dir: str
    status: str
    message: str | None = None
    quality_status: str | None = None
    frame_count: int | None = None
    pose_count: int | None = None
    depth_count: int | None = None
    camera_type: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViPE on WorldArena videos and export unified geometry sidecars."
    )
    parser.add_argument("--video", type=Path, help="Process one video file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Recursively discover and process videos from a directory.",
    )
    parser.add_argument("--output-dir", type=Path, help="Exact output directory for single-video mode.")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="External output root for batch mode; the input directory layout will be mirrored under it.",
    )
    parser.add_argument(
        "--vipe-root",
        type=Path,
        default=Path("/share/project/yesliu/hjd/model/WorldEvals/thirdparty/vipe"),
        help="ViPE repository root.",
    )
    parser.add_argument(
        "--vipe-python",
        type=Path,
        default=Path("/share/project/yesliu/miniconda3/envs/vipe/bin/python"),
        help="Python interpreter from a ViPE-compatible environment.",
    )
    parser.add_argument(
        "--pipeline",
        default="default",
        help="ViPE pipeline name to use. `default` matches the official ViPE default configuration.",
    )
    parser.add_argument(
        "--vipe-pretrained-models-dir",
        type=Path,
        help="Optional directory with official ViPE checkpoints exposed via VIPE_PRETRAINED_MODELS_DIR.",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many videos after prioritization.")
    parser.add_argument(
        "--num-shards",
        type=int,
        help="Split the discovered video list into this many stable shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Zero-based shard index to process when --num-shards > 1.",
    )
    parser.add_argument("--gpu", type=int, help="Optional CUDA_VISIBLE_DEVICES value for ViPE worker processes.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even when a complete sidecar already exists.",
    )
    parser.add_argument(
        "--keep-raw-artifacts",
        action="store_true",
        help="Keep raw ViPE artifacts under each exported sidecar.",
    )
    parser.add_argument("--summary-path", type=Path, help="Optional JSON file to write per-video quality summaries.")
    parser.add_argument(
        "--staging-root",
        type=Path,
        help="Optional root for temporary ViPE artifacts before final sidecar export.",
    )
    parser.add_argument(
        "--claim-root",
        type=Path,
        default=Path("/tmp/worldarena_vipe_claims"),
        help="Shared runtime-only directory used to claim videos across concurrent shard workers.",
    )
    parser.add_argument(
        "--prefer-cached-stream",
        action="store_true",
        help="Use the official cached raw_mp4 stream mode on the first attempt as a runtime-only stability knob.",
    )
    parser.add_argument(
        "--strict-low-thread",
        action="store_true",
        help="Force a 1-thread runtime environment for ViPE subprocesses as a runtime-only stability knob.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching inputs and resolved output directories without running ViPE.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    if (args.video is None) == (args.input_dir is None):
        parser.error("provide exactly one of --video or --input-dir")
    if args.video is not None and args.output_root is not None:
        parser.error("--output-root is only supported with --input-dir")
    if args.video is not None and args.output_dir is None:
        args.output_dir = default_sidecar_dir(args.video)
    if args.video is None and args.output_dir is not None:
        parser.error("--output-dir is only supported with --video")
    if args.num_shards is not None and args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.num_shards is None and args.shard_index is not None:
        parser.error("--shard-index requires --num-shards")
    if args.num_shards is not None and args.shard_index is None:
        parser.error("--shard-index is required when --num-shards is set")
    if args.shard_index is not None and not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num_shards)")
    return args


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepend_pythonpath(env: dict[str, str], path: Path) -> dict[str, str]:
    current = env.get("PYTHONPATH")
    if current:
        env["PYTHONPATH"] = f"{path}:{current}"
    else:
        env["PYTHONPATH"] = str(path)
    return env


def discover_batch(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.video is not None:
        return [(args.video.resolve(), args.output_dir.resolve())]

    videos = prioritize_videos(discover_videos(args.input_dir.resolve()))
    if args.limit is not None:
        videos = videos[: args.limit]
    if args.num_shards is not None:
        videos = videos[args.shard_index :: args.num_shards]
    output_root = args.output_root.resolve() if args.output_root is not None else args.input_dir.resolve()
    input_root = args.input_dir.resolve()
    return [(video, mirrored_sidecar_dir(video, input_root, output_root)) for video in videos]


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def claim_path_for(output_dir: Path, claim_root: Path) -> Path:
    digest = hashlib.sha1(str(output_dir).encode("utf-8"), usedforsecurity=False).hexdigest()
    return claim_root / f"{digest}.json"


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim_is_stale(claim_path: Path) -> bool:
    try:
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    pid = int(payload.get("pid", 0) or 0)
    return not pid_is_alive(pid)


def acquire_claim(output_dir: Path, claim_root: Path) -> Path | None:
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_path = claim_path_for(output_dir, claim_root)
    while True:
        try:
            fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if not claim_is_stale(claim_path):
                return None
            remove_path(claim_path)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "claimed_at_epoch": time.time(),
                    "output_dir": str(output_dir),
                },
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")
        return claim_path


def release_claim(claim_path: Path | None) -> None:
    if claim_path is None:
        return
    remove_path(claim_path)


def stash_existing_output(path: Path) -> Path | None:
    if not path.exists():
        return None
    suffix = f".delete-{os.getpid()}-{time.time_ns()}"
    target = path.with_name(path.name + suffix)
    path.rename(target)
    return target


def finalize_output(temp_output_dir: Path, output_dir: Path) -> None:
    staged_old = stash_existing_output(output_dir)
    temp_output_dir.rename(output_dir)
    if staged_old is not None:
        remove_path(staged_old)


def special_cached(video_path: Path) -> bool:
    return video_path.name in SPECIAL_CACHED_VIDEOS


def special_low_thread(video_path: Path) -> bool:
    return video_path.name in SPECIAL_LOW_THREAD_VIDEOS


def build_vipe_command(
    *,
    video_path: Path,
    raw_output_dir: Path,
    args: argparse.Namespace,
    use_cached_stream: bool,
) -> list[str]:
    command = [
        str(args.vipe_python.resolve()),
        str((args.vipe_root / "run.py").resolve()),
        f"pipeline={args.pipeline}",
        "streams=raw_mp4_stream",
        f"streams.base_path='{video_path}'",
        "pipeline.output.skip_exists=false",
        f"pipeline.output.path='{raw_output_dir}'",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
        "pipeline.output.save_slam_map=false",
    ]
    if use_cached_stream:
        command.append("streams.cached=true")
    return command


def run_vipe_worker(
    *,
    video_path: Path,
    raw_output_dir: Path,
    args: argparse.Namespace,
) -> None:
    prefer_cached = args.prefer_cached_stream or special_cached(video_path)
    attempt_specs: list[tuple[bool, bool]] = [
        (prefer_cached, args.strict_low_thread),
        (not prefer_cached, args.strict_low_thread),
        (prefer_cached, not args.strict_low_thread),
        (not prefer_cached, not args.strict_low_thread),
        (True, True),
    ]

    attempts: list[dict[str, bool]] = []
    seen_attempts: set[tuple[bool, bool]] = set()
    for use_cached_stream, strict_low_thread in attempt_specs:
        key = (use_cached_stream, strict_low_thread)
        if key in seen_attempts:
            continue
        seen_attempts.add(key)
        attempts.append(
            {
                "use_cached_stream": use_cached_stream,
                "strict_low_thread": strict_low_thread,
            }
        )

    for attempt_index, attempt in enumerate(attempts, start=1):
        env = dict(os.environ)
        env = prepend_pythonpath(env, BOOTSTRAP_DIR)
        if args.vipe_pretrained_models_dir is not None:
            env["VIPE_PRETRAINED_MODELS_DIR"] = str(args.vipe_pretrained_models_dir.resolve())
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if attempt["strict_low_thread"]:
            env.update(STRICT_LOW_THREAD_ENV)
        elif special_low_thread(video_path) or attempt["use_cached_stream"]:
            env.update(LOW_THREAD_ENV)
        command = build_vipe_command(
            video_path=video_path,
            raw_output_dir=raw_output_dir,
            args=args,
            use_cached_stream=attempt["use_cached_stream"],
        )
        LOG.info(
            "Running official ViPE CLI on %s with pipeline=%s%s%s",
            video_path,
            args.pipeline,
            " streams.cached=true" if attempt["use_cached_stream"] else "",
            " strict_low_thread=true" if attempt["strict_low_thread"] else "",
        )
        try:
            subprocess.run(command, check=True, env=env)
            return
        except subprocess.CalledProcessError as exc:
            if attempt_index == len(attempts):
                raise
            LOG.warning(
                "ViPE run failed for %s with return code %s; retrying with alternate runtime settings.",
                video_path,
                exc.returncode,
            )
            remove_path(raw_output_dir)


def process_video(
    *,
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> VideoResult:
    if sidecar_complete(output_dir) and not args.overwrite:
        LOG.info("Skipping existing sidecar %s", output_dir)
        return VideoResult(
            video_path=str(video_path),
            output_dir=str(output_dir),
            status="skipped",
            message="existing_complete_sidecar",
        )

    claim_path = acquire_claim(output_dir, args.claim_root.resolve())
    if claim_path is None and not args.overwrite:
        LOG.info("Skipping claimed sidecar %s", output_dir)
        return VideoResult(
            video_path=str(video_path),
            output_dir=str(output_dir),
            status="claimed",
            message="claimed_by_other_worker",
        )

    stage_base = args.staging_root.resolve() if args.staging_root is not None else output_dir.parent.resolve()
    stage_root = stage_base / f"{output_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    raw_output_dir = stage_root / "vipe_raw"
    temp_output_dir = output_dir.with_name(f"{output_dir.name}.copy-{os.getpid()}-{time.time_ns()}")
    remove_path(stage_root)
    remove_path(temp_output_dir)
    stage_root.mkdir(parents=True, exist_ok=True)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if sidecar_complete(output_dir) and not args.overwrite:
            LOG.info("Skipping existing sidecar %s", output_dir)
            return VideoResult(
                video_path=str(video_path),
                output_dir=str(output_dir),
                status="skipped",
                message="existing_complete_sidecar",
            )
        run_vipe_worker(video_path=video_path, raw_output_dir=raw_output_dir, args=args)
        export_summary = export_vipe_sidecar(
            video_path=video_path,
            artifact_root=raw_output_dir,
            output_dir=temp_output_dir,
            vipe_root=args.vipe_root.resolve(),
            pipeline_name=args.pipeline,
            vipe_python=args.vipe_python.resolve(),
        )
        if args.keep_raw_artifacts:
            shutil.copytree(raw_output_dir, temp_output_dir / "_vipe_raw", dirs_exist_ok=True)
        finalize_output(temp_output_dir, output_dir)
        LOG.info(
            "Completed %s | status=%s | frames=%s | poses=%s | depth=%s | out=%s",
            video_path.name,
            export_summary["status"],
            export_summary["frame_count"],
            export_summary["pose_count"],
            export_summary["depth_count"],
            output_dir,
        )
        return VideoResult(
            video_path=str(video_path),
            output_dir=str(output_dir),
            status="completed",
            quality_status=str(export_summary["status"]),
            frame_count=int(export_summary["frame_count"]),
            pose_count=int(export_summary["pose_count"]),
            depth_count=int(export_summary["depth_count"]),
            camera_type=str(export_summary["camera_type"]),
        )
    finally:
        release_claim(claim_path)
        remove_path(stage_root)
        remove_path(temp_output_dir)


def write_summary(path: Path, results: list[VideoResult]) -> None:
    ensure_parent(path)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    payload = {
        "generated_at_epoch": time.time(),
        "video_count": len(results),
        "counts": counts,
        "videos": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    jobs = discover_batch(args)
    if args.video is None and args.num_shards is not None:
        LOG.info("Selected %d video(s) for shard %d/%d", len(jobs), args.shard_index, args.num_shards)
    else:
        LOG.info("Selected %d video(s)", len(jobs))

    if args.dry_run:
        for video_path, output_dir in jobs:
            print(f"{video_path}\t{output_dir}")
        return

    results: list[VideoResult] = []
    had_failures = False
    for video_path, output_dir in jobs:
        try:
            result = process_video(video_path=video_path, output_dir=output_dir, args=args)
        except Exception as exc:  # noqa: BLE001
            had_failures = True
            LOG.exception("Failed to process %s", video_path)
            result = VideoResult(
                video_path=str(video_path),
                output_dir=str(output_dir),
                status="failed",
                message=str(exc),
            )
        results.append(result)
        if args.summary_path is not None:
            write_summary(args.summary_path.resolve(), results)

    if args.summary_path is not None:
        write_summary(args.summary_path.resolve(), results)

    completed = sum(result.status == "completed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    LOG.info(
        "Finished ViPE build: completed=%d skipped=%d failed=%d total=%d",
        completed,
        skipped,
        failed,
        len(results),
    )
    if had_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
