from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldarena.common.vipe_sidecars import default_sidecar_dir, discover_videos, probe_video, sidecar_complete


DEFAULT_INPUT_DIR = Path("/share/project/yesliu/hjd/data/videos_test_split_4971")
DEFAULT_PROXY_ROOT = Path("/videos_test_split_4971_video_proxies_1080p24_20s")
DEFAULT_BACKUP_ROOT = Path("/videos_test_split_4971_video_backups_original_tail")
DEFAULT_MANIFEST_PATH = REPO_ROOT / ".cache" / "vipe_tail_proxy_manifest.json"


@dataclass(slots=True)
class ProxyResult:
    video_path: str
    backup_path: str
    proxy_path: str
    status: str
    original_width: int | None = None
    original_height: int | None = None
    original_fps: float | None = None
    original_duration_sec: float | None = None
    proxy_width: int | None = None
    proxy_height: int | None = None
    proxy_fps: float | None = None
    proxy_duration_sec: float | None = None
    message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare low-cost proxy videos for pending ViPE tail samples.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--proxy-root", type=Path, default=DEFAULT_PROXY_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=1080)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--max-duration-sec", type=float, default=20.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--jobs", type=int, default=min(16, max(1, (os.cpu_count() or 16) // 8)))
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def discover_pending_videos(input_dir: Path) -> list[Path]:
    return [
        path
        for path in discover_videos(input_dir.resolve())
        if not sidecar_complete(default_sidecar_dir(path))
    ]


def proxy_path_for(video_path: Path, input_root: Path, proxy_root: Path) -> Path:
    return proxy_root / video_path.relative_to(input_root)


def backup_path_for(video_path: Path, input_root: Path, backup_root: Path) -> Path:
    return backup_root / video_path.relative_to(input_root)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def transcode_proxy(
    *,
    source_path: Path,
    proxy_path: Path,
    target_width: int,
    target_height: int,
    target_fps: float,
    max_duration_sec: float,
    crf: int,
    preset: str,
    ffmpeg_threads: int,
) -> None:
    ensure_parent(proxy_path)
    temp_path = proxy_path.with_name(
        f"{proxy_path.stem}.tmp-{os.getpid()}-{time.time_ns()}{proxy_path.suffix}"
    )
    if temp_path.exists():
        temp_path.unlink()
    fps_text = f"{target_fps:.6f}".rstrip("0").rstrip(".")
    filter_text = (
        f"fps={fps_text},"
        f"scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-nostdin",
        "-threads",
        str(ffmpeg_threads),
        "-i",
        str(source_path),
        "-t",
        str(max_duration_sec),
        "-an",
        "-vf",
        filter_text,
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(temp_path),
    ]
    subprocess.run(command, check=True)
    temp_path.rename(proxy_path)


def approx_equal(lhs: float, rhs: float, tol: float) -> bool:
    return abs(lhs - rhs) <= tol


def prepare_one(video_path: Path, args: argparse.Namespace) -> ProxyResult:
    video_path = video_path.absolute()
    proxy_path = proxy_path_for(video_path, args.input_dir.resolve(), args.proxy_root.resolve())
    backup_path = backup_path_for(video_path, args.input_dir.resolve(), args.backup_root.resolve())

    source_path = backup_path if backup_path.exists() else video_path
    original_probe = probe_video(source_path)
    original_duration = float(original_probe.frame_count) / float(original_probe.fps)

    if not proxy_path.exists() or args.overwrite:
        if proxy_path.exists():
            proxy_path.unlink()
        transcode_proxy(
            source_path=source_path,
            proxy_path=proxy_path,
            target_width=args.target_width,
            target_height=args.target_height,
            target_fps=args.target_fps,
            max_duration_sec=args.max_duration_sec,
            crf=args.crf,
            preset=args.preset,
            ffmpeg_threads=args.ffmpeg_threads,
        )

    proxy_probe = probe_video(proxy_path)
    proxy_duration = float(proxy_probe.frame_count) / float(proxy_probe.fps)
    if proxy_probe.width != args.target_width or proxy_probe.height != args.target_height:
        raise RuntimeError(f"proxy resolution mismatch for {video_path}: {proxy_probe.width}x{proxy_probe.height}")
    if not approx_equal(proxy_probe.fps, args.target_fps, 0.2):
        raise RuntimeError(f"proxy fps mismatch for {video_path}: {proxy_probe.fps}")
    if proxy_duration > args.max_duration_sec + 0.5:
        raise RuntimeError(f"proxy duration mismatch for {video_path}: {proxy_duration}")

    if video_path.is_symlink():
        current_target = video_path.resolve()
        if current_target != proxy_path.resolve():
            video_path.unlink()
            video_path.symlink_to(proxy_path)
        status = "already_linked"
    else:
        ensure_parent(backup_path)
        if not backup_path.exists():
            shutil.move(str(video_path), str(backup_path))
        elif video_path.exists():
            raise RuntimeError(f"backup already exists but original file still present: {video_path}")
        video_path.symlink_to(proxy_path)
        status = "prepared"

    return ProxyResult(
        video_path=str(video_path),
        backup_path=str(backup_path),
        proxy_path=str(proxy_path),
        status=status,
        original_width=int(original_probe.width),
        original_height=int(original_probe.height),
        original_fps=float(original_probe.fps),
        original_duration_sec=original_duration,
        proxy_width=int(proxy_probe.width),
        proxy_height=int(proxy_probe.height),
        proxy_fps=float(proxy_probe.fps),
        proxy_duration_sec=proxy_duration,
    )


def write_manifest(path: Path, args: argparse.Namespace, results: list[ProxyResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    payload = {
        "generated_at_epoch": time.time(),
        "input_dir": str(args.input_dir.resolve()),
        "proxy_root": str(args.proxy_root.resolve()),
        "backup_root": str(args.backup_root.resolve()),
        "target_width": args.target_width,
        "target_height": args.target_height,
        "target_fps": args.target_fps,
        "max_duration_sec": args.max_duration_sec,
        "crf": args.crf,
        "preset": args.preset,
        "jobs": args.jobs,
        "video_count": len(results),
        "counts": counts,
        "videos": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    pending = discover_pending_videos(args.input_dir)
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        write_manifest(args.manifest_path, args, [])
        print("No pending videos require proxy preparation.")
        return

    results: list[ProxyResult] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(prepare_one, video_path, args): video_path for video_path in pending}
        for future in as_completed(futures):
            video_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = ProxyResult(
                    video_path=str(video_path),
                    backup_path=str(backup_path_for(video_path, args.input_dir.resolve(), args.backup_root.resolve())),
                    proxy_path=str(proxy_path_for(video_path, args.input_dir.resolve(), args.proxy_root.resolve())),
                    status="failed",
                    message=str(exc),
                )
            results.append(result)
            write_manifest(args.manifest_path, args, results)

    write_manifest(args.manifest_path, args, results)
    failures = [result for result in results if result.status == "failed"]
    if failures:
        raise SystemExit(f"failed to prepare {len(failures)} proxy video(s)")


if __name__ == "__main__":
    main()
