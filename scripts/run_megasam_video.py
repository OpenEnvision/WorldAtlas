#!/usr/bin/env python3
"""Run MegaSAM's three inference stages for one video or frame directory.

MegaSAM's tracking demo writes its final artifacts relative to the repository
root.  This wrapper gives each benchmark sample an isolated cache directory and
hard-links the final artifacts into it, while retaining MegaSAM's expected
working directory during inference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VALID_STAGES = ("depth_anything", "unidepth", "tracking")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--frames-dir", type=Path)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mega-sam-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--stages",
        default=",".join(VALID_STAGES),
        help="Comma-separated subset of depth_anything,unidepth,tracking",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--depth-localhub", dest="depth_localhub", action="store_true")
    parser.add_argument("--no-depth-localhub", dest="depth_localhub", action="store_false")
    parser.set_defaults(depth_localhub=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None, env: dict[str, str], dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
    )


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        try:
            if source.samefile(target):
                return
        except OSError:
            # A broken symlink or a partially written target is replaced below.
            pass
        remove_path(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def link_tree(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        remove_path(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(os.readlink(path))
        else:
            link_or_copy(path, destination)


def materialize_frames(args: argparse.Namespace, frames_dir: Path, env: dict[str, str]) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = image_files(frames_dir)
    if existing and args.skip_existing:
        return len(existing)
    for path in existing:
        path.unlink()

    needs_window = args.start_frame != 0 or args.frame_stride != 1 or args.max_frames is not None
    if args.video is not None and not needs_window:
        ffmpeg = args.ffmpeg
        if ffmpeg is None:
            sibling = args.python.parent / "ffmpeg"
            ffmpeg = sibling if sibling.exists() else Path("ffmpeg")
        command = [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(args.video.resolve()),
            str(frames_dir / "%06d.png"),
        ]
        run(command, cwd=None, env=env, dry_run=args.dry_run)
        if args.dry_run:
            return 0
        count = len(image_files(frames_dir))
        if count == 0:
            raise RuntimeError(f"ffmpeg decoded no frames from {args.video}")
        return count

    if args.frames_dir is not None:
        source_frames = image_files(args.frames_dir.resolve())
    else:
        # Windowed inputs are decoded to a sibling staging directory first so
        # frame selection is deterministic and independent of ffmpeg filters.
        staging = frames_dir.parent / "decoded_all_frames"
        if staging.exists():
            remove_path(staging)
        staging.mkdir(parents=True)
        ffmpeg = args.ffmpeg
        if ffmpeg is None:
            sibling = args.python.parent / "ffmpeg"
            ffmpeg = sibling if sibling.exists() else Path("ffmpeg")
        run(
            [
                str(ffmpeg),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(args.video.resolve()),
                str(staging / "%06d.png"),
            ],
            cwd=None,
            env=env,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            return 0
        source_frames = image_files(staging)

    selected = source_frames[args.start_frame :: args.frame_stride]
    if args.max_frames is not None:
        selected = selected[: args.max_frames]
    if not selected:
        raise RuntimeError("frame selection produced no frames")
    for index, source_path in enumerate(selected, start=1):
        link_or_copy(source_path, frames_dir / f"{index:06d}{source_path.suffix.lower()}")
    return len(selected)


def count_outputs(directory: Path, suffixes: Iterable[str]) -> int:
    if not directory.exists():
        return 0
    suffix_set = set(suffixes)
    return sum(path.is_file() and path.suffix.lower() in suffix_set for path in directory.iterdir())


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be nonnegative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = sorted(set(stages) - set(VALID_STAGES))
    if unknown:
        raise ValueError(f"unknown stages: {unknown}")

    args.output_root = args.output_root.resolve()
    args.mega_sam_root = args.mega_sam_root.resolve()
    args.python = args.python.resolve()
    args.hf_home = args.hf_home.resolve()
    source_path = (args.video or args.frames_dir).resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not args.mega_sam_root.is_dir():
        raise FileNotFoundError(args.mega_sam_root)
    if not args.python.exists():
        raise FileNotFoundError(args.python)

    run_root = args.output_root / args.scene_name
    if args.overwrite and run_root.exists():
        remove_path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    frames_dir = run_root / "frames"
    depth_root = run_root / "depth_anything"
    depth_scene = depth_root / args.scene_name
    unidepth_root = run_root / "unidepth"
    unidepth_scene = unidepth_root / args.scene_name
    recon_scene = run_root / "reconstructions" / args.scene_name
    flow_scene = run_root / "cache_flow" / args.scene_name
    outputs_dir = run_root / "outputs"

    # A completed benchmark report can safely discard the large, reproducible
    # frame/depth/flow intermediates while retaining the final camera poses.
    # The report-side cache pruner writes this marker only after validating the
    # full report.  Honor it before decoding frames so resumed evaluations stay
    # cheap and do not recreate terabytes of intermediate data.
    pruned_complete = run_root / ".worldarena_megasam_pruned_complete.json"
    if (
        args.skip_existing
        and pruned_complete.is_file()
        and (recon_scene / "poses.npy").is_file()
        and (run_root / "run_config.json").is_file()
    ):
        print(
            f"[megasam] skip pruned complete scene: {args.scene_name}",
            flush=True,
        )
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["HF_HOME"] = str(args.hf_home)
    env.setdefault("TORCH_HOME", str(args.hf_home / "torch"))
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath_prefixes = [
        args.mega_sam_root / "UniDepth",
        args.mega_sam_root,
    ]
    pythonpath = os.pathsep.join(
        str(path) for path in pythonpath_prefixes if path.exists()
    )
    if pythonpath:
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = pythonpath + (os.pathsep + current if current else "")
    library_prefixes = [
        args.python.parent.parent / "lib",
        args.python.parent.parent / "lib" / "python3.10" / "site-packages" / "torch" / "lib",
    ]
    prefix = os.pathsep.join(str(path) for path in library_prefixes if path.exists())
    if prefix:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = prefix + (os.pathsep + current if current else "")

    frame_count = materialize_frames(args, frames_dir, env)

    metadata = {
        "scene": args.scene_name,
        "source": str(source_path),
        "frame_count": frame_count,
        "stages": stages,
        "gpu": str(args.gpu),
        "run_root": str(run_root),
        "paths": {
            "scene": args.scene_name,
            "frames": str(frames_dir),
            "depth": str(depth_root),
            "depth_scene": str(depth_scene),
            "unidepth": str(unidepth_root),
            "unidepth_scene": str(unidepth_scene),
            "recon_scene": str(recon_scene),
            "flow_scene": str(flow_scene),
            "cvd": str(run_root / "outputs_cvd"),
        },
        "pythonpath_prefix": [str(path) for path in pythonpath_prefixes if path.exists()],
        "ld_library_path_prefix": [str(path) for path in library_prefixes if path.exists()],
    }
    (run_root / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    depth_command = [
        str(args.python),
        "Depth-Anything/run_videos.py",
        "--encoder",
        "vitl",
        "--load-from",
        str(args.mega_sam_root / "Depth-Anything" / "checkpoints" / "depth_anything_vitl14.pth"),
        "--img-path",
        str(frames_dir),
        "--outdir",
        str(depth_scene),
    ]
    if args.depth_localhub:
        depth_command.append("--localhub")
    unidepth_command = [
        str(args.python),
        "UniDepth/scripts/demo_mega-sam.py",
        "--scene-name",
        args.scene_name,
        "--img-path",
        str(frames_dir),
        "--outdir",
        str(unidepth_root),
    ]
    tracking_command = [
        str(args.python),
        "camera_tracking_scripts/test_demo.py",
        "--datapath",
        str(frames_dir),
        "--weights",
        str(args.mega_sam_root / "checkpoints" / "megasam_final.pth"),
        "--scene_name",
        args.scene_name,
        "--mono_depth_path",
        str(depth_root),
        "--metric_depth_path",
        str(unidepth_root),
        "--disable_vis",
    ]

    if "depth_anything" in stages:
        complete = count_outputs(depth_scene, {".npy"}) >= frame_count
        if not (args.skip_existing and complete):
            depth_scene.mkdir(parents=True, exist_ok=True)
            run(depth_command, cwd=args.mega_sam_root, env=env, dry_run=args.dry_run)
    if "unidepth" in stages:
        complete = count_outputs(unidepth_scene, {".npz"}) >= frame_count
        if not (args.skip_existing and complete):
            unidepth_scene.mkdir(parents=True, exist_ok=True)
            run(unidepth_command, cwd=args.mega_sam_root, env=env, dry_run=args.dry_run)
    if "tracking" in stages:
        complete = (recon_scene / "poses.npy").exists()
        if not (args.skip_existing and complete):
            shared_recon = args.mega_sam_root / "reconstructions" / args.scene_name
            shared_flow = args.mega_sam_root / "cache_flow" / args.scene_name
            shared_output = args.mega_sam_root / "outputs" / f"{args.scene_name}_droid.npz"
            for path in (shared_recon, shared_flow, shared_output):
                if path.exists() or path.is_symlink():
                    remove_path(path)
            run(tracking_command, cwd=args.mega_sam_root, env=env, dry_run=args.dry_run)
            if not args.dry_run:
                if not (shared_recon / "poses.npy").exists():
                    raise FileNotFoundError(f"MegaSAM did not produce {shared_recon / 'poses.npy'}")
                link_tree(shared_recon, recon_scene)
                if shared_flow.exists():
                    link_tree(shared_flow, flow_scene)
                if shared_output.exists():
                    link_or_copy(shared_output, outputs_dir / shared_output.name)


if __name__ == "__main__":
    main()
