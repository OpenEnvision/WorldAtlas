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
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOG = logging.getLogger("build_vipe_pose_only_annotations")
BOOTSTRAP_DIR = REPO_ROOT / "runtimes" / "vipe-runner" / "bootstrap"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
POSE_REQUIRED = (
    "timestamps.npy",
    "frame_indices.npy",
    "poses.npy",
    "poses_c2w.npy",
    "poses_w2c.npy",
    "intrinsics.npy",
    "camera_type.txt",
    "quality_report.json",
)


@dataclass(frozen=True, slots=True)
class VideoProbe:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class VipeArtifactPaths:
    root: Path
    stem: str

    @property
    def pose_path(self) -> Path:
        return self.root / "pose" / f"{self.stem}.npz"

    @property
    def intrinsics_path(self) -> Path:
        return self.root / "intrinsics" / f"{self.stem}.npz"

    @property
    def camera_type_path(self) -> Path:
        return self.root / "intrinsics" / f"{self.stem}_camera.txt"


@dataclass(slots=True)
class PoseOnlyResult:
    video_path: str
    output_dir: str
    flashworld_output_dir: str
    status: str
    message: str | None = None
    quality_status: str | None = None
    frame_count: int | None = None
    pose_count: int | None = None
    intrinsics_count: int | None = None
    camera_type: str | None = None
    elapsed_seconds: float | None = None


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def probe_video(path: Path) -> VideoProbe:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count <= 0 or fps <= 0.0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"invalid video probe for {path}: frames={frame_count} fps={fps} size={width}x{height}"
        )
    return VideoProbe(frame_count=frame_count, fps=fps, width=width, height=height)


def discover_videos(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if is_video_file(root) else []
    return sorted(path for path in root.rglob("*") if is_video_file(path))


def prioritize_videos(paths: list[Path]) -> list[Path]:
    def key(path: Path) -> tuple[int, int, str]:
        normalized = path.as_posix()
        subset_rank = 0 if "/static/photorealistic/" in normalized else 1
        duration_rank = 0 if "/0-5s/" in normalized else 1 if "/5-10s/" in normalized else 2
        return subset_rank, duration_rank, normalized

    return sorted(paths, key=key)


def default_sidecar_dir(video_path: Path) -> Path:
    return video_path.parent / "annotations_vipe" / f"{video_path.stem}_vipe_ann"


def mirrored_sidecar_dir(video_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_parent = video_path.parent.relative_to(input_root)
    return output_root / relative_parent / "annotations_vipe" / f"{video_path.stem}_vipe_ann"


def read_vipe_pose_artifacts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    indices = np.asarray(payload["inds"], dtype=np.int32)
    poses = np.asarray(payload["data"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"unexpected pose shape in {path}: {poses.shape}")
    return indices, poses


def read_vipe_intrinsics_artifacts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    indices = np.asarray(payload["inds"], dtype=np.int32)
    intrinsics = np.asarray(payload["data"], dtype=np.float32)
    if intrinsics.ndim != 2 or intrinsics.shape[1] < 4:
        raise ValueError(f"unexpected intrinsics shape in {path}: {intrinsics.shape}")
    return indices, intrinsics[:, :4]


def read_vipe_camera_types(path: Path, indices: np.ndarray) -> dict[int, str]:
    if not path.exists():
        return {int(index): "PINHOLE" for index in indices.tolist()}
    camera_types: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index_text, camera_type = line.split(":", 1)
        camera_types[int(index_text.strip())] = camera_type.strip()
    return camera_types


def intrinsics_vec_to_matrix(vec: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(value) for value in vec[:4]]
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def intrinsics_sequence_to_matrices(intrinsics: np.ndarray) -> np.ndarray:
    return np.stack([intrinsics_vec_to_matrix(vec) for vec in intrinsics], axis=0).astype(
        np.float32
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViPE without depth and export pose/intrinsics sidecars."
    )
    parser.add_argument("--video", type=Path, help="Process one video file.")
    parser.add_argument("--input-dir", type=Path, help="Recursively process videos from a directory.")
    parser.add_argument("--output-dir", type=Path, help="Exact output directory for single-video mode.")
    parser.add_argument(
        "--flashworld-output-dir",
        type=Path,
        help="Exact FlashWorld/OpenGL output directory for single-video mode.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="External output root for batch mode; input layout is mirrored under it.",
    )
    parser.add_argument(
        "--vipe-root",
        type=Path,
        default=Path("/share/project/yesliu/hjd/model/WorldEvals/thirdparty/vipe"),
    )
    parser.add_argument(
        "--vipe-python",
        type=Path,
        default=Path("/share/project/yesliu/miniconda3/envs/vipe/bin/python"),
    )
    parser.add_argument("--pipeline", default="default")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--gpu", type=int, help="CUDA_VISIBLE_DEVICES value for the ViPE subprocess.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw-artifacts", action="store_true")
    parser.add_argument("--summary-jsonl", type=Path)
    parser.add_argument("--staging-root", type=Path, default=REPO_ROOT / ".cache" / "vipe_pose_only_stage")
    parser.add_argument(
        "--claim-root",
        type=Path,
        default=Path("/tmp/worldarena_vipe_pose_only_claims"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if (args.video is None) == (args.input_dir is None):
        parser.error("provide exactly one of --video or --input-dir")
    if args.video is not None and args.output_root is not None:
        parser.error("--output-root is only supported with --input-dir")
    if args.video is not None and args.output_dir is None:
        args.output_dir = default_sidecar_dir(args.video)
    if args.video is not None and args.flashworld_output_dir is None:
        args.flashworld_output_dir = flashworld_sidecar_dir(args.output_dir)
    if args.video is None and (args.output_dir is not None or args.flashworld_output_dir is not None):
        parser.error("--output-dir and --flashworld-output-dir are only supported with --video")
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


def flashworld_sidecar_dir(output_dir: Path) -> Path:
    parent = output_dir.parent
    if parent.name == "annotations_vipe":
        return parent.with_name("annotations_vipe_flashworld") / f"{output_dir.name}_opengl"
    return output_dir.with_name(f"{output_dir.name}_opengl")


def pose_sidecar_complete(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in POSE_REQUIRED)


def discover_batch(args: argparse.Namespace) -> list[tuple[Path, Path, Path]]:
    if args.video is not None:
        return [
            (
                args.video.resolve(),
                args.output_dir.resolve(),
                args.flashworld_output_dir.resolve(),
            )
        ]

    input_root = args.input_dir.resolve()
    videos = prioritize_videos(discover_videos(input_root))
    if args.limit is not None:
        videos = videos[: args.limit]
    if args.num_shards is not None:
        videos = videos[args.shard_index :: args.num_shards]
    output_root = args.output_root.resolve() if args.output_root is not None else input_root
    jobs: list[tuple[Path, Path, Path]] = []
    for video in videos:
        output_dir = mirrored_sidecar_dir(video, input_root, output_root)
        jobs.append((video, output_dir, flashworld_sidecar_dir(output_dir)))
    return jobs


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_dir(path: Path) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except FileExistsError:
        if not path.is_dir():
            raise


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


def claim_is_stale(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return not pid_is_alive(int(payload.get("pid", 0) or 0))


def acquire_claim(output_dir: Path, claim_root: Path) -> Path | None:
    ensure_dir(claim_root)
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
                {"pid": os.getpid(), "claimed_at_epoch": time.time(), "output_dir": str(output_dir)},
                handle,
            )
            handle.write("\n")
        return claim_path


def release_claim(path: Path | None) -> None:
    if path is not None:
        remove_path(path)


def stash_existing_output(path: Path) -> Path | None:
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.delete-{os.getpid()}-{time.time_ns()}")
    path.rename(target)
    return target


def finalize_output(temp_output_dir: Path, output_dir: Path) -> None:
    ensure_dir(output_dir.parent)
    staged_old = stash_existing_output(output_dir)
    temp_output_dir.rename(output_dir)
    if staged_old is not None:
        remove_path(staged_old)


def prepend_pythonpath(env: dict[str, str], path: Path) -> dict[str, str]:
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{path}:{current}" if current else str(path)
    return env


def build_vipe_command(video_path: Path, raw_output_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        str(args.vipe_python.resolve()),
        str((args.vipe_root / "run.py").resolve()),
        f"pipeline={args.pipeline}",
        "streams=raw_mp4_stream",
        f"streams.base_path='{video_path}'",
        "streams.cached=true",
        "pipeline.output.skip_exists=false",
        f"pipeline.output.path='{raw_output_dir}'",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
        "pipeline.output.save_slam_map=false",
        "pipeline.init.instance=null",
        "pipeline.slam.keyframe_depth=null",
        "pipeline.post.depth_align_model=null",
    ]


def run_vipe(video_path: Path, raw_output_dir: Path, args: argparse.Namespace) -> None:
    env = prepend_pythonpath(dict(os.environ), BOOTSTRAP_DIR)
    env["TOKENIZERS_PARALLELISM"] = "false"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    command = build_vipe_command(video_path, raw_output_dir, args)
    LOG.info("Running pose-only ViPE on %s", video_path)
    subprocess.run(command, check=True, env=env, cwd=str(args.vipe_root.resolve()))


def opencv_c2w_to_opengl_c2w(poses_c2w: np.ndarray) -> np.ndarray:
    cv_to_gl = np.eye(4, dtype=np.float32)
    cv_to_gl[1, 1] = -1.0
    cv_to_gl[2, 2] = -1.0
    return (poses_c2w.astype(np.float32) @ cv_to_gl).astype(np.float32)


def pose_quality_report(
    *,
    video_path: Path,
    raw_output_dir: Path,
    probe_frame_count: int,
    fps: float,
    width: int,
    height: int,
    indices: np.ndarray,
    poses_c2w: np.ndarray,
    poses_w2c: np.ndarray,
    intrinsics: np.ndarray,
    camera_types: list[str],
    pose_layout: str,
    note: str,
    conversion: str | None = None,
) -> dict[str, object]:
    aligned_count = int(poses_c2w.shape[0])
    inverse_residual = float(
        np.max(np.abs(np.matmul(poses_c2w, poses_w2c) - np.eye(4, dtype=np.float32)))
    )
    ortho_residual = float(
        np.max(
            np.abs(
                np.matmul(poses_c2w[:, :3, :3], np.transpose(poses_c2w[:, :3, :3], (0, 2, 1)))
                - np.eye(3, dtype=np.float32)
            )
        )
    )
    rotation_det = np.linalg.det(poses_c2w[:, :3, :3]).astype(np.float32)
    translation = poses_c2w[:, :3, 3]
    report: dict[str, object] = {
        "status": "passed",
        "source_video": str(video_path),
        "raw_vipe_artifacts": str(raw_output_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame_count": aligned_count,
        "source_video_frame_count": int(probe_frame_count),
        "coverage_ratio": float(aligned_count / max(int(probe_frame_count), 1)),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "pose_layout": pose_layout,
        "intrinsics_layout": "intrinsics.npy is per-frame 3x3 OpenCV intrinsics matrix",
        "frame_index_min": int(indices.min()),
        "frame_index_max": int(indices.max()),
        "translation_min": [float(value) for value in translation.min(axis=0)],
        "translation_max": [float(value) for value in translation.max(axis=0)],
        "translation_delta_first_last": [
            float(value) for value in (translation[-1] - translation[0])
        ],
        "pose_count": aligned_count,
        "intrinsics_count": int(intrinsics.shape[0]),
        "camera_type": camera_types[0] if len(set(camera_types)) == 1 else "VARYING",
        "camera_types_observed": sorted(set(camera_types)),
        "has_nan": bool(
            np.isnan(poses_c2w).any() or np.isnan(poses_w2c).any() or np.isnan(intrinsics).any()
        ),
        "has_inf": bool(
            np.isinf(poses_c2w).any() or np.isinf(poses_w2c).any() or np.isinf(intrinsics).any()
        ),
        "pose_inverse_residual_max": inverse_residual,
        "rotation_orthonormality_residual_max": ortho_residual,
        "rotation_det_min": float(rotation_det.min()),
        "rotation_det_max": float(rotation_det.max()),
        "note": note,
    }
    if conversion is not None:
        report["conversion"] = conversion
        report["conversion_matrix_right_multiply"] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    return report


def write_pose_sidecar(
    *,
    output_dir: Path,
    video_path: Path,
    raw_output_dir: Path,
    probe_frame_count: int,
    fps: float,
    width: int,
    height: int,
    indices: np.ndarray,
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    camera_types: list[str],
    pose_layout: str,
    note: str,
    conversion: str | None = None,
) -> dict[str, object]:
    poses_w2c = np.linalg.inv(poses_c2w).astype(np.float32)
    ensure_dir(output_dir)
    np.save(output_dir / "frame_indices.npy", indices.astype(np.int32))
    np.save(output_dir / "timestamps.npy", (indices.astype(np.float32) / float(fps)).astype(np.float32))
    np.save(output_dir / "poses.npy", poses_c2w.astype(np.float32))
    np.save(output_dir / "poses_c2w.npy", poses_c2w.astype(np.float32))
    np.save(output_dir / "poses_w2c.npy", poses_w2c)
    np.save(output_dir / "intrinsics.npy", intrinsics.astype(np.float32))
    with (output_dir / "camera_type.txt").open("w", encoding="utf-8") as handle:
        for index, camera_type in zip(indices.tolist(), camera_types, strict=True):
            handle.write(f"{int(index)}: {camera_type}\n")
    report = pose_quality_report(
        video_path=video_path,
        raw_output_dir=raw_output_dir,
        probe_frame_count=probe_frame_count,
        fps=fps,
        width=width,
        height=height,
        indices=indices,
        poses_c2w=poses_c2w,
        poses_w2c=poses_w2c,
        intrinsics=intrinsics,
        camera_types=camera_types,
        pose_layout=pose_layout,
        note=note,
        conversion=conversion,
    )
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def export_pose_only_sidecars(
    *,
    video_path: Path,
    raw_output_dir: Path,
    output_dir: Path,
    flashworld_output_dir: Path,
    vipe_root: Path,
    vipe_python: Path,
    pipeline_name: str,
) -> dict[str, object]:
    paths = VipeArtifactPaths(root=raw_output_dir, stem=video_path.stem)
    if not paths.pose_path.exists():
        raise FileNotFoundError(f"missing ViPE pose artifact: {paths.pose_path}")
    if not paths.intrinsics_path.exists():
        raise FileNotFoundError(f"missing ViPE intrinsics artifact: {paths.intrinsics_path}")

    probe = probe_video(video_path)
    pose_indices, poses_c2w = read_vipe_pose_artifacts(paths.pose_path)
    intr_indices, intrinsics_vec = read_vipe_intrinsics_artifacts(paths.intrinsics_path)
    common_indices = sorted(set(pose_indices.tolist()) & set(intr_indices.tolist()))
    if not common_indices:
        raise RuntimeError(f"no common pose/intrinsics frame indices found for {video_path}")

    pose_map = {int(index): pose for index, pose in zip(pose_indices.tolist(), poses_c2w, strict=True)}
    intr_map = {
        int(index): intr for index, intr in zip(intr_indices.tolist(), intrinsics_vec, strict=True)
    }
    aligned_indices = np.asarray(common_indices, dtype=np.int32)
    aligned_poses_c2w = np.stack([pose_map[index] for index in common_indices], axis=0).astype(
        np.float32
    )
    aligned_intrinsics = intrinsics_sequence_to_matrices(
        np.stack([intr_map[index] for index in common_indices], axis=0).astype(np.float32)
    )
    camera_types_by_index = read_vipe_camera_types(paths.camera_type_path, intr_indices)
    camera_types = [camera_types_by_index.get(int(index), "PINHOLE") for index in common_indices]

    base_note = (
        "Generated with official ViPE pipeline, overriding keyframe_depth and "
        "post depth_align_model to null; pose/intrinsics are from ViPE SLAM and GeoCalib. "
        f"repo_root={vipe_root}; python={vipe_python}; pipeline={pipeline_name}."
    )
    report = write_pose_sidecar(
        output_dir=output_dir,
        video_path=video_path,
        raw_output_dir=raw_output_dir,
        probe_frame_count=probe.frame_count,
        fps=probe.fps,
        width=probe.width,
        height=probe.height,
        indices=aligned_indices,
        poses_c2w=aligned_poses_c2w,
        intrinsics=aligned_intrinsics,
        camera_types=camera_types,
        pose_layout="poses.npy is per-frame 4x4 camera-to-world OpenCV matrix",
        note=base_note,
    )
    write_pose_sidecar(
        output_dir=flashworld_output_dir,
        video_path=video_path,
        raw_output_dir=raw_output_dir,
        probe_frame_count=probe.frame_count,
        fps=probe.fps,
        width=probe.width,
        height=probe.height,
        indices=aligned_indices,
        poses_c2w=opencv_c2w_to_opengl_c2w(aligned_poses_c2w),
        intrinsics=aligned_intrinsics,
        camera_types=camera_types,
        pose_layout=(
            "poses.npy is per-frame 4x4 camera-to-world OpenGL matrix converted from "
            "VIPE OpenCV c2w by right-multiplying diag(1,-1,-1)"
        ),
        note=base_note,
        conversion="opencv_c2w_to_flashworld_opengl_c2w",
    )
    return report


def process_video(
    *,
    video_path: Path,
    output_dir: Path,
    flashworld_output_dir: Path,
    args: argparse.Namespace,
) -> PoseOnlyResult:
    start = time.time()
    if (
        pose_sidecar_complete(output_dir)
        and pose_sidecar_complete(flashworld_output_dir)
        and not args.overwrite
    ):
        LOG.info("Skipping existing pose sidecars %s", output_dir)
        return PoseOnlyResult(
            video_path=str(video_path),
            output_dir=str(output_dir),
            flashworld_output_dir=str(flashworld_output_dir),
            status="skipped",
            message="existing_complete_sidecar",
            elapsed_seconds=time.time() - start,
        )

    claim_path = acquire_claim(output_dir, args.claim_root.resolve())
    if claim_path is None and not args.overwrite:
        return PoseOnlyResult(
            video_path=str(video_path),
            output_dir=str(output_dir),
            flashworld_output_dir=str(flashworld_output_dir),
            status="claimed",
            message="claimed_by_other_worker",
            elapsed_seconds=time.time() - start,
        )

    stage_root = args.staging_root.resolve() / f"{output_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    raw_output_dir = stage_root / "vipe_raw"
    temp_output_dir = output_dir.with_name(f"{output_dir.name}.copy-{os.getpid()}-{time.time_ns()}")
    temp_flashworld_output_dir = flashworld_output_dir.with_name(
        f"{flashworld_output_dir.name}.copy-{os.getpid()}-{time.time_ns()}"
    )
    remove_path(stage_root)
    remove_path(temp_output_dir)
    remove_path(temp_flashworld_output_dir)
    ensure_dir(stage_root)
    ensure_dir(temp_output_dir)
    ensure_dir(temp_flashworld_output_dir)

    try:
        run_vipe(video_path, raw_output_dir, args)
        report = export_pose_only_sidecars(
            video_path=video_path,
            raw_output_dir=raw_output_dir,
            output_dir=temp_output_dir,
            flashworld_output_dir=temp_flashworld_output_dir,
            vipe_root=args.vipe_root.resolve(),
            vipe_python=args.vipe_python.resolve(),
            pipeline_name=args.pipeline,
        )
        if args.keep_raw_artifacts:
            shutil.copytree(raw_output_dir, temp_output_dir / "_vipe_raw", dirs_exist_ok=True)
        finalize_output(temp_output_dir, output_dir)
        finalize_output(temp_flashworld_output_dir, flashworld_output_dir)
        elapsed = time.time() - start
        LOG.info(
            "Completed %s | frames=%s | out=%s | flashworld=%s | %.1fs",
            video_path.name,
            report["frame_count"],
            output_dir,
            flashworld_output_dir,
            elapsed,
        )
        return PoseOnlyResult(
            video_path=str(video_path),
            output_dir=str(output_dir),
            flashworld_output_dir=str(flashworld_output_dir),
            status="completed",
            quality_status=str(report["status"]),
            frame_count=int(report["frame_count"]),
            pose_count=int(report["pose_count"]),
            intrinsics_count=int(report["intrinsics_count"]),
            camera_type=str(report["camera_type"]),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.time() - start
        LOG.exception("Failed to process %s", video_path)
        return PoseOnlyResult(
            video_path=str(video_path),
            output_dir=str(output_dir),
            flashworld_output_dir=str(flashworld_output_dir),
            status="failed",
            message=str(exc),
            elapsed_seconds=elapsed,
        )
    finally:
        release_claim(claim_path)
        remove_path(stage_root)
        remove_path(temp_output_dir)
        remove_path(temp_flashworld_output_dir)


def append_summary(path: Path | None, result: PoseOnlyResult) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    jobs = discover_batch(args)
    LOG.info("Selected %d video(s)", len(jobs))
    if args.dry_run:
        for video, output_dir, flashworld_output_dir in jobs:
            print(f"{video}\t{output_dir}\t{flashworld_output_dir}")
        return 0

    counts: dict[str, int] = {}
    for video_path, output_dir, flashworld_output_dir in jobs:
        result = process_video(
            video_path=video_path,
            output_dir=output_dir,
            flashworld_output_dir=flashworld_output_dir,
            args=args,
        )
        counts[result.status] = counts.get(result.status, 0) + 1
        append_summary(args.summary_jsonl, result)
    LOG.info("Finished pose-only ViPE build: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
