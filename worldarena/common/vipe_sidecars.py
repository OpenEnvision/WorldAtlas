"""Load ViPE annotation sidecars (depth, flow, segmentation) for metric evaluation."""

from __future__ import annotations

import json
import zipfile

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import OpenEXR


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
REQUIRED_SIDECAR_FILES = (
    "timestamps.npy",
    "frame_indices.npy",
    "poses_c2w.npy",
    "poses_w2c.npy",
    "intrinsics.npy",
    "depth_metric.npy",
    "depth_valid_mask.npz",
    "camera_model.json",
    "provenance.json",
    "quality_report.json",
)


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """Describe basic metadata required to align ViPE sidecars with a source video."""

    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class VipeArtifactPaths:
    """Resolve the standard ViPE artifact file layout for one processed video."""

    root: Path
    stem: str

    @property
    def pose_path(self) -> Path:
        """Pose path -> Path."""
        return self.root / "pose" / f"{self.stem}.npz"

    @property
    def intrinsics_path(self) -> Path:
        """Intrinsics path -> Path."""
        return self.root / "intrinsics" / f"{self.stem}.npz"

    @property
    def camera_type_path(self) -> Path:
        """Camera type path -> Path."""
        return self.root / "intrinsics" / f"{self.stem}_camera.txt"

    @property
    def depth_path(self) -> Path:
        """Depth path -> Path."""
        return self.root / "depth" / f"{self.stem}.zip"


def is_video_file(path: Path) -> bool:

    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def probe_video(path: Path) -> VideoProbe:
    """Read frame count, FPS, and frame size from a decodable video file."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count <= 0:
        raise RuntimeError(f"video has no decodable frames: {path}")
    if fps <= 0.0:
        raise RuntimeError(f"video reported a non-positive FPS: {path}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"video reported an invalid frame size: {path}")
    return VideoProbe(frame_count=frame_count, fps=fps, width=width, height=height)


def discover_videos(root: Path) -> list[Path]:
    """Recursively discover supported video files under a directory."""

    if root.is_file():
        return [root] if is_video_file(root) else []
    return sorted(path for path in root.rglob("*") if is_video_file(path))


def prioritize_videos(paths: Iterable[Path]) -> list[Path]:
    """Sort videos so static/photorealistic and shorter buckets are processed first."""

    def key(path: Path) -> tuple[int, int, str]:
        normalized = path.as_posix()
        subset_rank = 0 if "/static/photorealistic/" in normalized else 1
        duration_rank = 0 if "/0-5s/" in normalized else 1 if "/5-10s/" in normalized else 2
        return subset_rank, duration_rank, normalized

    return sorted(paths, key=key)


def default_sidecar_dir(video_path: Path) -> Path:
    """Return the default inline output directory for one video."""

    return video_path.parent / "annotations_vipe" / f"{video_path.stem}_vipe_ann"


def mirrored_sidecar_dir(video_path: Path, input_root: Path, output_root: Path) -> Path:
    """Mirror a source video under an external output root using the same sidecar naming."""

    relative_parent = video_path.parent.relative_to(input_root)
    return output_root / relative_parent / "annotations_vipe" / f"{video_path.stem}_vipe_ann"


def read_vipe_pose_artifacts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ViPE camera-to-world pose matrices and their source frame indices."""

    payload = np.load(path)
    indices = np.asarray(payload["inds"], dtype=np.int32)
    poses = np.asarray(payload["data"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"unexpected pose shape in {path}: {poses.shape}")
    return indices, poses


def read_vipe_intrinsics_artifacts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ViPE per-frame intrinsics vectors and their source frame indices."""

    payload = np.load(path)
    indices = np.asarray(payload["inds"], dtype=np.int32)
    intrinsics = np.asarray(payload["data"], dtype=np.float32)
    if intrinsics.ndim != 2 or intrinsics.shape[1] < 4:
        raise ValueError(f"unexpected intrinsics shape in {path}: {intrinsics.shape}")
    return indices, intrinsics[:, :4]


def read_vipe_camera_types(path: Path, indices: np.ndarray) -> dict[int, str]:
    """Load ViPE camera type annotations keyed by source frame index."""

    if not path.exists():
        return {int(index): "PINHOLE" for index in indices.tolist()}
    camera_types: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index_text, camera_type = line.split(":", 1)
        camera_types[int(index_text.strip())] = camera_type.strip()
    return camera_types


def list_vipe_depth_indices(path: Path) -> list[int]:
    """Read ViPE depth frame indices without decoding the EXR payloads."""

    with zipfile.ZipFile(path, "r") as archive:
        return [int(Path(name).stem) for name in sorted(archive.namelist())]


def read_vipe_depth_frame(archive: zipfile.ZipFile, member_name: str) -> np.ndarray:
    """Decode one zipped ViPE EXR depth frame as float32 metric depth."""

    with archive.open(member_name) as handle:
        exr = OpenEXR.InputFile(handle)
        header = exr.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        channel = exr.channels(["Z"])[0]
        return np.frombuffer(channel, dtype=np.float16).reshape(height, width).astype(np.float32)


def intrinsics_vec_to_matrix(vec: np.ndarray) -> np.ndarray:
    """Convert one ViPE `[fx, fy, cx, cy]` intrinsics vector to a full 3x3 matrix."""

    fx, fy, cx, cy = [float(value) for value in vec[:4]]
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def intrinsics_sequence_to_matrices(intrinsics: np.ndarray) -> np.ndarray:
    """Convert a sequence of ViPE intrinsics vectors to `[N, 3, 3]` matrices."""

    return np.stack([intrinsics_vec_to_matrix(vec) for vec in intrinsics], axis=0).astype(np.float32)


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    """Decode one RGB frame from a video."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video for preview: {path}")
    if frame_index > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"failed to decode frame {frame_index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def write_depth_preview(
    output_path: Path,
    *,
    video_path: Path,
    frame_index: int,
    depth: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Write a simple RGB/depth side-by-side preview for quick qualitative inspection."""

    rgb = read_video_frame(video_path, frame_index)
    if rgb.shape[:2] != depth.shape:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    preview = np.zeros(depth.shape + (3,), dtype=np.uint8)
    valid_values = depth[valid_mask]
    if valid_values.size > 0:
        lo, hi = np.percentile(valid_values, [5.0, 95.0])
        if hi <= lo:
            hi = lo + 1e-6
        normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        preview = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
        preview[~valid_mask] = 0
    combined = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), preview], axis=1)
    cv2.putText(combined, "RGB", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        combined,
        "ViPE Depth",
        (depth.shape[1] + 24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), combined)


def sidecar_complete(path: Path) -> bool:

    if not path.is_dir():
        return False
    return all((path / name).exists() for name in REQUIRED_SIDECAR_FILES)


def validate_export(
    *,
    source_video: VideoProbe,
    frame_indices: np.ndarray,
    poses_c2w: np.ndarray,
    poses_w2c: np.ndarray,
    intrinsics: np.ndarray,
    depth_metric: np.ndarray,
    depth_valid_mask: np.ndarray,
) -> dict[str, object]:
    """Validate a converted ViPE export and summarize its structural quality."""

    aligned_count = int(frame_indices.shape[0])
    counts_consistent = (
        aligned_count
        == int(poses_c2w.shape[0])
        == int(poses_w2c.shape[0])
        == int(intrinsics.shape[0])
        == int(depth_metric.shape[0])
        == int(depth_valid_mask.shape[0])
    )
    pose_shape_ok = poses_c2w.ndim == 3 and poses_c2w.shape[1:] == (4, 4)
    pose_inverse_shape_ok = poses_w2c.ndim == 3 and poses_w2c.shape[1:] == (4, 4)
    intrinsics_shape_ok = intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3)
    depth_shape_ok = depth_metric.ndim == 3 and depth_valid_mask.ndim == 3
    height = int(depth_metric.shape[1]) if depth_shape_ok else 0
    width = int(depth_metric.shape[2]) if depth_shape_ok else 0
    pose_has_nan = bool(np.isnan(poses_c2w).any() or np.isnan(poses_w2c).any())
    pose_has_inf = bool(np.isinf(poses_c2w).any() or np.isinf(poses_w2c).any())
    intrinsics_has_nan = bool(np.isnan(intrinsics).any())
    intrinsics_has_inf = bool(np.isinf(intrinsics).any())
    depth_has_nan = bool(np.isnan(depth_metric).any())
    depth_has_inf = bool(np.isinf(depth_metric).any())
    has_nan = bool(pose_has_nan or intrinsics_has_nan or depth_has_nan)
    has_inf = bool(pose_has_inf or intrinsics_has_inf or depth_has_inf)
    inverse_residual = float(
        np.max(np.abs(np.matmul(poses_c2w, poses_w2c) - np.eye(4, dtype=np.float32)))
    ) if pose_shape_ok and pose_inverse_shape_ok and aligned_count > 0 else float("inf")
    rotation_ortho_residual = float(
        np.max(
            np.abs(
                np.matmul(poses_c2w[:, :3, :3], np.transpose(poses_c2w[:, :3, :3], (0, 2, 1)))
                - np.eye(3, dtype=np.float32)
            )
        )
    ) if pose_shape_ok and aligned_count > 0 else float("inf")
    rotation_det = np.linalg.det(poses_c2w[:, :3, :3]).astype(np.float32) if pose_shape_ok and aligned_count > 0 else np.array([], dtype=np.float32)
    valid_fraction = float(depth_valid_mask.mean()) if depth_shape_ok and depth_valid_mask.size > 0 else 0.0
    coverage_ratio = float(aligned_count / source_video.frame_count) if source_video.frame_count > 0 else 0.0
    status = "passed"
    if not counts_consistent or not pose_shape_ok or not pose_inverse_shape_ok or not intrinsics_shape_ok or not depth_shape_ok:
        status = "failed"
    elif has_inf or inverse_residual > 1e-3:
        status = "failed"
    elif has_nan or rotation_ortho_residual > 1e-3 or valid_fraction <= 0.0:
        status = "warn"
    report = {
        "status": status,
        "frame_count": aligned_count,
        "source_video_frame_count": int(source_video.frame_count),
        "coverage_ratio": coverage_ratio,
        "pose_count": int(poses_c2w.shape[0]) if pose_shape_ok else 0,
        "intrinsics_count": int(intrinsics.shape[0]) if intrinsics_shape_ok else 0,
        "depth_count": int(depth_metric.shape[0]) if depth_shape_ok else 0,
        "image_height": height,
        "image_width": width,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "inverse_residual_max": inverse_residual,
        "rotation_orthonormality_residual_max": rotation_ortho_residual,
        "rotation_det_min": float(rotation_det.min()) if rotation_det.size > 0 else None,
        "rotation_det_max": float(rotation_det.max()) if rotation_det.size > 0 else None,
        "depth_valid_fraction": valid_fraction,
        "fps": float(source_video.fps),
    }
    return report


def export_vipe_sidecar(
    *,
    video_path: Path,
    artifact_root: Path,
    output_dir: Path,
    vipe_root: Path,
    pipeline_name: str,
    vipe_python: Path,
) -> dict[str, object]:
    """Convert one ViPE artifact bundle into a WorldAtlas Arena-friendly sidecar directory."""

    source_video = probe_video(video_path)
    paths = VipeArtifactPaths(root=artifact_root, stem=video_path.stem)
    if not paths.pose_path.exists():
        raise FileNotFoundError(f"missing ViPE pose artifact: {paths.pose_path}")
    if not paths.intrinsics_path.exists():
        raise FileNotFoundError(f"missing ViPE intrinsics artifact: {paths.intrinsics_path}")
    if not paths.depth_path.exists():
        raise FileNotFoundError(f"missing ViPE depth artifact: {paths.depth_path}")

    pose_indices, poses_c2w = read_vipe_pose_artifacts(paths.pose_path)
    intr_indices, intrinsics_vec = read_vipe_intrinsics_artifacts(paths.intrinsics_path)
    depth_indices = list_vipe_depth_indices(paths.depth_path)
    camera_types_by_index = read_vipe_camera_types(paths.camera_type_path, intr_indices)
    common_indices = sorted(set(pose_indices.tolist()) & set(intr_indices.tolist()) & set(depth_indices))
    if not common_indices:
        raise RuntimeError(f"no common pose/intrinsics/depth frame indices found for {video_path}")

    pose_map = {int(index): pose for index, pose in zip(pose_indices.tolist(), poses_c2w, strict=True)}
    intr_map = {int(index): intr for index, intr in zip(intr_indices.tolist(), intrinsics_vec, strict=True)}
    frame_indices = np.asarray(common_indices, dtype=np.int32)
    aligned_poses_c2w = np.stack([pose_map[index] for index in common_indices], axis=0).astype(np.float32)
    aligned_poses_w2c = np.linalg.inv(aligned_poses_c2w).astype(np.float32)
    aligned_intrinsics = intrinsics_sequence_to_matrices(
        np.stack([intr_map[index] for index in common_indices], axis=0).astype(np.float32)
    )
    timestamps = (frame_indices.astype(np.float64) / float(source_video.fps)).astype(np.float64)
    depth_positions = {index: position for position, index in enumerate(common_indices)}
    depth_memmap: np.memmap | None = None
    valid_memmap = None
    written_depth = 0
    preview_index: int | None = None
    preview_depth: np.ndarray | None = None
    preview_mask: np.ndarray | None = None

    with zipfile.ZipFile(paths.depth_path, "r") as archive:
        for member_name in sorted(archive.namelist()):
            frame_index = int(Path(member_name).stem)
            if frame_index not in depth_positions:
                continue
            depth_frame = read_vipe_depth_frame(archive, member_name)
            valid_mask = np.isfinite(depth_frame) & (depth_frame > 0.0)
            if depth_memmap is None:
                depth_memmap = np.lib.format.open_memmap(
                    output_dir / "depth_metric.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(common_indices), depth_frame.shape[0], depth_frame.shape[1]),
                )
                valid_memmap = np.lib.format.open_memmap(
                    output_dir / "_depth_valid_mask.npy",
                    mode="w+",
                    dtype=np.bool_,
                    shape=(len(common_indices), depth_frame.shape[0], depth_frame.shape[1]),
                )
            if depth_frame.shape != depth_memmap.shape[1:]:
                raise ValueError(
                    f"inconsistent depth frame shape for {video_path}: "
                    f"expected {depth_memmap.shape[1:]}, got {depth_frame.shape}"
                )
            position = depth_positions[frame_index]
            depth_memmap[position] = depth_frame
            valid_memmap[position] = valid_mask
            written_depth += 1
            if preview_index is None:
                preview_index = frame_index
                preview_depth = depth_frame.copy()
                preview_mask = valid_mask.copy()
    if depth_memmap is None or valid_memmap is None:
        raise RuntimeError(f"no common depth frames were written for {video_path}")
    if written_depth != len(common_indices):
        raise RuntimeError(
            f"depth export count mismatch for {video_path}: expected {len(common_indices)}, wrote {written_depth}"
        )

    depth_metric = depth_memmap
    depth_valid_mask = valid_memmap
    np.save(output_dir / "timestamps.npy", timestamps)
    np.save(output_dir / "frame_indices.npy", frame_indices.astype(np.int32))
    np.save(output_dir / "poses_c2w.npy", aligned_poses_c2w)
    np.save(output_dir / "poses_w2c.npy", aligned_poses_w2c)
    np.save(output_dir / "intrinsics.npy", aligned_intrinsics)
    np.savez_compressed(output_dir / "depth_valid_mask.npz", mask=np.asarray(depth_valid_mask))

    camera_types = [camera_types_by_index.get(index, "PINHOLE") for index in common_indices]
    camera_model_payload = {
        "camera_type": camera_types[0] if len(set(camera_types)) == 1 else "VARYING",
        "camera_types_observed": sorted(set(camera_types)),
        "varying_intrinsics": bool(
            np.max(np.abs(aligned_intrinsics - aligned_intrinsics[0][None, ...])) > 1e-6
        ),
        "matrix_layout": "per-frame 3x3 OpenCV intrinsics matrix",
        "pose_layout": "per-frame 4x4 camera-to-world OpenCV matrix",
        "frame_width": int(source_video.width),
        "frame_height": int(source_video.height),
    }
    (output_dir / "camera_model.json").write_text(
        json.dumps(camera_model_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = validate_export(
        source_video=source_video,
        frame_indices=frame_indices,
        poses_c2w=aligned_poses_c2w,
        poses_w2c=aligned_poses_w2c,
        intrinsics=aligned_intrinsics,
        depth_metric=depth_metric,
        depth_valid_mask=depth_valid_mask,
    )
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    provenance_payload = {
        "source_video_path": str(video_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_pipeline_defaults": pipeline_name == "default",
        "estimator": {
            "name": "ViPE",
            "pipeline": pipeline_name,
            "repo_root": str(vipe_root),
            "python": str(vipe_python),
            "entrypoint": str(vipe_root / "run.py"),
            "official_pipeline_defaults": pipeline_name == "default",
        },
        "source_video": {
            "frame_count": int(source_video.frame_count),
            "fps": float(source_video.fps),
            "width": int(source_video.width),
            "height": int(source_video.height),
        },
        "export": {
            "frame_count": int(frame_indices.shape[0]),
            "frame_indices_path": "frame_indices.npy",
            "timestamps_path": "timestamps.npy",
            "poses_c2w_path": "poses_c2w.npy",
            "poses_w2c_path": "poses_w2c.npy",
            "intrinsics_path": "intrinsics.npy",
            "depth_metric_path": "depth_metric.npy",
            "depth_valid_mask_path": "depth_valid_mask.npz",
        },
        "notes": [
            "These geometry annotations are estimated by ViPE and are not native ground-truth.",
            "timestamps.npy is derived from the source frame index sequence and the decoded video FPS.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if preview_index is not None and preview_depth is not None and preview_mask is not None:
        write_depth_preview(
            output_dir / "preview_rgb_depth.png",
            video_path=video_path,
            frame_index=preview_index,
            depth=preview_depth,
            valid_mask=preview_mask,
        )

    del depth_valid_mask
    del valid_memmap
    (output_dir / "_depth_valid_mask.npy").unlink()

    return {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "camera_type": camera_model_payload["camera_type"],
        **report,
    }


__all__ = [
    "REQUIRED_SIDECAR_FILES",
    "VIDEO_EXTENSIONS",
    "VipeArtifactPaths",
    "VideoProbe",
    "default_sidecar_dir",
    "discover_videos",
    "export_vipe_sidecar",
    "intrinsics_sequence_to_matrices",
    "intrinsics_vec_to_matrix",
    "is_video_file",
    "mirrored_sidecar_dir",
    "prioritize_videos",
    "probe_video",
    "sidecar_complete",
    "validate_export",
]
