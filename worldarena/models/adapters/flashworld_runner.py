from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

from worldarena.benchmark.annotations import load_camera_matrices, load_intrinsics_sequence
from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena FlashWorld subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--annotation_path", required=True, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--num_frames", default=24, type=int)
    parser.add_argument("--image_height", default=480, type=int)
    parser.add_argument("--image_width", default=704, type=int)
    parser.add_argument("--image_index", default=0, type=int)
    parser.add_argument("--video_fps", default=15, type=int)
    parser.add_argument("--offload_t5", action="store_true")
    parser.add_argument("--offload_vae", action="store_true")
    parser.add_argument("--offload_transformer_during_vae", action="store_true")
    parser.add_argument("--export_ply", action="store_true")
    parser.add_argument("--export_spz", action="store_true")
    return parser.parse_args()


def _rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quaternion / norm


def _flashworld_quaternion(matrix: np.ndarray) -> list[float]:
    qx, qy, qz, qw = _rotation_matrix_to_quaternion_xyzw(matrix[:3, :3])
    return [float(qw), float(qx), float(qy), float(qz)]


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def flashworld_render_frame_count(*, num_frames: int, video_fps: int) -> int:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")
    return (int(num_frames) - 1) * int(video_fps) + 1


def flashworld_center_crop_resize_image(
    image: Image.Image,
    *,
    image_height: int,
    image_width: int,
) -> Image.Image:
    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            f"image_height and image_width must be positive, got {image_height}x{image_width}"
        )
    width, height = image.size
    if height <= 0 or width <= 0:
        raise ValueError(f"image has invalid size: {width}x{height}")

    if image_height / height > image_width / width:
        scale = image_height / height
    else:
        scale = image_width / width
    crop_height = min(height, int(image_height / scale))
    crop_width = min(width, int(image_width / scale))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(
        (image_width, image_height)
    )


def flashworld_center_crop_resize_frame(
    frame_rgb: np.ndarray,
    *,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8)).convert("RGB")
    resized = flashworld_center_crop_resize_image(
        image,
        image_height=image_height,
        image_width=image_width,
    )
    return np.asarray(resized, dtype=np.uint8)


def _target_source_indices(*, source_frame_count: int, target_frame_count: int) -> list[int]:
    if source_frame_count <= 0:
        raise ValueError(f"source_frame_count must be positive, got {source_frame_count}")
    if target_frame_count <= 0:
        raise ValueError(f"target_frame_count must be positive, got {target_frame_count}")
    positions = np.linspace(0, source_frame_count - 1, num=target_frame_count)
    return [
        min(max(int(round(float(position))), 0), source_frame_count - 1)
        for position in positions
    ]


def _decode_needed_frames(video_path: Path, target_indices: list[int]) -> tuple[dict[int, np.ndarray], int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open reference video: {video_path}")
    try:
        native_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if native_count <= 0:
            decoded: list[np.ndarray] = []
            while True:
                success, frame = capture.read()
                if not success:
                    break
                decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not decoded:
                raise RuntimeError(f"reference video contains no decodable frames: {video_path}")
            missing = sorted(index for index in set(target_indices) if index >= len(decoded))
            if missing:
                raise RuntimeError(
                    f"failed to decode {len(missing)} required reference frames from {video_path}"
                )
            return {index: decoded[index] for index in set(target_indices)}, len(decoded)

        needed = set(target_indices)
        decoded_frames: dict[int, np.ndarray] = {}
        max_needed = max(needed) if needed else -1
        frame_index = 0
        while frame_index <= max_needed:
            success, frame = capture.read()
            if not success:
                break
            if frame_index in needed:
                decoded_frames[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_index += 1
    finally:
        capture.release()

    missing = sorted(needed - set(decoded_frames))
    if missing:
        raise RuntimeError(
            f"failed to decode {len(missing)} required reference frames from {video_path}"
        )
    return decoded_frames, native_count


def _decode_all_frames(video_path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open reference video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"reference video contains no decodable frames: {video_path}")
    return frames


def write_flashworld_aligned_reference_video(
    *,
    input_video: Path,
    output_path: Path,
    image_height: int,
    image_width: int,
    target_frame_count: int,
    video_fps: int,
) -> dict[str, int | str]:
    if target_frame_count <= 0:
        raise ValueError(f"target_frame_count must be positive, got {target_frame_count}")
    input_video = input_video.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    probe_capture = cv2.VideoCapture(str(input_video))
    if not probe_capture.isOpened():
        raise RuntimeError(f"failed to open reference video: {input_video}")
    source_frame_count = int(probe_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    probe_capture.release()
    if source_frame_count <= 0:
        all_frames = _decode_all_frames(input_video)
        source_frame_count = len(all_frames)
        target_indices = _target_source_indices(
            source_frame_count=source_frame_count,
            target_frame_count=target_frame_count,
        )
        decoded_frames = {index: all_frames[index] for index in set(target_indices)}
        decoded_source_count = source_frame_count
    else:
        target_indices = _target_source_indices(
            source_frame_count=source_frame_count,
            target_frame_count=target_frame_count,
        )
        decoded_frames, decoded_source_count = _decode_needed_frames(input_video, target_indices)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_path.with_name(f"{output_path.stem}.mpeg4.tmp{output_path.suffix}")
    raw_output_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(video_fps),
        (int(image_width), int(image_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for aligned reference video: {raw_output_path}")
    try:
        for frame_index in target_indices:
            frame = decoded_frames[frame_index]
            aligned = flashworld_center_crop_resize_frame(
                frame,
                image_height=image_height,
                image_width=image_width,
            )
            writer.write(cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    output_path.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(raw_output_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        raw_output_path.unlink(missing_ok=True)

    return {
        "input_video": str(input_video),
        "output_path": str(output_path),
        "codec": "h264",
        "source_frame_count": int(decoded_source_count),
        "target_frame_count": int(target_frame_count),
        "image_height": int(image_height),
        "image_width": int(image_width),
        "video_fps": int(video_fps),
    }


def _intrinsics_are_normalized(intrinsic: np.ndarray) -> bool:
    values = np.asarray(intrinsic, dtype=np.float32)
    if values.shape != (4,) or not np.isfinite(values).all():
        return False
    fx, fy, cx, cy = [float(value) for value in values]
    return 0.0 < abs(fx) <= 4.0 and 0.0 < abs(fy) <= 4.0 and 0.0 <= cx <= 2.0 and 0.0 <= cy <= 2.0


def _flashworld_source_intrinsics(intrinsic: np.ndarray, *, image_width: int, image_height: int) -> np.ndarray:
    values = np.asarray(intrinsic, dtype=np.float32)
    if values.shape != (4,):
        raise ValueError(f"expected intrinsics vector with shape (4,), got {values.shape}")
    if not _intrinsics_are_normalized(values):
        return values.astype(np.float32)
    scale = np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
    return (values * scale).astype(np.float32)


def _build_request_payload(
    *,
    annotation_path: str,
    conditioning_image: Path,
    prompt: str,
    num_frames: int,
    image_height: int,
    image_width: int,
    image_index: int,
) -> dict[str, object]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            f"image_height and image_width must be positive, got {image_height}x{image_width}"
        )
    if image_index < 0 or image_index >= num_frames:
        raise ValueError(f"image_index must be within [0, {num_frames - 1}], got {image_index}")

    camera_matrices = load_camera_matrices(annotation_path, target_frames=num_frames)
    if camera_matrices is None:
        raise FileNotFoundError(f"poses.npy not found under annotation path: {annotation_path}")

    intrinsics = load_intrinsics_sequence(annotation_path, target_frames=num_frames)
    if intrinsics is None:
        raise FileNotFoundError(f"intrinsics.npy not found under annotation path: {annotation_path}")

    source_width, source_height = _image_size(conditioning_image)
    cameras: list[dict[str, object]] = []
    for matrix, intrinsic in zip(camera_matrices, intrinsics):
        fx, fy, cx, cy = [
            float(value)
            for value in _flashworld_source_intrinsics(
                np.asarray(intrinsic, dtype=np.float32),
                image_width=source_width,
                image_height=source_height,
            )
        ]
        cameras.append(
            {
                "quaternion": _flashworld_quaternion(matrix),
                "position": [float(value) for value in matrix[:3, 3]],
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }
        )

    return {
        "image_prompt": str(conditioning_image),
        "text_prompt": str(prompt).strip(),
        "resolution": [num_frames, image_height, image_width],
        "image_index": image_index,
        "cameras": cameras,
    }


def _resolve_checkpoint_path(checkpoint_dir: str) -> Path:
    path = resolve_checkpoint_path(checkpoint_dir, required=True)
    if path is None:
        raise FileNotFoundError("FlashWorld checkpoint_dir is missing")
    if path.is_file():
        return path
    candidate = path / "model.ckpt"
    if not candidate.is_file():
        raise FileNotFoundError(f"FlashWorld checkpoint file not found: {candidate}")
    return candidate.resolve()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cli_path = (repo_root / "cli.py").resolve()
    if not cli_path.exists():
        raise FileNotFoundError(f"FlashWorld CLI entrypoint not found: {cli_path}")

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_dir)

    with tempfile.TemporaryDirectory(prefix="worldarena_flashworld_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        input_dir = temp_dir / "inputs"
        output_dir = temp_dir / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_source = Path(args.conditioning_image).expanduser().resolve()
        image_copy = temp_dir / f"conditioning{image_source.suffix or '.png'}"
        shutil.copy2(image_source, image_copy)

        request_payload = _build_request_payload(
            annotation_path=args.annotation_path,
            conditioning_image=image_copy,
            prompt=args.prompt,
            num_frames=args.num_frames,
            image_height=args.image_height,
            image_width=args.image_width,
            image_index=args.image_index,
        )
        request_path = input_dir / f"{args.sample_name}.json"
        request_path.write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            str(cli_path),
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--video",
            "--video_fps",
            str(args.video_fps),
        ]
        command.extend(["--ckpt", str(checkpoint_path)])
        for flag_name in (
            "offload_t5",
            "offload_vae",
            "offload_transformer_during_vae",
            "export_ply",
            "export_spz",
        ):
            if getattr(args, flag_name):
                command.append(f"--{flag_name}")

        subprocess.run(
            command,
            check=True,
            cwd=str(repo_root),
            env=apply_checkpoint_env(),
        )

        generated_root = output_dir / args.sample_name
        generated_video = generated_root / "video.mp4"
        if not generated_video.exists():
            raise FileNotFoundError(f"FlashWorld output video was not written: {generated_video}")
        shutil.copy2(generated_video, output_path)

        if args.export_ply:
            generated_ply = generated_root / "gaussians.ply"
            if not generated_ply.exists():
                raise FileNotFoundError(f"FlashWorld PLY output was not written: {generated_ply}")
            shutil.copy2(generated_ply, output_path.with_suffix(".ply"))

        if args.export_spz:
            generated_spz = generated_root / "gaussians.spz"
            if not generated_spz.exists():
                raise FileNotFoundError(f"FlashWorld SPZ output was not written: {generated_spz}")
            shutil.copy2(generated_spz, output_path.with_suffix(".spz"))


if __name__ == "__main__":
    main()
