"""Video frame extraction, windowing, and freeze-frame utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def probe_video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return max(frame_count, 0)


def probe_video_fps(path: Path, *, default: float = 8.0) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if fps <= 0.0:
        return float(default)
    return fps


def read_video_frame(
    path: Path,
    *,
    frame_index: int | None = None,
    frame_ratio: float | None = None,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    native_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_index = int(frame_index or 0)
    if frame_ratio is not None and native_count > 0:
        clamped = min(max(float(frame_ratio), 0.0), 1.0)
        target_index = int(round(clamped * max(native_count - 1, 0)))
    if target_index > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(target_index))

    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"failed to decode frame {target_index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    frames: list[np.ndarray] = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {path}")
    return frames


def write_video_frames(
    output_path: Path,
    frames: list[np.ndarray],
    *,
    fps: float,
) -> Path:
    if not frames:
        raise RuntimeError(f"cannot write empty video clip: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0], dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(first.shape[1]), int(first.shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for video clip: {output_path}")
    for frame in frames:
        rgb = np.asarray(frame, dtype=np.uint8)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return output_path


def freeze_frame_to_video(
    image_path: Path,
    output_path: Path,
    *,
    fps: float = 8.0,
    frame_count: int = 8,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(Image.open(image_path).convert("RGB"))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame.shape[1], frame.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for video clip: {output_path}")
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for _ in range(max(int(frame_count), 1)):
        writer.write(bgr)
    writer.release()
    return output_path


def extract_video_window(
    video_path: Path,
    output_path: Path,
    *,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    min_frame_count: int = 1,
    fps: float | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    native_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        source_fps = 8.0
    if native_count <= 0:
        raise RuntimeError(f"video has no decodable frames: {video_path}")

    start = min(max(float(start_ratio), 0.0), 1.0)
    end = min(max(float(end_ratio), 0.0), 1.0)
    if end < start:
        end = start

    start_index = int(round(start * max(native_count - 1, 0)))
    end_index = int(round(end * max(native_count - 1, 0)))
    if end_index <= start_index and native_count > 1:
        end_index = min(native_count - 1, start_index + max(int(min_frame_count), 1) - 1)
    end_index = min(end_index, native_count - 1)

    frames: list[np.ndarray] = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if start_index <= frame_index <= end_index:
            frames.append(frame)
        frame_index += 1
    capture.release()

    if not frames:
        raise RuntimeError(
            f"no frames extracted from {video_path} for window [{start_ratio}, {end_ratio}]"
        )
    while len(frames) < max(int(min_frame_count), 1):
        frames.append(frames[-1].copy())

    first = frames[0]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps or source_fps),
        (first.shape[1], first.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for video clip: {output_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return output_path


__all__ = [
    "extract_video_window",
    "freeze_frame_to_video",
    "probe_video_fps",
    "probe_video_frame_count",
    "read_video_frame",
    "read_video_frames",
    "write_video_frames",
]
