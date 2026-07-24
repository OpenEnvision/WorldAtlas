"""Media loading, probing, and frame extraction for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from worldarena.common.video_io import probe_video_frame_count


@dataclass(slots=True)
class LoadedMedia:
    modality: str
    path: str
    frames: list[np.ndarray]
    anchor_frames: list[np.ndarray]
    evaluation_protocol: str = "anchor_compat"
    sampled_frame_indices: list[int] = field(default_factory=list)
    protocol_frame_indices: list[int] = field(default_factory=list)
    native_frame_count: int | None = None
    window_start_ratio: float = 0.0
    window_end_ratio: float = 1.0


def protocol_native_frame_indices(media: LoadedMedia) -> list[int]:
    indices: list[int] = []
    for protocol_index in media.protocol_frame_indices:
        if 0 <= int(protocol_index) < len(media.sampled_frame_indices):
            indices.append(int(media.sampled_frame_indices[int(protocol_index)]))
    return indices


def protocol_frame_details(media: LoadedMedia) -> dict[str, object]:
    return {
        "evaluation_protocol": media.evaluation_protocol,
        "frame_indices": list(media.protocol_frame_indices),
        "sampled_frame_indices": list(media.sampled_frame_indices),
        "protocol_native_frame_indices": protocol_native_frame_indices(media),
        "protocol_frame_count": len(media.anchor_frames),
        "loaded_frame_count": len(media.frames),
        "native_frame_count": media.native_frame_count,
    }


def _sample_positions(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 1 or length == 1:
        return [0]
    indices = np.linspace(0, length - 1, num=min(count, length))
    return sorted({int(round(value)) for value in indices})


def _anchor_indices(length: int, positions: list[float]) -> list[int]:
    if length <= 0:
        return []
    indices = []
    for position in positions:
        clamped = min(max(position, 0.0), 1.0)
        indices.append(int(round(clamped * (length - 1))))
    return sorted({value for value in indices})


def _segment_random_indices(length: int, count: int, sample_key: str, seed: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 1 or length == 1:
        return [0]
    segments = min(max(count, 1), length)
    indices: list[int] = []
    for segment_idx in range(segments):
        start = int(np.floor(segment_idx * length / segments))
        end = int(np.floor((segment_idx + 1) * length / segments)) - 1
        end = max(end, start)
        digest = hashlib.sha256(f"{sample_key}:{seed}:{segment_idx}".encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        offset = int(np.floor(unit * (end - start + 1)))
        indices.append(start + min(offset, end - start))
    return sorted({value for value in indices})


def _protocol_indices(
    length: int,
    anchor_positions: list[float],
    protocol_name: str,
    sample_key: str,
    protocol_seed: int,
    segment_count: int,
) -> list[int]:
    if protocol_name == "segment_random":
        return _segment_random_indices(length, segment_count, sample_key, protocol_seed)
    return _anchor_indices(length, anchor_positions)


def _window_bounds(length: int, start_ratio: float, end_ratio: float) -> tuple[int, int]:
    if length <= 0:
        return 0, 0
    start = min(max(float(start_ratio), 0.0), 1.0)
    end = min(max(float(end_ratio), 0.0), 1.0)
    if end < start:
        end = start
    start_index = int(round(start * max(length - 1, 0)))
    end_index = int(round(end * max(length - 1, 0)))
    return start_index, max(start_index, end_index)


def _load_image(
    path: Path,
    protocol_name: str,
    *,
    start_ratio: float,
    end_ratio: float,
) -> LoadedMedia:
    frame = np.asarray(Image.open(path).convert("RGB"))
    return LoadedMedia(
        modality="image",
        path=str(path),
        frames=[frame],
        anchor_frames=[frame],
        evaluation_protocol=protocol_name,
        sampled_frame_indices=[0],
        protocol_frame_indices=[0],
        native_frame_count=1,
        window_start_ratio=start_ratio,
        window_end_ratio=end_ratio,
    )


def _load_video(
    path: Path,
    anchor_positions: list[float],
    frame_count: int,
    sample_key: str,
    protocol_name: str,
    protocol_seed: int,
    segment_count: int,
    *,
    start_ratio: float,
    end_ratio: float,
) -> LoadedMedia:
    capture = None
    for attempt in range(5):
        candidate = cv2.VideoCapture(str(path))
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()
        if attempt < 4:
            time.sleep(0.2 * (attempt + 1))
    if capture is None:
        raise RuntimeError(f"failed to open video: {path}")

    native_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray]
    sampled_frame_indices: list[int]
    if native_count > 0:
        start_index, end_index = _window_bounds(native_count, start_ratio, end_ratio)
        target_indices = [
            start_index + offset
            for offset in _sample_positions(end_index - start_index + 1, frame_count)
        ]
        target_set = set(target_indices)
        frames = []
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index in target_set:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_index += 1
        sampled_frame_indices = target_indices[: len(frames)]
    else:
        decoded_frames: list[np.ndarray] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            decoded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        native_count = len(decoded_frames)
        start_index, end_index = _window_bounds(native_count, start_ratio, end_ratio)
        window_frames = decoded_frames[start_index : end_index + 1]
        relative_indices = _sample_positions(len(window_frames), frame_count)
        frames = [window_frames[index] for index in relative_indices]
        sampled_frame_indices = [start_index + index for index in relative_indices]
    capture.release()

    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {path}")

    protocol_frame_indices = _protocol_indices(
        len(frames),
        anchor_positions,
        protocol_name,
        sample_key,
        protocol_seed,
        segment_count,
    )
    anchors = [frames[index] for index in protocol_frame_indices] or [frames[0]]
    return LoadedMedia(
        modality="video",
        path=str(path),
        frames=frames,
        anchor_frames=anchors,
        evaluation_protocol=protocol_name,
        sampled_frame_indices=sampled_frame_indices,
        protocol_frame_indices=protocol_frame_indices,
        native_frame_count=native_count,
        window_start_ratio=start_ratio,
        window_end_ratio=end_ratio,
    )


def load_media(
    path: Path,
    modality: str,
    anchor_positions: list[float],
    frame_count: int,
    *,
    sample_key: str,
    protocol_name: str = "anchor_compat",
    protocol_seed: int = 17,
    segment_count: int = 4,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> LoadedMedia:
    if modality == "image":
        return _load_image(path, protocol_name, start_ratio=start_ratio, end_ratio=end_ratio)
    return _load_video(
        path,
        anchor_positions,
        frame_count,
        sample_key,
        protocol_name,
        protocol_seed,
        segment_count,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
    )


def probe_media_frame_count(path: Path, modality: str) -> int:
    if modality == "image":
        return 1
    return probe_video_frame_count(path)


__all__ = [
    "LoadedMedia",
    "load_media",
    "probe_media_frame_count",
    "protocol_frame_details",
    "protocol_native_frame_indices",
]
