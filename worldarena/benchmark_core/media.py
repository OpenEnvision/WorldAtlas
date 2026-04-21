from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(slots=True)
class LoadedMedia:
    modality: str
    path: str
    frames: list[np.ndarray]
    anchor_frames: list[np.ndarray]
    evaluation_protocol: str = "anchor_compat"
    sampled_frame_indices: list[int] = field(default_factory=list)
    protocol_frame_indices: list[int] = field(default_factory=list)


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


def _load_image(path: Path, protocol_name: str) -> LoadedMedia:
    frame = np.asarray(Image.open(path).convert("RGB"))
    return LoadedMedia(
        modality="image",
        path=str(path),
        frames=[frame],
        anchor_frames=[frame],
        evaluation_protocol=protocol_name,
        sampled_frame_indices=[0],
        protocol_frame_indices=[0],
    )


def _load_video(
    path: Path,
    anchor_positions: list[float],
    frame_count: int,
    sample_key: str,
    protocol_name: str,
    protocol_seed: int,
    segment_count: int,
) -> LoadedMedia:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    native_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = _sample_positions(native_count if native_count > 0 else frame_count, frame_count)
    target_set = set(target_indices) if native_count > 0 else None

    frames: list[np.ndarray] = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if target_set is None or frame_index in target_set:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_index += 1
    capture.release()

    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {path}")

    sampled_frame_indices = target_indices[: len(frames)] if native_count > 0 else list(range(len(frames)))
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
) -> LoadedMedia:
    if modality == "image":
        return _load_image(path, protocol_name)
    return _load_video(
        path,
        anchor_positions,
        frame_count,
        sample_key,
        protocol_name,
        protocol_seed,
        segment_count,
    )
