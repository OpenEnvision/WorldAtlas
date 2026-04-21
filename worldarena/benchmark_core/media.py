from __future__ import annotations

from dataclasses import dataclass
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


def _load_image(path: Path, anchor_positions: list[float]) -> LoadedMedia:
    frame = np.asarray(Image.open(path).convert("RGB"))
    return LoadedMedia(
        modality="image",
        path=str(path),
        frames=[frame],
        anchor_frames=[frame],
    )


def _load_video(path: Path, anchor_positions: list[float], frame_count: int) -> LoadedMedia:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    native_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = _sample_positions(native_count if native_count > 0 else frame_count, frame_count)
    if native_count <= 0:
        target_set = None
    else:
        target_set = set(target_indices)

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

    anchors = [frames[index] for index in _anchor_indices(len(frames), anchor_positions)]
    return LoadedMedia(
        modality="video",
        path=str(path),
        frames=frames,
        anchor_frames=anchors or [frames[0]],
    )


def load_media(path: Path, modality: str, anchor_positions: list[float], frame_count: int) -> LoadedMedia:
    if modality == "image":
        return _load_image(path, anchor_positions)
    return _load_video(path, anchor_positions, frame_count)
