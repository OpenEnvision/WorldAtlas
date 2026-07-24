"""Dynamic region mask loading and alignment for region-based metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


MASK_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def iter_mask_image_paths(mask_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in MASK_IMAGE_SUFFIXES
        ),
        key=_mask_sort_key,
    )


def load_index_frame_indices(index_path: Path) -> list[int]:
    indices: list[int] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        indices.append(int(parts[1]))
    return indices


def mask_dir_has_nonzero(mask_dir: Path) -> bool:
    if not mask_dir.exists():
        return False
    for path in iter_mask_image_paths(mask_dir):
        with Image.open(path) as image:
            if image.getbbox() is not None:
                return True
    return False


def dyn_masks_has_content(mask_path: Path) -> bool:
    if not mask_path.exists():
        return False
    payload = np.load(mask_path)
    try:
        for key in payload.files:
            if key.startswith("f_") and key.endswith("_data") and payload[key].size:
                return True
    finally:
        payload.close()
    return False


def write_zero_dyn_masks(mask_path: Path, shape: tuple[int, int], frame_count: int) -> None:
    payload: dict[str, Any] = {"shape": np.asarray(shape, dtype=np.int32)}
    zero_data = np.zeros((0,), dtype=bool)
    zero_indices = np.zeros((0,), dtype=np.int32)
    zero_indptr = np.zeros((shape[0] + 1,), dtype=np.int32)
    for frame_index in range(frame_count):
        payload[f"f_{frame_index}_data"] = zero_data
        payload[f"f_{frame_index}_indices"] = zero_indices
        payload[f"f_{frame_index}_indptr"] = zero_indptr
    np.savez_compressed(mask_path, **payload)


def write_sampled_dyn_masks_from_mask_dir(
    mask_path: Path,
    mask_dir: Path,
    frame_indices: Sequence[int],
) -> None:
    mask_paths = iter_mask_image_paths(mask_dir)
    if not mask_paths:
        raise FileNotFoundError(f"no mask images found in {mask_dir}")

    resolved_paths = _resolve_sampled_mask_paths(mask_paths, frame_indices)
    first_mask = _load_binary_mask(resolved_paths[0])
    payload: dict[str, Any] = {"shape": np.asarray(first_mask.shape, dtype=np.int32)}
    for ordinal, path in enumerate(resolved_paths):
        mask = _load_binary_mask(path, target_shape=first_mask.shape)
        data, indices, indptr = _binary_mask_to_csr(mask)
        payload[f"f_{ordinal}_data"] = data
        payload[f"f_{ordinal}_indices"] = indices
        payload[f"f_{ordinal}_indptr"] = indptr
    np.savez_compressed(mask_path, **payload)


def _mask_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), path.name
    return 10**18, path.name


def _resolve_sampled_mask_paths(mask_paths: Sequence[Path], frame_indices: Sequence[int]) -> list[Path]:
    numeric_map = {int(path.stem): path for path in mask_paths if path.stem.isdigit()}
    resolved: list[Path] = []
    for frame_index in frame_indices:
        if numeric_map:
            path = numeric_map.get(int(frame_index))
            if path is None:
                raise FileNotFoundError(
                    f"missing source mask for sampled frame {frame_index} in {mask_paths[0].parent}"
                )
            resolved.append(path)
            continue
        if int(frame_index) >= len(mask_paths):
            raise FileNotFoundError(
                f"sampled frame {frame_index} exceeds available mask count {len(mask_paths)} "
                f"in {mask_paths[0].parent}"
            )
        resolved.append(mask_paths[int(frame_index)])
    return resolved


def _load_binary_mask(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        if target_shape is not None and grayscale.size != (target_shape[1], target_shape[0]):
            grayscale = grayscale.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)
        array = np.asarray(grayscale)
    return array > 0


def _resize_binary_mask(mask: np.ndarray, target_shape: tuple[int, int] | None) -> np.ndarray:
    if target_shape is None or tuple(mask.shape) == tuple(target_shape):
        return mask.astype(bool, copy=False)
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    resized = image.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _dense_mask_from_archive(payload: np.lib.npyio.NpzFile, frame_index: int) -> np.ndarray:
    height, width = (int(value) for value in payload["shape"])
    data = payload[f"f_{frame_index}_data"]
    indices = payload[f"f_{frame_index}_indices"]
    indptr = payload[f"f_{frame_index}_indptr"]
    frame = np.zeros((height, width), dtype=bool)
    for row in range(height):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        if start == end:
            continue
        frame[row, indices[start:end]] = data[start:end]
    return frame


def _archive_frame_ordinals(mask_path: Path, frame_indices: Sequence[int] | None, frame_count: int) -> list[int]:
    if frame_indices is None:
        return list(range(frame_count))
    index_path = mask_path.parent / "indexes.txt"
    if not index_path.exists():
        return [min(max(int(index), 0), frame_count - 1) for index in frame_indices]
    source_indices = load_index_frame_indices(index_path)
    if not source_indices:
        return [min(max(int(index), 0), frame_count - 1) for index in frame_indices]
    ordinals: list[int] = []
    for frame_index in frame_indices:
        candidates = [
            (abs(int(source_index) - int(frame_index)), ordinal)
            for ordinal, source_index in enumerate(source_indices[:frame_count])
        ]
        if not candidates:
            ordinals.append(0)
            continue
        _, best_ordinal = min(candidates, key=lambda item: (item[0], item[1]))
        ordinals.append(int(best_ordinal))
    return ordinals


def load_dyn_mask_sequence(
    mask_source: Path,
    *,
    frame_indices: Sequence[int] | None = None,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if mask_source.is_dir():
        mask_paths = iter_mask_image_paths(mask_source)
        if not mask_paths:
            raise FileNotFoundError(f"no mask images found in {mask_source}")
        resolved_paths = (
            _resolve_sampled_mask_paths(mask_paths, frame_indices)
            if frame_indices is not None
            else mask_paths
        )
        masks = [
            _load_binary_mask(path, target_shape=target_shape)
            for path in resolved_paths
        ]
        return np.stack(masks, axis=0).astype(bool)

    if not mask_source.exists():
        raise FileNotFoundError(f"mask source not found: {mask_source}")
    payload = np.load(mask_source)
    try:
        archive_frame_count = len(
            [name for name in payload.files if name.startswith("f_") and name.endswith("_data")]
        )
        if archive_frame_count <= 0:
            raise RuntimeError(f"mask archive contains no frames: {mask_source}")
        ordinals = _archive_frame_ordinals(mask_source, frame_indices, archive_frame_count)
        masks = [
            _resize_binary_mask(_dense_mask_from_archive(payload, ordinal), target_shape)
            for ordinal in ordinals
        ]
        return np.stack(masks, axis=0).astype(bool)
    finally:
        payload.close()


def _binary_mask_to_csr(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(mask)
    counts = np.bincount(rows, minlength=mask.shape[0]).astype(np.int32, copy=False)
    indptr = np.empty((mask.shape[0] + 1,), dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    data = np.ones((rows.shape[0],), dtype=bool)
    indices = cols.astype(np.int32, copy=False)
    return data, indices, indptr


__all__ = [
    "dyn_masks_has_content",
    "iter_mask_image_paths",
    "load_dyn_mask_sequence",
    "load_index_frame_indices",
    "mask_dir_has_nonzero",
    "write_sampled_dyn_masks_from_mask_dir",
    "write_zero_dyn_masks",
]
