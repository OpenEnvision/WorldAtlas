"""Validate image-mask provenance and instance counts."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

MASK_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def mask_source_relative(meta: dict) -> str:
    raw = str(meta.get("relative_image_path") or "").replace("\\", "/")
    if not raw:
        raise ValueError("Mask metadata has no relative_image_path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe mask source path: {raw}")
    if relative.parts[0] != "dynamic":
        relative = PurePosixPath("dynamic") / relative
    return relative.as_posix()


def mask_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS)


def validate_mask_metadata(directory: Path, relative_path: str) -> dict:
    meta = json.loads((directory / "meta.json").read_text())
    if mask_source_relative(meta) != relative_path:
        raise ValueError(f"Mask belongs to another source image: {directory}")
    files = mask_files(directory)
    if meta.get("num_saved_masks") != len(files) or len(meta.get("saved_masks", [])) != len(files):
        raise ValueError(f"Mask files do not match metadata count: {directory}")
    return meta
