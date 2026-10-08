"""Resolve independently numbered video masks through their source mapping."""

from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath

VIDEO_MASK_MAPPING = "dynamic/mask_mapping.csv"
VIDEO_MASK_RENUMBER_LOCK = "dynamic/.mask-renumber.lock"


def _relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "\\" in raw:
        raise ValueError(f"Unsafe video-mask mapping path: {raw!r}")
    return path


def build_video_mask_index(video_root: Path) -> dict[str, Path] | None:
    """Return source-video -> frame directory, or None for the legacy layout.

    An existing mapping is authoritative, including when no rows are matched.
    Unknown-source groups must not be paired by their new directory number.
    """
    if (video_root / VIDEO_MASK_RENUMBER_LOCK).exists():
        raise RuntimeError("Video masks are being renumbered; retry after the migration finishes")
    mapping = video_root / VIDEO_MASK_MAPPING
    if not mapping.exists():
        return None

    index: dict[str, Path] = {}
    mask_paths: set[str] = set()
    with mapping.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"mask_relative_path", "video_relative_path"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns in {mapping}")
        for row in reader:
            mask_relative = _relative_path(row["mask_relative_path"])
            if mask_relative.parts[:2] != ("dynamic", "masks"):
                raise ValueError(f"Unexpected video mask location: {mask_relative}")
            if mask_relative.as_posix() in mask_paths:
                raise ValueError(f"Duplicate video mask directory: {mask_relative}")
            mask_paths.add(mask_relative.as_posix())
            group = video_root / mask_relative
            if not (group / "masks").is_dir():
                raise FileNotFoundError(group / "masks")
            meta = json.loads((group / "meta.json").read_text(encoding="utf-8"))
            video_raw = row["video_relative_path"]
            if not video_raw:
                if meta.get("relative_video_path"):
                    raise ValueError(f"Unmatched group declares a video source: {group}")
                continue
            video_relative = _relative_path(video_raw)
            if video_relative.parts[:2] != ("dynamic", "videos"):
                raise ValueError(f"Unexpected source video location: {video_relative}")
            if meta.get("relative_video_path") != video_relative.as_posix():
                raise ValueError(f"Mask metadata belongs to another source video: {group}")
            if not (video_root / video_relative).is_file():
                raise FileNotFoundError(video_root / video_relative)
            if video_relative.as_posix() in index:
                raise ValueError(f"Multiple primary masks claim source video: {video_relative}")
            index[video_relative.as_posix()] = group / "masks"
    return index
