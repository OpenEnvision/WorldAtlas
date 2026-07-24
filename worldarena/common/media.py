"""Image and video probing utilities shared across benchmark and datasets."""

from __future__ import annotations

import concurrent.futures
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from worldarena.common.serialization import ensure_dir


class MediaProbeCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            self._entries = payload.get("entries", {})

    @staticmethod
    def _signature(path: Path) -> dict[str, int]:
        stat = path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def get(self, path: Path) -> dict[str, Any] | None:
        key = str(path)
        cached = self._entries.get(key)
        if not cached:
            return None
        if cached.get("signature") != self._signature(path):
            return None
        return cached.get("data")

    def set(self, path: Path, data: dict[str, Any]) -> None:
        self._entries[str(path)] = {
            "signature": self._signature(path),
            "data": data,
        }

    def prune(self, valid_paths: set[str]) -> None:
        stale = [key for key in self._entries if key not in valid_paths]
        for key in stale:
            del self._entries[key]

    def save(self) -> None:
        """Save."""
        ensure_dir(self.path.parent)
        payload = {"entries": self._entries}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_divide(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right in (None, 0):
        return None
    return round(float(left) / float(right), 4)


def _parse_fps(raw: str | None) -> float | None:
    if not raw or raw in {"0/0", "N/A"}:
        return None
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        if float(denominator) == 0:
            return None
        return round(float(numerator) / float(denominator), 4)
    return round(float(raw), 4)


def probe_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        return {
            "width": width,
            "height": height,
            "aspect_ratio": _safe_divide(width, height),
            "image_format": image.format,
            "image_mode": image.mode,
            "pixel_count": width * height,
        }


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "probe_error": result.stderr.strip() or "ffprobe failed",
            "width": None,
            "height": None,
            "fps": None,
            "duration_seconds": None,
            "aspect_ratio": None,
        }

    payload = json.loads(result.stdout or "{}")
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "video"), {})
    width = stream.get("width")
    height = stream.get("height")
    duration_raw = stream.get("duration") or payload.get("format", {}).get("duration")
    duration_seconds = round(float(duration_raw), 4) if duration_raw else None
    fps = _parse_fps(stream.get("avg_frame_rate"))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "aspect_ratio": _safe_divide(width, height),
        "video_codec": stream.get("codec_name"),
        "pixel_count": (width * height) if width and height else None,
    }


def _probe_many(
    paths: list[Path],
    probe_fn: Callable[[Path], dict[str, Any]],
    cache: MediaProbeCache,
    max_workers: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    pending: list[Path] = []

    for path in paths:
        cached = cache.get(path)
        if cached is not None:
            results[str(path)] = cached
        else:
            pending.append(path)

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(probe_fn, path): path for path in pending}
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                data = future.result()
                results[str(path)] = data
                cache.set(path, data)

    return results


def probe_media(
    image_paths: list[Path],
    video_paths: list[Path],
    cache_path: Path,
    max_workers: int = 8,
) -> dict[str, dict[str, Any]]:
    cache = MediaProbeCache(cache_path)
    image_results = _probe_many(image_paths, probe_image, cache, max_workers=max_workers)
    video_results = _probe_many(video_paths, probe_video, cache, max_workers=max_workers)

    valid_paths = {str(path) for path in image_paths + video_paths}
    cache.prune(valid_paths)
    cache.save()
    merged = {}
    merged.update(image_results)
    merged.update(video_results)
    return merged


def orientation(width: float | int | None, height: float | int | None) -> str | None:
    if width is None or height is None:
        return None
    if isinstance(width, float) and math.isnan(width):
        return None
    if isinstance(height, float) and math.isnan(height):
        return None
    if width == 0 or height == 0:
        return None
    if math.isclose(width, height):
        return "square"
    return "landscape" if width > height else "portrait"
