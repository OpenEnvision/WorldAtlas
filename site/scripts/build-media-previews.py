#!/usr/bin/env python3
"""Build lightweight WebP previews for the /uncertain WebGL archive."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps


def latest_catalog(cache_dir: Path) -> Path:
    candidates = sorted(
        cache_dir.glob("media-catalog-*.json"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No media catalog found under {cache_dir}")
    return candidates[0]


def read_items(catalog_path: Path) -> list[dict]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog = payload.get("catalog", payload)
    items = catalog.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Invalid media catalog: {catalog_path}")
    return [item for item in items if item.get("source") in {"images", "videos"}]


def safe_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe catalog path: {value}")
    return relative


def preview_path(output_root: Path, item: dict) -> Path:
    relative = safe_relative_path(str(item.get("path", "")))
    return output_root / "preview" / str(item["source"]) / f"{relative.as_posix()}.webp"


def cached_derivative_path(cache_root: Path, item: dict, asset_version: str) -> Path:
    derivative_kind = "image" if item["source"] == "images" else "poster"
    cache_key = f"{asset_version}/{derivative_kind}/{item['source']}/{item.get('path', '')}"
    encoded = base64.urlsafe_b64encode(cache_key.encode("utf-8")).decode("ascii").rstrip("=")
    cache_directory = "image-thumbs" if item["source"] == "images" else "posters"
    return cache_root / cache_directory / f"{encoded}.jpg"


def save_image_preview(source: Path, destination: Path, width: int, quality: int, force: bool) -> str:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.webp")

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

        if image.width > width:
            height = max(1, round(image.height * width / image.width))
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        image.save(temporary, "WEBP", quality=quality, method=5, optimize=True)

    os.replace(temporary, destination)
    return "written"


def save_video_preview(
    ffmpeg: str,
    source: Path,
    destination: Path,
    width: int,
    quality: int,
    force: bool,
) -> str:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.webp")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        "0.1",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"scale='min({width},iw)':-2:flags=lanczos",
        "-c:v",
        "libwebp",
        "-q:v",
        str(quality),
        "-compression_level",
        "4",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        message = result.stderr.strip() or f"ffmpeg exited with {result.returncode}"
        raise RuntimeError(message)

    os.replace(temporary, destination)
    return "written"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    site_root = script_dir.parent
    local_ffmpeg = site_root.parent / ".tools" / "bin" / "ffmpeg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="TOS upload staging root")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument(
        "--derivative-cache",
        type=Path,
        default=None,
        help="Optional existing site/.cache root to use when original media is unavailable",
    )
    parser.add_argument("--asset-version", default="20260502-fps-v4")
    parser.add_argument("--kind", choices=("all", "images", "videos"), default="all")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--quality", type=int, default=72)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--ffmpeg",
        default=os.environ.get("FFMPEG", str(local_ffmpeg) if local_ffmpeg.exists() else "ffmpeg"),
    )
    args = parser.parse_args()
    args.site_root = site_root
    return args


def main() -> int:
    args = parse_args()
    catalog_path = (args.catalog or latest_catalog(args.site_root / ".cache")).resolve()
    output_root = args.output.resolve()
    items = read_items(catalog_path)
    if args.kind != "all":
        items = [item for item in items if item.get("source") == args.kind]
    if args.limit > 0:
        items = items[: args.limit]

    roots = {
        "images": args.image_root.resolve() if args.image_root else None,
        "videos": args.video_root.resolve() if args.video_root else None,
    }
    required_sources = {str(item["source"]) for item in items}
    derivative_cache = args.derivative_cache.resolve() if args.derivative_cache else None
    missing_roots = sorted(
        source for source in required_sources if roots[source] is None and derivative_cache is None
    )
    if missing_roots:
        raise ValueError(f"Missing source root(s): {', '.join('--' + source[:-1] + '-root' for source in missing_roots)}")

    ffmpeg = shutil.which(args.ffmpeg) if "videos" in required_sources else ""
    if (
        "videos" in required_sources
        and not ffmpeg
        and (derivative_cache is None or roots["videos"] is not None)
    ):
        raise FileNotFoundError("ffmpeg is required for video posters; install it or pass --ffmpeg /path/to/ffmpeg")

    jobs: list[tuple[dict, Path, Path, bool]] = []
    missing_sources: list[str] = []
    for item in items:
        source_kind = str(item["source"])
        cached_source = (
            cached_derivative_path(derivative_cache, item, args.asset_version)
            if derivative_cache
            else None
        )
        original_root = roots[source_kind]
        original_source = (
            original_root / safe_relative_path(str(item.get("path", "")))
            if original_root
            else None
        )
        source = cached_source if cached_source and cached_source.is_file() else original_source
        destination = preview_path(output_root, item)
        if source and source.is_file():
            jobs.append((item, source, destination, source == cached_source))
        else:
            missing_sources.append(str(original_source or cached_source))

    counters = {"written": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []

    def run(job: tuple[dict, Path, Path, bool]) -> str:
        item, source, destination, source_is_derivative = job
        if item["source"] == "images" or source_is_derivative:
            return save_image_preview(source, destination, args.width, args.quality, args.force)
        return save_video_preview(ffmpeg, source, destination, args.width, args.quality, args.force)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            _, source, _, _ = futures[future]
            try:
                counters[future.result()] += 1
            except Exception as error:  # noqa: BLE001 - report every bad asset without dropping the batch
                counters["failed"] += 1
                failures.append(f"{source}: {error}")

            if completed % 100 == 0 or completed == len(futures):
                print(f"processed {completed}/{len(futures)}", file=sys.stderr)

    report = {
        "catalog": str(catalog_path),
        "output": str(output_root),
        "selected": len(items),
        "matched": len(jobs),
        "missing": len(missing_sources),
        **counters,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if missing_sources:
        print("Missing source examples:", file=sys.stderr)
        print("\n".join(missing_sources[:20]), file=sys.stderr)
    if failures:
        print("Failed preview examples:", file=sys.stderr)
        print("\n".join(failures[:20]), file=sys.stderr)

    return 1 if counters["failed"] or (missing_sources and not args.allow_missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
