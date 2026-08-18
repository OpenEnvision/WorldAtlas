#!/usr/bin/env python3
"""Curate, link, and render the homepage video-static dataset atlases.

The official 4,000-case video-static split contains ten scene categories in
photorealistic and stylized styles. This script ranks a technically strong
probe pool per category, measures real source frames, removes near duplicates,
exports six cases per group as symlinks, and renders one 2401 x 1350 WebP atlas
per style. No generated pixels are introduced.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_PATH = Path(__file__).with_name("build-dynamic-dataset-atlases.py")
SPEC = importlib.util.spec_from_file_location("dataset_atlas_shared", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load shared atlas helpers: {SCRIPT_PATH}")
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_ROOT = shared.DEFAULT_VIDEO_ROOT
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "dataset_showcase_selection"
DEFAULT_SITE_ASSETS = REPO_ROOT / "site" / "assets"
DEFAULT_CURATION = Path(__file__).with_name("dataset-showcase-curation.json")

STYLES = ("photorealistic", "stylized")
CATEGORIES = (
    "dining_spaces",
    "living_spaces",
    "passageways",
    "public_spaces",
    "workspaces",
    "city",
    "suburb",
    "aquatic_landscape",
    "terrestrial_landscape",
    "verdant_landscape",
)
CATEGORY_LABELS = {
    "dining_spaces": "Dining spaces",
    "living_spaces": "Living spaces",
    "passageways": "Passageways",
    "public_spaces": "Public spaces",
    "workspaces": "Workspaces",
    "city": "City",
    "suburb": "Suburb",
    "aquatic_landscape": "Aquatic landscape",
    "terrestrial_landscape": "Terrestrial landscape",
    "verdant_landscape": "Verdant landscape",
}
DURATION_HINTS = {"0-5s": 5.0, "5-10s": 10.0, "10-20s": 20.0}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--site-assets", type=Path, default=DEFAULT_SITE_ASSETS)
    parser.add_argument("--curation", type=Path, default=DEFAULT_CURATION)
    parser.add_argument("--workers", type=int, default=min(16, max(4, os.cpu_count() or 4)))
    parser.add_argument("--probe-per-group", type=int, default=18)
    parser.add_argument("--selected-per-group", type=int, default=6)
    parser.add_argument("--shortlist-per-group", type=int, default=12)
    parser.add_argument("--webp-quality", type=int, default=88)
    return parser.parse_args()


def discover(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    static_root = root / "static"
    for path in sorted(static_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in shared.VIDEO_EXTENSIONS:
            continue
        relative = path.relative_to(static_root)
        if len(relative.parts) < 5:
            continue
        style, setting, category, bucket = relative.parts[:4]
        if style not in STYLES or category not in CATEGORIES:
            continue
        size = path.stat().st_size
        duration_hint = DURATION_HINTS.get(bucket, 5.0)
        records.append(
            {
                "suite": "video_static",
                "style": style,
                "setting": setting,
                "category": category,
                "duration_bucket": bucket,
                "source_path": str(path.resolve()),
                "relative_path": relative.as_posix(),
                "filename": path.name,
                "size_bytes": size,
                "bytes_per_second": size / duration_hint,
                "duration_seconds": 0.0,
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "frame_count": 0,
                "aspect_ratio": 16 / 9,
                "aspect_score": 1.0,
                "resolution_score": 0.45,
                "analysis_error": "",
            }
        )
    return records


def grouped(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        groups[(str(item["style"]), str(item["category"]))].append(item)
    return dict(groups)


def prepare_probe_pool(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for _, group in grouped(records).items():
        logs = [math.log1p(float(item["bytes_per_second"])) for item in group]
        for item in group:
            bitrate = shared.robust_unit(logs, math.log1p(float(item["bytes_per_second"])))
            bucket_bonus = {"0-5s": 0.88, "5-10s": 1.0, "10-20s": 0.96}.get(
                str(item["duration_bucket"]), 0.90
            )
            item["cheap_score"] = 0.86 * bitrate + 0.14 * bucket_bonus

        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in group:
            by_bucket[str(item["duration_bucket"])].append(item)
        if len(by_bucket) == 1:
            selected.extend(sorted(group, key=lambda item: -float(item["cheap_score"]))[:count])
            continue

        quota = max(1, math.ceil(count / len(by_bucket)))
        candidates: list[dict[str, Any]] = []
        for bucket in sorted(by_bucket):
            candidates.extend(
                sorted(by_bucket[bucket], key=lambda item: -float(item["cheap_score"]))[:quota]
            )
        candidates.sort(key=lambda item: -float(item["cheap_score"]))
        selected.extend(candidates[:count])
    return selected


def assign_scores(records: list[dict[str, Any]]) -> None:
    for _, group in grouped(records).items():
        valid = [item for item in group if not item.get("analysis_error") and "motion_change" in item]
        metric_names = (
            "sharpness",
            "entropy",
            "contrast",
            "edge_density",
            "colorfulness",
            "bytes_per_second",
        )
        metrics = {name: [float(item[name]) for item in valid] for name in metric_names}
        for item in group:
            reasons: list[str] = []
            if item.get("analysis_error") or "motion_change" not in item:
                reasons.append("analysis_error")
            if min(int(item.get("width", 0)), int(item.get("height", 0))) < 360:
                reasons.append("low_resolution")
            if not 0.72 <= float(item.get("aspect_ratio", 0)) <= 2.45:
                reasons.append("unsupported_aspect")
            if float(item.get("duration_seconds", 0)) < 2.0:
                reasons.append("too_short")
            if float(item.get("exposure", 0)) < 0.14:
                reasons.append("bad_exposure")
            normalized = {
                name: shared.robust_unit(values, float(item.get(name, 0)))
                for name, values in metrics.items()
            }
            camera_motion = shared.clamp(float(item.get("motion_change", 0)) / 0.055)
            duration = shared.clamp(float(item.get("duration_seconds", 0)) / 10.0)
            item["selection_score"] = (
                0.19 * normalized["sharpness"]
                + 0.11 * normalized["entropy"]
                + 0.10 * normalized["contrast"]
                + 0.08 * normalized["edge_density"]
                + 0.06 * normalized["colorfulness"]
                + 0.11 * normalized["bytes_per_second"]
                + 0.12 * float(item.get("exposure", 0))
                + 0.08 * float(item.get("resolution_score", 0))
                + 0.07 * float(item.get("aspect_score", 0))
                + 0.05 * camera_motion
                + 0.03 * duration
            )
            item["eligible"] = not reasons
            item["exclusion_reason"] = ";".join(reasons)


def load_curation(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("video_static", {}))


def apply_curation(
    key: tuple[str, str],
    automatic: list[dict[str, Any]],
    group: list[dict[str, Any]],
    curation: dict[str, list[str]],
    count: int,
) -> list[dict[str, Any]]:
    requested = curation.get("/".join(key), [])
    if not requested:
        return automatic[:count]
    by_filename = {str(item["filename"]): item for item in group}
    by_relative = {str(item["relative_path"]): item for item in group}
    selected: list[dict[str, Any]] = []
    for value in requested:
        item = by_relative.get(str(value)) or by_filename.get(str(value))
        if item is None:
            raise ValueError(f"Curated video-static case not found for {'/'.join(key)}: {value}")
        selected.append(item)
    used = {str(item["relative_path"]) for item in selected}
    selected.extend(item for item in automatic if str(item["relative_path"]) not in used)
    return selected[:count]


def draw_header(
    canvas: Image.Image,
    position: tuple[int, int],
    style: str,
    category: str,
    index: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = position
    draw.text(
        (x, y + 2),
        f"{shared.STYLE_LABELS[style]}  ·  STATIC VIDEO",
        font=shared.font(14, True),
        fill=shared.BLUE,
    )
    circle = (x, y + 25, x + 34, y + 59)
    draw.ellipse(circle, outline=shared.HAIRLINE, width=2)
    number = f"{index:02d}"
    number_font = shared.font(13, True)
    box = draw.textbbox((0, 0), number, font=number_font)
    draw.text(
        (x + 17 - (box[2] - box[0]) / 2, y + 42 - (box[3] - box[1]) / 2 - 1),
        number,
        font=number_font,
        fill=shared.MUTED,
    )
    draw.text((x + 46, y + 27), CATEGORY_LABELS[category], font=shared.font(23), fill=shared.INK)


def record_frames(record: dict[str, Any]) -> list[Image.Image]:
    frames = list(record.get("_review_frames", []))
    if frames:
        return frames
    frames, _ = shared.read_video_frames(
        Path(str(record["source_path"])),
        fractions=(0.15, 0.50, 0.85),
        max_width=720,
        metadata=record,
    )
    return frames


def render_atlas(
    style: str,
    selected: dict[tuple[str, str], list[dict[str, Any]]],
) -> Image.Image:
    canvas = Image.new("RGB", (shared.ATLAS_WIDTH, shared.ATLAS_HEIGHT), shared.WHITE)
    outer_x, outer_y = 18, 18
    column_gap, row_gap = 16, 18
    cell_width, cell_height = 460, 648
    header_height, hero_height = 72, 390
    thumb_gap, thumb_height = 6, 154
    thumb_width = (cell_width - 4 * thumb_gap) // 5
    for index, category in enumerate(CATEGORIES):
        row, column = divmod(index, 5)
        x = outer_x + column * (cell_width + column_gap)
        y = outer_y + row * (cell_height + row_gap)
        draw_header(canvas, (x, y), style, category, index + 1)
        records = selected[(style, category)]
        hero_frames = record_frames(records[0])
        if not hero_frames:
            raise ValueError(f"No cached frames for {records[0]['source_path']}")
        middle = len(hero_frames) // 2
        canvas.paste(shared.crop(hero_frames[middle], (cell_width, hero_height)), (x, y + header_height))

        overlay_width, overlay_height = 94, 62
        overlay_gap, overlay_padding = 4, 5
        strip_width = 3 * overlay_width + 2 * overlay_gap + 2 * overlay_padding
        strip_height = overlay_height + 2 * overlay_padding
        strip_x = x + cell_width - strip_width - 10
        strip_y = y + header_height + hero_height - strip_height - 10
        ImageDraw.Draw(canvas).rounded_rectangle(
            (strip_x, strip_y, strip_x + strip_width, strip_y + strip_height),
            radius=7,
            fill=shared.WHITE,
        )
        storyboard = [hero_frames[0], hero_frames[middle], hero_frames[-1]]
        for frame_index, frame in enumerate(storyboard):
            tile = shared.crop(frame, (overlay_width, overlay_height))
            canvas.paste(
                tile,
                (strip_x + overlay_padding + frame_index * (overlay_width + overlay_gap), strip_y + overlay_padding),
            )

        thumb_y = y + header_height + hero_height + thumb_gap
        for thumb_index, record in enumerate(records[1:6]):
            frames = record_frames(record)
            if not frames:
                continue
            tile = shared.crop(frames[len(frames) // 2], (thumb_width, thumb_height))
            canvas.paste(tile, (x + thumb_index * (thumb_width + thumb_gap), thumb_y))
        if row == 0:
            ImageDraw.Draw(canvas).line(
                (x, y + cell_height + row_gap // 2, x + cell_width, y + cell_height + row_gap // 2),
                fill=(239, 241, 244),
                width=1,
            )
    return canvas


def render_review_sheet(
    style: str,
    shortlist: dict[tuple[str, str], list[dict[str, Any]]],
    output: Path,
    count: int,
) -> None:
    width, header_height, row_height = 2400, 54, 196
    height = header_height + len(CATEGORIES) * row_height + 18
    margin, label_width, gap = 18, 220, 5
    tile_width = (width - margin * 2 - label_width - gap * (count - 1)) // count
    canvas = Image.new("RGB", (width, height), shared.WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 12),
        f"{shared.STYLE_LABELS[style].title()} · video static shortlist",
        font=shared.font(26, True),
        fill=shared.INK,
    )
    for row, category in enumerate(CATEGORIES):
        y = header_height + row * row_height
        draw.text((margin, y + 8), CATEGORY_LABELS[category], font=shared.font(20, True), fill=shared.INK)
        draw.text((margin, y + 38), "Technical quality + scene diversity", font=shared.font(13), fill=shared.MUTED)
        for index, record in enumerate(shortlist[(style, category)][:count]):
            x = margin + label_width + index * (tile_width + gap)
            media_y, media_h = y + 5, row_height - 42
            frames = record_frames(record)
            if frames:
                while len(frames) < 3:
                    frames.append(frames[-1].copy())
                frame_width = max(1, (tile_width - 4) // 3)
                tile = Image.new("RGB", (tile_width, media_h), shared.WHITE)
                for frame_index, frame in enumerate((frames[0], frames[len(frames) // 2], frames[-1])):
                    tile.paste(
                        shared.crop(frame, (frame_width, media_h)),
                        (frame_index * (frame_width + 2), 0),
                    )
                canvas.paste(tile, (x, media_y))
            draw.text(
                (x, y + row_height - 29),
                f"{index + 1:02d}  {record['filename'][:18]}",
                font=shared.font(11),
                fill=shared.MUTED,
            )
        draw.line((margin, y + row_height - 2, width - margin, y + row_height - 2), fill=shared.HAIRLINE)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)


def write_selection(
    selected: dict[tuple[str, str], list[dict[str, Any]]],
    output_dir: Path,
    video_root: Path,
) -> list[dict[str, Any]]:
    root = output_dir / "selected" / "video_static"
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                path.unlink()
    rows: list[dict[str, Any]] = []
    for (style, category), records in selected.items():
        for rank, record in enumerate(records, 1):
            source = Path(str(record["source_path"]))
            destination = root / style / category / f"{rank:02d}_{source.name}"
            shared.ensure_symlink(source, destination)
            row = shared.clean_record(record)
            row.update(
                selection_rank=rank,
                selection_link=str(destination.relative_to(output_dir)),
            )
            rows.append(row)
    payload = {
        "schema_version": 1,
        "selection_method": "deterministic video quality + scene diversity + human curation overrides",
        "generated_pixels": False,
        "source_root": str((video_root / "static").resolve()),
        "selected_count": len(rows),
        "items": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "video_static_selection_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "video_static_selection_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    args = arguments()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    site_assets = args.site_assets.resolve()
    records = discover(video_root)
    print(f"Inventory: {len(records)} video-static cases", flush=True)
    groups = grouped(records)
    expected = {(style, category) for style in STYLES for category in CATEGORIES}
    if set(groups) != expected:
        raise ValueError(f"Unexpected video-static groups: missing={sorted(expected - set(groups))}")
    probe_pool = prepare_probe_pool(records, args.probe_per_group)
    print(f"Scoring source frames for {len(probe_pool)} strong candidates", flush=True)
    with ThreadPoolExecutor(max_workers=max(2, min(args.workers, 12))) as executor:
        analyzed = list(executor.map(shared.analyze_video_visual, probe_pool))
    assign_scores(analyzed)
    analyzed_groups = grouped(analyzed)
    curation = load_curation(args.curation)
    selected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    shortlist: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, group in analyzed_groups.items():
        shortlist[key] = shared.diverse_selection(group, args.shortlist_per_group)
        automatic = shared.diverse_selection(group, args.selected_per_group)
        selected[key] = apply_curation(key, automatic, group, curation, args.selected_per_group)
        if len(selected[key]) < args.selected_per_group:
            raise ValueError(f"Not enough eligible video-static cases for {'/'.join(key)}")

    review_dir = output_dir / "review"
    for style in STYLES:
        render_review_sheet(
            style,
            shortlist,
            review_dir / f"video_static_{style}_shortlist.jpg",
            args.shortlist_per_group,
        )
    rows = write_selection(selected, output_dir, video_root)
    site_assets.mkdir(parents=True, exist_ok=True)
    atlas_paths = []
    for style in STYLES:
        path = site_assets / f"dataset-atlas-video-static-{style}.webp"
        render_atlas(style, selected).save(path, format="WEBP", quality=args.webp_quality, method=6)
        atlas_paths.append(str(path))
    summary = {
        "source_count": len(records),
        "visual_probe_count": len(analyzed),
        "selected_count": len(rows),
        "selected_per_group": {"/".join(key): len(items) for key, items in selected.items()},
        "atlas_paths": atlas_paths,
        "selection_dir": str(output_dir / "selected" / "video_static"),
        "review_dir": str(review_dir),
    }
    (output_dir / "video_static_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
