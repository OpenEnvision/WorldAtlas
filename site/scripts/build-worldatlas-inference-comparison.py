#!/usr/bin/env python3
"""Build the category-led, same-case WorldAtlas Arena inference showcase.

The selection unit is a benchmark case, not a model.  For every official
static-image data category this builder selects one photorealistic and one
stylized multi-stage camera case.  Every available evaluated model result for
that case is then shown on a synchronized, consistently ordered comparison
grid.  This makes the edit promotional while keeping the comparison honest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_BUILDER = REPO_ROOT / "site" / "scripts" / "build-model-inference-showcase.py"
BENCHMARK_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "benchmark"
    / "manifest"
    / "worldarena_manifest_local_data_20260722.jsonl"
)
DEFAULT_BUILD_DIR = REPO_ROOT / "artifacts" / "worldatlas_inference_comparison"
DEFAULT_OUTPUT = REPO_ROOT / "site" / "assets" / "worldatlas_arena_inference_showcase.mp4"
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")

WIDTH = 1920
HEIGHT = 1080
FPS = 24
GRID_COLUMNS = 4
GRID_ROWS = 3
PAGE_SIZE = GRID_COLUMNS * GRID_ROWS
PAGE_DURATION = 2.5
CATEGORY_DURATION = 1.5
INTRO_DURATION = 5.0
END_DURATION = 5.0

SCENE_ORDER = [
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
]

SCENE_LABELS = {
    "dining_spaces": "DINING SPACES",
    "living_spaces": "LIVING SPACES",
    "passageways": "PASSAGEWAYS",
    "public_spaces": "PUBLIC SPACES",
    "workspaces": "WORKSPACES",
    "city": "CITY",
    "suburb": "SUBURB",
    "aquatic_landscape": "AQUATIC LANDSCAPE",
    "terrestrial_landscape": "TERRESTRIAL LANDSCAPE",
    "verdant_landscape": "VERDANT LANDSCAPE",
}

SCENE_COLORS = {
    "dining_spaces": (255, 173, 82),
    "living_spaces": (244, 126, 111),
    "passageways": (167, 139, 250),
    "public_spaces": (92, 190, 255),
    "workspaces": (75, 224, 209),
    "city": (255, 199, 87),
    "suburb": (127, 210, 124),
    "aquatic_landscape": (75, 170, 255),
    "terrestrial_landscape": (224, 153, 91),
    "verdant_landscape": (78, 218, 145),
}

# These camera-specialist models make the case-level motion probe comparable
# across categories.  Strong generalists fill any unavailable probe slots.
MOTION_PROBE_LABELS = [
    "CameraCtrl",
    "Stable Virtual Camera V2",
    "MotionCtrl",
    "Matrix-Game 3.5",
    "AC3D",
    "LingBot-World",
]

# Motion ranking is followed by a visual audit.  These overrides favor scenes
# whose shared visual anchors remain legible across the comparison grid.
MANUAL_CASE_OVERRIDES = {
    # The numerically strongest dining candidate drifts from a kitchen into a
    # sunflower garden in several model outputs.  This visually steadier case
    # keeps the shared kitchen/dining anchors recognizable across the grid.
    ("photorealistic", "dining_spaces"): (
        "image_static_photorealistic_dining_spaces_0076"
    ),
    ("photorealistic", "terrestrial_landscape"): (
        "image_static_photorealistic_terrestrial_landscape_0076"
    ),
    ("stylized", "verdant_landscape"): "image_static_stylized_verdant_landscape_0160",
}

# Remove these model outputs from every comparison without mixing sample IDs.
# The remaining models stay unique and naturally close the grid gaps.
GLOBAL_CLIP_EXCLUSIONS = {"CameraCtrl", "MotionCtrl"}

MANUAL_CLIP_PROMOTIONS = {
    "image_static_photorealistic_dining_spaces_0076": {"Matrix-Game 1": 2},
}


def load_legacy_builder():
    spec = importlib.util.spec_from_file_location("worldatlas_legacy_showcase", LEGACY_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LEGACY_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_legacy_builder()
PROBE_FFMPEG = CORE.ffmpeg_path(None)


def run(command: list[str], *, quiet: bool = False) -> None:
    if not quiet:
        print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ffmpeg_path(explicit: str | None = None) -> str:
    return CORE.ffmpeg_path(explicit)


def load_metadata() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with BENCHMARK_MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("suite") == "image_static":
                rows[row["sample_id"]] = row
    return rows


def build_clip_index() -> tuple[dict[str, dict[str, Any]], list[str], dict[str, int]]:
    manifest = json.loads(CORE.FINAL_RESULTS_MANIFEST.read_text(encoding="utf-8"))
    static_pool = CORE.load_canonical_pool(
        manifest["reports"], "image_static", CORE.STATIC_WEIGHTS
    )
    reconstruction_pool = CORE.load_reconstruction_pool()

    by_sample: dict[str, dict[str, Any]] = defaultdict(dict)
    # Canonical benchmark scores win ties; reconstruction runs only expand the
    # model universe when a display label is not already represented.
    for source_index, source in enumerate((static_pool, reconstruction_pool)):
        for clips in source.values():
            for clip in clips:
                existing = by_sample[clip.sample_id].get(clip.model_label)
                if existing is None:
                    by_sample[clip.sample_id][clip.model_label] = clip
                elif source_index == 0 and clip.score > existing.score:
                    by_sample[clip.sample_id][clip.model_label] = clip

    model_coverage: dict[str, int] = defaultdict(int)
    for clips in by_sample.values():
        for label in clips:
            model_coverage[label] += 1
    model_order = sorted(model_coverage, key=lambda label: (-model_coverage[label], label.lower()))
    stats = {
        "canonical_static_models": len(static_pool),
        "reconstruction_extension_models": len(reconstruction_pool),
        "display_model_universe": len(model_order),
    }
    return by_sample, model_order, stats


def decode_probe_frames(path: str, *, pixel_format: str, max_frames: int = 15) -> list[np.ndarray]:
    channels = 1 if pixel_format == "gray" else 3
    width, height = 320, 180
    command = [
        PROBE_FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-an",
        "-sn",
        "-dn",
        "-vf",
        "fps=1.5,scale=320:180:force_original_aspect_ratio=decrease:flags=bilinear,"
        "pad=320:180:(ow-iw)/2:(oh-ih)/2:color=black",
        "-frames:v",
        str(max_frames),
        "-pix_fmt",
        pixel_format,
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    frame_size = width * height * channels
    frame_count = len(result.stdout) // frame_size
    frames: list[np.ndarray] = []
    for index in range(frame_count):
        payload = result.stdout[index * frame_size : (index + 1) * frame_size]
        if channels == 1:
            frames.append(np.frombuffer(payload, dtype=np.uint8).reshape(height, width).copy())
        else:
            frames.append(
                np.frombuffer(payload, dtype=np.uint8).reshape(height, width, channels).copy()
            )
    return frames


def sampled_frames(path: str, count: int = 9) -> tuple[list[np.ndarray], float]:
    frames = decode_probe_frames(path, pixel_format="gray", max_frames=max(9, count))
    return frames, 0.0


def measure_video_motion(path: str) -> dict[str, float]:
    frames, duration = sampled_frames(path)
    if len(frames) < 4:
        return {"motion": 0.0, "duration": duration, "samples": float(len(frames))}
    pair_scores: list[float] = []
    for left, right in zip(frames, frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            left,
            right,
            None,
            0.5,
            4,
            25,
            4,
            7,
            1.5,
            0,
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        margin_y = max(1, magnitude.shape[0] // 12)
        margin_x = max(1, magnitude.shape[1] // 12)
        center = magnitude[margin_y:-margin_y, margin_x:-margin_x]
        if center.size == 0:
            center = magnitude
        diagonal = math.hypot(center.shape[1], center.shape[0])
        median_flow = float(np.median(center)) / max(1.0, diagonal)
        broad_flow = float(np.percentile(center, 70)) / max(1.0, diagonal)
        frame_delta = float(np.mean(cv2.absdiff(left, right))) / 255.0
        pair_scores.append((median_flow + 0.35 * broad_flow) * 1000.0 + 0.12 * frame_delta)
    return {
        "motion": round(float(statistics.median(pair_scores)), 6),
        "duration": round(duration, 4),
        "samples": float(len(frames)),
    }


def candidate_probe_paths(clips: list[Any], limit: int = 5) -> list[str]:
    by_label = {clip.model_label: clip for clip in clips}
    selected: list[Any] = []
    used: set[str] = set()
    for label in MOTION_PROBE_LABELS:
        clip = by_label.get(label)
        if clip is not None and Path(clip.path).exists():
            selected.append(clip)
            used.add(label)
        if len(selected) == limit:
            break
    if len(selected) >= limit:
        return [clip.path for clip in selected[:limit]]
    for clip in sorted(clips, key=lambda item: item.score, reverse=True):
        if clip.model_label in used or not Path(clip.path).exists():
            continue
        selected.append(clip)
        used.add(clip.model_label)
        if len(selected) == limit:
            break
    return [clip.path for clip in selected]


def select_command(args: argparse.Namespace) -> None:
    build_dir = Path(args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    print("Loading official static inference pools...", flush=True)
    by_sample, model_order, pool_stats = build_clip_index()
    metadata = load_metadata()

    shortlisted: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_probe_paths: set[str] = set()
    for scene in SCENE_ORDER:
        for style in ("photorealistic", "stylized"):
            candidates: list[dict[str, Any]] = []
            for sample_id, row in metadata.items():
                if row.get("scene") != scene or row.get("style") != style:
                    continue
                camera_path = row.get("camera_path") or []
                if len(camera_path) < 3:
                    continue
                clips = [
                    clip
                    for clip in by_sample.get(sample_id, {}).values()
                    if (clip.model, clip.sample_id) not in CORE.MANUAL_REJECTIONS
                    and Path(clip.path).exists()
                ]
                if len(clips) < 12:
                    continue
                mean_quality = sum(clip.score for clip in clips) / len(clips)
                candidates.append(
                    {
                        "sample_id": sample_id,
                        "row": row,
                        "clips": clips,
                        "coverage": len(clips),
                        "mean_quality": mean_quality,
                    }
                )
            if not candidates:
                raise RuntimeError(f"no multi-stage candidates for {style}/{scene}")
            max_coverage = max(item["coverage"] for item in candidates)
            coverage_floor = max(12, max_coverage - 3)
            eligible = [item for item in candidates if item["coverage"] >= coverage_floor]
            eligible.sort(
                key=lambda item: (item["mean_quality"], item["coverage"]), reverse=True
            )
            shortlist = eligible[: int(args.shortlist)]
            for item in shortlist:
                item["probe_paths"] = candidate_probe_paths(item["clips"])
                all_probe_paths.update(item["probe_paths"])
            shortlisted[(style, scene)] = shortlist

    cache_path = build_dir / "motion_cache_ffmpeg_v2.json"
    if cache_path.exists():
        motion_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        motion_cache = {}
    missing = sorted(path for path in all_probe_paths if path not in motion_cache)
    print(
        f"Measuring camera motion for {len(missing)} new videos "
        f"({len(all_probe_paths)} probe videos total)...",
        flush=True,
    )
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(measure_video_motion, path): path for path in missing}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    motion_cache[path] = future.result()
                except Exception as exc:  # pragma: no cover - defensive audit path
                    print(f"motion probe failed: {path}: {exc}", file=sys.stderr)
                    motion_cache[path] = {"motion": 0.0, "duration": 0.0, "samples": 0.0}
                completed += 1
                if completed % 100 == 0:
                    print(f"  measured {completed}/{len(missing)}", flush=True)
        cache_path.write_text(
            json.dumps(motion_cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    selected_cases: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(SCENE_ORDER):
        category_cases: list[dict[str, Any]] = []
        for style in ("photorealistic", "stylized"):
            candidates = shortlisted[(style, scene)]
            for item in candidates:
                scores = [
                    float(motion_cache.get(path, {}).get("motion", 0.0))
                    for path in item["probe_paths"]
                ]
                item["motion_score"] = statistics.median(scores) if scores else 0.0
            candidates.sort(
                key=lambda item: (
                    item["motion_score"],
                    item["coverage"],
                    item["mean_quality"],
                ),
                reverse=True,
            )
            override_sample = MANUAL_CASE_OVERRIDES.get((style, scene))
            winner = next(
                (item for item in candidates if item["sample_id"] == override_sample),
                candidates[0],
            )
            row = winner["row"]
            ordered_clips = sorted(
                winner["clips"],
                key=lambda clip: model_order.index(clip.model_label),
            )
            ordered_clips = [
                clip
                for clip in ordered_clips
                if clip.model_label not in GLOBAL_CLIP_EXCLUSIONS
            ]
            for model_label, target_index in MANUAL_CLIP_PROMOTIONS.get(
                winner["sample_id"], {}
            ).items():
                promoted_clip = next(
                    (clip for clip in ordered_clips if clip.model_label == model_label),
                    None,
                )
                if promoted_clip is not None:
                    ordered_clips.remove(promoted_clip)
                    ordered_clips.insert(target_index, promoted_clip)
            case = {
                "case_index": len(selected_cases) + 1,
                "category_index": scene_index + 1,
                "sample_id": winner["sample_id"],
                "scene": scene,
                "scene_label": SCENE_LABELS[scene],
                "style": style,
                "environment": row.get("environment") or "",
                "camera_path": row.get("camera_path") or [],
                "prompt": row.get("prompt_target") or row.get("prompt_current") or "",
                "reference_path": row.get("reference_path") or row.get("path") or "",
                "available_model_count": len(ordered_clips),
                "mean_quality": round(winner["mean_quality"], 6),
                "motion_score": round(winner["motion_score"], 6),
                "motion_probe_paths": winner["probe_paths"],
                "clips": [clip.to_json() for clip in ordered_clips],
            }
            selected_cases.append(case)
            category_cases.append(case)
            print(
                f"Selected {SCENE_LABELS[scene]:24s} {style:14s} "
                f"{winner['sample_id']}  models={len(ordered_clips):2d} "
                f"motion={winner['motion_score']:.3f}",
                flush=True,
            )
        categories.append(
            {
                "category_index": scene_index + 1,
                "scene": scene,
                "scene_label": SCENE_LABELS[scene],
                "cases": category_cases,
            }
        )

    payload = {
        "schema_version": 2,
        "brand": "WorldAtlas Arena",
        "selection_policy": {
            "unit": "same_case_all_available_models",
            "category_order": SCENE_ORDER,
            "styles_per_category": ["photorealistic", "stylized"],
            "required_camera_stages": 3,
            "motion_metric": "median dense optical-flow displacement across fixed model probes",
            "page_size": PAGE_SIZE,
        },
        "pool_stats": pool_stats,
        "model_universe": model_order,
        "category_count": len(categories),
        "case_count": len(selected_cases),
        "unique_inference_views": sum(case["available_model_count"] for case in selected_cases),
        "categories": categories,
    }
    selection_path = build_dir / "selection.json"
    selection_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_motion_review_sheets(build_dir, selected_cases)
    print(
        f"Wrote {selection_path}: {payload['case_count']} cases, "
        f"{payload['unique_inference_views']} same-case inference views",
        flush=True,
    )


def read_preview_frames(path: str) -> list[Image.Image]:
    decoded = decode_probe_frames(path, pixel_format="rgb24", max_frames=15)
    if not decoded:
        return []
    frames: list[Image.Image] = []
    for index in (0, len(decoded) // 2, len(decoded) - 1):
        image = Image.fromarray(decoded[index])
        image.thumbnail((430, 205), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (430, 205), (5, 9, 16))
        canvas.paste(image, ((430 - image.width) // 2, (205 - image.height) // 2))
        frames.append(canvas)
    return frames


def make_motion_review_sheets(build_dir: Path, cases: list[dict[str, Any]]) -> None:
    review_dir = build_dir / "motion_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    regular = ImageFont.truetype(str(FONT_REGULAR), 17)
    bold = ImageFont.truetype(str(FONT_BOLD), 22)
    mono = ImageFont.truetype(str(FONT_MONO), 14)
    for page_index in range(math.ceil(len(cases) / 4)):
        page_cases = cases[page_index * 4 : (page_index + 1) * 4]
        sheet = Image.new("RGB", (1920, 1030), (4, 8, 15))
        draw = ImageDraw.Draw(sheet)
        draw.text((42, 24), "WorldAtlas Arena  //  CAMERA-MOTION REVIEW", font=bold, fill=(245, 249, 255))
        draw.text((1878, 28), f"SHEET {page_index + 1:02d}", font=mono, fill=(79, 226, 211), anchor="ra")
        for row_index, case in enumerate(page_cases):
            y = 72 + row_index * 236
            color = SCENE_COLORS[case["scene"]]
            draw.rounded_rectangle((28, y, 1892, y + 218), radius=10, fill=(10, 16, 26), outline=color, width=2)
            draw.rectangle((28, y, 36, y + 218), fill=color)
            draw.text((58, y + 20), f"CASE {case['case_index']:02d}  {case['scene_label']}", font=bold, fill=(255, 255, 255))
            draw.text((58, y + 55), case["style"].upper(), font=mono, fill=color)
            camera = "  →  ".join(str(item).replace("_", " ").upper() for item in case["camera_path"])
            draw.text((58, y + 86), camera, font=mono, fill=(189, 207, 227))
            draw.text((58, y + 116), f"{case['available_model_count']} MODELS  //  MOTION {case['motion_score']:.3f}", font=mono, fill=(255, 199, 87))
            draw.text((58, y + 150), case["sample_id"][-44:], font=regular, fill=(136, 155, 177))
            probe_path = case["motion_probe_paths"][0]
            for frame_index, frame in enumerate(read_preview_frames(probe_path)):
                x = 530 + frame_index * 448
                sheet.paste(frame, (x, y + 7))
                draw.text((x + 12, y + 18), ("START", "MID", "END")[frame_index], font=mono, fill=(255, 255, 255))
        sheet.save(review_dir / f"motion_review_{page_index + 1:02d}.jpg", quality=92)


def image_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def fit_text(draw: ImageDraw.ImageDraw, text: str, max_size: int, width: int) -> ImageFont.FreeTypeFont:
    for size in range(max_size, 13, -2):
        candidate = image_font(FONT_BOLD, size)
        box = draw.textbbox((0, 0), text, font=candidate)
        if box[2] - box[0] <= width:
            return candidate
    return image_font(FONT_BOLD, 14)


def comparison_grid_layout(count: int) -> list[tuple[int, int, int, int]]:
    """Return gap-free (x, y, width, height) boxes for one comparison page."""
    row_counts_by_total = {
        1: [1],
        2: [2],
        3: [3],
        4: [2, 2],
        5: [3, 2],
        6: [3, 3],
        7: [4, 3],
        8: [4, 4],
        9: [3, 3, 3],
        10: [4, 3, 3],
        11: [4, 4, 3],
        12: [4, 4, 4],
    }
    if count not in row_counts_by_total:
        raise ValueError(f"comparison page requires 1-{PAGE_SIZE} clips, found {count}")
    row_counts = row_counts_by_total[count]
    row_total = len(row_counts)
    boxes: list[tuple[int, int, int, int]] = []
    for row_index, column_total in enumerate(row_counts):
        y = row_index * HEIGHT // row_total
        bottom = (row_index + 1) * HEIGHT // row_total
        for column_index in range(column_total):
            x = column_index * WIDTH // column_total
            right = (column_index + 1) * WIDTH // column_total
            boxes.append((x, y, right - x, bottom - y))
    return boxes


def save_case_overlay(
    case: dict[str, Any],
    clips: list[dict[str, Any]],
    page_index: int,
    page_count: int,
    destination: Path,
) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    color = SCENE_COLORS[case["scene"]]
    draw.rectangle((0, 0, WIDTH, 105), fill=(3, 8, 15, 226))
    draw.rectangle((0, 101, WIDTH, 106), fill=(*color, 255))
    draw.text((42, 35), "WorldAtlas Arena", font=image_font(FONT_BOLD, 29), fill=(250, 252, 255, 255))
    heading = f"{case['scene_label']}  ·  {case['style'].upper()}  ·  CASE {case['case_index']:02d}"
    draw.text((960, 25), heading, font=fit_text(draw, heading, 26, 850), fill=(255, 255, 255, 255), anchor="ma")
    camera = "  →  ".join(str(item).replace("_", " ").upper() for item in case["camera_path"])
    draw.text((960, 64), camera, font=image_font(FONT_MONO, 17), fill=(199, 215, 232, 255), anchor="ma")

    layout = comparison_grid_layout(len(clips))
    label_size = 30 if len(clips) == 1 else 24 if len(clips) <= 3 else 21 if len(clips) <= 6 else 18
    label_font = image_font(FONT_BOLD, label_size)
    bar_height = 78 if len(clips) == 1 else 66 if len(clips) <= 3 else 54 if len(clips) <= 6 else 43
    for clip, (x, y, tile_width, tile_height) in zip(clips, layout):
        label_y = y + tile_height - bar_height
        draw.rectangle((x, label_y, x + tile_width, y + tile_height), fill=(3, 8, 15, 188))
        draw.rectangle((x, label_y, x + 5, y + tile_height), fill=(*color, 255))
        display = clip["model_label"]
        if len(display) > 31:
            display = display[:29] + "…"
        draw.text(
            (x + 15, label_y + bar_height / 2),
            display,
            font=label_font,
            fill=(255, 255, 255, 255),
            anchor="lm",
        )
        draw.rectangle(
            (x, y, x + tile_width - 1, y + tile_height - 1),
            outline=(255, 255, 255, 35),
            width=1,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def probe_duration(path: str) -> float:
    try:
        result = subprocess.run(
            [PROBE_FFMPEG, "-hide_banner", "-i", path],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 5.0
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        return 5.0
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if not math.isfinite(duration) or duration < 0.25:
        return 5.0
    return min(20.0, duration)


def normalized_filter(
    index: int,
    duration: float,
    output: str,
    *,
    tile_width: int,
    tile_height: int,
) -> str:
    speed = PAGE_DURATION / max(0.25, duration)
    return (
        f"[{index}:v]trim=start=0:end={duration:.4f},"
        f"setpts=(PTS-STARTPTS)*{speed:.8f},fps={FPS},"
        f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={tile_width}:{tile_height},setsar=1,"
        "eq=contrast=1.025:saturation=1.055:brightness=-0.004,"
        f"tpad=stop_mode=clone:stop_duration=0.2,trim=duration={PAGE_DURATION:.3f}[{output}]"
    )


def encode_args(threads: int = 10) -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level:v",
        "4.2",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-sc_threshold",
        "0",
        "-threads",
        str(threads),
        "-movflags",
        "+faststart",
    ]


def render_case_page(
    ffmpeg: str,
    case: dict[str, Any],
    clips: list[dict[str, Any]],
    page_index: int,
    page_count: int,
    destination: Path,
    overlay_path: Path,
    force: bool,
) -> Path:
    if destination.exists() and not force:
        validation = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(destination), "-frames:v", "1", "-f", "null", "-"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if destination.stat().st_size > 10_000 and validation.returncode == 0:
            return destination
    save_case_overlay(case, clips, page_index, page_count, overlay_path)
    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    durations: list[float] = []
    for clip in clips:
        command.extend(["-i", clip["path"]])
        durations.append(probe_duration(clip["path"]))
    command.extend(["-loop", "1", "-framerate", str(FPS), "-i", str(overlay_path)])
    layout = comparison_grid_layout(len(clips))
    filters = [
        normalized_filter(
            index,
            duration,
            f"v{index}",
            tile_width=layout[index][2],
            tile_height=layout[index][3],
        )
        for index, duration in enumerate(durations)
    ]
    positions = "|".join(
        f"{x}_{y}" for x, y, _tile_width, _tile_height in layout
    )
    inputs = "".join(f"[v{index}]" for index in range(len(clips)))
    if len(clips) == 1:
        filters.append("[v0]null[grid]")
    else:
        filters.append(
            f"{inputs}xstack=inputs={len(clips)}:layout={positions}:fill=0x050912[stacked]"
        )
        filters.append(
            f"[stacked]pad={WIDTH}:{HEIGHT}:0:0:color=0x050912[grid]"
        )
    filters.append(f"[{len(clips)}:v]format=rgba[ui]")
    filters.append("[grid][ui]overlay=0:0:shortest=1[composite]")
    filters.append(
        f"[composite]format=yuv420p,fade=t=in:st=0:d=0.10,"
        f"fade=t=out:st={PAGE_DURATION - 0.10:.3f}:d=0.10[out]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-t",
            f"{PAGE_DURATION:.3f}",
            "-r",
            str(FPS),
            "-an",
            *encode_args(10),
            str(destination),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(command, quiet=True)
    return destination


def save_title_overlay(
    destination: Path,
    *,
    title: str,
    subtitle: str,
    color: tuple[int, int, int],
    darken: int,
) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (2, 6, 12, darken))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((280, 365, 1640, 715), radius=20, fill=(3, 8, 15, 214), outline=(*color, 145), width=2)
    draw.rectangle((280, 365, 960, 373), fill=(*color, 255))
    draw.rectangle((960, 365, 1640, 373), fill=(255, 199, 87, 255))
    draw.text((960, 485), title, font=fit_text(draw, title, 66, 1240), fill=(255, 255, 255, 255), anchor="mm")
    draw.text((960, 605), subtitle, font=image_font(FONT_REGULAR, 30), fill=(203, 219, 236, 255), anchor="mm")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def render_title_segment(
    ffmpeg: str,
    background: Path,
    destination: Path,
    overlay_path: Path,
    *,
    duration: float,
    title: str,
    subtitle: str,
    color: tuple[int, int, int],
    darken: int,
    force: bool,
) -> Path:
    if destination.exists() and not force:
        if destination.stat().st_mtime >= background.stat().st_mtime:
            return destination
    save_title_overlay(
        overlay_path,
        title=title,
        subtitle=subtitle,
        color=color,
        darken=darken,
    )
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stream_loop",
            "-1",
            "-i",
            str(background),
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-i",
            str(overlay_path),
            "-filter_complex",
            f"[0:v]fps={FPS},scale={WIDTH}:{HEIGHT},trim=duration={duration:.3f},setpts=PTS-STARTPTS[base];"
            f"[1:v]format=rgba[ui];[base][ui]overlay=0:0:shortest=1,format=yuv420p[out]",
            "-map",
            "[out]",
            "-t",
            f"{duration:.3f}",
            "-r",
            str(FPS),
            "-an",
            *encode_args(10),
            str(destination),
        ],
        quiet=True,
    )
    return destination


def concat_videos(ffmpeg: str, videos: list[Path], destination: Path, force: bool) -> Path:
    if destination.exists() and not force:
        newest_input = max(path.stat().st_mtime for path in videos)
        if destination.stat().st_mtime >= newest_input:
            return destination
    list_path = destination.with_suffix(".concat.txt")
    list_path.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in videos), encoding="utf-8"
    )
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        quiet=True,
    )
    return destination


def render_command(args: argparse.Namespace) -> None:
    build_dir = Path(args.build_dir).resolve()
    selection_path = build_dir / "selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"missing {selection_path}; run select first")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    categories = payload["categories"]
    cases = [case for category in categories for case in category["cases"]]
    if len(cases) != 20:
        raise RuntimeError(f"expected 20 category/style cases, found {len(cases)}")
    ffmpeg = ffmpeg_path(args.ffmpeg)
    force = bool(args.force)
    render_root = build_dir / "render_v2"
    page_dir = render_root / "pages"
    segment_dir = render_root / "segments"
    overlay_dir = render_root / "overlays"
    for path in (page_dir, segment_dir, overlay_dir):
        path.mkdir(parents=True, exist_ok=True)

    page_jobs: list[tuple[dict[str, Any], list[dict[str, Any]], int, int, Path, Path]] = []
    pages_by_case: dict[str, list[Path]] = {}
    for case in cases:
        chunks = [case["clips"][start : start + PAGE_SIZE] for start in range(0, len(case["clips"]), PAGE_SIZE)]
        outputs: list[Path] = []
        for page_index, clips in enumerate(chunks):
            destination = page_dir / f"case_{case['case_index']:02d}_page_{page_index + 1:02d}.mp4"
            overlay = overlay_dir / f"case_{case['case_index']:02d}_page_{page_index + 1:02d}.png"
            page_jobs.append((case, clips, page_index, len(chunks), destination, overlay))
            outputs.append(destination)
        pages_by_case[case["sample_id"]] = outputs

    print(
        f"Rendering {len(page_jobs)} synchronized comparison pages "
        f"for {payload['unique_inference_views']} inference views...",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, 3))) as executor:
        futures = [
            executor.submit(
                render_case_page,
                ffmpeg,
                case,
                clips,
                page_index,
                page_count,
                destination,
                overlay,
                force,
            )
            for case, clips, page_index, page_count, destination, overlay in page_jobs
        ]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            print(f"  comparison pages {completed}/{len(page_jobs)}", flush=True)

    first_page = pages_by_case[cases[0]["sample_id"]][0]
    intro = render_title_segment(
        ffmpeg,
        first_page,
        segment_dir / "00_intro.mp4",
        overlay_dir / "00_intro.png",
        duration=INTRO_DURATION,
        title="WorldAtlas Arena",
        subtitle="Model Inference · Direct Comparison",
        color=(75, 224, 209),
        darken=104,
        force=force,
    )

    timeline: list[Path] = [intro]
    sections: list[dict[str, Any]] = [
        {"name": "intro", "duration_seconds": INTRO_DURATION, "path": str(intro)}
    ]
    for category in categories:
        first_case = category["cases"][0]
        background = pages_by_case[first_case["sample_id"]][0]
        color = SCENE_COLORS[category["scene"]]
        category_segment = render_title_segment(
            ffmpeg,
            background,
            segment_dir / f"category_{category['category_index']:02d}.mp4",
            overlay_dir / f"category_{category['category_index']:02d}.png",
            duration=CATEGORY_DURATION,
            title=category["scene_label"],
            subtitle="Photorealistic + Stylized · Multi-stage Camera Motion",
            color=color,
            darken=122,
            force=force,
        )
        timeline.append(category_segment)
        sections.append(
            {
                "name": f"category_{category['scene']}",
                "duration_seconds": CATEGORY_DURATION,
                "path": str(category_segment),
            }
        )
        for case in category["cases"]:
            case_pages = pages_by_case[case["sample_id"]]
            timeline.extend(case_pages)
            sections.append(
                {
                    "name": f"case_{case['case_index']:02d}",
                    "sample_id": case["sample_id"],
                    "style": case["style"],
                    "model_count": case["available_model_count"],
                    "page_count": len(case_pages),
                    "duration_seconds": PAGE_DURATION * len(case_pages),
                    "paths": [str(path) for path in case_pages],
                }
            )

    last_page = pages_by_case[cases[-1]["sample_id"]][-1]
    endcard = render_title_segment(
        ffmpeg,
        last_page,
        segment_dir / "99_endcard.mp4",
        overlay_dir / "99_endcard.png",
        duration=END_DURATION,
        title="WorldAtlas Arena",
        subtitle="A Unified Arena for Generative World Models",
        color=(255, 199, 87),
        darken=132,
        force=force,
    )
    timeline.append(endcard)
    sections.append(
        {"name": "endcard", "duration_seconds": END_DURATION, "path": str(endcard)}
    )

    video_only = render_root / "worldatlas_arena_inference_showcase_video_only.mp4"
    print("Assembling category-led timeline...", flush=True)
    concat_videos(ffmpeg, timeline, video_only, force)
    expected_duration = sum(
        INTRO_DURATION if index == 0 else END_DURATION if index == len(timeline) - 1 else
        CATEGORY_DURATION if path.name.startswith("category_") else PAGE_DURATION
        for index, path in enumerate(timeline)
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_audio:
        shutil.copy2(video_only, output)
        soundtrack = None
    else:
        soundtrack = CORE.render_soundtrack(
            ffmpeg,
            expected_duration + 1.0,
            render_root / "procedural_soundtrack.m4a",
            force,
        )
        run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_only),
                "-i",
                str(soundtrack),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output),
            ],
            quiet=True,
        )

    poster = output.with_name(f"{output.stem}_poster.jpg")
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "2.0",
            "-i",
            str(output),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(poster),
        ],
        quiet=True,
    )
    render_manifest = {
        "schema_version": 2,
        "brand": "WorldAtlas Arena",
        "output": str(output),
        "poster": str(poster),
        "resolution": [WIDTH, HEIGHT],
        "fps": FPS,
        "expected_duration_seconds": round(expected_duration, 3),
        "category_count": payload["category_count"],
        "case_count": payload["case_count"],
        "model_universe_count": len(payload["model_universe"]),
        "unique_inference_views": payload["unique_inference_views"],
        "comparison_page_count": len(page_jobs),
        "has_procedural_soundtrack": soundtrack is not None,
        "sections": sections,
    }
    (build_dir / "render_manifest.json").write_text(
        json.dumps(render_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Finished {output} ({expected_duration:.1f}s, {payload['case_count']} cases, "
        f"{payload['unique_inference_views']} inference views)",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--ffmpeg")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 8))
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--shortlist", type=int, default=12)
    select_parser.set_defaults(func=select_command)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    render_parser.add_argument("--force", action="store_true")
    render_parser.add_argument("--no-audio", action="store_true")
    render_parser.set_defaults(func=render_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
