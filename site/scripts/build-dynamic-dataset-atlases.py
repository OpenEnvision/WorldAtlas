#!/usr/bin/env python3
"""Curate and render the homepage dynamic-image and dynamic-video atlases.

The script intentionally keeps the showcase grounded in real benchmark inputs:

* rank every dynamic-image case with deterministic, no-reference visual signals;
* rank a technically strong video shortlist, then score sampled source frames;
* remove near duplicates and repository-known image/mask failures;
* optionally lock human-reviewed choices in ``dataset-showcase-curation.json``;
* export the selected source cases as symlinks with a machine-readable manifest;
* compose two 2401 x 1350 WebP boards matching the homepage dataset atlas.

No generated pixels are introduced. Video poster frames are decoded directly
from the selected source clips.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORLD_DATA = REPO_ROOT / "data_external" / "OpenEnvision" / "worldarena"
DEFAULT_IMAGE_ROOT = DEFAULT_WORLD_DATA / "image_test_balanced_5000"
DEFAULT_VIDEO_ROOT = DEFAULT_WORLD_DATA / "videos_test_split_4971"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "dataset_showcase_selection"
DEFAULT_SITE_ASSETS = REPO_ROOT / "site" / "assets"
DEFAULT_CURATION = Path(__file__).with_name("dataset-showcase-curation.json")
DEFAULT_KNOWN_FAILURES = REPO_ROOT / "config" / "image_mask_known_failures.json"
DEFAULT_IMAGE_MASK_AUDIT = DEFAULT_OUTPUT / "image_mask_audit" / "mask_records.jsonl"
FFMPEG = REPO_ROOT / ".tools" / "bin" / "ffmpeg"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
STYLES = ("photorealistic", "stylized")
CATEGORIES = ("articulated", "deformable", "fluid", "multi_motion", "rigid")
CATEGORY_LABELS = {
    "articulated": "Articulated motion",
    "deformable": "Deformable motion",
    "fluid": "Fluid dynamics",
    "multi_motion": "Multi-motion",
    "rigid": "Rigid-body motion",
}
STYLE_LABELS = {
    "photorealistic": "PHOTOREALISTIC",
    "stylized": "STYLIZED",
}

ATLAS_WIDTH = 2401
ATLAS_HEIGHT = 1350
BLUE = (52, 120, 229)
INK = (22, 26, 33)
MUTED = (94, 101, 112)
HAIRLINE = (223, 227, 232)
WHITE = (255, 255, 255)
MIN_SHOWCASE_VIDEO_MASK_AREA = 0.003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--site-assets", type=Path, default=DEFAULT_SITE_ASSETS)
    parser.add_argument("--curation", type=Path, default=DEFAULT_CURATION)
    parser.add_argument("--known-failures", type=Path, default=DEFAULT_KNOWN_FAILURES)
    parser.add_argument("--image-mask-audit", type=Path, default=DEFAULT_IMAGE_MASK_AUDIT)
    parser.add_argument("--workers", type=int, default=min(16, max(4, os.cpu_count() or 4)))
    parser.add_argument("--selected-per-group", type=int, default=6)
    parser.add_argument("--shortlist-per-group", type=int, default=12)
    parser.add_argument("--video-probe-pool", type=int, default=40)
    parser.add_argument("--video-mask-probe-per-group", type=int, default=24)
    parser.add_argument("--webp-quality", type=int, default=88)
    return parser.parse_args()


def clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def robust_unit(values: Sequence[float], value: float) -> float:
    array = np.asarray([x for x in values if math.isfinite(x)], dtype=np.float64)
    if array.size < 2:
        return 0.5
    low, high = np.percentile(array, [10, 90])
    if high <= low + 1e-10:
        return 0.5
    return clamp((value - float(low)) / float(high - low))


def dct_matrix(size: int) -> np.ndarray:
    n = np.arange(size, dtype=np.float32)
    k = np.arange(size, dtype=np.float32)[:, None]
    matrix = np.cos(math.pi / size * (n + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


DCT = dct_matrix(32)


def phash(gray: Image.Image) -> int:
    pixels = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    block = (DCT @ pixels @ DCT.T)[:8, :8]
    threshold = float(np.median(block.reshape(-1)[1:]))
    result = 0
    for bit in (block >= threshold).reshape(-1):
        result = (result << 1) | int(bit)
    return result


def hamming(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def image_features(image: Image.Image) -> dict[str, Any]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    thumb = image.copy()
    thumb.thumbnail((360, 360), Image.Resampling.LANCZOS)
    rgb8 = np.asarray(thumb, dtype=np.uint8)
    rgb = rgb8.astype(np.float32) / 255.0
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    if min(gray.shape) < 3:
        raise ValueError("image is too small")

    laplace = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    gx = np.abs(np.diff(gray, axis=1))[:-1]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    gradient = np.sqrt(gx * gx + gy * gy)
    histogram, _ = np.histogram(gray, bins=64, range=(0, 1))
    probability = histogram / max(float(histogram.sum()), 1.0)
    probability = probability[probability > 0]
    entropy = float(-(probability * np.log2(probability)).sum() / 6.0)
    mean_luma = float(np.mean(gray))
    clipped = float(np.mean((gray <= 0.02) | (gray >= 0.98)))
    exposure = clamp(
        math.exp(-((mean_luma - 0.5) / 0.36) ** 4) * math.exp(-3.2 * clipped)
    )
    aspect = width / max(height, 1)
    aspect_score = math.exp(-((math.log(max(aspect, 1e-5) / 1.60)) / 0.58) ** 2)
    resolution_score = clamp(min(width, height) / 1080)
    colorfulness = float(
        np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2))
    )

    layout = np.asarray(thumb.resize((4, 4), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255
    low_hist, _ = np.histogram(gray, bins=8, range=(0, 1))
    descriptor = np.concatenate([layout, 2 * low_hist / max(float(low_hist.sum()), 1.0)])
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-8)

    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": float(aspect),
        "mean_luma": mean_luma,
        "contrast": float(np.std(gray)),
        "entropy": entropy,
        "sharpness": float(np.var(laplace)),
        "edge_density": float(np.mean(gradient >= 0.055)),
        "colorfulness": colorfulness,
        "exposure": exposure,
        "aspect_score": float(aspect_score),
        "resolution_score": resolution_score,
        "_descriptor": descriptor,
        "_phash": phash(Image.fromarray(np.uint8(gray * 255), mode="L")),
    }


def analyze_image(path: Path, image_root: Path, known_failure_images: set[str]) -> dict[str, Any]:
    relative = path.relative_to(image_root).as_posix()
    parts = path.relative_to(image_root).parts
    style, category = parts[1], parts[3]
    record: dict[str, Any] = {
        "suite": "image_dynamic",
        "style": style,
        "category": category,
        "source_path": str(path.resolve()),
        "relative_path": relative,
        "filename": path.name,
        "analysis_error": "",
    }
    try:
        with Image.open(path) as opened:
            record.update(image_features(opened))
    except Exception as error:  # one bad asset must not block the full showcase
        record["analysis_error"] = f"{type(error).__name__}: {error}"
    record["known_failure"] = relative in known_failure_images
    return record


VIDEO_SIZE_PATTERN = re.compile(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)")
DURATION_PATTERN = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
FPS_PATTERN = re.compile(r"(?:,|\s)(\d+(?:\.\d+)?)\s+fps(?:,|\s)")


def probe_video_ffmpeg(path: Path) -> dict[str, Any]:
    if not FFMPEG.exists():
        raise FileNotFoundError(f"Bundled ffmpeg not found: {FFMPEG}")
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "0",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    output = completed.stderr.decode("utf-8", errors="replace")
    duration_match = DURATION_PATTERN.search(output)
    size_matches = VIDEO_SIZE_PATTERN.findall(output)
    fps_match = FPS_PATTERN.search(output)
    if not duration_match or not size_matches:
        raise ValueError(f"ffmpeg could not probe video metadata: {output[-500:]}")
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    # The first plausible WxH pair on the video stream is the encoded frame size.
    width, height = next(
        ((int(left), int(right)) for left, right in size_matches if int(left) >= 64 and int(right) >= 64),
        (0, 0),
    )
    if width <= 0 or height <= 0:
        raise ValueError("ffmpeg returned no valid video dimensions")
    fps = float(fps_match.group(1)) if fps_match else 0.0
    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": int(round(duration * fps)) if fps > 0 else 0,
    }


def read_video_frames(
    path: Path,
    fractions: Sequence[float] = (0.10, 0.30, 0.50, 0.70, 0.90),
    max_width: int = 360,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[Image.Image], dict[str, Any]]:
    metadata = dict(metadata or probe_video_ffmpeg(path))
    duration = float(metadata.get("duration_seconds", 0))
    source_width = int(metadata.get("width", 0))
    source_height = int(metadata.get("height", 0))
    if duration <= 0 or source_width <= 0 or source_height <= 0:
        raise ValueError("invalid video metadata")
    output_width = min(max_width, source_width) if max_width else source_width
    output_width = max(2, output_width - output_width % 2)
    output_height = max(2, round(source_height * output_width / source_width))
    output_height -= output_height % 2

    frame_bytes = output_width * output_height * 3
    if len(fractions) > 1 and duration <= 30.0 and max(source_width, source_height) <= 1920:
        # Short clips are fastest in one sequential decode; process startup and
        # repeated keyframe seeks otherwise cost more than reading the clip.
        frame_rate = len(fractions) / max(duration, 0.1)
        command = [
            str(FFMPEG),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={frame_rate:.9f},scale={output_width}:{output_height}",
            "-frames:v",
            str(len(fractions)),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0 and len(completed.stdout) < frame_bytes:
            error = completed.stderr.decode("utf-8", errors="replace")
            raise ValueError(f"ffmpeg frame decode failed: {error[-500:]}")
        frame_count = min(len(fractions), len(completed.stdout) // frame_bytes)
        frames: list[Image.Image] = []
        for index in range(frame_count):
            start = index * frame_bytes
            array = np.frombuffer(completed.stdout[start : start + frame_bytes], dtype=np.uint8)
            frames.append(Image.fromarray(array.reshape(output_height, output_width, 3).copy(), mode="RGB"))
        return frames, metadata

    frames = []
    for fraction in fractions:
        command = [
            str(FFMPEG),
            "-v",
            "error",
            "-ss",
            f"{duration * clamp(float(fraction)):.6f}",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={output_width}:{output_height}",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        # Input-side seeking avoids decoding an entire long-form 4K source just
        # to obtain a few review frames. It is dramatically faster for curation.
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0 and len(completed.stdout) < frame_bytes:
            error = completed.stderr.decode("utf-8", errors="replace")
            raise ValueError(f"ffmpeg frame decode failed: {error[-500:]}")
        if len(completed.stdout) < frame_bytes:
            continue
        array = np.frombuffer(completed.stdout[:frame_bytes], dtype=np.uint8)
        frames.append(Image.fromarray(array.reshape(output_height, output_width, 3).copy(), mode="RGB"))
    return frames, metadata


RESOLUTION_PATTERN = re.compile(r"(?<!\d)(\d{3,4})_(\d{3,4})(?!\d)")


def cheap_video_probe(path: Path, video_root: Path) -> dict[str, Any]:
    parts = path.relative_to(video_root).parts
    style, category = parts[1], parts[3]
    record: dict[str, Any] = {
        "suite": "video_dynamic",
        "style": style,
        "category": category,
        "source_path": str(path.resolve()),
        "relative_path": path.relative_to(video_root).as_posix(),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "analysis_error": "",
    }
    match = RESOLUTION_PATTERN.search(path.stem)
    width, height = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    fps_match = re.search(r"(?<!\d)(\d{2,3})fps(?!\d)", path.stem)
    fps = float(fps_match.group(1)) if fps_match else 0.0
    aspect = width / max(height, 1) if width and height else 16 / 9
    record.update(
        frame_count=0,
        fps=fps,
        width=width,
        height=height,
        duration_seconds=0.0,
        aspect_ratio=aspect,
        aspect_score=math.exp(-((math.log(max(aspect, 1e-5) / 1.60)) / 0.58) ** 2),
        resolution_score=(clamp(min(width, height) / 1080) if width and height else 0.45),
        bytes_per_second=float(path.stat().st_size),
    )
    return record


def video_mask_source_stem(mask_directory_name: str) -> str:
    return re.sub(r"__scene_\d+$", "", mask_directory_name)


def analyze_video_mask_directory(path: Path) -> dict[str, Any]:
    mask_root = path / "masks"
    files = sorted(mask_root.glob("*.png")) if mask_root.exists() else []
    result: dict[str, Any] = {
        "video_mask_path": str(path.resolve()),
        "video_mask_frame_count": len(files),
        "video_mask_quality_score": 0.0,
        "video_mask_high_quality": False,
        "video_mask_nonempty_fraction": 0.0,
        "video_mask_median_area": 0.0,
        "video_mask_resolution": "",
        "video_mask_error": "",
    }
    if not files:
        result["video_mask_error"] = "no_mask_frames"
        return result
    sample_indices = sorted(set(int(value) for value in np.linspace(0, len(files) - 1, min(9, len(files)))))
    areas: list[float] = []
    edge_ratios: list[float] = []
    sizes: list[tuple[int, int]] = []
    unreadable = 0
    for index in sample_indices:
        try:
            with Image.open(files[index]) as opened:
                raw = np.asarray(opened.convert("L"), dtype=np.uint8)
        except Exception:
            unreadable += 1
            continue
        mask = raw >= 128
        height, width = mask.shape
        sizes.append((width, height))
        area = float(mask.mean())
        areas.append(area)
        if area > 0:
            horizontal = int(np.count_nonzero(mask[:, 1:] != mask[:, :-1]))
            vertical = int(np.count_nonzero(mask[1:, :] != mask[:-1, :]))
            edge_ratios.append((horizontal + vertical) / max(int(mask.sum()), 1))
    if not areas or not sizes:
        result["video_mask_error"] = "unreadable_mask_samples"
        return result
    nonempty = [area for area in areas if area > 0.0002]
    nonempty_fraction = len(nonempty) / len(areas)
    median_area = float(np.median(nonempty)) if nonempty else 0.0
    plausible_fraction = (
        sum(MIN_SHOWCASE_VIDEO_MASK_AREA <= area <= 0.95 for area in nonempty) / len(nonempty)
        if nonempty
        else 0.0
    )
    area_stability = (
        math.exp(-float(np.std(np.log(np.asarray(nonempty) + 1e-5))) / 1.4)
        if len(nonempty) >= 2
        else 0.4
    )
    edge_ratio = float(np.median(edge_ratios)) if edge_ratios else 1.0
    compactness = math.exp(-max(0.0, edge_ratio - 0.12) * 3.0)
    width, height = sizes[0]
    consistent_dimensions = len(set(sizes)) == 1
    resolution_score = clamp(min(width, height) / 360.0)
    metadata_path = path / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    expected_frames = int(metadata.get("num_mask_frames_written") or len(files))
    frame_completeness = clamp(len(files) / max(expected_frames, 1))
    score = (
        0.27 * nonempty_fraction
        + 0.20 * plausible_fraction
        + 0.16 * area_stability
        + 0.14 * compactness
        + 0.13 * resolution_score
        + 0.10 * frame_completeness
    )
    high_quality = (
        unreadable == 0
        and consistent_dimensions
        and len(files) >= 3
        and nonempty_fraction >= 0.60
        and plausible_fraction >= 0.75
        and MIN_SHOWCASE_VIDEO_MASK_AREA <= median_area <= 0.95
        and edge_ratio <= 0.75
        and min(width, height) >= 256
    )
    result.update(
        video_mask_quality_score=float(score),
        video_mask_high_quality=bool(high_quality),
        video_mask_nonempty_fraction=float(nonempty_fraction),
        video_mask_median_area=float(median_area),
        video_mask_edge_ratio=float(edge_ratio),
        video_mask_resolution=f"{width}x{height}",
        video_mask_prompt=str(metadata.get("text_prompt") or ""),
        video_mask_error="" if high_quality else "mask_quality_gate_failed",
    )
    return result


def attach_video_mask_audit(
    records: list[dict[str, Any]],
    video_root: Path,
    workers: int,
) -> None:
    directory_index: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    record_keys = {
        (str(item["style"]), str(item["category"]), Path(str(item["source_path"])).stem)
        for item in records
    }
    for style in STYLES:
        for category in CATEGORIES:
            root = video_root / "dynamic" / style / "masks" / category
            if not root.exists():
                continue
            for path in root.iterdir():
                if not path.is_dir():
                    continue
                key = (style, category, video_mask_source_stem(path.name))
                if key in record_keys:
                    directory_index[key].append(path)
    unique_directories = sorted({path for paths in directory_index.values() for path in paths})
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        diagnostics = dict(zip(unique_directories, executor.map(analyze_video_mask_directory, unique_directories)))
    empty = {
        "video_mask_path": "",
        "video_mask_frame_count": 0,
        "video_mask_quality_score": 0.0,
        "video_mask_high_quality": False,
        "video_mask_nonempty_fraction": 0.0,
        "video_mask_median_area": 0.0,
        "video_mask_resolution": "",
        "video_mask_error": "no_matching_mask_directory",
    }
    for item in records:
        key = (str(item["style"]), str(item["category"]), Path(str(item["source_path"])).stem)
        options = [diagnostics[path] for path in directory_index.get(key, [])]
        best = max(
            options,
            key=lambda value: (
                bool(value.get("video_mask_high_quality")),
                float(value.get("video_mask_quality_score", 0)),
            ),
            default=empty,
        )
        item.update(best)


def analyze_video_visual(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    if output.get("analysis_error"):
        return output
    try:
        metadata = probe_video_ffmpeg(Path(output["source_path"]))
        frames, metadata = read_video_frames(
            Path(output["source_path"]),
            fractions=(0.15, 0.50, 0.85),
            metadata=metadata,
        )
        if len(frames) < 3:
            raise ValueError(f"decoded only {len(frames)} review frames")
        features = [image_features(frame) for frame in frames]
        grays = []
        for frame in frames:
            gray = np.asarray(frame.convert("L").resize((192, 108), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            grays.append(gray)
        temporal_changes = [float(np.mean(np.abs(right - left))) for left, right in zip(grays, grays[1:])]
        middle = features[len(features) // 2]
        output.update(metadata)
        output["aspect_ratio"] = output["width"] / max(output["height"], 1)
        output["aspect_score"] = math.exp(
            -((math.log(max(output["aspect_ratio"], 1e-5) / 1.60)) / 0.58) ** 2
        )
        output["resolution_score"] = clamp(min(output["width"], output["height"]) / 1080)
        output["bytes_per_second"] = float(Path(output["source_path"]).stat().st_size) / max(
            float(output["duration_seconds"]), 0.1
        )
        for key in ("mean_luma", "contrast", "entropy", "sharpness", "edge_density", "colorfulness", "exposure"):
            output[key] = float(np.mean([item[key] for item in features]))
        output["motion_change"] = float(np.mean(temporal_changes))
        output["motion_peak"] = float(max(temporal_changes))
        output["_review_frames"] = frames
        output["_descriptor"] = middle["_descriptor"]
        output["_phash"] = middle["_phash"]
    except Exception as error:
        output["analysis_error"] = f"{type(error).__name__}: {error}"
    return output


def load_known_failure_images(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures: set[str] = set()
    for entry in payload.get("failures", []):
        if entry.get("relative_image_path"):
            failures.add(str(entry["relative_image_path"]))
        mask_path = entry.get("relative_mask_path")
        if mask_path:
            parts = Path(mask_path).parts
            if len(parts) >= 5:
                image_dir = Path(parts[0]) / parts[1] / "images" / parts[3]
                failures.update(f"{(image_dir / parts[4].split('-')[0]).as_posix()}{extension}" for extension in IMAGE_EXTENSIONS)
    for mask_path in payload.get("hashed_mask_failures", {}):
        parts = Path(mask_path).parts
        if len(parts) >= 5:
            image_dir = Path(parts[0]) / parts[1] / "images" / parts[3]
            failures.update(f"{(image_dir / parts[4].split('-')[0]).as_posix()}{extension}" for extension in IMAGE_EXTENSIONS)
    return failures


def load_image_mask_audit(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Image-mask audit is required for showcase curation: {path}. "
            "Run `python -m scripts.audit_image_masks --output "
            "artifacts/dataset_showcase_selection/image_mask_audit`."
        )
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        records[str(item.get("relative_image_path") or "")].append(item)
    return dict(records)


def mask_diagnostic_score(record: dict[str, Any]) -> float:
    features = record.get("features") or {}
    metadata = record.get("legacy_metadata") or {}
    values: list[float] = []
    if metadata.get("sam_score") is not None:
        values.append(clamp(float(metadata["sam_score"])))
    if metadata.get("box_iou") is not None:
        values.append(clamp(float(metadata["box_iou"])))
    if metadata.get("mask_quality") is not None:
        values.append(clamp((float(metadata["mask_quality"]) + 2.0) / 7.0))
    if features.get("largest_component_ratio") is not None:
        values.append(clamp(float(features["largest_component_ratio"])))
    if features.get("bbox_fill") is not None:
        values.append(clamp(float(features["bbox_fill"]) / 0.70))
    return float(np.mean(values)) if values else 0.72


def attach_image_mask_audit(
    records: list[dict[str, Any]],
    audit: dict[str, list[dict[str, Any]]],
    image_root: Path,
) -> None:
    for item in records:
        audited = audit.get(str(item["relative_path"]), [])
        decisions = [str(entry.get("assessment", {}).get("decision") or "") for entry in audited]
        mask_dir = (
            image_root
            / "dynamic"
            / str(item["style"])
            / "masks"
            / str(item["category"])
            / Path(str(item["source_path"])).stem
        )
        item.update(
            image_mask_path=str(mask_dir.resolve()) if mask_dir.exists() else "",
            mask_count=len(audited),
            mask_keep_count=sum(decision == "keep" for decision in decisions),
            mask_review_count=sum(decision == "review" for decision in decisions),
            mask_regenerate_count=sum(decision == "regenerate" for decision in decisions),
            mask_all_grade_a=bool(audited) and all(decision == "keep" for decision in decisions),
            mask_quality_score=(
                float(np.mean([mask_diagnostic_score(entry) for entry in audited]))
                if audited
                else 0.0
            ),
        )


def assign_image_scores(records: list[dict[str, Any]]) -> None:
    for _, group in grouped(records).items():
        valid = [item for item in group if not item.get("analysis_error")]
        metrics = {
            key: [float(item[key]) for item in valid]
            for key in ("sharpness", "entropy", "contrast", "edge_density", "colorfulness")
        }
        for item in group:
            reasons: list[str] = []
            if item.get("analysis_error"):
                reasons.append("analysis_error")
            if item.get("known_failure"):
                reasons.append("known_image_or_mask_failure")
            if not item.get("mask_all_grade_a"):
                reasons.append("mask_not_all_grade_a")
            if min(int(item.get("width", 0)), int(item.get("height", 0))) < 480:
                reasons.append("low_resolution")
            if not 0.72 <= float(item.get("aspect_ratio", 0)) <= 2.45:
                reasons.append("unsupported_aspect")
            if float(item.get("exposure", 0)) < 0.16:
                reasons.append("bad_exposure")
            normalized = {
                key: robust_unit(values, float(item.get(key, 0)))
                for key, values in metrics.items()
            }
            visual_score = (
                0.20 * normalized["sharpness"]
                + 0.14 * normalized["entropy"]
                + 0.12 * normalized["contrast"]
                + 0.11 * normalized["edge_density"]
                + 0.08 * normalized["colorfulness"]
                + 0.13 * float(item.get("exposure", 0))
                + 0.12 * float(item.get("resolution_score", 0))
                + 0.10 * float(item.get("aspect_score", 0))
            )
            item["selection_score"] = 0.90 * visual_score + 0.10 * float(
                item.get("mask_quality_score", 0)
            )
            item["eligible"] = not reasons
            item["exclusion_reason"] = ";".join(reasons)


def assign_video_scores(records: list[dict[str, Any]]) -> None:
    for _, group in grouped(records).items():
        valid = [item for item in group if not item.get("analysis_error") and "motion_change" in item]
        metrics = {
            key: [float(item[key]) for item in valid]
            for key in (
                "sharpness",
                "entropy",
                "contrast",
                "edge_density",
                "colorfulness",
                "motion_change",
                "bytes_per_second",
            )
        }
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
                key: robust_unit(values, float(item.get(key, 0)))
                for key, values in metrics.items()
            }
            motion = 0.75 * normalized["motion_change"] + 0.25 * clamp(float(item.get("motion_peak", 0)) / 0.16)
            duration = clamp(float(item.get("duration_seconds", 0)) / 5.0)
            item["selection_score"] = (
                0.16 * normalized["sharpness"]
                + 0.10 * normalized["entropy"]
                + 0.09 * normalized["contrast"]
                + 0.08 * normalized["edge_density"]
                + 0.06 * normalized["colorfulness"]
                + 0.18 * motion
                + 0.09 * normalized["bytes_per_second"]
                + 0.10 * float(item.get("exposure", 0))
                + 0.07 * float(item.get("resolution_score", 0))
                + 0.04 * float(item.get("aspect_score", 0))
                + 0.03 * duration
            )
            item["eligible"] = not reasons
            item["exclusion_reason"] = ";".join(reasons)


def grouped(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        result[(str(item["style"]), str(item["category"]))].append(item)
    return dict(result)


def visual_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    descriptor_left = left.get("_descriptor")
    descriptor_right = right.get("_descriptor")
    if descriptor_left is None or descriptor_right is None:
        return 1.0
    return float(np.linalg.norm(descriptor_left - descriptor_right))


def diverse_selection(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    pool = sorted(
        (item for item in candidates if item.get("eligible")),
        key=lambda item: (-float(item["selection_score"]), str(item["relative_path"])),
    )[: max(72, count * 10)]
    chosen: list[dict[str, Any]] = []
    while pool and len(chosen) < count:
        best: dict[str, Any] | None = None
        best_value = -1.0
        for item in pool:
            if any(hamming(int(item.get("_phash", 0)), int(old.get("_phash", 0))) <= 4 for old in chosen):
                continue
            diversity = 1.0 if not chosen else clamp(min(visual_distance(item, old) for old in chosen) / 0.42)
            value = 0.86 * float(item["selection_score"]) + 0.14 * diversity
            if value > best_value:
                best, best_value = item, value
        if best is None:
            break
        chosen.append(best)
        pool.remove(best)
    if len(chosen) < count:
        used = {str(item["relative_path"]) for item in chosen}
        for item in sorted(candidates, key=lambda value: -float(value.get("selection_score", 0))):
            if item.get("eligible") and str(item["relative_path"]) not in used:
                chosen.append(item)
                used.add(str(item["relative_path"]))
                if len(chosen) == count:
                    break
    return chosen


def load_curation(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"image_dynamic": {}, "video_dynamic": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "image_dynamic": dict(payload.get("image_dynamic", {})),
        "video_dynamic": dict(payload.get("video_dynamic", {})),
    }


def apply_curation(
    suite: str,
    key: tuple[str, str],
    auto: list[dict[str, Any]],
    group: list[dict[str, Any]],
    curation: dict[str, Any],
    count: int,
) -> list[dict[str, Any]]:
    requested = curation.get(suite, {}).get("/".join(key), [])
    if not requested:
        return auto[:count]
    by_filename = {str(item["filename"]): item for item in group}
    by_relative = {str(item["relative_path"]): item for item in group}
    selected: list[dict[str, Any]] = []
    for value in requested:
        item = by_relative.get(str(value)) or by_filename.get(str(value))
        if item is None:
            raise ValueError(f"Curated {suite} case not found for {'/'.join(key)}: {value}")
        selected.append(item)
    if len(selected) < count:
        used = {str(item["relative_path"]) for item in selected}
        selected.extend(item for item in auto if str(item["relative_path"]) not in used)
    return selected[:count]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), size, method=Image.Resampling.LANCZOS)


def open_source_image(record: dict[str, Any]) -> Image.Image:
    with Image.open(record["source_path"]) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def draw_cell_header(
    canvas: Image.Image,
    position: tuple[int, int],
    style: str,
    category: str,
    index: int,
    modality: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = position
    draw.text((x, y + 2), f"{STYLE_LABELS[style]}  ·  {modality}", font=font(14, True), fill=BLUE)
    circle_box = (x, y + 25, x + 34, y + 59)
    draw.ellipse(circle_box, outline=HAIRLINE, width=2)
    number = f"{index:02d}"
    number_font = font(13, True)
    number_box = draw.textbbox((0, 0), number, font=number_font)
    number_x = x + 17 - (number_box[2] - number_box[0]) / 2
    number_y = y + 42 - (number_box[3] - number_box[1]) / 2 - 1
    draw.text((number_x, number_y), number, font=number_font, fill=MUTED)
    draw.text((x + 46, y + 27), CATEGORY_LABELS[category], font=font(24), fill=INK)


def render_image_atlas(selected: dict[tuple[str, str], list[dict[str, Any]]]) -> Image.Image:
    canvas = Image.new("RGB", (ATLAS_WIDTH, ATLAS_HEIGHT), WHITE)
    outer_x, outer_y = 18, 18
    column_gap, row_gap = 16, 18
    cell_width = 460
    cell_height = 648
    header_height = 72
    hero_height = 390
    thumb_gap = 6
    thumb_height = 154
    thumb_width = (cell_width - 4 * thumb_gap) // 5

    for row, style in enumerate(STYLES):
        for column, category in enumerate(CATEGORIES):
            x = outer_x + column * (cell_width + column_gap)
            y = outer_y + row * (cell_height + row_gap)
            draw_cell_header(canvas, (x, y), style, category, column + 1, "IMAGE")
            records = selected[(style, category)]
            hero = crop(open_source_image(records[0]), (cell_width, hero_height))
            canvas.paste(hero, (x, y + header_height))
            thumb_y = y + header_height + hero_height + thumb_gap
            for index, record in enumerate(records[1:6]):
                thumb = crop(open_source_image(record), (thumb_width, thumb_height))
                canvas.paste(thumb, (x + index * (thumb_width + thumb_gap), thumb_y))
            if row == 0:
                ImageDraw.Draw(canvas).line(
                    (x, y + cell_height + row_gap // 2, x + cell_width, y + cell_height + row_gap // 2),
                    fill=(239, 241, 244),
                    width=1,
                )
    return canvas


def video_storyboard(record: dict[str, Any]) -> tuple[Image.Image, list[Image.Image]]:
    frames = list(record.get("_review_frames", []))
    if len(frames) < 3:
        frames, _ = read_video_frames(
            Path(record["source_path"]),
            fractions=(0.12, 0.50, 0.88),
            max_width=960,
            metadata=record,
        )
    if not frames:
        raise ValueError(f"Could not decode selected video: {record['source_path']}")
    while len(frames) < 3:
        frames.append(frames[-1].copy())
    middle = len(frames) // 2
    return frames[middle], [frames[0], frames[middle], frames[-1]]


def render_video_atlas(selected: dict[tuple[str, str], list[dict[str, Any]]]) -> Image.Image:
    canvas = Image.new("RGB", (ATLAS_WIDTH, ATLAS_HEIGHT), WHITE)
    outer_x, outer_y = 18, 18
    column_gap, row_gap = 16, 18
    cell_width = 460
    cell_height = 648
    header_height = 72
    hero_height = 390
    thumb_gap = 6
    thumb_height = 154
    thumb_width = (cell_width - 4 * thumb_gap) // 5

    for row, style in enumerate(STYLES):
        for column, category in enumerate(CATEGORIES):
            x = outer_x + column * (cell_width + column_gap)
            y = outer_y + row * (cell_height + row_gap)
            draw_cell_header(canvas, (x, y), style, category, column + 1, "VIDEO")
            records = selected[(style, category)]
            hero_frame, storyboard = video_storyboard(records[0])
            hero = crop(hero_frame, (cell_width, hero_height))
            canvas.paste(hero, (x, y + header_height))

            overlay_width, overlay_height = 94, 62
            overlay_gap, overlay_padding = 4, 5
            strip_width = 3 * overlay_width + 2 * overlay_gap + 2 * overlay_padding
            strip_height = overlay_height + 2 * overlay_padding
            strip_x = x + cell_width - strip_width - 10
            strip_y = y + header_height + hero_height - strip_height - 10
            ImageDraw.Draw(canvas).rounded_rectangle(
                (strip_x, strip_y, strip_x + strip_width, strip_y + strip_height),
                radius=7,
                fill=(255, 255, 255),
            )
            for index, frame in enumerate(storyboard):
                tile = crop(frame, (overlay_width, overlay_height))
                canvas.paste(tile, (strip_x + overlay_padding + index * (overlay_width + overlay_gap), strip_y + overlay_padding))

            thumb_frames: list[Image.Image] = []
            for index, record in enumerate(records[1:6]):
                frames = list(record.get("_review_frames", []))
                if not frames:
                    frames, _ = read_video_frames(
                        Path(record["source_path"]), fractions=(0.50,), max_width=640, metadata=record
                    )
                if frames:
                    thumb_frames.append(frames[len(frames) // 2])

            # Some official groups are intentionally sparse (the current
            # stylized multi-motion split has only one unique clip). Fill the
            # visual strip with distinct moments from the hero clip while the
            # exported case list remains honest and contains no duplicates.
            missing = 5 - len(thumb_frames)
            if missing > 0:
                cached = list(records[0].get("_review_frames", []))
                thumb_frames.extend(cached[:missing])
                missing = 5 - len(thumb_frames)
                if missing > 0:
                    fallback_fractions = tuple(np.linspace(0.08, 0.92, missing))
                    fallback, _ = read_video_frames(
                        Path(records[0]["source_path"]),
                        fractions=fallback_fractions,
                        max_width=640,
                        metadata=records[0],
                    )
                    thumb_frames.extend(fallback)
            while len(thumb_frames) < 5:
                thumb_frames.append(hero_frame.copy())

            thumb_y = y + header_height + hero_height + thumb_gap
            for index, frame in enumerate(thumb_frames[:5]):
                thumb = crop(frame, (thumb_width, thumb_height))
                canvas.paste(thumb, (x + index * (thumb_width + thumb_gap), thumb_y))
            if row == 0:
                ImageDraw.Draw(canvas).line(
                    (x, y + cell_height + row_gap // 2, x + cell_width, y + cell_height + row_gap // 2),
                    fill=(239, 241, 244),
                    width=1,
                )
    return canvas


def render_review_sheet(
    suite: str,
    style: str,
    shortlist: dict[tuple[str, str], list[dict[str, Any]]],
    output: Path,
    count: int,
) -> None:
    width = 2400
    header_height = 54
    row_height = 184 if suite == "image_dynamic" else 196
    height = header_height + len(CATEGORIES) * row_height + 18
    margin = 18
    label_width = 220
    gap = 5
    tile_width = (width - margin * 2 - label_width - gap * (count - 1)) // count
    canvas = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(canvas)
    title = f"{STYLE_LABELS[style].title()} · {'dynamic image' if suite == 'image_dynamic' else 'dynamic video'} shortlist"
    draw.text((margin, 12), title, font=font(26, True), fill=INK)

    for row, category in enumerate(CATEGORIES):
        y = header_height + row * row_height
        draw.text((margin, y + 8), CATEGORY_LABELS[category], font=font(20, True), fill=INK)
        draw.text((margin, y + 38), f"Top {len(shortlist[(style, category)])} technical + diverse", font=font(13), fill=MUTED)
        for index, record in enumerate(shortlist[(style, category)][:count]):
            x = margin + label_width + index * (tile_width + gap)
            media_y = y + 5
            media_h = row_height - 42
            try:
                if suite == "image_dynamic":
                    tile = crop(open_source_image(record), (tile_width, media_h))
                else:
                    frames = list(record.get("_review_frames", []))
                    if not frames:
                        frames, _ = read_video_frames(
                            Path(record["source_path"]),
                            fractions=(0.14, 0.50, 0.86),
                            max_width=360,
                            metadata=record,
                        )
                    if not frames:
                        raise ValueError("no frames")
                    while len(frames) < 3:
                        frames.append(frames[-1].copy())
                    frame_width = max(1, (tile_width - 4) // 3)
                    tile = Image.new("RGB", (tile_width, media_h), WHITE)
                    for frame_index, frame in enumerate(frames[:3]):
                        tile.paste(crop(frame, (frame_width, media_h)), (frame_index * (frame_width + 2), 0))
                canvas.paste(tile, (x, media_y))
            except Exception:
                draw.rectangle((x, media_y, x + tile_width, media_y + media_h), fill=(238, 240, 243))
            label = f"{index + 1:02d}  {record['filename'][:18]}"
            draw.text((x, y + row_height - 29), label, font=font(11), fill=MUTED)
        draw.line((margin, y + row_height - 2, width - margin, y + row_height - 2), fill=HAIRLINE, width=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)


def selected_mask_overlay(record: dict[str, Any], suite: str) -> tuple[Image.Image, bool]:
    if suite == "image_dynamic":
        source = open_source_image(record)
        mask_files = sorted(Path(str(record.get("image_mask_path") or "")).glob("*.png"))
        mask_ready = bool(record.get("mask_all_grade_a") and mask_files)
    else:
        frames = list(record.get("_review_frames", []))
        if not frames:
            frames, _ = read_video_frames(
                Path(str(record["source_path"])),
                fractions=(0.50,),
                max_width=720,
                metadata=record,
            )
        source = frames[len(frames) // 2]
        mask_root = Path(str(record.get("video_mask_path") or "")) / "masks"
        candidates = sorted(mask_root.glob("*.png")) if mask_root.exists() else []
        mask_files = [candidates[len(candidates) // 2]] if candidates else []
        mask_ready = bool(record.get("video_mask_high_quality") and mask_files)
    source = ImageOps.exif_transpose(source).convert("RGB")
    source.thumbnail((620, 360), Image.Resampling.LANCZOS)
    if not mask_ready:
        return source, False
    combined = np.zeros((source.height, source.width), dtype=bool)
    for path in mask_files:
        with Image.open(path) as opened:
            mask = np.asarray(
                opened.convert("L").resize(source.size, Image.Resampling.NEAREST),
                dtype=np.uint8,
            ) >= 128
        combined |= mask
    rgb = np.asarray(source).copy()
    rgb[combined] = (
        0.58 * rgb[combined] + 0.42 * np.asarray([255, 34, 52], dtype=np.float32)
    ).astype(np.uint8)
    inner = combined.copy()
    inner[1:-1, 1:-1] &= (
        combined[:-2, 1:-1]
        & combined[2:, 1:-1]
        & combined[1:-1, :-2]
        & combined[1:-1, 2:]
    )
    rgb[combined & ~inner] = np.asarray([255, 235, 48], dtype=np.uint8)
    return Image.fromarray(rgb), True


def render_selected_mask_sheet(
    suite: str,
    style: str,
    selected: dict[tuple[str, str], list[dict[str, Any]]],
    output: Path,
) -> None:
    width, header_height, row_height = 2400, 58, 224
    height = header_height + len(CATEGORIES) * row_height + 18
    margin, label_width, gap = 18, 220, 6
    tile_width = (width - margin * 2 - label_width - gap * 5) // 6
    canvas = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 12),
        f"{STYLE_LABELS[style].title()} · {suite.replace('_', ' ')} · selected mask overlays",
        font=font(26, True),
        fill=INK,
    )
    for row, category in enumerate(CATEGORIES):
        y = header_height + row * row_height
        records = selected[(style, category)]
        draw.text((margin, y + 8), CATEGORY_LABELS[category], font=font(20, True), fill=INK)
        ready_count = sum(
            bool(item.get("mask_all_grade_a") if suite == "image_dynamic" else item.get("video_mask_high_quality"))
            for item in records
        )
        draw.text(
            (margin, y + 38),
            f"{ready_count}/{len(records)} Grade-A masks paired",
            font=font(13),
            fill=MUTED,
        )
        for index, record in enumerate(records[:6]):
            x = margin + label_width + index * (tile_width + gap)
            overlay, ready = selected_mask_overlay(record, suite)
            tile = crop(overlay, (tile_width, row_height - 48))
            if not ready:
                gray = ImageOps.grayscale(tile).convert("RGB")
                tile = Image.blend(tile, gray, 0.72)
            canvas.paste(tile, (x, y + 4))
            status = "GRADE A" if ready else "NO GRADE-A MASK"
            color = (33, 128, 78) if ready else (172, 77, 65)
            draw.rounded_rectangle(
                (x + 6, y + 10, x + 6 + (68 if ready else 130), y + 31),
                radius=5,
                fill=(255, 255, 255),
            )
            draw.text((x + 11, y + 13), status, font=font(10, True), fill=color)
            draw.text(
                (x, y + row_height - 29),
                f"{index + 1:02d}  {record['filename'][:22]}",
                font=font(11),
                fill=MUTED,
            )
        draw.line((margin, y + row_height - 2, width - margin, y + row_height - 2), fill=HAIRLINE)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)


def ensure_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace non-symlink selection artifact: {destination}")
    destination.symlink_to(source.resolve())


def maybe_link_companion(record: dict[str, Any], destination: Path, image_root: Path, video_root: Path) -> str | None:
    if record["suite"] == "image_dynamic":
        companion_value = str(record.get("image_mask_path") or "")
    else:
        if not record.get("video_mask_high_quality"):
            return None
        companion_value = str(record.get("video_mask_path") or "")
    if not companion_value:
        return None
    companion = Path(companion_value)
    if not companion.exists():
        return None
    link_path = destination.with_name(f"{destination.stem}_masks")
    ensure_symlink(companion, link_path)
    return str(link_path)


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        output[key] = value
    return output


def write_selection(
    selected_by_suite: dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
    output_dir: Path,
    image_root: Path,
    video_root: Path,
) -> list[dict[str, Any]]:
    for suite in selected_by_suite:
        suite_root = output_dir / "selected" / suite
        if not suite_root.exists():
            continue
        for path in suite_root.rglob("*"):
            if path.is_symlink():
                path.unlink()
    rows: list[dict[str, Any]] = []
    for suite, groups in selected_by_suite.items():
        for (style, category), records in groups.items():
            for rank, record in enumerate(records, 1):
                extension = Path(str(record["source_path"])).suffix.lower()
                filename = f"{rank:02d}_{Path(str(record['source_path'])).stem}{extension}"
                destination = output_dir / "selected" / suite / style / category / filename
                ensure_symlink(Path(str(record["source_path"])), destination)
                companion = maybe_link_companion(record, destination, image_root, video_root)
                row = clean_record(record)
                row.update(
                    selection_rank=rank,
                    selection_link=str(destination.relative_to(output_dir)),
                    companion_mask_link=(str(Path(companion).relative_to(output_dir)) if companion else None),
                )
                rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "selection_method": "technical quality + visual diversity + Grade-A mask priority + human curation overrides",
        "generated_pixels": False,
        "source_roots": {
            "image_dynamic": str(image_root.resolve()),
            "video_dynamic": str(video_root.resolve()),
        },
        "selected_count": len(rows),
        "grade_a_mask_pairs": {
            "image_dynamic": sum(
                bool(item.get("mask_all_grade_a"))
                for items in selected_by_suite.get("image_dynamic", {}).values()
                for item in items
            ),
            "video_dynamic": sum(
                bool(item.get("video_mask_high_quality"))
                for items in selected_by_suite.get("video_dynamic", {}).values()
                for item in items
            ),
        },
        "items": rows,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "selection_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def shortlist_groups(
    records: list[dict[str, Any]],
    count: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, group in grouped(records).items():
        result[key] = diverse_selection(group, count)
    return result


def mask_priority_video_selection(
    candidates: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    masked = [
        item
        for item in candidates
        if item.get("eligible") and item.get("video_mask_high_quality")
    ]
    chosen = diverse_selection(masked, count)
    used = {str(item["relative_path"]) for item in chosen}
    if len(chosen) < count:
        for item in diverse_selection(candidates, max(count * 2, 16)):
            relative = str(item["relative_path"])
            if relative in used:
                continue
            chosen.append(item)
            used.add(relative)
            if len(chosen) == count:
                break
    return chosen


def prepare_video_probe_pool(
    records: list[dict[str, Any]],
    pool_size: int,
    mask_pool_size: int,
    forced_cases: set[str] | None = None,
) -> list[dict[str, Any]]:
    forced_cases = forced_cases or set()
    selected: list[dict[str, Any]] = []
    for _, group in grouped(records).items():
        valid = [item for item in group if not item.get("analysis_error")]
        bitrates = [math.log1p(float(item.get("bytes_per_second", 0))) for item in valid]
        for item in valid:
            item["cheap_score"] = (
                0.42 * float(item.get("resolution_score", 0))
                + 0.22 * float(item.get("aspect_score", 0))
                + 0.18 * robust_unit(bitrates, math.log1p(float(item.get("bytes_per_second", 0))))
                + 0.10 * clamp(float(item.get("duration_seconds", 0)) / 5.0)
                + 0.08 * clamp(float(item.get("fps", 0)) / 30.0)
            )
        group_selected = sorted(valid, key=lambda item: -float(item["cheap_score"]))[:pool_size]
        masked = sorted(
            (item for item in valid if item.get("video_mask_high_quality")),
            key=lambda item: -(
                0.76 * float(item["cheap_score"])
                + 0.24 * float(item.get("video_mask_quality_score", 0))
            ),
        )[:mask_pool_size]
        used = {str(item["relative_path"]) for item in group_selected}
        group_selected.extend(item for item in masked if str(item["relative_path"]) not in used)
        used = {str(item["relative_path"]) for item in group_selected}
        group_selected.extend(
            item
            for item in valid
            if (
                str(item["filename"]) in forced_cases
                or str(item["relative_path"]) in forced_cases
            )
            and str(item["relative_path"]) not in used
        )
        selected.extend(group_selected)
    return selected


def validate_groups(records: dict[tuple[str, str], list[dict[str, Any]]], count: int, label: str) -> None:
    expected = {(style, category) for style in STYLES for category in CATEGORIES}
    missing = expected - set(records)
    if missing:
        raise ValueError(f"Missing {label} groups: {sorted(missing)}")
    short = {key: len(value) for key, value in records.items() if len(value) < count}
    if short:
        raise ValueError(f"Not enough {label} cases for atlas: {short}")


def main() -> None:
    args = parse_args()
    image_root = args.image_root.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    site_assets = args.site_assets.resolve()
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not video_root.exists():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")

    known_failure_images = load_known_failure_images(args.known_failures)
    image_paths = sorted(
        path
        for style in STYLES
        for category in CATEGORIES
        for path in (image_root / "dynamic" / style / "images" / category).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"Analyzing {len(image_paths)} dynamic-image cases", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        image_records = list(
            executor.map(
                lambda path: analyze_image(path, image_root, known_failure_images),
                image_paths,
            )
        )
    image_mask_audit = load_image_mask_audit(args.image_mask_audit.resolve())
    attach_image_mask_audit(image_records, image_mask_audit, image_root)
    assign_image_scores(image_records)

    video_paths = sorted(
        path
        for style in STYLES
        for category in CATEGORIES
        for path in (video_root / "dynamic" / style / "videos" / category).iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    print(f"Reading metadata for {len(video_paths)} dynamic-video cases", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        video_metadata = list(executor.map(lambda path: cheap_video_probe(path, video_root), video_paths))
    print("Auditing available dynamic-video mask sequences", flush=True)
    attach_video_mask_audit(video_metadata, video_root, args.workers)
    curation = load_curation(args.curation)
    forced_video_cases = {
        str(value)
        for values in curation.get("video_dynamic", {}).values()
        for value in values
    }
    probe_pool = prepare_video_probe_pool(
        video_metadata,
        args.video_probe_pool,
        args.video_mask_probe_per_group,
        forced_video_cases,
    )
    print(f"Scoring sampled frames for {len(probe_pool)} technically strong videos", flush=True)
    with ThreadPoolExecutor(max_workers=max(2, min(args.workers, 10))) as executor:
        video_records = list(executor.map(analyze_video_visual, probe_pool))
    assign_video_scores(video_records)

    image_shortlist = shortlist_groups(image_records, args.shortlist_per_group)
    video_shortlist = {
        key: mask_priority_video_selection(group, args.shortlist_per_group)
        for key, group in grouped(video_records).items()
    }
    image_selected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    video_selected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, group in grouped(image_records).items():
        auto = diverse_selection(group, args.selected_per_group)
        image_selected[key] = apply_curation(
            "image_dynamic", key, auto, group, curation, args.selected_per_group
        )
    for key, group in grouped(video_records).items():
        auto = mask_priority_video_selection(group, args.selected_per_group)
        video_selected[key] = apply_curation(
            "video_dynamic", key, auto, group, curation, args.selected_per_group
        )

    validate_groups(image_shortlist, min(args.shortlist_per_group, args.selected_per_group), "image shortlist")
    validate_groups(video_shortlist, 1, "video shortlist")
    validate_groups(image_selected, args.selected_per_group, "image selection")
    validate_groups(video_selected, 1, "video selection")

    review_dir = output_dir / "review"
    for style in STYLES:
        render_review_sheet(
            "image_dynamic",
            style,
            image_shortlist,
            review_dir / f"image_dynamic_{style}_shortlist.jpg",
            args.shortlist_per_group,
        )
        render_review_sheet(
            "video_dynamic",
            style,
            video_shortlist,
            review_dir / f"video_dynamic_{style}_shortlist.jpg",
            args.shortlist_per_group,
        )
        render_selected_mask_sheet(
            "image_dynamic",
            style,
            image_selected,
            review_dir / f"image_dynamic_{style}_selected_masks.jpg",
        )
        render_selected_mask_sheet(
            "video_dynamic",
            style,
            video_selected,
            review_dir / f"video_dynamic_{style}_selected_masks.jpg",
        )

    rows = write_selection(
        {"image_dynamic": image_selected, "video_dynamic": video_selected},
        output_dir,
        image_root,
        video_root,
    )
    site_assets.mkdir(parents=True, exist_ok=True)
    image_atlas_path = site_assets / "dataset-atlas-image-dynamic.webp"
    video_atlas_path = site_assets / "dataset-atlas-video-dynamic.webp"
    render_image_atlas(image_selected).save(
        image_atlas_path, format="WEBP", quality=args.webp_quality, method=6
    )
    render_video_atlas(video_selected).save(
        video_atlas_path, format="WEBP", quality=args.webp_quality, method=6
    )

    summary = {
        "image_source_count": len(image_records),
        "video_source_count": len(video_metadata),
        "video_visual_probe_count": len(video_records),
        "selected_count": len(rows),
        "grade_a_mask_pairs": {
            "image_dynamic": sum(
                bool(item.get("mask_all_grade_a"))
                for items in image_selected.values()
                for item in items
            ),
            "video_dynamic": sum(
                bool(item.get("video_mask_high_quality"))
                for items in video_selected.values()
                for item in items
            ),
        },
        "selected_per_suite": {
            suite: sum(len(items) for items in groups.values())
            for suite, groups in {
                "image_dynamic": image_selected,
                "video_dynamic": video_selected,
            }.items()
        },
        "selected_per_group": {
            suite: {
                "/".join(key): len(items)
                for key, items in groups.items()
            }
            for suite, groups in {
                "image_dynamic": image_selected,
                "video_dynamic": video_selected,
            }.items()
        },
        "atlas_paths": [str(image_atlas_path), str(video_atlas_path)],
        "review_dir": str(review_dir),
        "selection_dir": str(output_dir / "selected"),
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
