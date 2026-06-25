#!/usr/bin/env python3
"""Generate a cinematic hero cover video from WorldArena benchmark data."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT.parent / "data/OpenEnvision/worldarena/image_test_balanced_5000"
VIDEO_ROOT = ROOT.parent / "data/OpenEnvision/worldarena/videos"
ANNOTATIONS = IMAGE_ROOT / "image_annotations.json"
OUTPUT = ROOT / "site/assets/benchmark_hero_cover.mp4"
PROMO_COVER = ROOT / "site/assets/worldarena-hero-cover.png"
WORK = ROOT / "artifacts/hero_cover_build"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

WIDTH, HEIGHT = 1920, 1080
FPS = 24
SEED = 20260620
MIN_VIDEO_QUALITY = 120.0  # prefer 4K sources
MIN_VIDEO_QUALITY_FALLBACK = 75.0
VIDEO_CLIP_SECONDS = 2.4
IMAGE_GALLERY_SECONDS = 6.5
COVER_SECONDS = 5.0
SHOWCASE_CLIP_COUNT = 8
SHOWCASE_CLIP_SECONDS = 2.0
IMAGES_PER_GALLERY = 48

FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_SERIF = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}
VIDEO_SCAN_ROOTS = [
    VIDEO_ROOT / "dynamic/photorealistic/videos",
    VIDEO_ROOT / "dynamic/stylized/videos",
    VIDEO_ROOT / "indoor",
    VIDEO_ROOT / "outdoor",
]


@dataclass(frozen=True)
class CategorySpec:
    key: str
    headline: str
    tags: tuple[str, ...]
    count: int
    accent: tuple[int, int, int]
    image_filter: str
    video_hint: str | None = None


def humanize(value: str | None) -> str:
    if not value:
        return "Scene"
    return value.replace("_", " ").title()


def run(cmd: list[str], *, timeout: int | None = None) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n"
            + (result.stderr or result.stdout or "")
        )


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def annotation_category_key(item: dict) -> str:
    regime = item.get("motion_regime") or "unknown"
    style = item.get("style") or "unknown"
    if regime == "static":
        environment = item.get("environment") or "unknown"
        scene = item.get("scene") or "unknown"
        return f"{regime}/{style}/{environment}/{scene}"
    motion = item.get("motion_category") or "unknown"
    return f"{regime}/{style}/{motion}"


def build_category_specs(items: list[dict]) -> list[CategorySpec]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[annotation_category_key(item)].append(item)

    specs: list[CategorySpec] = []
    for key in sorted(grouped):
        sample = grouped[key][0]
        count = len(grouped[key])
        regime = sample.get("motion_regime") or "unknown"
        style = sample.get("style") or "unknown"
        if regime == "static":
            environment = sample.get("environment") or "unknown"
            scene = sample.get("scene") or "unknown"
            headline = f"{humanize(environment)} · {humanize(scene)}"
            tags = (regime.title(), style.title(), environment.title())
            accent = {
                ("indoor", "photorealistic"): (78, 168, 255),
                ("outdoor", "photorealistic"): (72, 201, 129),
                ("indoor", "stylized"): (176, 132, 255),
                ("outdoor", "stylized"): (255, 196, 72),
            }.get((environment, style), (120, 180, 255))
            video_hint = f"{environment}/{scene}" if environment in {"indoor", "outdoor"} else None
        else:
            motion = sample.get("motion_category") or "unknown"
            headline = humanize(motion)
            tags = (regime.title(), style.title(), "Motion")
            accent = (255, 118, 88) if style == "photorealistic" else (255, 88, 196)
            video_hint = f"dynamic/{style}/videos/{motion}"

        specs.append(
            CategorySpec(
                key=key,
                headline=headline,
                tags=tags,
                count=count,
                accent=accent,
                image_filter=key,
                video_hint=video_hint,
            )
        )
    return specs


def index_annotations() -> tuple[list[dict], dict[str, list[Path]]]:
    payload = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    buckets: dict[str, list[Path]] = defaultdict(list)
    for item in items:
        rel = item.get("relative_path")
        if not rel:
            continue
        path = IMAGE_ROOT / rel
        if path.exists():
            buckets[annotation_category_key(item)].append(path)
    rng = random.Random(SEED)
    for paths in buckets.values():
        rng.shuffle(paths)
    return items, buckets


def video_quality_score(path: Path) -> float:
    name = path.name.lower()
    score = min(path.stat().st_size / (1024 * 1024), 80.0)
    if any(token in name for token in ("3840", "2160", "4k")):
        score += 200.0
    elif any(token in name for token in ("1920", "1080", "-hd_", "_hd_", "-hd.")):
        score += 95.0
    if any(token in name for token in ("sd_", "_sd_", "540_", "960_540", "540_960")):
        score -= 200.0
    if "pexels" in name:
        score += 25.0
    if "60fps" in name:
        score += 10.0
    elif "30fps" in name:
        score += 7.0
    elif "24fps" in name:
        score += 5.0
    return score


def is_ultra_hd(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("3840", "2160", "4k"))


def collect_videos(limit_per_leaf: int = 8) -> list[Path]:
    found: list[Path] = []
    leaf_buckets: dict[str, list[Path]] = defaultdict(list)

    def consider(path: Path) -> None:
        if path.suffix.lower() not in VIDEO_EXTS:
            return
        leaf = str(path.parent)
        leaf_buckets[leaf].append(path)

    def walk_bounded(directory: Path, depth: int) -> None:
        if depth < 0 or not directory.exists():
            return
        try:
            entries = list(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_file():
                consider(entry)
            elif entry.is_dir() and depth > 0:
                walk_bounded(entry, depth - 1)

    for scan_root in VIDEO_SCAN_ROOTS:
        depth = 3 if scan_root.name in {"indoor", "outdoor"} else 1
        walk_bounded(scan_root, depth)

    for paths in leaf_buckets.values():
        ranked = sorted(paths, key=video_quality_score, reverse=True)
        found.extend(ranked[:limit_per_leaf])
    return found


def videos_for_hint(videos: list[Path], hint: str | None) -> list[Path]:
    if not hint:
        return []
    matches = [video for video in videos if hint in str(video.relative_to(VIDEO_ROOT))]
    return sorted(matches, key=video_quality_score, reverse=True)


def pick_video_for_category(videos: list[Path], hint: str | None, *, used: set[str], ultra_only: bool = False) -> Path | None:
    candidates = videos_for_hint(videos, hint)
    thresholds = [MIN_VIDEO_QUALITY, MIN_VIDEO_QUALITY_FALLBACK] if not ultra_only else [MIN_VIDEO_QUALITY]
    for threshold in thresholds:
        for video in candidates:
            if str(video) in used:
                continue
            if ultra_only and not is_ultra_hd(video):
                continue
            if video_quality_score(video) >= threshold:
                return video
    return None


def load_image_fitted(path: Path, size: tuple[int, int]) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    except OSError:
        return None


def load_category_images(paths: list[Path], limit: int = IMAGES_PER_GALLERY) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths[:limit]:
        img = load_image_fitted(path, (640, 360))
        if img is not None:
            images.append(img)
    return images


def load_thumb(path: Path, size: tuple[int, int]) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    except OSError:
        return None


_CINEMATIC_OVERLAY: Image.Image | None = None


def cinematic_overlay() -> Image.Image:
    global _CINEMATIC_OVERLAY
    if _CINEMATIC_OVERLAY is not None:
        return _CINEMATIC_OVERLAY
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(HEIGHT):
        t = y / max(HEIGHT - 1, 1)
        alpha = int(255 * (0.34 * (0.15 + 0.85 * t)))
        draw.line([(0, y), (WIDTH, y)], fill=(8, 10, 18, alpha))
    vignette = Image.new("L", (WIDTH, HEIGHT), 0)
    vdraw = ImageDraw.Draw(vignette)
    vdraw.ellipse((-220, -120, WIDTH + 220, HEIGHT + 220), fill=210)
    vignette = vignette.filter(ImageFilter.GaussianBlur(90))
    dark = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 120))
    transparent = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    composed = Image.composite(dark, transparent, vignette)
    _CINEMATIC_OVERLAY = Image.alpha_composite(overlay, composed)
    return _CINEMATIC_OVERLAY


def add_cinematic_overlay(frame: Image.Image) -> Image.Image:
    return Image.alpha_composite(frame.convert("RGBA"), cinematic_overlay()).convert("RGB")


def build_mosaic_canvas(thumbs: list[Image.Image], *, cell_size: tuple[int, int], cols: int, rows: int) -> Image.Image:
    cell_w, cell_h = cell_size
    gap = 5
    canvas_cols = cols + 5
    canvas_rows = rows + 5
    canvas = Image.new("RGB", (canvas_cols * (cell_w + gap), canvas_rows * (cell_h + gap)), (8, 10, 16))
    for row in range(canvas_rows):
        for col in range(canvas_cols):
            idx = (row * canvas_cols + col) % len(thumbs)
            canvas.paste(thumbs[idx], (col * (cell_w + gap), row * (cell_h + gap)))
    return canvas


def render_mosaic_frame(
    canvas: Image.Image,
    frame_idx: int,
    total_frames: int,
    *,
    cell_size: tuple[int, int],
    zoom_start: float,
    zoom_end: float,
) -> Image.Image:
    cell_w, cell_h = cell_size
    gap = 5
    progress = frame_idx / max(total_frames - 1, 1)
    zoom = zoom_start + (zoom_end - zoom_start) * (0.5 - 0.5 * math.cos(progress * math.pi))
    drift_x = int((frame_idx * 3.1) % (cell_w + gap))
    drift_y = int((frame_idx * 2.2) % (cell_h + gap))
    scaled = canvas.resize((int(canvas.width * zoom), int(canvas.height * zoom)), Image.Resampling.BILINEAR)
    left = max((scaled.width - WIDTH) // 2 + drift_x, 0)
    top = max((scaled.height - HEIGHT) // 2 + drift_y, 0)
    crop = scaled.crop((left, top, left + WIDTH, top + HEIGHT))
    if crop.size != (WIDTH, HEIGHT):
        crop = ImageOps.fit(crop, (WIDTH, HEIGHT), method=Image.Resampling.BILINEAR)
    crop = ImageEnhance.Contrast(crop).enhance(1.1)
    crop = ImageEnhance.Color(crop).enhance(1.14)
    return add_cinematic_overlay(crop)


def draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    *,
    shadow: tuple[int, int, int, int] = (0, 0, 0, 180),
    offset: tuple[int, int] = (2, 2),
) -> None:
    draw.text((xy[0] + offset[0], xy[1] + offset[1]), text, font=font, fill=shadow)
    draw.text(xy, text, font=font, fill=fill)


def render_subtitle_overlay(
    frame: Image.Image,
    *,
    kicker: str,
    headline: str,
    tags: tuple[str, ...],
    count: int,
    accent: tuple[int, int, int],
    progress: float,
) -> Image.Image:
    eased = min(max(progress, 0.0), 1.0)
    fade = 1.0
    if eased < 0.12:
        fade = eased / 0.12
    elif eased > 0.88:
        fade = max(0.0, (1.0 - eased) / 0.12)

    slide = int((1.0 - min(eased / 0.25, 1.0)) * 36)
    overlay = frame.convert("RGBA")
    card_w, card_h = 760, 250
    x0, y0 = 84, HEIGHT - card_h - 110 + slide
    card = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(card)

    cdraw.rounded_rectangle((0, 0, card_w, card_h), radius=22, fill=(8, 12, 22, int(168 * fade)))
    cdraw.rounded_rectangle((0, 0, card_w, card_h), radius=22, outline=(*accent, int(120 * fade)), width=2)
    cdraw.rectangle((0, 18, 8, card_h - 18), fill=(*accent, int(255 * fade)))

    font_kicker = load_font(FONT_REGULAR, 22)
    font_headline = load_font(FONT_SERIF, 54)
    font_count = load_font(FONT_BOLD, 34)
    font_tag = load_font(FONT_BOLD, 18)

    draw_text_with_shadow(
        cdraw,
        (34, 28),
        kicker.upper(),
        font_kicker,
        (*accent, int(230 * fade)),
    )
    draw_text_with_shadow(
        cdraw,
        (34, 62),
        headline,
        font_headline,
        (255, 255, 255, int(255 * fade)),
    )

    tag_x = 34
    for tag in tags:
        pad_x, pad_y = 12, 7
        bbox = cdraw.textbbox((0, 0), tag.upper(), font=font_tag)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pill_w, pill_h = tw + pad_x * 2, th + pad_y * 2
        cdraw.rounded_rectangle(
            (tag_x, 142, tag_x + pill_w, 142 + pill_h),
            radius=999,
            fill=(*accent, int(42 * fade)),
            outline=(*accent, int(130 * fade)),
        )
        cdraw.text((tag_x + pad_x, 142 + pad_y - 1), tag.upper(), font=font_tag, fill=(245, 248, 255, int(240 * fade)))
        tag_x += pill_w + 10

    count_text = f"{count:,}+ high-quality samples"
    draw_text_with_shadow(
        cdraw,
        (34, 196),
        count_text,
        font_count,
        (240, 244, 255, int(255 * fade)),
    )

    overlay.alpha_composite(card, (x0, y0))
    return overlay.convert("RGB")


def render_stats_overlay(frame: Image.Image, *, lines: tuple[str, ...], progress: float) -> Image.Image:
    eased = min(max(progress, 0.0), 1.0)
    fade = min(eased / 0.2, 1.0) if eased < 0.85 else max(0.0, (1.0 - eased) / 0.15)
    overlay = frame.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    title_font = load_font(FONT_SERIF, 78)
    line_font = load_font(FONT_BOLD, 36)
    draw_text_with_shadow(draw, (84, 130), "WorldArena Benchmark", title_font, (255, 255, 255, int(255 * fade)))
    y = 240
    for line in lines:
        draw_text_with_shadow(draw, (88, y), line, line_font, (210, 230, 255, int(240 * fade)))
        y += 52
    return overlay.convert("RGB")


def render_cover_ken_burns(canvas: Image.Image, progress: float) -> Image.Image:
    zoom = 1.0 + 0.08 * progress
    scaled_w = int(canvas.width * zoom)
    scaled_h = int(canvas.height * zoom)
    scaled = canvas.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    left = max((scaled_w - WIDTH) // 2, 0)
    top = max((scaled_h - HEIGHT) // 2, 0)
    return scaled.crop((left, top, left + WIDTH, top + HEIGHT))


def render_cover_overlay(
    frame: Image.Image,
    progress: float,
    *,
    image_total: int,
    video_total: int,
    category_total: int,
) -> Image.Image:
    eased = min(max(progress, 0.0), 1.0)
    if eased < 0.18:
        ui_fade = eased / 0.18
    elif eased > 0.92:
        ui_fade = max(0.0, (1.0 - eased) / 0.08)
    else:
        ui_fade = 1.0

    slide = int((1.0 - min(max((eased - 0.05) / 0.25, 0.0), 1.0)) * 36)
    overlay = frame.convert("RGBA")

    scrim = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(scrim)
    scrim_w = WIDTH // 2 + 260
    for x in range(scrim_w):
        t = x / max(scrim_w - 1, 1)
        alpha = int(190 * (1.0 - t) * ui_fade)
        sdraw.line([(x, 0), (x, HEIGHT)], fill=(6, 8, 16, alpha))
    overlay = Image.alpha_composite(overlay, scrim)
    draw = ImageDraw.Draw(overlay)

    accent_y = 108 + slide
    draw.rectangle((84, accent_y, 84 + int(128 * ui_fade), accent_y + 4), fill=(255, 150, 80, int(220 * ui_fade)))

    kicker_font = load_font(FONT_BOLD, 20)
    title_font = load_font(FONT_SERIF, 108)
    sub_font = load_font(FONT_REGULAR, 30)
    stat_val_font = load_font(FONT_BOLD, 28)
    stat_lab_font = load_font(FONT_REGULAR, 16)
    tag_font = load_font(FONT_REGULAR, 22)

    draw_text_with_shadow(
        draw,
        (84, 128 + slide),
        "BENCHMARK FOR CONTROLLABLE WORLD GENERATION",
        kicker_font,
        (255, 176, 120, int(230 * ui_fade)),
    )
    draw_text_with_shadow(draw, (84, 166 + slide), "WorldArena", title_font, (255, 255, 255, int(255 * ui_fade)))
    draw_text_with_shadow(
        draw,
        (88, 288 + slide),
        "High-quality · Diverse · Large-scale Evaluation Data",
        sub_font,
        (200, 220, 255, int(235 * ui_fade)),
    )

    stats = [
        (f"{image_total:,}+", "Images"),
        (f"{video_total:,}+", "Videos"),
        (f"{category_total}", "Categories"),
    ]
    x = 84
    pill_y = 368 + slide
    for value, label in stats:
        pill_w, pill_h = 180, 72
        pill = Image.new("RGBA", (pill_w, pill_h), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(pill)
        pdraw.rounded_rectangle(
            (0, 0, pill_w, pill_h),
            radius=14,
            fill=(10, 14, 24, int(150 * ui_fade)),
            outline=(120, 180, 255, int(100 * ui_fade)),
        )
        draw_text_with_shadow(pdraw, (16, 12), value, stat_val_font, (255, 255, 255, int(255 * ui_fade)))
        draw_text_with_shadow(pdraw, (16, 44), label.upper(), stat_lab_font, (180, 200, 230, int(220 * ui_fade)))
        overlay.alpha_composite(pill, (x, pill_y))
        x += pill_w + 14

    draw_text_with_shadow(
        draw,
        (84, HEIGHT - 78),
        "Explore the benchmark universe →",
        tag_font,
        (180, 210, 255, int(200 * ui_fade)),
    )
    return overlay.convert("RGB")


def encode_frames(frames_dir: Path, output: Path) -> None:
    run(
        [
            FFMPEG,
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "frame_%05d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-preset",
            "veryfast",
            str(output),
        ],
        timeout=600,
    )


def build_segment_from_frames(frames: list[Image.Image], out_dir: Path, tag: str) -> Path:
    frames_dir = out_dir / f"segment_{tag}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{index:05d}.jpg", quality=88, optimize=True)
    output = out_dir / f"{tag}.mp4"
    encode_frames(frames_dir, output)
    return output


def category_thumbs(paths: list[Path], limit: int = 220) -> list[Image.Image]:
    thumbs: list[Image.Image] = []
    for path in paths[:limit]:
        thumb = load_thumb(path, (168, 96))
        if thumb is not None:
            thumbs.append(thumb)
    return thumbs


def compose_image_gallery_frame(
    images: list[Image.Image],
    frame_idx: int,
    total_frames: int,
    *,
    cols: int = 5,
    rows: int = 4,
) -> Image.Image:
    if not images:
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (8, 10, 16))
        return add_cinematic_overlay(canvas)

    # Use a denser grid for a more continuous scrolling feel
    gap = 8
    # Calculate tile height based on 4 rows leaving room for bottom elements
    tile_h = (HEIGHT - gap * 6) // rows
    tile_w = int(tile_h * 16 / 9)
    y_offset = gap * 2

    # Remove blur and darkening, use the raw image directly for maximum fidelity
    backdrop_idx = (frame_idx // 48) % len(images)
    backdrop = images[backdrop_idx].resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    canvas = backdrop.convert("RGB")

    draw = ImageDraw.Draw(canvas)

    # Continuous horizontal scroll
    # 24 frames per second means we need to move by some pixels per frame
    speed = 2.4

    # We draw extra columns so they can scroll across smoothly
    draw_cols = (WIDTH // (tile_w + gap)) + 3

    for row in range(rows):
        # Alternate scrolling direction for rows: even rows go left, odd rows go right
        direction = -1 if row % 2 == 0 else 1

        # Stagger rows so they aren't aligned vertically
        row_stagger = (row % 2) * ((tile_w + gap) // 2)

        # Calculate continuous current scroll offset in pixels
        raw_offset = frame_idx * speed * direction
        current_offset_x = raw_offset % (tile_w + gap)

        if direction == 1:
            current_offset_x -= (tile_w + gap)

        for col in range(draw_cols):
            x = int(col * (tile_w + gap) + current_offset_x + row_stagger - (tile_w + gap))
            y = y_offset + row * (tile_h + gap)

            # Use the virtual column index to keep the image consistent as it moves
            virtual_col = col - int(raw_offset // (tile_w + gap))

            # Map this to a stable image index
            image_idx = (row * 13 + virtual_col * 7) % len(images)
            tile = images[image_idx].resize((tile_w, tile_h), Image.Resampling.LANCZOS)

            canvas.paste(tile, (x, y))
            draw.rounded_rectangle((x, y, x + tile_w, y + tile_h), radius=6, outline=(220, 230, 245), width=1)

    # Bottom filmstrip with more images (scrolls horizontally)
    strip_h = 76
    strip_y = HEIGHT - strip_h - 20
    strip_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    strip_draw = ImageDraw.Draw(strip_layer)
    strip_draw.rounded_rectangle((0, strip_y, WIDTH, strip_y + strip_h), radius=14, fill=(6, 8, 14, 150))
    thumb_w, thumb_h = 106, 60
    spacing = 8
    total_strip_w = len(images[:18]) * (thumb_w + spacing)
    scroll = int((frame_idx * 1.5) % max(total_strip_w, 1))
    x = 24 - scroll
    for index in range(len(images[:18]) * 3):
        image = images[index % len(images[:18])]
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        strip_layer.paste(thumb, (x, strip_y + (strip_h - thumb_h) // 2))
        x += thumb_w + spacing

    canvas = Image.alpha_composite(canvas.convert("RGBA"), strip_layer).convert("RGB")
    # Removed contrast/color enhancement here to present the raw image natively
    return canvas


def build_image_gallery_segment(
    spec: CategorySpec,
    paths: list[Path],
    out_dir: Path,
    *,
    seconds: float,
    fallback_paths: list[Path] | None = None,
) -> Path:
    source_paths = paths if paths else (fallback_paths or [])
    images = load_category_images(source_paths, limit=IMAGES_PER_GALLERY)
    if len(images) < 8 and fallback_paths:
        images = load_category_images(fallback_paths, limit=IMAGES_PER_GALLERY)
    if len(images) < 4:
        images = images * max(1, 4 // max(len(images), 1))

    total_frames = max(int(seconds * FPS), 1)
    frames: list[Image.Image] = []
    for index in range(total_frames):
        progress = index / max(total_frames - 1, 1)
        frame = compose_image_gallery_frame(images, index, total_frames)
        frame = render_subtitle_overlay(
            frame,
            kicker="Image Benchmark · Gallery",
            headline=spec.headline,
            tags=spec.tags + ("Gallery",),
            count=spec.count,
            accent=spec.accent,
            progress=progress,
        )
        frames.append(frame)
    tag = f"gallery_{spec.key.replace('/', '_')}"
    print(f"[gallery] {spec.headline} ({len(images)} images, {seconds:.1f}s)", flush=True)
    return build_segment_from_frames(frames, out_dir, tag)


def apply_cover_fade(frame: Image.Image, progress: float) -> Image.Image:
    if progress < 0.15:
        alpha = progress / 0.15
    elif progress > 0.9:
        alpha = max(0.0, (1.0 - progress) / 0.1)
    else:
        alpha = 1.0
    if alpha >= 0.999:
        return frame
    dark = Image.new("RGBA", frame.size, (0, 0, 0, int(255 * (1.0 - alpha))))
    return Image.alpha_composite(frame.convert("RGBA"), dark).convert("RGB")


def build_cover_segment(
    all_paths: list[Path],
    out_dir: Path,
    *,
    seconds: float,
    image_total: int,
    video_total: int,
    category_total: int,
) -> Path:
    if PROMO_COVER.exists():
        with Image.open(PROMO_COVER) as img:
            cover_canvas = ImageOps.exif_transpose(img).convert("RGB")
            cover_canvas = ImageOps.fit(
                cover_canvas,
                (int(WIDTH * 1.1), int(HEIGHT * 1.1)),
                method=Image.Resampling.LANCZOS,
            )
            # Enhance cover to make it less pale and more vibrant (Removed per user request to show raw video/image)
    else:
        images = load_category_images(all_paths, limit=60)
        cover_canvas = compose_image_gallery_frame(images, 0, 1, cols=6, rows=4)

    total_frames = max(int(seconds * FPS), 1)
    frames: list[Image.Image] = []
    for index in range(total_frames):
        progress = index / max(total_frames - 1, 1)
        frame = render_cover_ken_burns(cover_canvas, progress)
        frame = apply_cover_fade(frame, progress)
        frames.append(frame)
    print(f"[cover] title card ({seconds:.1f}s)", flush=True)
    return build_segment_from_frames(frames, out_dir, "cover")


def build_finale_segment(all_paths: list[Path], out_dir: Path, *, image_total: int, category_total: int, video_total: int) -> Path:
    images = load_category_images(all_paths, limit=120)
    total_frames = int(2.8 * FPS)
    lines = (
        f"{image_total:,}+ curated test images",
        f"{video_total:,}+ benchmark videos",
        f"{category_total} fine-grained categories",
    )
    frames: list[Image.Image] = []
    for index in range(total_frames):
        progress = index / max(total_frames - 1, 1)
        frame = compose_image_gallery_frame(images, index, total_frames, cols=6, rows=4)
        frame = render_stats_overlay(frame, lines=lines, progress=progress)
        frames.append(frame)
    return build_segment_from_frames(frames, out_dir, "finale")


def find_video_for_category(videos: list[Path], hint: str | None, *, used: set[str] | None = None) -> Path | None:
    return pick_video_for_category(videos, hint, used=used or set())


def clip_start_seconds(source: Path) -> float:
    digest = int(hashlib.md5(str(source).encode()).hexdigest()[:8], 16)
    return 0.9 + (digest % 360) / 100.0


def extract_video_clip(source: Path, target: Path, *, start: float, duration: float) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-y",
        "-ss",
        f"{start:.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(source),
        "-vf",
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,crop={WIDTH}:{HEIGHT},"
        "fps=24,format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "17",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        run(cmd, timeout=180)
        return target.exists() and target.stat().st_size > 10_000
    except RuntimeError:
        return False


def clip_frames_to_images(clip: Path, count: int) -> list[Image.Image]:
    tmp_dir = WORK / "clip_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pattern = tmp_dir / f"{clip.stem}_%03d.jpg"
    run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(clip),
            "-vf",
            f"fps={FPS},scale={WIDTH}:{HEIGHT}",
            str(pattern),
        ],
        timeout=120,
    )
    frames = sorted(tmp_dir.glob(f"{clip.stem}_*.jpg"))[:count]
    images: list[Image.Image] = []
    for path in frames:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def build_video_showcase_segment(
    clips: list[tuple[CategorySpec, Path]],
    out_dir: Path,
) -> Path:
    frames: list[Image.Image] = []
    per_clip_frames = max(int(SHOWCASE_CLIP_SECONDS * FPS), 1)
    for spec, clip in clips:
        clip_frames = clip_frames_to_images(clip, per_clip_frames)
        if not clip_frames:
            continue
        for index in range(per_clip_frames):
            progress = index / max(per_clip_frames - 1, 1)
            frame = clip_frames[min(index, len(clip_frames) - 1)]
            frame = render_subtitle_overlay(
                frame,
                kicker="Video Benchmark · HQ Selection",
                headline=spec.headline,
                tags=spec.tags + ("Video",),
                count=spec.count,
                accent=spec.accent,
                progress=progress,
            )
            frames.append(frame)
    tag = "video_showcase"
    print(f"[showcase] {len(clips)} high-quality clips", flush=True)
    return build_segment_from_frames(frames, out_dir, tag)


def try_build_category_video_segment(
    spec: CategorySpec,
    videos: list[Path],
    out_dir: Path,
    *,
    used_videos: set[str],
    ultra_only: bool = True,
) -> Path | None:
    video = pick_video_for_category(videos, spec.video_hint, used=used_videos, ultra_only=ultra_only)
    if video is None:
        return None
    start = clip_start_seconds(video)
    clip_path = out_dir / "clips" / f"{spec.key.replace('/', '_')}.mp4"
    if not extract_video_clip(video, clip_path, start=start, duration=VIDEO_CLIP_SECONDS):
        return None
    used_videos.add(str(video))
    print(
        f"[video] {spec.headline} <- {video.name} (score={video_quality_score(video):.0f})",
        flush=True,
    )
    return build_video_pulse_segment(spec, clip_path, out_dir, seconds=VIDEO_CLIP_SECONDS)


def build_video_pulse_segment(spec: CategorySpec, clip: Path, out_dir: Path, *, seconds: float = 1.4) -> Path:
    total_frames = max(int(seconds * FPS), 1)
    clip_frames = clip_frames_to_images(clip, total_frames)
    if not clip_frames:
        return build_image_gallery_segment(spec, [], out_dir, seconds=IMAGE_GALLERY_SECONDS)
    frames: list[Image.Image] = []
    for index in range(total_frames):
        progress = index / max(total_frames - 1, 1)
        frame = clip_frames[min(index, len(clip_frames) - 1)]
        frame = render_subtitle_overlay(
            frame,
            kicker="Video Benchmark",
            headline=spec.headline,
            tags=spec.tags + ("Video",),
            count=spec.count,
            accent=spec.accent,
            progress=progress,
        )
        frames.append(frame)
    tag = f"video_{spec.key.replace('/', '_')}"
    return build_segment_from_frames(frames, out_dir, tag)


def final_concat(segments: list[Path], out_path: Path) -> None:
    list_file = out_path.with_suffix(".txt")
    list_file.write_text("\n".join(f"file '{segment}'" for segment in segments) + "\n", encoding="utf-8")
    run(
        [
            FFMPEG,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(out_path),
        ],
        timeout=600,
    )


def main() -> int:
    if not IMAGE_ROOT.exists() or not VIDEO_ROOT.exists():
        print("Dataset roots missing:", IMAGE_ROOT, VIDEO_ROOT, file=sys.stderr)
        return 1

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    items, buckets = index_annotations()
    specs = build_category_specs(items)
    videos = collect_videos(limit_per_leaf=8)
    print(f"Categories: {len(specs)}, HQ video pool: {len(videos)}", flush=True)

    all_paths = [path for paths in buckets.values() for path in paths]
    rng = random.Random(SEED)
    rng.shuffle(all_paths)

    used_videos: set[str] = set()
    image_segments: list[Path] = []
    video_segments: list[Path] = []

    image_segments.append(
        build_cover_segment(
            all_paths,
            WORK,
            seconds=COVER_SECONDS,
            image_total=len(items),
            video_total=6390,
            category_total=len(specs),
        )
    )

    gallery_count = 0
    # Curate a representative subset for a faster, lighter background video
    # Expanding to 10 categories for longer image presentation
    curated_gallery_keys = [
        "dynamic/photorealistic/fluid",
        "dynamic/photorealistic/multi_motion",
        "dynamic/stylized/articulated",
        "dynamic/stylized/deformable",
        "static/photorealistic/indoor/living_spaces",
        "static/photorealistic/indoor/dining_spaces",
        "static/photorealistic/outdoor/city",
        "static/photorealistic/outdoor/terrestrial_landscape",
        "static/photorealistic/outdoor/aquatic_landscape",
        "static/stylized/outdoor/suburb",
    ]
    curated_specs = [s for s in specs if s.key in curated_gallery_keys]

    for spec in curated_specs:
        paths = buckets.get(spec.key, [])
        image_segments.append(
            build_image_gallery_segment(
                spec,
                paths,
                WORK,
                seconds=IMAGE_GALLERY_SECONDS,
                fallback_paths=all_paths,
            )
        )
        gallery_count += 1

    curated_video_keys = [
        "dynamic/photorealistic/fluid",
        "dynamic/photorealistic/articulated",
        "dynamic/photorealistic/multi_motion",
        "dynamic/photorealistic/deformable",
        "dynamic/photorealistic/rigid",
        "dynamic/stylized/articulated",
        "static/photorealistic/outdoor/city",
        "static/photorealistic/indoor/living_spaces",
    ]
    curated_video_specs = [s for s in specs if s.key in curated_video_keys]

    video_count = 0
    for spec in curated_video_specs:
        video_segment = try_build_category_video_segment(
            spec,
            videos,
            WORK,
            used_videos=used_videos,
            ultra_only=True,
        )
        if video_segment is None:
            # fallback if no ultra 4k found
            video_segment = try_build_category_video_segment(
                spec,
                videos,
                WORK,
                used_videos=used_videos,
                ultra_only=False,
            )
        if video_segment is not None:
            video_segments.append(video_segment)
            video_count += 1

    showcase_specs: list[tuple[CategorySpec, Path]] = []
    ranked_videos = sorted(
        [video for video in videos if is_ultra_hd(video)],
        key=video_quality_score,
        reverse=True,
    )
    for video in ranked_videos:
        if str(video) in used_videos:
            continue
        rel = str(video.relative_to(VIDEO_ROOT))
        matched = next((spec for spec in specs if spec.video_hint and spec.video_hint in rel), None)
        if matched is None:
            continue
        clip_path = WORK / "clips" / f"showcase_{len(showcase_specs):02d}.mp4"
        if not extract_video_clip(video, clip_path, start=clip_start_seconds(video), duration=SHOWCASE_CLIP_SECONDS):
            continue
        showcase_specs.append((matched, clip_path))
        used_videos.add(str(video))
        if len(showcase_specs) >= SHOWCASE_CLIP_COUNT:
            break
    if showcase_specs:
        video_segments.append(build_video_showcase_segment(showcase_specs, WORK))

    segments = image_segments + video_segments + [
        build_finale_segment(
            all_paths,
            WORK,
            image_total=len(items),
            category_total=len(specs),
            video_total=6390,
        )
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final_concat(segments, OUTPUT)
    print(
        f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / (1024 * 1024):.1f} MB, "
        f"galleries={gallery_count}, videos={video_count}, showcase={len(showcase_specs)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
