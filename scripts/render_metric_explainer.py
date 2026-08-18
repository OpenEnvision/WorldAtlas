#!/usr/bin/env python3
"""Render the WorldAtlas Arena metric explainer used by the homepage carousel.

The animation is intentionally generated from code: it keeps the visual language
close to an executable-diagram / Manim presentation while embedding one real,
verified WorldArena benchmark case for the quality comparison.
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "docs/assets/metric_examples/curated_20260801"
INPUT_PATH = CASE_DIR / "838324ca602a__image_static_photorealistic_workspaces_0020__input.jpg"
HIGH_PATH = CASE_DIR / "46ef56e3b4ed__image_static_photorealistic_workspaces_0020__high.mp4"
LOW_PATH = CASE_DIR / "6e4b918a70e4__image_static_photorealistic_workspaces_0020__low.mp4"

REFERENCE_W = 1920
REFERENCE_H = 1080
TOTAL_DURATION = 56.0
SCENE_TIMES = (0.0, 4.0, 10.0, 17.0, 24.0, 31.0, 38.0, 45.0, 52.0, 56.0)

BG = "#07131f"
BG_2 = "#0b1c2a"
PANEL = "#0c2132"
PANEL_2 = "#102a3d"
STROKE = "#294860"
OFFWHITE = "#f3eadf"
MUTED = "#8da1b3"
GRID = "#173044"
CYAN = "#58d6d6"
GREEN = "#80e0a7"
AMBER = "#ffbd66"
CORAL = "#ff6b70"
LILAC = "#b9a2ff"
BLUE_GRAY = "#5a7187"

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT_MONO_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def smooth(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def ease_out(value: float) -> float:
    value = clamp(value)
    return 1.0 - (1.0 - value) ** 3


def reveal(progress: float, start: float, end: float) -> float:
    if end <= start:
        return float(progress >= end)
    return smooth((progress - start) / (end - start))


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def mix_color(a: str | tuple[int, int, int], b: str | tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    ca = hex_rgb(a) if isinstance(a, str) else a
    cb = hex_rgb(b) if isinstance(b, str) else b
    amount = clamp(amount)
    return tuple(round(x * (1.0 - amount) + y * amount) for x, y in zip(ca, cb))


def bezier_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    count: int = 100,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(count):
        t = index / max(1, count - 1)
        u = 1.0 - t
        points.append(
            (
                u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return points


def partial_polyline(points: Sequence[tuple[float, float]], progress: float) -> list[tuple[float, float]]:
    if not points:
        return []
    if len(points) == 1 or progress <= 0:
        return [points[0]]
    if progress >= 1:
        return list(points)
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    target = sum(lengths) * progress
    output = [points[0]]
    travelled = 0.0
    for start, end, length in zip(points, points[1:], lengths):
        if travelled + length <= target:
            output.append(end)
            travelled += length
            continue
        ratio = (target - travelled) / max(length, 1e-6)
        output.append((start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio))
        break
    return output


def point_on_path(points: Sequence[tuple[float, float]], progress: float) -> tuple[float, float]:
    visible = partial_polyline(points, clamp(progress))
    return visible[-1]


def sample_video(path: Path, count: int = 28) -> list[Image.Image]:
    """Decode browser media deterministically without relying on OpenCV codecs."""
    container = av.open(str(path))
    stream = container.streams.video[0]
    decoded = [frame.to_image().convert("RGB") for frame in container.decode(stream)]
    container.close()
    if not decoded:
        raise RuntimeError(f"No decodable frames in {path}")
    indices = np.linspace(0, len(decoded) - 1, count, dtype=int)
    return [decoded[int(index)] for index in indices]


def sift_correspondences(source: Image.Image, target: Image.Image, limit: int = 44) -> list[tuple[float, float, float, float]]:
    source_gray = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(np.asarray(target.convert("RGB")), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=2400)
    source_keypoints, source_desc = sift.detectAndCompute(source_gray, None)
    target_keypoints, target_desc = sift.detectAndCompute(target_gray, None)
    if source_desc is None or target_desc is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates = matcher.knnMatch(source_desc, target_desc, k=2)
    good = [a for a, b in candidates if a.distance < 0.72 * b.distance]
    if len(good) >= 6:
        source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target_points = np.float32([target_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)
        if mask is not None:
            good = [match for match, keep in zip(good, mask.ravel()) if keep]
    good.sort(key=lambda match: match.distance)
    if len(good) > limit:
        positions = np.linspace(0, len(good) - 1, limit, dtype=int)
        good = [good[int(index)] for index in positions]
    return [
        (*source_keypoints[match.queryIdx].pt, *target_keypoints[match.trainIdx].pt)
        for match in good
    ]


@lru_cache(maxsize=128)
def load_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    font_path = {
        "regular": FONT_REGULAR,
        "bold": FONT_BOLD,
        "mono": FONT_MONO,
        "mono-bold": FONT_MONO_BOLD,
    }[kind]
    return ImageFont.truetype(str(font_path), size=max(8, size))


class Canvas:
    def __init__(self, width: int, height: int, base: Image.Image | None = None) -> None:
        self.width = width
        self.height = height
        self.scale = width / REFERENCE_W
        self.image = (base.copy() if base is not None else Image.new("RGB", (width, height), hex_rgb(BG))).convert("RGB")
        self.draw = ImageDraw.Draw(self.image)

    def x(self, value: float) -> int:
        return round(value * self.scale)

    def y(self, value: float) -> int:
        return round(value * self.scale)

    def pt(self, point: tuple[float, float]) -> tuple[int, int]:
        return self.x(point[0]), self.y(point[1])

    def box(self, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        x, y, width, height = box
        return self.x(x), self.y(y), self.x(x + width), self.y(y + height)

    def color(self, color: str, opacity: float = 1.0) -> tuple[int, int, int]:
        return mix_color(BG, color, opacity)

    def font(self, size: float, kind: str = "regular") -> ImageFont.FreeTypeFont:
        return load_font(kind, round(size * self.scale))

    def text(
        self,
        position: tuple[float, float],
        value: str,
        size: float,
        color: str = OFFWHITE,
        *,
        kind: str = "regular",
        anchor: str = "la",
        opacity: float = 1.0,
        spacing: float = 10,
        align: str = "left",
    ) -> None:
        self.draw.multiline_text(
            self.pt(position),
            value,
            fill=self.color(color, opacity),
            font=self.font(size, kind),
            anchor=anchor,
            spacing=self.y(spacing),
            align=align,
        )

    def line(
        self,
        points: Iterable[tuple[float, float]],
        color: str,
        width: float = 2.0,
        opacity: float = 1.0,
        joint: str = "curve",
    ) -> None:
        scaled = [self.pt(point) for point in points]
        if len(scaled) >= 2:
            self.draw.line(scaled, fill=self.color(color, opacity), width=max(1, self.x(width)), joint=joint)

    def dashed_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str,
        width: float = 2.0,
        dash: float = 12.0,
        gap: float = 10.0,
        opacity: float = 1.0,
    ) -> None:
        distance = math.dist(start, end)
        if distance <= 0:
            return
        ux = (end[0] - start[0]) / distance
        uy = (end[1] - start[1]) / distance
        cursor = 0.0
        while cursor < distance:
            stop = min(distance, cursor + dash)
            a = (start[0] + ux * cursor, start[1] + uy * cursor)
            b = (start[0] + ux * stop, start[1] + uy * stop)
            self.line((a, b), color, width, opacity)
            cursor += dash + gap

    def rect(
        self,
        box: tuple[float, float, float, float],
        *,
        outline: str | None = None,
        fill: str | None = None,
        width: float = 2.0,
        radius: float = 0.0,
        opacity: float = 1.0,
    ) -> None:
        scaled = self.box(box)
        fill_color = self.color(fill, opacity) if fill else None
        outline_color = self.color(outline, opacity) if outline else None
        if radius > 0:
            self.draw.rounded_rectangle(
                scaled,
                radius=self.x(radius),
                fill=fill_color,
                outline=outline_color,
                width=max(1, self.x(width)),
            )
        else:
            self.draw.rectangle(scaled, fill=fill_color, outline=outline_color, width=max(1, self.x(width)))

    def panel(
        self,
        box: tuple[float, float, float, float],
        *,
        accent: str | None = None,
        opacity: float = 1.0,
        radius: float = 18.0,
    ) -> None:
        """Draw a restrained glass-like card with depth and a top highlight."""
        if opacity <= 0:
            return
        x, y, width, height = box
        self.rect((x + 10, y + 14, width, height), fill="#020912", radius=radius + 2, opacity=0.72 * opacity)
        self.rect(box, outline=STROKE, fill=PANEL, width=2, radius=radius, opacity=0.96 * opacity)
        self.line(((x + radius, y + 1), (x + width - radius, y + 1)), OFFWHITE, 1, 0.10 * opacity)
        if accent:
            self.line(((x + radius, y + 1), (x + min(width * 0.32, 164), y + 1)), accent, 4, 0.92 * opacity)

    def glow_line(
        self,
        points: Sequence[tuple[float, float]],
        color: str,
        width: float = 3.0,
        opacity: float = 1.0,
    ) -> None:
        if len(points) < 2 or opacity <= 0:
            return
        self.line(points, color, width + 18, opacity * 0.045)
        self.line(points, color, width + 9, opacity * 0.09)
        self.line(points, color, width, opacity)

    def glow_ellipse(
        self,
        box: tuple[float, float, float, float],
        color: str,
        *,
        width: float = 3.0,
        opacity: float = 1.0,
    ) -> None:
        x, y, box_w, box_h = box
        for spread, local_opacity in ((22, 0.035), (12, 0.07), (5, 0.13)):
            self.ellipse(
                (x - spread, y - spread, box_w + spread * 2, box_h + spread * 2),
                outline=color,
                width=width + spread * 0.15,
                opacity=opacity * local_opacity,
            )
        self.ellipse(box, outline=color, width=width, opacity=opacity)

    def ellipse(
        self,
        box: tuple[float, float, float, float],
        *,
        outline: str | None = None,
        fill: str | None = None,
        width: float = 2.0,
        opacity: float = 1.0,
    ) -> None:
        self.draw.ellipse(
            self.box(box),
            fill=self.color(fill, opacity) if fill else None,
            outline=self.color(outline, opacity) if outline else None,
            width=max(1, self.x(width)),
        )

    def polygon(self, points: Iterable[tuple[float, float]], fill: str, opacity: float = 1.0) -> None:
        self.draw.polygon([self.pt(point) for point in points], fill=self.color(fill, opacity))

    def arrow(
        self,
        points: Sequence[tuple[float, float]],
        color: str,
        *,
        width: float = 3.0,
        opacity: float = 1.0,
        head: float = 15.0,
    ) -> None:
        if len(points) < 2:
            return
        self.line(points, color, width, opacity)
        tip = points[-1]
        previous = points[-2]
        angle = math.atan2(tip[1] - previous[1], tip[0] - previous[0])
        wing = math.radians(28)
        left = (tip[0] - head * math.cos(angle - wing), tip[1] - head * math.sin(angle - wing))
        right = (tip[0] - head * math.cos(angle + wing), tip[1] - head * math.sin(angle + wing))
        self.polygon((tip, left, right), color, opacity)

    def paste_cover(
        self,
        source: Image.Image,
        box: tuple[float, float, float, float],
        opacity: float = 1.0,
        radius: float = 0.0,
    ) -> None:
        x, y, width, height = box
        target_w, target_h = self.x(width), self.y(height)
        source = source.convert("RGB")
        source_ratio = source.width / source.height
        target_ratio = target_w / target_h
        if source_ratio > target_ratio:
            crop_w = round(source.height * target_ratio)
            offset = (source.width - crop_w) // 2
            source = source.crop((offset, 0, offset + crop_w, source.height))
        else:
            crop_h = round(source.width / target_ratio)
            offset = (source.height - crop_h) // 2
            source = source.crop((0, offset, source.width, offset + crop_h))
        source = source.resize((target_w, target_h), Image.Resampling.LANCZOS)
        destination = (self.x(x), self.y(y))
        if opacity >= 0.999 and radius <= 0:
            self.image.paste(source, destination)
        else:
            mask = Image.new("L", (target_w, target_h), 0)
            mask_draw = ImageDraw.Draw(mask)
            fill = round(255 * clamp(opacity))
            if radius > 0:
                mask_draw.rounded_rectangle((0, 0, target_w - 1, target_h - 1), radius=self.x(radius), fill=fill)
            else:
                mask_draw.rectangle((0, 0, target_w - 1, target_h - 1), fill=fill)
            self.image.paste(source, destination, mask)
        self.draw = ImageDraw.Draw(self.image)


class MetricExplainer:
    def __init__(self, width: int, height: int) -> None:
        if width * 9 != height * 16:
            raise ValueError("The explainer must use a 16:9 frame size")
        self.width = width
        self.height = height
        self.scale = width / REFERENCE_W
        self.input_image = Image.open(INPUT_PATH).convert("RGB")
        self.high_frames = sample_video(HIGH_PATH)
        self.low_frames = sample_video(LOW_PATH)
        self.matches = sift_correspondences(self.input_image, self.high_frames[0])
        self.base = self._make_background()

    def _make_background(self) -> Image.Image:
        top = np.array(hex_rgb("#0b2132"), dtype=np.float32)
        bottom = np.array(hex_rgb("#040c15"), dtype=np.float32)
        yy, xx = np.mgrid[0 : self.height, 0 : self.width].astype(np.float32)
        nx = xx / max(1, self.width - 1)
        ny = yy / max(1, self.height - 1)
        vertical = ny[..., None]
        gradient = top[None, None, :] * (1.0 - vertical) + bottom[None, None, :] * vertical

        def add_glow(center_x: float, center_y: float, spread_x: float, spread_y: float, color: str, strength: float) -> None:
            nonlocal gradient
            distance = ((nx - center_x) / spread_x) ** 2 + ((ny - center_y) / spread_y) ** 2
            glow = np.exp(-distance * 2.35)[..., None] * strength
            target = np.array(hex_rgb(color), dtype=np.float32)[None, None, :]
            gradient = gradient * (1.0 - glow) + target * glow

        add_glow(-0.04, 0.18, 0.54, 0.58, CYAN, 0.12)
        add_glow(1.03, 0.10, 0.46, 0.50, AMBER, 0.085)
        add_glow(0.68, 1.05, 0.62, 0.48, LILAC, 0.055)

        vignette = np.clip(((nx - 0.5) ** 2 / 0.42 + (ny - 0.48) ** 2 / 0.58), 0.0, 1.0)[..., None]
        gradient *= 1.0 - vignette * 0.17
        rng = np.random.default_rng(2408)
        noise = rng.normal(0.0, 0.82, (self.height, self.width, 1)).astype(np.float32)
        gradient = np.clip(gradient + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(gradient, "RGB")
        draw = ImageDraw.Draw(image)
        line_color = mix_color(BG, GRID, 0.48)
        step_x = max(52, round(120 * self.scale))
        step_y = max(44, round(96 * self.scale))
        for x in range(step_x // 2, self.width, step_x):
            draw.line((x, 0, x, self.height), fill=line_color, width=max(1, round(self.scale)))
        for y in range(step_y // 2, self.height, step_y):
            draw.line((0, y, self.width, y), fill=line_color, width=max(1, round(self.scale)))

        dot_color = mix_color(BG, MUTED, 0.36)
        for _ in range(46):
            x = int(rng.uniform(40, self.width - 40))
            y = int(rng.uniform(35, self.height - 35))
            radius = max(1, round(rng.choice((1.0, 1.0, 1.6)) * self.scale))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=dot_color)
        return image

    def _chrome(self, canvas: Canvas, scene_index: int, progress: float, label: str) -> None:
        opacity = reveal(progress, 0.0, 0.14)
        canvas.rect((72, 55, 18, 18), outline=AMBER, fill=AMBER, radius=4, opacity=opacity)
        canvas.rect((77, 60, 8, 8), fill=BG, radius=2, opacity=opacity)
        canvas.text((105, 65), "WORLDATLAS ARENA", 19, OFFWHITE, kind="mono-bold", opacity=opacity)
        canvas.text((106, 96), "EVALUATION, MADE VISIBLE", 13, MUTED, kind="mono", opacity=opacity)

        canvas.rect((770, 51, 380, 46), outline=STROKE, fill=PANEL, radius=23, opacity=opacity)
        canvas.ellipse((793, 68, 8, 8), fill=CYAN, opacity=opacity)
        canvas.text((960, 74), "METRIC EXPLAINER", 14, OFFWHITE, kind="mono-bold", anchor="mm", opacity=opacity)

        canvas.text((1838, 66), f"{scene_index + 1:02d} / 09", 15, MUTED, kind="mono", anchor="ra", opacity=opacity)
        canvas.text((1838, 96), label, 13, AMBER, kind="mono-bold", anchor="ra", opacity=opacity)

        scan_x = 110 + (1680 * ((progress * 0.72 + scene_index * 0.13) % 1.0))
        canvas.line(((scan_x, 136), (scan_x, 940)), CYAN, 1, 0.035 * opacity)

        canvas.line(((52, 52), (52, 126)), CYAN, 2, 0.32 * opacity)
        canvas.line(((52, 52), (126, 52)), CYAN, 2, 0.32 * opacity)
        canvas.line(((1868, 52), (1794, 52)), AMBER, 2, 0.28 * opacity)
        canvas.line(((1868, 52), (1868, 126)), AMBER, 2, 0.28 * opacity)

        canvas.text((78, 1014), f"0{scene_index + 1}", 14, MUTED, kind="mono-bold", anchor="lm", opacity=opacity)
        canvas.text((132, 1014), label, 12, OFFWHITE, kind="mono", anchor="lm", opacity=opacity)
        canvas.line(((610, 1014), (1310, 1014)), STROKE, 2, opacity)
        start_x = 654
        for index in range(9):
            color = AMBER if index == scene_index else BLUE_GRAY
            width = 55 if index == scene_index else 37
            canvas.line(((start_x + index * 69, 1014), (start_x + index * 69 + width, 1014)), color, 5 if index == scene_index else 3, opacity)
        canvas.text((1842, 1014), "TRACEABLE / REPRODUCIBLE", 11, MUTED, kind="mono", anchor="rm", opacity=opacity)

    def _title(self, canvas: Canvas, kicker: str, title: str, subtitle: str, progress: float) -> None:
        a = reveal(progress, 0.02, 0.16)
        kicker_width = max(210, min(390, 58 + len(kicker) * 12))
        canvas.rect((104, 148, kicker_width, 40), outline=STROKE, fill=PANEL, radius=20, opacity=a)
        canvas.ellipse((124, 164, 8, 8), fill=AMBER, opacity=a)
        canvas.text((145, 168), kicker, 15, AMBER, kind="mono-bold", anchor="lm", opacity=a)
        canvas.text((104, 224), title, 60, OFFWHITE, kind="bold", opacity=a)
        canvas.text((107, 298), subtitle, 21, MUTED, opacity=a)
        canvas.line(((106, 333), (214, 333)), CYAN, 4, a)
        canvas.line(((226, 333), (286, 333)), STROKE, 2, a)

    def _image_frame(
        self,
        canvas: Canvas,
        image: Image.Image,
        box: tuple[float, float, float, float],
        label: str,
        color: str,
        opacity: float,
    ) -> None:
        if opacity <= 0:
            return
        x, y, width, height = box
        canvas.rect((x - 12 + 10, y - 45 + 13, width + 24, height + 68), fill="#020912", radius=17, opacity=0.72 * opacity)
        canvas.rect((x - 12, y - 45, width + 24, height + 68), outline=STROKE, fill=PANEL, width=2, radius=17, opacity=opacity)
        canvas.ellipse((x + 2, y - 27, 8, 8), fill=color, opacity=opacity)
        canvas.text((x + 22, y - 23), label, 14, OFFWHITE, kind="mono-bold", anchor="lm", opacity=opacity)
        canvas.paste_cover(image, box, opacity, radius=10)
        canvas.rect(box, outline=color, width=2.5, radius=10, opacity=opacity)
        canvas.line(((x + 8, y + height + 11), (x + 74, y + height + 11)), color, 4, opacity)
        canvas.line(((x + 82, y + height + 11), (x + width - 8, y + height + 11)), STROKE, 2, opacity)

    def _camera(self, canvas: Canvas, position: tuple[float, float], angle: float, color: str, opacity: float = 1.0) -> None:
        x, y = position
        radius = 18
        forward = (math.cos(angle), math.sin(angle))
        side = (-forward[1], forward[0])
        body = [
            (x + side[0] * radius, y + side[1] * radius),
            (x - side[0] * radius, y - side[1] * radius),
            (x - side[0] * radius + forward[0] * 34, y - side[1] * radius + forward[1] * 34),
            (x + side[0] * radius + forward[0] * 34, y + side[1] * radius + forward[1] * 34),
        ]
        canvas.polygon(body, color, opacity)
        nose = (x + forward[0] * 54, y + forward[1] * 54)
        wing_a = (x + forward[0] * 32 + side[0] * 28, y + forward[1] * 32 + side[1] * 28)
        wing_b = (x + forward[0] * 32 - side[0] * 28, y + forward[1] * 32 - side[1] * 28)
        canvas.polygon((nose, wing_a, wing_b), color, opacity * 0.72)

    def _scene_intro(self, progress: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.base)
        self._chrome(canvas, 0, progress, "ONE ROLLOUT")
        title_a = reveal(progress, 0.02, 0.28)
        halo = reveal(progress, 0.04, 0.38)
        pulse = 9 * math.sin(progress * math.tau)
        canvas.glow_ellipse((960 - 320 - pulse, 344 - 138 - pulse, 640 + pulse * 2, 276 + pulse * 2), CYAN, width=2, opacity=0.14 * halo)
        canvas.ellipse((960 - 270, 344 - 106, 540, 212), outline=STROKE, width=1, opacity=0.38 * halo)
        canvas.text((960, 285), "One rollout.", 88, OFFWHITE, kind="bold", anchor="mm", opacity=title_a)
        canvas.text((960, 385), "Many truths.", 88, CYAN, kind="bold", anchor="mm", opacity=title_a)
        canvas.text(
            (960, 482),
            "A world model is not one score. We test what changes—and what should stay true.",
            23,
            MUTED,
            anchor="mm",
            opacity=reveal(progress, 0.18, 0.42),
        )
        film_a = reveal(progress, 0.34, 0.72)
        frame_w, frame_h, gap = 330, 185, 24
        left = (REFERENCE_W - (frame_w * 4 + gap * 3)) / 2
        for index, frame in enumerate((self.input_image, self.high_frames[6], self.high_frames[13], self.high_frames[21])):
            local = reveal(progress, 0.33 + index * 0.07, 0.60 + index * 0.07)
            box = (left + index * (frame_w + gap), 610, frame_w, frame_h)
            self._image_frame(canvas, frame, box, "INPUT" if index == 0 else f"T+{index:02d}", CYAN, local * film_a)
        labels = (("CONTROL", CYAN), ("GEOMETRY", AMBER), ("PHYSICS", GREEN), ("QUALITY", CORAL), ("MEMORY", OFFWHITE))
        total_width = 740
        for index, (label, color) in enumerate(labels):
            x = 960 - total_width / 2 + index * (total_width / 4)
            local = reveal(progress, 0.55 + index * 0.035, 0.79 + index * 0.02)
            canvas.rect((x - 74, 861, 148, 48), outline=STROKE, fill=PANEL, radius=24, opacity=local)
            canvas.ellipse((x - 51, 882, 7, 7), fill=color, opacity=local)
            canvas.text((x + 8, 885), label, 13, color, kind="mono-bold", anchor="mm", opacity=local)
        return canvas.image

    def _scene_pipeline(self, progress: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.base)
        self._chrome(canvas, 1, progress, "FROM PROMPT TO EVIDENCE")
        self._title(canvas, "00 / TEST PROTOCOL", "Generate once. Measure many ways.", "The same rollout feeds every evaluation dimension.", progress)

        shell_a = reveal(progress, 0.06, 0.22)
        canvas.panel((70, 365, 410, 475), accent=CYAN, opacity=shell_a)
        canvas.panel((655, 365, 420, 475), accent=AMBER, opacity=shell_a)
        canvas.panel((1130, 365, 720, 475), accent=GREEN, opacity=shell_a)
        canvas.text((102, 397), "01 / CONDITION", 12, MUTED, kind="mono-bold", opacity=shell_a)
        canvas.text((687, 397), "02 / GENERATE", 12, MUTED, kind="mono-bold", opacity=shell_a)
        canvas.text((1162, 397), "03 / OBSERVE", 12, MUTED, kind="mono-bold", opacity=shell_a)

        input_a = reveal(progress, 0.12, 0.29)
        self._image_frame(canvas, self.input_image, (105, 455, 340, 205), "INPUT IMAGE", CYAN, input_a)
        canvas.rect((105, 706, 340, 82), outline=STROKE, fill=PANEL_2, radius=11, opacity=input_a)
        canvas.ellipse((126, 729, 8, 8), fill=AMBER, opacity=input_a)
        canvas.text((148, 733), "Camera orbit left", 16, OFFWHITE, kind="mono-bold", anchor="lm", opacity=input_a)
        canvas.text((126, 768), "ACTION / PROMPT TARGET", 12, MUTED, kind="mono", anchor="lm", opacity=input_a)

        first_arrow = partial_polyline(((498, 602), (628, 602)), reveal(progress, 0.22, 0.39))
        if len(first_arrow) >= 2:
            canvas.arrow(first_arrow, CYAN, width=4, opacity=0.9)

        model_a = reveal(progress, 0.28, 0.46)
        pulse = 8 * math.sin(progress * math.tau * 2.0) * model_a
        canvas.glow_ellipse((710 - pulse, 422 - pulse, 310 + pulse * 2, 310 + pulse * 2), AMBER, width=3, opacity=model_a)
        canvas.ellipse((748, 460, 234, 234), outline=STROKE, fill=BG_2, width=2, opacity=model_a)
        orbit_nodes = ((865, 474), (942, 530), (915, 625), (815, 634), (781, 540))
        for index, node in enumerate(orbit_nodes):
            next_node = orbit_nodes[(index + 2) % len(orbit_nodes)]
            canvas.line((node, next_node), BLUE_GRAY, 2, 0.54 * model_a)
            canvas.ellipse((node[0] - 7, node[1] - 7, 14, 14), fill=(CYAN if index % 2 == 0 else AMBER), opacity=model_a)
        canvas.text((865, 548), "WORLD", 23, OFFWHITE, kind="mono-bold", anchor="mm", opacity=model_a)
        canvas.text((865, 585), "MODEL", 23, AMBER, kind="mono-bold", anchor="mm", opacity=model_a)
        canvas.text((865, 763), "predict the next state", 14, MUTED, kind="mono", anchor="mm", opacity=model_a)

        second_arrow = partial_polyline(((1096, 602), (1115, 602)), reveal(progress, 0.39, 0.52))
        if len(second_arrow) >= 2:
            canvas.arrow(second_arrow, AMBER, width=4, opacity=0.9)

        rollout_a = reveal(progress, 0.43, 0.72)
        frames = (self.high_frames[1], self.high_frames[8], self.high_frames[15], self.high_frames[23])
        for index, frame in enumerate(frames):
            local = reveal(progress, 0.42 + index * 0.045, 0.61 + index * 0.045)
            col, row = index % 2, index // 2
            box = (1170 + col * 325, 455 + row * 175, 290, 125)
            self._image_frame(canvas, frame, box, f"T+{index + 1:02d}", GREEN, local * rollout_a)
        canvas.text((1490, 808), "SHARED EVIDENCE STREAM", 13, GREEN, kind="mono-bold", anchor="mm", opacity=rollout_a)
        return canvas.image

    def _scene_control(self, progress: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.base)
        self._chrome(canvas, 2, progress, "CONTROL")
        self._title(canvas, "01 / CONTROL", "Did the world follow the command?", "Compare the requested camera motion with the motion observed in the rollout.", progress)

        shell_a = reveal(progress, 0.07, 0.23)
        canvas.panel((105, 355, 1460, 530), accent=CYAN, opacity=shell_a)
        canvas.panel((1590, 410, 260, 390), accent=CORAL, opacity=shell_a)
        canvas.text((145, 390), "CAMERA TRAJECTORY / WORLD COORDINATES", 12, MUTED, kind="mono-bold", opacity=shell_a)
        canvas.rect((1330, 375, 190, 36), outline=STROKE, fill=PANEL_2, radius=18, opacity=shell_a)
        canvas.text((1425, 393), "TARGET vs PREDICTED", 11, OFFWHITE, kind="mono-bold", anchor="mm", opacity=shell_a)

        for x in range(180, 1521, 134):
            canvas.dashed_line((x, 425), (x, 810), GRID, width=1, dash=7, gap=11, opacity=0.75 * shell_a)
        for y in range(370, 821, 90):
            canvas.dashed_line((145, y + 55), (1525, y + 55), GRID, width=1, dash=7, gap=11, opacity=0.75 * shell_a)

        target = bezier_points((230, 730), (520, 390), (960, 415), (1420, 500))
        predicted = bezier_points((230, 730), (500, 500), (1040, 720), (1420, 620))
        target_p = reveal(progress, 0.18, 0.55)
        predicted_p = reveal(progress, 0.32, 0.72)
        visible_target = partial_polyline(target, target_p)
        visible_predicted = partial_polyline(predicted, predicted_p)
        canvas.glow_line(visible_target, CYAN, 6, 1.0)
        canvas.glow_line(visible_predicted, AMBER, 6, 1.0)
        if len(visible_target) > 1:
            prev, current = visible_target[-2], visible_target[-1]
            self._camera(canvas, current, math.atan2(current[1] - prev[1], current[0] - prev[0]), CYAN)
        if len(visible_predicted) > 1:
            prev, current = visible_predicted[-2], visible_predicted[-1]
            self._camera(canvas, current, math.atan2(current[1] - prev[1], current[0] - prev[0]), AMBER)

        legend_a = reveal(progress, 0.26, 0.48)
        canvas.rect((175, 828, 276, 42), outline=STROKE, fill=PANEL_2, radius=21, opacity=legend_a)
        canvas.line(((196, 849), (248, 849)), CYAN, 5, legend_a)
        canvas.text((268, 849), "TARGET MOTION", 13, CYAN, kind="mono-bold", anchor="lm", opacity=legend_a)
        canvas.rect((472, 828, 310, 42), outline=STROKE, fill=PANEL_2, radius=21, opacity=legend_a)
        canvas.line(((493, 849), (545, 849)), AMBER, 5, legend_a)
        canvas.text((565, 849), "PREDICTED MOTION", 13, AMBER, kind="mono-bold", anchor="lm", opacity=legend_a)

        measure_a = reveal(progress, 0.48, 0.69)
        target_end = point_on_path(target, 1.0)
        predicted_end = point_on_path(predicted, 1.0)
        canvas.dashed_line(target_end, predicted_end, CORAL, width=3, dash=10, gap=8, opacity=measure_a)
        canvas.text((1720, 460), "MEASURE", 12, MUTED, kind="mono-bold", anchor="mm", opacity=measure_a)
        canvas.text((1720, 518), "Δ PATH", 27, CORAL, kind="mono-bold", anchor="mm", opacity=measure_a)
        canvas.line(((1630, 552), (1810, 552)), STROKE, 2, measure_a)
        for index, value in enumerate(("DIRECTION", "SPEED", "PATH")):
            canvas.ellipse((1635, 591 + index * 43, 7, 7), fill=(CYAN, AMBER, CORAL)[index], opacity=measure_a)
            canvas.text((1655, 595 + index * 43), value, 12, OFFWHITE, kind="mono-bold", anchor="lm", opacity=measure_a)
        canvas.text((1720, 746), "smaller error", 13, MUTED, kind="mono", anchor="mm", opacity=measure_a)
        canvas.text((1720, 775), "BETTER CONTROL", 14, CYAN, kind="mono-bold", anchor="mm", opacity=measure_a)
        return canvas.image

    def _scene_geometry(self, progress: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.base)
        self._chrome(canvas, 3, progress, "GEOMETRY")
        self._title(canvas, "02 / 3D + 4D GEOMETRY", "Does the scene remain spatially coherent?", "Track correspondences, depth and camera-relative structure through time.", progress)

        shell_a = reveal(progress, 0.06, 0.22)
        canvas.panel((70, 355, 970, 520), accent=CYAN, opacity=shell_a)
        canvas.panel((1065, 355, 785, 520), accent=AMBER, opacity=shell_a)
        canvas.text((100, 389), "FEATURE CORRESPONDENCE", 12, MUTED, kind="mono-bold", opacity=shell_a)
        canvas.text((1095, 389), "CAMERA-RELATIVE STRUCTURE", 12, MUTED, kind="mono-bold", opacity=shell_a)

        box_a = (100, 440, 390, 275)
        box_b = (590, 440, 390, 275)
        image_a = reveal(progress, 0.12, 0.28)
        image_b = reveal(progress, 0.20, 0.36)
        self._image_frame(canvas, self.input_image, box_a, "INPUT FRAME", CYAN, image_a)
        self._image_frame(canvas, self.high_frames[0], box_b, "ROLLOUT T+00", AMBER, image_b)

        match_a = reveal(progress, 0.30, 0.66)
        count = max(1, round(len(self.matches) * match_a))
        source_w, source_h = self.input_image.size
        target_w, target_h = self.high_frames[0].size
        for index, (sx, sy, tx, ty) in enumerate(self.matches[:count]):
            source_point = (box_a[0] + sx / source_w * box_a[2], box_a[1] + sy / source_h * box_a[3])
            target_point = (box_b[0] + tx / target_w * box_b[2], box_b[1] + ty / target_h * box_b[3])
            color = CYAN if index % 3 else AMBER
            canvas.line((source_point, target_point), color, 1.2, 0.44)
            canvas.ellipse((source_point[0] - 3, source_point[1] - 3, 6, 6), fill=color, opacity=0.85)
            canvas.ellipse((target_point[0] - 3, target_point[1] - 3, 6, 6), fill=color, opacity=0.85)

        verified_a = reveal(progress, 0.48, 0.67)
        canvas.rect((100, 766, 880, 76), outline=STROKE, fill=PANEL_2, radius=13, opacity=verified_a)
        canvas.ellipse((124, 789, 10, 10), fill=GREEN, opacity=verified_a)
        canvas.text((151, 793), "575 VERIFIED INLIERS", 17, GREEN, kind="mono-bold", anchor="lm", opacity=verified_a)
        canvas.text((151, 819), "same input confirmed before comparison", 12, MUTED, kind="mono", anchor="lm", opacity=verified_a)
        canvas.text((948, 804), "SIFT + RANSAC", 12, OFFWHITE, kind="mono-bold", anchor="rm", opacity=verified_a)

        geom_a = reveal(progress, 0.18, 0.45)
        origin = (1320, 705)
        plane = ((1080, 690), (1490, 535), (1780, 660), (1360, 835))
        canvas.polygon(plane, BLUE_GRAY, 0.32 * geom_a)
        for fraction in (0.2, 0.4, 0.6, 0.8):
            a = (plane[0][0] * (1 - fraction) + plane[1][0] * fraction, plane[0][1] * (1 - fraction) + plane[1][1] * fraction)
            b = (plane[3][0] * (1 - fraction) + plane[2][0] * fraction, plane[3][1] * (1 - fraction) + plane[2][1] * fraction)
            canvas.line((a, b), BLUE_GRAY, 2, 0.72 * geom_a)
        rng = np.random.default_rng(420)
        for index in range(42):
            x = float(rng.normal(1375, 125))
            y = float(rng.normal(630, 78))
            radius = 2.5 + (index % 4) * 0.6
            canvas.ellipse((x - radius, y - radius, radius * 2, radius * 2), fill=(CYAN if index % 5 else OFFWHITE), opacity=(0.34 + (index % 3) * 0.16) * geom_a)
        frustums = ((1120, 520, CYAN), (1320, 435, AMBER), (1540, 500, GREEN))
        for index, (x, y, color) in enumerate(frustums):
            local = reveal(progress, 0.22 + index * 0.055, 0.42 + index * 0.055)
            canvas.ellipse((x - 12, y - 12, 24, 24), fill=color, opacity=local)
            canvas.glow_line(((x, y), (origin[0] - 120 + index * 120, origin[1])), color, 3, local)
            canvas.glow_line(((x, y), (origin[0] - 20 + index * 120, origin[1] + 60)), color, 3, local)
        canvas.rect((1228, 401, 405, 58), outline=STROKE, fill=PANEL_2, radius=29, opacity=geom_a)
        canvas.text((1430, 430), "2D  →  3D  →  4D", 19, OFFWHITE, kind="mono-bold", anchor="mm", opacity=geom_a)
        canvas.text((1430, 835), "FEATURES  •  DEPTH  •  TIME", 12, AMBER, kind="mono-bold", anchor="mm", opacity=geom_a)
        for index in range(12):
            amount = index / 11
            color = mix_color(CYAN, OFFWHITE, amount)
            canvas.rect((1787, 510 + index * 20, 13, 20), fill=color, opacity=geom_a)
        canvas.text((1810, 510), "FAR", 10, MUTED, kind="mono-bold", anchor="lm", opacity=geom_a)
        canvas.text((1810, 742), "NEAR", 10, OFFWHITE, kind="mono-bold", anchor="lm", opacity=geom_a)
        return canvas.image

    def _scene_physics(self, progress: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.base)
        self._chrome(canvas, 4, progress, "PHYSICS")
        self._title(canvas, "03 / PHYSICS", "Is the rollout plausible?", "Test motion, stability and collisions against the expected dynamics.", progress)

        shell_a = reveal(progress, 0.06, 0.22)
        canvas.panel((90, 355, 1425, 525), accent=GREEN, opacity=shell_a)
        canvas.panel((1545, 420, 305, 330), accent=CORAL, opacity=shell_a)
        canvas.text((125, 390), "DYNAMICS PROBE / TRAJECTORY TEST", 12, MUTED, kind="mono-bold", opacity=shell_a)
        canvas.text((1698, 458), "EVENT MONITOR", 12, MUTED, kind="mono-bold", anchor="mm", opacity=shell_a)
        canvas.rect((1616, 482, 164, 36), outline=STROKE, fill=PANEL_2, radius=18, opacity=shell_a)
        canvas.ellipse((1640, 496, 8, 8), fill=GREEN, opacity=shell_a)
        canvas.text((1706, 500), "MONITORING", 11, OFFWHITE, kind="mono-bold", anchor="mm", opacity=shell_a)
        for index, value in enumerate(("MOTION", "STABILITY", "COLLISION")):
            canvas.text((1580, 560 + index * 52), value, 11, MUTED, kind="mono-bold", anchor="lm", opacity=shell_a)
            canvas.line(((1690, 560 + index * 52), (1815, 560 + index * 52)), STROKE, 4, shell_a)

        world_a = reveal(progress, 0.12, 0.29)
        canvas.line(((150, 795), (1490, 795)), BLUE_GRAY, 5, world_a)
        canvas.line(((1290, 430), (1290, 795)), BLUE_GRAY, 5, world_a)
        canvas.text((1314, 458), "OBSTACLE", 14, MUTED, kind="mono-bold", opacity=world_a)
        for x in range(190, 1490, 120):
            canvas.line(((x, 795), (x + 70, 850)), GRID, 2, world_a)

        expected = bezier_points((240, 690), (540, 320), (950, 370), (1450, 690))
        observed = bezier_points((240, 690), (560, 395), (970, 470), (1290, 650))
        expected_a = reveal(progress, 0.20, 0.58)
        observed_a = reveal(progress, 0.32, 0.72)
        canvas.glow_line(partial_polyline(expected, expected_a), GREEN, 6, 1.0)
        canvas.glow_line(partial_polyline(observed, observed_a), CORAL, 6, 1.0)

        if expected_a > 0:
            x, y = point_on_path(expected, expected_a)
            canvas.ellipse((x - 24, y - 24, 48, 48), fill=GREEN, opacity=1.0)
        if observed_a > 0:
            x, y = point_on_path(observed, observed_a)
            canvas.rect((x - 25, y - 25, 50, 50), fill=CORAL, radius=7, opacity=1.0)

        impact_a = reveal(progress, 0.68, 0.78)
        if impact_a > 0:
            center = (1290, 650)
            for angle in np.linspace(0, math.tau, 12, endpoint=False):
                inner = (center[0] + math.cos(angle) * 42, center[1] + math.sin(angle) * 42)
                outer = (center[0] + math.cos(angle) * (42 + 55 * impact_a), center[1] + math.sin(angle) * (42 + 55 * impact_a))
                canvas.line((inner, outer), CORAL, 4, 1.0 - impact_a * 0.45)
            canvas.rect((1564, 477, 267, 248), fill=PANEL_2, radius=15, opacity=impact_a)
            canvas.ellipse((1580, 500, 12, 12), fill=CORAL, opacity=impact_a)
            canvas.text((1605, 506), "COLLISION", 17, CORAL, kind="mono-bold", anchor="lm", opacity=impact_a)
            canvas.line(((1580, 542), (1815, 542)), STROKE, 2, impact_a)
            canvas.text((1580, 579), "expected", 12, MUTED, kind="mono", opacity=impact_a)
            canvas.text((1815, 579), "clear", 13, GREEN, kind="mono-bold", anchor="ra", opacity=impact_a)
            canvas.text((1580, 620), "observed", 12, MUTED, kind="mono", opacity=impact_a)
            canvas.text((1815, 620), "impact", 13, CORAL, kind="mono-bold", anchor="ra", opacity=impact_a)
            canvas.text((1698, 692), "trajectory diverges", 12, OFFWHITE, kind="mono-bold", anchor="mm", opacity=impact_a)

        legend_a = reveal(progress, 0.25, 0.46)
        canvas.line(((200, 850), (270, 850)), GREEN, 6, legend_a)
        canvas.text((292, 850), "EXPECTED", 14, GREEN, kind="mono-bold", anchor="lm", opacity=legend_a)
        canvas.line(((480, 850), (550, 850)), CORAL, 6, legend_a)
        canvas.text((572, 850), "OBSERVED", 14, CORAL, kind="mono-bold", anchor="lm", opacity=legend_a)
        for index, value in enumerate(("MOTION", "STABILITY", "COLLISION")):
            x = 900 + index * 205
            local = reveal(progress, 0.54 + index * 0.05, 0.72 + index * 0.05)
            canvas.rect((x - 82, 827, 164, 44), outline=STROKE, fill=PANEL_2, radius=22, opacity=local)
            canvas.text((x, 850), value, 12, OFFWHITE, kind="mono-bold", anchor="mm", opacity=local)
        return canvas.image

    def _waveform(self, canvas: Canvas, box: tuple[float, float, float, float], color: str, phase: float, coherence: float, opacity: float) -> None:
        x, y, width, height = box
        points = []
        for index in range(100):
            u = index / 99
            envelope = 0.34 + 0.66 * math.sin(u * math.pi) ** 2
            signal = math.sin(u * math.tau * (3.0 + coherence * 3.0) + phase)
            signal += (1.0 - coherence) * 0.55 * math.sin(u * math.tau * 17.0 + phase * 0.7)
            points.append((x + u * width, y + height / 2 + signal * height * 0.35 * envelope))
        canvas.line(points, color, 3, opacity)

    def _scene_quality(self, progress: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.base)
        self._chrome(canvas, 5, progress, "VISUAL QUALITY")
        self._title(canvas, "04 / QUALITY", "Which rollout looks better?", "A real same-input example: preference and image quality can separate two outputs.", progress)

        high_index = min(len(self.high_frames) - 1, round(clamp(progress * 1.12) * (len(self.high_frames) - 1)))
        low_index = min(len(self.low_frames) - 1, round(clamp(progress * 1.12) * (len(self.low_frames) - 1)))
        high_a = reveal(progress, 0.12, 0.31)
        low_a = reveal(progress, 0.20, 0.39)
        self._image_frame(canvas, self.high_frames[high_index], (110, 395, 790, 380), "SAME INPUT / OUTPUT A", GREEN, high_a)
        self._image_frame(canvas, self.low_frames[low_index], (1020, 395, 790, 380), "SAME INPUT / OUTPUT B", CORAL, low_a)

        compare_a = reveal(progress, 0.30, 0.49)
        canvas.glow_ellipse((924, 545, 72, 72), OFFWHITE, width=2, opacity=compare_a)
        canvas.ellipse((932, 553, 56, 56), fill=PANEL_2, opacity=compare_a)
        canvas.text((960, 581), "VS", 14, OFFWHITE, kind="mono-bold", anchor="mm", opacity=compare_a)

        score_a = reveal(progress, 0.40, 0.64)
        canvas.rect((110, 817, 790, 78), outline=STROKE, fill=PANEL_2, radius=14, opacity=score_a)
        canvas.rect((1020, 817, 790, 78), outline=STROKE, fill=PANEL_2, radius=14, opacity=score_a)
        canvas.text((135, 856), "HPSv3", 13, MUTED, kind="mono-bold", anchor="lm", opacity=score_a)
        canvas.text((242, 856), "1.00", 28, GREEN, kind="mono-bold", anchor="lm", opacity=score_a)
        canvas.text((1045, 856), "HPSv3", 13, MUTED, kind="mono-bold", anchor="lm", opacity=score_a)
        canvas.text((1152, 856), "0.00", 28, CORAL, kind="mono-bold", anchor="lm", opacity=score_a)
        self._waveform(canvas, (345, 831, 515, 48), GREEN, progress * 9.0, 0.94, score_a)
        self._waveform(canvas, (1255, 831, 515, 48), CORAL, progress * 12.0, 0.32, score_a)
        verified_a = reveal(progress, 0.58, 0.79)
        canvas.rect((675, 919, 570, 40), outline=STROKE, fill=PANEL, radius=20, opacity=verified_a)
        canvas.ellipse((704, 935, 8, 8), fill=GREEN, opacity=verified_a)
        canvas.text((960, 939), "VERIFIED CASE  •  ONE INPUT  •  TWO PREDICTIONS", 12, OFFWHITE, kind="mono-bold", anchor="mm", opacity=verified_a)
        return canvas.image

    def _scene_memory(self, progress: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.base)
        self._chrome(canvas, 6, progress, "LONG HORIZON")
        self._title(canvas, "05 / MEMORY", "What survives a long rollout?", "Leave the scene, explore, then return: identity and structure should persist.", progress)

        rail_a = reveal(progress, 0.06, 0.22)
        canvas.line(((345, 355), (1575, 355)), STROKE, 3, rail_a)
        for index, (x, color) in enumerate(((345, CYAN), (960, AMBER), (1575, GREEN))):
            canvas.glow_ellipse((x - 10, 345, 20, 20), color, width=2, opacity=rail_a)
            canvas.text((x, 374), f"T{index}", 11, color, kind="mono-bold", anchor="mm", opacity=rail_a)

        boxes = ((125, 385, 440, 300), (740, 385, 440, 300), (1355, 385, 440, 300))
        frames = (self.input_image, self.high_frames[14], self.high_frames[26])
        labels = ("LEAVE / T+00", "EXPLORE / T+N", "RETURN / T+2N")
        colors = (CYAN, AMBER, GREEN)
        for index, (box, frame, label, color) in enumerate(zip(boxes, frames, labels, colors)):
            local = reveal(progress, 0.12 + index * 0.12, 0.32 + index * 0.12)
            self._image_frame(canvas, frame, box, label, color, local)
            marker_x = box[0] + box[2] * (0.58 if index != 1 else 0.42)
            marker_y = box[1] + box[3] * 0.34
            canvas.ellipse((marker_x - 22, marker_y - 22, 44, 44), outline=OFFWHITE, width=4, opacity=local * (0.35 if index == 1 else 0.95))
            canvas.rect((marker_x - 38, marker_y + 34, 76, 30), fill=PANEL_2, radius=15, opacity=local * (0.48 if index == 1 else 0.95))
            canvas.text((marker_x, marker_y + 49), "ID-A", 11, OFFWHITE, kind="mono-bold", anchor="mm", opacity=local * (0.35 if index == 1 else 0.95))

        arrow_1 = partial_polyline(((575, 535), (710, 535)), reveal(progress, 0.28, 0.47))
        arrow_2 = partial_polyline(((1190, 535), (1325, 535)), reveal(progress, 0.46, 0.65))
        if len(arrow_1) >= 2:
            canvas.glow_line(arrow_1, AMBER, width=4)
            canvas.arrow(arrow_1, AMBER, width=4)
        if len(arrow_2) >= 2:
            canvas.glow_line(arrow_2, GREEN, width=4)
            canvas.arrow(arrow_2, GREEN, width=4)

        loop_a = reveal(progress, 0.44, 0.74)
        loop = bezier_points((1575, 725), (1420, 920), (450, 920), (345, 725), count=90)
        visible_loop = partial_polyline(loop, loop_a)
        if len(visible_loop) >= 2:
            canvas.arrow(visible_loop, CYAN, width=4, opacity=0.9, head=17)
        canvas.rect((600, 842, 720, 82), outline=STROKE, fill=PANEL, radius=18, opacity=loop_a)
        canvas.text((960, 872), "LEAVE  →  EXPLORE  →  RETURN", 16, CYAN, kind="mono-bold", anchor="mm", opacity=loop_a)
        canvas.text((960, 903), "compare persistent identity, layout and appearance", 13, MUTED, kind="mono", anchor="mm", opacity=loop_a)
        return canvas.image

    def _scene_aggregate(self, progress: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.base)
        self._chrome(canvas, 7, progress, "AGGREGATION")
        self._title(canvas, "06 / SCORE", "Evidence becomes a comparable score.", "Each metric is normalized, grouped by dimension, then aggregated for the leaderboard.", progress)

        shell_a = reveal(progress, 0.05, 0.20)
        canvas.panel((90, 355, 630, 515), accent=CYAN, opacity=shell_a)
        canvas.panel((755, 355, 520, 515), accent=AMBER, opacity=shell_a)
        canvas.panel((1385, 355, 465, 515), accent=LILAC, opacity=shell_a)
        canvas.text((125, 390), "DIMENSION EVIDENCE", 12, MUTED, kind="mono-bold", opacity=shell_a)
        canvas.text((1015, 390), "NORMALIZATION", 12, MUTED, kind="mono-bold", anchor="mm", opacity=shell_a)
        canvas.text((1618, 390), "COMPARABLE OUTPUT", 12, MUTED, kind="mono-bold", anchor="mm", opacity=shell_a)

        metrics = (
            ("CONTROL", CYAN),
            ("GEOMETRY", AMBER),
            ("PHYSICS", GREEN),
            ("QUALITY", CORAL),
            ("MEMORY", OFFWHITE),
        )
        center = (1110, 570)
        for index, (label, color) in enumerate(metrics):
            y = 375 + index * 105
            local = reveal(progress, 0.10 + index * 0.075, 0.31 + index * 0.075)
            canvas.text((150, y), label, 19, color, kind="mono-bold", anchor="lm", opacity=local)
            canvas.line(((330, y), (520, y)), BLUE_GRAY, 8, local)
            signal = 72 + (index * 29) % 105
            canvas.glow_line(((330, y), (330 + signal, y)), color, 8, local)
            path = ((540, y), (790, y), (950, center[1]), center)
            visible = partial_polyline(path, reveal(progress, 0.25 + index * 0.045, 0.55 + index * 0.045))
            canvas.glow_line(visible, color, 3, local)

        aggregate_a = reveal(progress, 0.32, 0.54)
        canvas.glow_ellipse((995, 455, 230, 230), AMBER, width=5, opacity=aggregate_a)
        canvas.ellipse((1018, 478, 184, 184), outline=STROKE, fill=BG_2, width=2, opacity=aggregate_a)
        canvas.text(center, "NORMALIZE\n+\nAGGREGATE", 18, OFFWHITE, kind="mono-bold", anchor="mm", align="center", opacity=aggregate_a, spacing=4)

        final_path = partial_polyline(((1235, 570), (1430, 570)), reveal(progress, 0.46, 0.62))
        if len(final_path) >= 2:
            canvas.glow_line(final_path, LILAC, width=5)
            canvas.arrow(final_path, AMBER, width=5)

        score_a = reveal(progress, 0.41, 0.59)
        canvas.glow_ellipse((1480, 405, 330, 330), CYAN, width=7, opacity=score_a)
        canvas.ellipse((1515, 440, 260, 260), outline=GRID, width=2, opacity=score_a)
        canvas.text((1645, 522), "SCORE", 19, MUTED, kind="mono-bold", anchor="mm", opacity=score_a)
        canvas.text((1645, 590), "[0, 1]", 52, CYAN, kind="mono-bold", anchor="mm", opacity=score_a)
        canvas.text((1645, 646), "traceable evidence", 14, OFFWHITE, kind="mono", anchor="mm", opacity=score_a)
        footer_a = reveal(progress, 0.57, 0.78)
        canvas.rect((598, 904, 724, 46), outline=STROKE, fill=PANEL, radius=23, opacity=footer_a)
        canvas.text((960, 927), "METRIC EVIDENCE  →  DIMENSION SCORES  →  LEADERBOARD", 13, OFFWHITE, kind="mono-bold", anchor="mm", opacity=footer_a)
        return canvas.image

    def _compass(self, canvas: Canvas, center: tuple[float, float], radius: float, opacity: float) -> None:
        x, y = center
        canvas.ellipse((x - radius, y - radius, radius * 2, radius * 2), outline=AMBER, width=4, opacity=opacity)
        canvas.ellipse((x - radius * 0.72, y - radius * 0.72, radius * 1.44, radius * 1.44), outline=GRID, width=2, opacity=opacity)
        canvas.polygon(((x, y - radius * 0.88), (x + radius * 0.20, y + radius * 0.10), (x, y), (x - radius * 0.20, y + radius * 0.10)), AMBER, opacity)
        canvas.polygon(((x, y + radius * 0.88), (x + radius * 0.18, y - radius * 0.08), (x, y), (x - radius * 0.18, y - radius * 0.08)), CYAN, opacity)
        canvas.ellipse((x - 7, y - 7, 14, 14), fill=OFFWHITE, opacity=opacity)

    def _scene_outro(self, progress: float) -> Image.Image:
        canvas = Canvas(self.width, self.height, self.base)
        self._chrome(canvas, 8, progress, "WORLDATLAS ARENA")
        enter = reveal(progress, 0.02, 0.24)
        ring = 180 + 10 * math.sin(progress * math.tau)
        canvas.glow_ellipse((960 - ring, 305 - ring, ring * 2, ring * 2), AMBER, width=2, opacity=0.22 * enter)
        canvas.ellipse((960 - 142, 305 - 142, 284, 284), outline=STROKE, width=1, opacity=0.52 * enter)
        for index, color in enumerate((CYAN, AMBER, GREEN, CORAL)):
            angle = progress * math.tau * (0.32 + index * 0.05) + index * math.pi / 2
            radius = 150 + index * 9
            x = 960 + math.cos(angle) * radius
            y = 305 + math.sin(angle) * radius
            canvas.ellipse((x - 5, y - 5, 10, 10), fill=color, opacity=enter)
        self._compass(canvas, (960, 305), 88, enter)
        canvas.text((960, 468), "WorldAtlas", 76, OFFWHITE, kind="bold", anchor="mm", opacity=enter)
        canvas.text((960, 548), "ARENA", 44, AMBER, kind="mono-bold", anchor="mm", opacity=enter)
        statement_a = reveal(progress, 0.18, 0.38)
        canvas.rect((592, 621, 736, 78), outline=STROKE, fill=PANEL, radius=39, opacity=statement_a)
        canvas.ellipse((630, 655, 10, 10), fill=CYAN, opacity=statement_a)
        canvas.text((960, 660), "Every score traces back to evidence.", 25, CYAN, kind="regular", anchor="mm", opacity=statement_a)

        line_a = reveal(progress, 0.28, 0.58)
        labels = ("DATA", "ROLLOUT", "METRICS", "LEADERBOARD")
        xs = (470, 795, 1125, 1450)
        for index, (label, x) in enumerate(zip(labels, xs)):
            local = reveal(progress, 0.26 + index * 0.05, 0.45 + index * 0.05)
            color = (CYAN, AMBER, GREEN, CORAL)[index]
            canvas.glow_ellipse((x - 12, 774, 24, 24), color, width=3, opacity=local)
            canvas.rect((x - 88, 815, 176, 46), outline=STROKE, fill=PANEL_2, radius=23, opacity=local)
            canvas.text((x, 838), label, 13, OFFWHITE, kind="mono-bold", anchor="mm", opacity=local)
            if index < len(xs) - 1:
                path = partial_polyline(((x + 20, 786), (xs[index + 1] - 20, 786)), reveal(progress, 0.32 + index * 0.05, 0.52 + index * 0.05))
                if len(path) >= 2:
                    canvas.glow_line(path, BLUE_GRAY, width=3, opacity=line_a)
                    canvas.arrow(path, BLUE_GRAY, width=3, opacity=line_a, head=13)
        return canvas.image

    def render_scene(self, index: int, progress: float) -> Image.Image:
        renderer = (
            self._scene_intro,
            self._scene_pipeline,
            self._scene_control,
            self._scene_geometry,
            self._scene_physics,
            self._scene_quality,
            self._scene_memory,
            self._scene_aggregate,
            self._scene_outro,
        )[index]
        return renderer(clamp(progress))

    def render_frame(self, time_seconds: float) -> Image.Image:
        time_seconds %= TOTAL_DURATION
        index = max(0, min(len(SCENE_TIMES) - 2, np.searchsorted(SCENE_TIMES, time_seconds, side="right") - 1))
        start, end = SCENE_TIMES[index], SCENE_TIMES[index + 1]
        progress = (time_seconds - start) / (end - start)
        frame = self.render_scene(index, progress)
        fade_seconds = 0.64
        if end - time_seconds < fade_seconds:
            blend = smooth((time_seconds - (end - fade_seconds)) / fade_seconds)
            next_index = (index + 1) % 9
            next_duration = SCENE_TIMES[next_index + 1] - SCENE_TIMES[next_index]
            next_progress = blend * fade_seconds / next_duration
            next_frame = self.render_scene(next_index, next_progress)
            frame = Image.blend(frame, next_frame, blend)
            sweep = int((-0.18 + blend * 1.36) * self.width)
            band = max(36, round(92 * self.scale))
            slant = round(170 * self.scale)
            overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            cyan = hex_rgb(CYAN)
            overlay_draw.polygon(
                (
                    (sweep - band, -slant),
                    (sweep + band, -slant),
                    (sweep + band + slant, self.height + slant),
                    (sweep - band + slant, self.height + slant),
                ),
                fill=(*cyan, round(34 * math.sin(blend * math.pi))),
            )
            overlay = overlay.filter(ImageFilter.GaussianBlur(max(4, round(24 * self.scale))))
            frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        return frame


def render_video(
    explainer: MetricExplainer,
    output: Path,
    webm: Path,
    poster: Path,
    fps: int,
    duration: float,
    crf: int,
    preset: str,
    webm_crf: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    webm.parent.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=fps)
    stream.width = explainer.width
    stream.height = explainer.height
    stream.pix_fmt = "yuv420p"
    # FFmpeg's numeric H.264 profile id is stable across PyAV versions; the
    # symbolic "high" value is parsed as an expression by newer builds.
    stream.options = {"crf": str(crf), "preset": preset, "profile": "100"}

    webm_container = av.open(str(webm), "w", format="webm")
    webm_stream = webm_container.add_stream("libvpx-vp9", rate=fps)
    webm_stream.width = explainer.width
    webm_stream.height = explainer.height
    webm_stream.pix_fmt = "yuv420p"
    webm_stream.bit_rate = 0
    webm_stream.options = {
        "crf": str(webm_crf),
        "deadline": "good",
        "cpu-used": "3",
        "row-mt": "1",
    }

    total_frames = round(duration * fps)
    poster_frame = explainer.render_frame(53.6)
    poster_frame.save(poster, quality=92, optimize=True, progressive=True)
    for frame_index in range(total_frames):
        time_seconds = frame_index / fps
        image = explainer.render_frame(time_seconds)
        pixels = np.asarray(image)
        video_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
        webm_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in webm_stream.encode(webm_frame):
            webm_container.mux(packet)
        if frame_index % max(fps * 2, 1) == 0:
            print(f"rendered {frame_index:04d}/{total_frames} frames", flush=True)
    for packet in stream.encode():
        container.mux(packet)
    for packet in webm_stream.encode():
        webm_container.mux(packet)
    container.close()
    webm_container.close()
    print(f"video:  {output}")
    print(f"webm:   {webm}")
    print(f"poster: {poster}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site/assets/worldatlas_metric_explainer.mp4")
    parser.add_argument("--webm", type=Path, default=ROOT / "site/assets/worldatlas_metric_explainer.webm")
    parser.add_argument("--poster", type=Path, default=ROOT / "site/assets/worldatlas_metric_explainer_poster.jpg")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=TOTAL_DURATION)
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--webm-crf", type=int, default=30)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--frame-time", type=float, help="Render a single PNG/JPEG instead of a video")
    parser.add_argument("--frame-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explainer = MetricExplainer(args.width, args.height)
    if args.frame_time is not None:
        destination = args.frame_output or Path("/tmp/worldatlas_metric_explainer_frame.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        explainer.render_frame(args.frame_time).save(destination)
        print(destination)
        return
    render_video(explainer, args.output, args.webm, args.poster, args.fps, args.duration, args.crf, args.preset, args.webm_crf)


if __name__ == "__main__":
    main()
