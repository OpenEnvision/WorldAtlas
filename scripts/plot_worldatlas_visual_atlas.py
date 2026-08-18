#!/usr/bin/env python3
"""Render a low-text, publication-ready visual atlas for WorldArena.

The atlas intentionally mixes chart families—unit charts, radial bars, word
clouds, donut small multiples, bubble plots, treemaps, ridgelines, heatmaps,
and a specification matrix—while preserving one data scope and one color
language.  Every figure is exported as PDF/SVG/400-dpi PNG, plus a combined
multipage PDF.

Usage:
    python scripts/plot_worldatlas_visual_atlas.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import ImageFont
from scipy.stats import gaussian_kde


REPO_ROOT = Path(__file__).resolve().parents[1]
STATS_ROOT = REPO_ROOT / "outputs" / "worldatlas_statistics"
IMAGE_RECORDS = STATS_ROOT / "image" / "image_records.csv"
VIDEO_RECORDS = STATS_ROOT / "video" / "video_records.csv"
IMAGE_ANNOTATIONS = REPO_ROOT / "data_external" / "OpenEnvision" / "worldarena" / "image" / "image_annotations.json"
DEFAULT_OUTPUT = STATS_ROOT / "visual_atlas"


INK = "#1F2933"
MUTED = "#667085"
GRID = "#D9E1E8"
PAPER = "#FFFFFF"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
TEAL = "#009E73"
GOLD = "#E69F00"
PURPLE = "#CC79A7"
UNKNOWN = "#98A2B3"
PALE = "#EAF0F4"


GROUP_COLORS = {
    "Static / Photorealistic": BLUE,
    "Static / Stylized": VERMILLION,
    "Dynamic / Photorealistic": TEAL,
    "Dynamic / Stylized": PURPLE,
    "AAA Games": GOLD,
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def pretty(value: object) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def pct(value: float, total: float) -> str:
    return f"{100 * value / total:.1f}%" if total else "0.0%"


def ensure_records() -> None:
    if IMAGE_RECORDS.exists() and VIDEO_RECORDS.exists():
        return
    base = REPO_ROOT / "scripts" / "plot_worldatlas_dataset_statistics.py"
    subprocess.run([sys.executable, str(base), "--section", "all"], cwd=REPO_ROOT, check=True)


def load_object_annotations() -> pd.DataFrame:
    payload = json.loads(IMAGE_ANNOTATIONS.read_text())
    rows: list[dict[str, object]] = []
    for item in payload["items"]:
        relative_path = item["relative_path"]
        parts = Path(relative_path).parts
        generation = item.get("generation_spec") or {}
        rows.append(
            {
                "relative_path": relative_path,
                "mode": parts[0],
                "style": parts[1],
                "context_objects": generation.get("current_objects") or [],
                "target_objects": generation.get("objects") or [],
            }
        )
    return pd.DataFrame(rows)


def video_group(row: pd.Series) -> str:
    if row["branch"] == "AAA Games":
        return "AAA Games"
    return f"{row['branch']} / {pretty(row['style'])}"


def finish_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    title: str,
    source: str,
    book: PdfPages,
    *,
    rect: tuple[float, float, float, float] = (0, 0.055, 1, 0.93),
) -> None:
    fig.suptitle(title, x=0.055, y=0.985, ha="left", va="top", fontsize=14.5, weight="semibold", color=INK)
    fig.text(0.012, 0.015, source, ha="left", va="bottom", fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=rect)
    for ext, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 400})):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    book.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def largest_remainder(values: Iterable[float], total_units: int = 100) -> list[int]:
    values = np.asarray(list(values), dtype=float)
    raw = values / values.sum() * total_units
    base = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - base))[: total_units - base.sum()]:
        base[index] += 1
    return base.tolist()


def draw_waffle(
    ax: plt.Axes,
    labels: list[str],
    values: list[int],
    colors: list[str],
    panel_title: str,
) -> None:
    units = largest_remainder(values)
    cell_colors: list[str] = []
    for color, count in zip(colors, units):
        cell_colors.extend([color] * count)
    for index, color in enumerate(cell_colors):
        row, col = divmod(index, 10)
        x, y = col, 9 - row
        ax.add_patch(FancyBboxPatch((x, y), 0.78, 0.78, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=color, edgecolor=PAPER, linewidth=0.45))
    ax.text(0, 10.6, panel_title, ha="left", va="bottom", fontsize=11.5, weight="semibold", color=INK)
    total = sum(values)
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = 8.6 - index * 1.65
        ax.scatter([11.2], [y], s=54, color=color, edgecolor="none")
        ax.text(11.7, y + 0.18, label, ha="left", va="center", fontsize=8.1, color=INK, weight="semibold")
        ax.text(11.7, y - 0.35, f"{value:,} · {pct(value, total)}", ha="left", va="center", fontsize=7.2, color=MUTED)
    ax.set_xlim(-0.3, 18.2)
    ax.set_ylim(-0.3, 11.4)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_composition_waffles(image: pd.DataFrame, video: pd.DataFrame, out: Path, book: PdfPages) -> None:
    image_groups = image.assign(group=image["mode"].map(pretty) + " / " + image["style"].map(pretty))["group"].value_counts()
    video_work = video.copy()
    video_work["group"] = video_work.apply(video_group, axis=1)
    video_groups = video_work["group"].value_counts()
    image_order = ["Static / Photorealistic", "Static / Stylized", "Dynamic / Photorealistic", "Dynamic / Stylized"]
    video_order = ["AAA Games", "Static / Photorealistic", "Static / Stylized", "Dynamic / Photorealistic", "Dynamic / Stylized"]
    short = {
        "Static / Photorealistic": "Static · photo",
        "Static / Stylized": "Static · stylized",
        "Dynamic / Photorealistic": "Dynamic · photo",
        "Dynamic / Stylized": "Dynamic · stylized",
        "AAA Games": "AAA Games",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4))
    draw_waffle(axes[0], [short[x] for x in image_order], [int(image_groups[x]) for x in image_order], [GROUP_COLORS[x] for x in image_order], "Images · 5,000")
    draw_waffle(axes[1], [short[x] for x in video_order], [int(video_groups[x]) for x in video_order], [GROUP_COLORS[x] for x in video_order], "Videos · 5,426")
    finish_figure(fig, out, "visual_01_composition_waffles", "Balanced images, uneven videos", "1 tile ≈ 1% · current local files.", book)


def plot_image_taxonomy_radial(image: pd.DataFrame, out: Path, book: PdfPages) -> None:
    order = [
        "Indoor · Dining Spaces",
        "Indoor · Living Spaces",
        "Indoor · Passageways",
        "Indoor · Public Spaces",
        "Indoor · Workspaces",
        "Outdoor · Aquatic Landscape",
        "Outdoor · City",
        "Outdoor · Suburb",
        "Outdoor · Terrestrial Landscape",
        "Outdoor · Verdant Landscape",
        "Articulated",
        "Rigid",
        "Fluid",
        "Deformable",
        "Multi Motion",
    ]
    labels = ["Dining", "Living", "Passage", "Public", "Work", "Aquatic", "City", "Suburb", "Terrestrial", "Verdant", "Articulated", "Rigid", "Fluid", "Deformable", "Multi"]
    table = image.pivot_table(index="taxonomy", columns="style", values="relative_path", aggfunc="count", fill_value=0).reindex(order)
    angles = np.linspace(0, 2 * np.pi, len(order), endpoint=False)
    sector = 2 * np.pi / len(order)
    width = sector * 0.34
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="polar")
    photo = table["photorealistic"].to_numpy()
    style = table["stylized"].to_numpy()
    ax.bar(angles - width * 0.58, photo, width=width, color=BLUE, alpha=0.92, label="Photorealistic")
    ax.bar(angles + width * 0.58, style, width=width, color=VERMILLION, alpha=0.88, label="Stylized")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 205)
    ax.set_yticks([50, 100, 150], labels=[])
    ax.set_xticks([])
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.spines["polar"].set_visible(False)
    for index, (angle, label) in enumerate(zip(angles, labels)):
        display_angle = (90 - np.degrees(angle)) % 360
        rotation = display_angle
        if 90 < display_angle < 270:
            rotation += 180
            align = "right"
        else:
            align = "left"
        color = TEAL if index >= 10 else MUTED
        ax.text(angle, 196, label, rotation=rotation, rotation_mode="anchor", ha=align, va="center", fontsize=7.4, color=color, weight="semibold" if index >= 10 else "normal")
    rigid_index = order.index("Rigid")
    ax.text(angles[rigid_index] - width * 0.58, photo[rigid_index] + 10, "108", ha="center", va="bottom", fontsize=7.5, weight="bold", color=BLUE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)
    finish_figure(fig, out, "visual_02_image_taxonomy_radial", "Image taxonomy is almost flat", "Image count · Rigid / photo = 108 · n = 5,000.", book, rect=(0.02, 0.04, 0.98, 0.90))


def normalize_tag(tag: object) -> str:
    return re.sub(r"\s+", " ", str(tag).strip().lower())


def _overlap(rect: tuple[float, float, float, float], others: list[tuple[float, float, float, float]], padding: float = 5) -> bool:
    x0, y0, x1, y1 = rect
    for ox0, oy0, ox1, oy1 in others:
        if not (x1 + padding < ox0 or x0 - padding > ox1 or y1 + padding < oy0 or y0 - padding > oy1):
            return True
    return False


def place_words(
    frequencies: Counter[str],
    *,
    seed: int,
    global_max: int,
    canvas: tuple[int, int] = (1050, 650),
    max_words: int = 65,
) -> list[tuple[str, float, float, float, int]]:
    width, height = canvas
    font_path = findfont(FontProperties(family="DejaVu Sans", weight="bold"))
    items = frequencies.most_common(max_words)
    if not items:
        return []
    min_frequency = min(value for _, value in items)
    rng = np.random.default_rng(seed)
    placed: list[tuple[str, float, float, float, int]] = []
    occupied: list[tuple[float, float, float, float]] = []
    for rank, (word, frequency) in enumerate(items):
        scaled = (math.sqrt(frequency) - math.sqrt(min_frequency)) / max(math.sqrt(global_max) - math.sqrt(min_frequency), 1e-9)
        size = int(14 + 70 * scaled)
        accepted = False
        for shrink in range(7):
            trial_size = max(11, int(size * (0.90**shrink)))
            font = ImageFont.truetype(font_path, trial_size)
            bbox = font.getbbox(word)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            phase = rng.uniform(0, 2 * np.pi)
            for step in range(3600):
                angle = phase + step * 0.47
                radial_fraction = math.sqrt((step + 1) / 3600)
                cx = width / 2 + radial_fraction * width * 0.46 * math.cos(angle)
                cy = height / 2 + radial_fraction * height * 0.44 * math.sin(angle)
                rect = (cx - text_width / 2, cy - text_height / 2, cx + text_width / 2, cy + text_height / 2)
                if rect[0] < 8 or rect[1] < 8 or rect[2] > width - 8 or rect[3] > height - 8:
                    continue
                if not _overlap(rect, occupied, padding=2):
                    occupied.append(rect)
                    placed.append((word, cx / width, 1 - cy / height, trial_size * 0.24, rank))
                    accepted = True
                    break
            if accepted:
                break
    return placed


def draw_word_cloud(ax: plt.Axes, placed: list[tuple[str, float, float, float, int]], palette: list[str], title: str) -> None:
    for word, x, y, fontsize, rank in placed:
        color = palette[rank % len(palette)]
        ax.text(x, y, word, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize, color=color, weight="semibold" if rank < 12 else "normal", alpha=0.96)
    ax.set_title(title, loc="left", fontsize=11.2, pad=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def plot_image_word_clouds(annotations: pd.DataFrame, out: Path, book: PdfPages) -> None:
    counters: dict[str, Counter[str]] = {}
    for style in ["photorealistic", "stylized"]:
        counter: Counter[str] = Counter()
        for values in annotations.loc[annotations["style"] == style, "context_objects"]:
            counter.update(tag for tag in (normalize_tag(value) for value in values) if tag)
        counters[style] = counter
    global_max = max(max(counter.values()) for counter in counters.values())
    photo = place_words(counters["photorealistic"], seed=7, global_max=global_max)
    stylized = place_words(counters["stylized"], seed=13, global_max=global_max)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.7))
    draw_word_cloud(axes[0], photo, [BLUE, TEAL, "#2E6687", "#5B8FA8"], "Photorealistic")
    draw_word_cloud(axes[1], stylized, [VERMILLION, PURPLE, GOLD, "#9F6449"], "Stylized")
    finish_figure(fig, out, "visual_03_image_object_clouds", "What fills the images", "Context-object frequency · shared type scale · n = 5,000.", book)


def blend(color: str, fraction: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple((1 - fraction) * np.ones(3) + fraction * rgb)


def plot_image_environment_rings(image: pd.DataFrame, out: Path, book: PdfPages) -> None:
    fields = [
        ("brightness", "Brightness"),
        ("time_of_day", "Time"),
        ("weather", "Weather"),
        ("crowd_density", "Crowd"),
    ]
    rank_colors = [BLUE, TEAL, GOLD, PURPLE]
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 4.35))
    for ax, (field, title) in zip(axes, fields):
        counts = image[field].value_counts()
        top = counts.head(3)
        other = int(counts.iloc[3:].sum())
        labels = list(top.index)
        values = list(map(int, top.values))
        if other:
            labels.append("Other")
            values.append(other)
        colors = []
        for index, label in enumerate(labels):
            if label == "Unknown":
                colors.append(UNKNOWN)
            elif label == "Other":
                colors.append(PALE)
            else:
                colors.append(rank_colors[index % len(rank_colors)])
        wedges, _ = ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.28, "edgecolor": PAPER, "linewidth": 1.2},
        )
        total = sum(values)
        for index, (wedge, value) in enumerate(zip(wedges, values)):
            percentage = 100 * value / total
            if index == 0 or percentage < 10:
                continue
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            ax.text(0.84 * np.cos(angle), 0.84 * np.sin(angle), f"{percentage:.0f}%", ha="center", va="center", fontsize=6.8, color=INK)
        ax.text(0, 0.08, f"{100 * values[0] / sum(values):.0f}%", ha="center", va="center", fontsize=15, weight="bold", color=INK)
        ax.text(0, -0.18, str(labels[0]), ha="center", va="center", fontsize=7.1, color=MUTED)
        ax.set_title(title, fontsize=10.5, pad=8)
        handles = [Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none") for color in colors]
        ax.legend(handles, [str(label) for label in labels], loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False, fontsize=6.2, handlelength=0.9, labelspacing=0.30)
    finish_figure(fig, out, "visual_04_image_environment_rings", "Bright, clear and mostly empty", "Top 3 + Other · Unknown retained · n = 5,000.", book, rect=(0, 0.19, 1, 0.90))


def plot_image_geometry_bubbles(image: pd.DataFrame, out: Path, book: PdfPages) -> None:
    work = image.copy()
    work["group"] = work["mode"].map(pretty) + " / " + work["style"].map(pretty)
    grouped = work.groupby(["width", "height", "group"], as_index=False).size().rename(columns={"size": "count"})
    max_count = grouped["count"].max()
    offsets = {"Static / Photorealistic": -0.018, "Static / Stylized": -0.006, "Dynamic / Photorealistic": 0.006, "Dynamic / Stylized": 0.018}
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for group in GROUP_COLORS:
        if group == "AAA Games":
            continue
        part = grouped[grouped["group"] == group]
        jitter = offsets[group]
        sizes = 18 + 750 * np.sqrt(part["count"] / max_count)
        ax.scatter(part["width"] * (1 + jitter), part["height"] * (1 - jitter), s=sizes, color=GROUP_COLORS[group], alpha=0.67, edgecolor=PAPER, linewidth=0.7, label=group.replace(" / ", " · "))
    dimensions = work.groupby(["width", "height"]).size()
    callouts = [
        ((1024, 1024), (8, 7)),
        ((1880, 1253), (8, 7)),
        ((3840, 2560), (8, -13)),
        ((6000, 4000), (8, 7)),
    ]
    for (width, height), offset in callouts:
        if (width, height) not in dimensions.index:
            continue
        count = int(dimensions.loc[(width, height)])
        ax.annotate(
            f"{width}×{height} · {count:,}",
            (width, height),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.0,
            color=INK,
            weight="semibold" if count > 500 else "normal",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": PAPER, "edgecolor": "none", "alpha": 0.78},
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = [256, 512, 1024, 2048, 4096, 8192]
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.set_yticks(ticks, [str(value) for value in ticks])
    ax.set_xlabel("Width (pixels, log₂)")
    ax.set_ylabel("Height (pixels, log₂)")
    ax.grid(color=GRID, linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor=PAPER,
            markeredgewidth=0.7,
            markersize=7.2,
            alpha=0.76,
            label=group.replace(" / ", " · "),
        )
        for group in GROUP_COLORS
        if group != "AAA Games"
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=7.2, handletextpad=0.45)
    finish_figure(fig, out, "visual_05_image_geometry_bubbles", "Stylized images cluster at 1024²", "Area = repeated raster-size count · n = 5,000.", book)


def binary_treemap(items: list[tuple[str, float]], x: float, y: float, width: float, height: float) -> list[tuple[str, float, float, float, float, float]]:
    if not items:
        return []
    if len(items) == 1:
        label, value = items[0]
        return [(label, value, x, y, width, height)]
    total = sum(value for _, value in items)
    cumulative = 0.0
    split = 1
    for index, (_, value) in enumerate(items[:-1], start=1):
        cumulative += value
        if cumulative >= total / 2:
            before = abs(cumulative - total / 2)
            previous = abs((cumulative - value) - total / 2)
            split = index if before <= previous or index == 1 else index - 1
            break
    left_items, right_items = items[:split], items[split:]
    left_total = sum(value for _, value in left_items)
    ratio = left_total / total
    if width >= height:
        first = binary_treemap(left_items, x, y, width * ratio, height)
        second = binary_treemap(right_items, x + width * ratio, y, width * (1 - ratio), height)
    else:
        first = binary_treemap(left_items, x, y, width, height * ratio)
        second = binary_treemap(right_items, x, y + height * ratio, width, height * (1 - ratio))
    return first + second


def draw_treemap(ax: plt.Axes, values: pd.Series, title: str, unit: Callable[[float], str]) -> None:
    items = sorted([(str(label), float(value)) for label, value in values.items()], key=lambda item: item[1], reverse=True)
    rectangles = binary_treemap(items, 0, 0, 1, 1)
    total = sum(value for _, value in items)
    short = {
        "AAA Games": "AAA",
        "Static / Photorealistic": "Static\nphoto",
        "Static / Stylized": "Static\nstylized",
        "Dynamic / Photorealistic": "Dynamic\nphoto",
        "Dynamic / Stylized": "Dynamic\nstylized",
    }
    for label, value, x, y, width, height in rectangles:
        color = GROUP_COLORS[label]
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor=PAPER, linewidth=2.2))
        area = width * height
        if area > 0.055:
            text = f"{short[label]}\n{unit(value)}\n{pct(value, total)}"
            fontsize = 9.0 if area > 0.12 else 7.2
        elif area > 0.012:
            text = pct(value, total)
            fontsize = 5.4
        else:
            continue
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color=PAPER, weight="semibold", linespacing=1.08, clip_on=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=11.2)
    ax.axis("off")


def plot_video_treemaps(video: pd.DataFrame, out: Path, book: PdfPages) -> None:
    work = video.copy()
    work["group"] = work.apply(video_group, axis=1)
    count = work.groupby("group").size()
    storage = work.groupby("group")["size_bytes"].sum() / 2**30
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.6))
    draw_treemap(axes[0], count, "Clip count", lambda value: f"{int(value):,}")
    draw_treemap(axes[1], storage, "Storage", lambda value: f"{value:.1f} GiB")
    finish_figure(fig, out, "visual_06_video_count_storage_treemaps", "Storage tells a different story", "Area = clips / GiB · n = 5,426 · purple storage sliver = 0.4%.", book)


def plot_video_size_ridgelines(video: pd.DataFrame, out: Path, book: PdfPages) -> None:
    work = video.copy()
    work["group"] = work.apply(video_group, axis=1)
    order = ["AAA Games", "Static / Photorealistic", "Static / Stylized", "Dynamic / Photorealistic", "Dynamic / Stylized"]
    grid = np.linspace(-1.45, 3.55, 500)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for row_index, group in enumerate(order[::-1]):
        values = np.log10(work.loc[work["group"] == group, "size_mib"].clip(lower=0.02).to_numpy())
        density = gaussian_kde(values, bw_method=0.23)(grid)
        density = density / density.max() * 0.78
        base = row_index
        color = GROUP_COLORS[group]
        ax.fill_between(grid, base, base + density, color=color, alpha=0.72, linewidth=0)
        ax.plot(grid, base + density, color=color, linewidth=1.6)
        median = float(np.median(values))
        p90 = float(np.quantile(values, 0.9))
        ax.plot([median, p90], [base + 0.05, base + 0.05], color=INK, linewidth=1.0, alpha=0.65)
        ax.scatter([median, p90], [base + 0.05, base + 0.05], s=[22, 12], color=[INK, PAPER], edgecolor=INK, linewidth=0.7, zorder=4)
        ax.text(grid.min() - 0.08, base + 0.18, group.replace(" / ", " · "), ha="right", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(grid.min() - 0.9, grid.max())
    ax.set_ylim(-0.15, len(order) - 0.15)
    ticks = [-1, 0, 1, 2, 3]
    ax.set_xticks(ticks, ["0.1", "1", "10", "100", "1,000"])
    ax.set_xlabel("File size (MiB, log scale)   • median   ○ P90")
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    finish_figure(fig, out, "visual_07_video_size_ridgelines", "Dynamic photorealistic has the longest tail", "log₁₀(file size) · dot = median · ○ = P90 · n = 5,426.", book)


def color_mix(base: str, intensity: float) -> tuple[float, float, float, float]:
    rgb = np.asarray(to_rgb(base))
    mixed = np.ones(3) * (1 - intensity) + rgb * intensity
    return (*mixed, 1.0)


def plot_video_taxonomy_heatmap(video: pd.DataFrame, out: Path, book: PdfPages) -> None:
    static = video[video["branch"] == "Static"]
    dynamic = video[video["branch"] == "Dynamic"]
    static_order = [
        "Indoor · Dining Spaces",
        "Indoor · Living Spaces",
        "Indoor · Passageways",
        "Indoor · Workspaces",
        "Outdoor · Aquatic Landscape",
        "Outdoor · City",
        "Outdoor · Public Spaces",
        "Outdoor · Suburb",
        "Outdoor · Terrestrial Landscape",
        "Outdoor · Verdant Landscape",
    ]
    dynamic_order = ["Articulated", "Rigid", "Fluid", "Deformable", "Multi Motion"]
    static_table = static.pivot_table(index="category_label", columns="style", values="relative_path", aggfunc="count", fill_value=0).reindex(static_order)
    dynamic_table = dynamic.pivot_table(index="category_label", columns="style", values="relative_path", aggfunc="count", fill_value=0).reindex(dynamic_order)
    table = pd.concat([static_table, dynamic_table]).reindex(columns=["photorealistic", "stylized"], fill_value=0)
    values = table.to_numpy(dtype=float)
    max_value = values.max()
    rgba = np.ones((len(table), 2, 4))
    for row in range(len(table)):
        rgba[row, 0] = color_mix(BLUE, 0.12 + 0.82 * math.sqrt(values[row, 0] / max_value))
        rgba[row, 1] = color_mix(VERMILLION, 0.12 + 0.82 * math.sqrt(values[row, 1] / max_value))
    labels = [label.replace("Indoor · ", "In · ").replace("Outdoor · ", "Out · ").replace(" Landscape", "") for label in static_order] + dynamic_order
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.imshow(rgba, aspect="auto")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            intensity = math.sqrt(values[row, col] / max_value)
            ax.text(col, row, f"{int(values[row, col])}", ha="center", va="center", fontsize=8.0, color=PAPER if intensity > 0.64 else INK, weight="semibold")
    ax.set_xticks([0, 1], ["Photorealistic", "Stylized"])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(length=0)
    ax.axhline(9.5, color=PAPER, linewidth=4)
    ax.text(-1.10, 4.5, "STATIC", rotation=90, ha="center", va="center", fontsize=6.8, color=MUTED, weight="bold")
    ax.text(-1.10, 12, "DYNAMIC", rotation=90, ha="center", va="center", fontsize=6.8, color=TEAL, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    finish_figure(fig, out, "visual_08_video_taxonomy_heatmap", "Stylized motion clips are sparse", "MP4 count · AAA Games omitted.", book, rect=(0.16, 0.05, 0.96, 0.93))


def plot_video_duration_orbit(video: pd.DataFrame, out: Path, book: PdfPages) -> None:
    static = video[video["branch"] == "Static"]
    categories = [
        ("0–5 s", int((static["duration_bucket"] == "0-5s").sum()), BLUE),
        ("5–10 s", int((static["duration_bucket"] == "5-10s").sum()), TEAL),
        ("10–20 s", int((static["duration_bucket"] == "10-20s").sum()), GOLD),
        ("Dynamic · unlabeled", int((video["branch"] == "Dynamic").sum()), PURPLE),
        ("AAA · unlabeled", int((video["branch"] == "AAA Games").sum()), UNKNOWN),
    ]
    units = largest_remainder([value for _, value, _ in categories])
    colors: list[str] = []
    for (_, _, color), count in zip(categories, units):
        colors.extend([color] * count)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for index, color in enumerate(colors):
        ring, slot = divmod(index, 20)
        radius = 0.45 + ring * 0.18
        angle = 2 * np.pi * slot / 20 + ring * 0.12
        ax.scatter(radius * np.cos(angle), radius * np.sin(angle), s=55, color=color, edgecolor=PAPER, linewidth=0.55)
    labeled = int((video["branch"] == "Static").sum())
    ax.text(0, 0.08, pct(labeled, len(video)), ha="center", va="center", fontsize=24, weight="bold", color=INK)
    ax.text(0, -0.16, "duration-labeled", ha="center", va="center", fontsize=8.0, color=MUTED)
    for index, (label, value, color) in enumerate(categories):
        y = 0.72 - index * 0.34
        ax.scatter([1.36], [y], s=55, color=color, edgecolor="none")
        ax.text(1.48, y + 0.04, label, ha="left", va="center", fontsize=8.0, color=INK, weight="semibold")
        ax.text(1.48, y - 0.10, f"{value:,} · {pct(value, len(video))}", ha="left", va="center", fontsize=7.0, color=MUTED)
    ax.set_xlim(-1.15, 2.65)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    finish_figure(fig, out, "visual_09_video_duration_orbit", "Duration labels cover static clips only", "1 dot ≈ 1% · directory duration labels · n = 5,426.", book)


def plot_video_specs_bubbles(video: pd.DataFrame, out: Path, book: PdfPages) -> None:
    known = video[video["spec_source"] == "filename"].copy()
    known["resolution"] = known.apply(lambda row: f"{int(row['width'])}×{int(row['height'])}", axis=1)
    resolution_order = list(known["resolution"].value_counts().head(8).index)
    known["resolution_group"] = known["resolution"].where(known["resolution"].isin(resolution_order), "Other")
    fps_order = sorted(known["fps"].astype(int).unique())
    table = known.pivot_table(index="resolution_group", columns=known["fps"].astype(int), values="relative_path", aggfunc="count", fill_value=0)
    row_order = resolution_order + (["Other"] if "Other" in table.index else [])
    table = table.reindex(index=row_order, columns=fps_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    xs, ys, sizes, counts = [], [], [], []
    max_count = table.to_numpy().max()
    for y, row in enumerate(table.index):
        for x, fps in enumerate(table.columns):
            count = int(table.loc[row, fps])
            if not count:
                continue
            xs.append(x)
            ys.append(y)
            counts.append(count)
            sizes.append(26 + 780 * math.sqrt(count / max_count))
    scatter = ax.scatter(xs, ys, s=sizes, c=counts, cmap=LinearSegmentedColormap.from_list("teal", ["#DDF2EB", TEAL]), vmin=0, vmax=max_count, edgecolor=PAPER, linewidth=1.0)
    for x, y, count in zip(xs, ys, counts):
        if count >= 10:
            ax.text(x, y, str(count), ha="center", va="center", fontsize=6.8, color=PAPER if count > max_count * 0.35 else INK, weight="bold")
    ax.set_xticks(range(len(fps_order)), [str(value) for value in fps_order])
    ax.set_yticks(range(len(row_order)), row_order)
    ax.invert_yaxis()
    ax.set_xlabel("Frames per second")
    ax.set_ylabel("Resolution")
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.98, f"{len(known):,} known\n{pct(len(known), len(video))} of clips", transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=MUTED)
    handles = [plt.scatter([], [], s=26 + 780 * math.sqrt(value / max_count), color=TEAL, alpha=0.75, edgecolor=PAPER) for value in [10, 50, 100]]
    ax.legend(handles, ["10", "50", "100 clips"], loc="lower right", frameon=False, fontsize=7.0, labelspacing=0.8)
    finish_figure(fig, out, "visual_10_video_specs_bubbles", "Technical specs appear in 665 filenames", "Area = clips · filename-encoded specs · known = 665 / 5,426.", book)


def write_readme(output_dir: Path, stems: list[tuple[str, str]]) -> None:
    lines = [
        "# WorldArena low-text visual atlas",
        "",
        "Ten visually distinct, paper-ready figures. Each is exported as PNG (400 dpi), PDF, and SVG; `worldarena_visual_atlas.pdf` contains all pages.",
        "",
    ]
    lines.extend(f"- `{stem}` — {title}" for stem, title in stems)
    lines.extend(
        [
            "",
            "Counting scope: 5,000 annotated images and 5,426 local MP4 files. TP is excluded. Video duration uses directory buckets; resolution/FPS uses explicit filename tokens only.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    ensure_records()
    image = pd.read_csv(IMAGE_RECORDS)
    video = pd.read_csv(VIDEO_RECORDS)
    annotations = load_object_annotations()

    stems = [
        ("visual_01_composition_waffles", "Balanced images, uneven videos"),
        ("visual_02_image_taxonomy_radial", "Image taxonomy is almost flat"),
        ("visual_03_image_object_clouds", "What fills the images"),
        ("visual_04_image_environment_rings", "Bright, clear and mostly empty"),
        ("visual_05_image_geometry_bubbles", "Stylized images cluster at 1024²"),
        ("visual_06_video_count_storage_treemaps", "Storage tells a different story"),
        ("visual_07_video_size_ridgelines", "Dynamic photorealistic has the longest tail"),
        ("visual_08_video_taxonomy_heatmap", "Stylized motion clips are sparse"),
        ("visual_09_video_duration_orbit", "Duration labels cover static clips only"),
        ("visual_10_video_specs_bubbles", "Technical specs appear in 665 filenames"),
    ]
    book_path = output_dir / "worldarena_visual_atlas.pdf"
    with PdfPages(book_path) as book:
        plot_composition_waffles(image, video, output_dir, book)
        plot_image_taxonomy_radial(image, output_dir, book)
        plot_image_word_clouds(annotations, output_dir, book)
        plot_image_environment_rings(image, output_dir, book)
        plot_image_geometry_bubbles(image, output_dir, book)
        plot_video_treemaps(video, output_dir, book)
        plot_video_size_ridgelines(video, output_dir, book)
        plot_video_taxonomy_heatmap(video, output_dir, book)
        plot_video_duration_orbit(video, output_dir, book)
        plot_video_specs_bubbles(video, output_dir, book)

    manifest = {
        "image_records": len(image),
        "video_records": len(video),
        "figure_count": len(stems),
        "figures": [stem for stem, _ in stems],
        "formats": ["png (400 dpi)", "pdf", "svg"],
        "combined_pdf": book_path.name,
    }
    (output_dir / "visual_atlas_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_readme(output_dir, stems)
    print(f"visual atlas: {len(stems)} figures -> {output_dir}")


if __name__ == "__main__":
    main()
