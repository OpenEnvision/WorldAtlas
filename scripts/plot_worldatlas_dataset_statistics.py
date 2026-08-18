#!/usr/bin/env python3
"""Publication-quality statistics for the local WorldAtlas image/video data.

The script deliberately separates *count* from *storage* and keeps unknown or
filename-inferred metadata visible.  It produces vector PDF/SVG plus 400-dpi
PNG figures, and CSV/JSON tables that can be cited or regenerated in a paper.

Usage:
    python scripts/plot_worldatlas_dataset_statistics.py --section image
    python scripts/plot_worldatlas_dataset_statistics.py --section video
    python scripts/plot_worldatlas_dataset_statistics.py --section all
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data_external" / "OpenEnvision" / "worldarena"
IMAGE_ROOT = DATA_ROOT / "image"
VIDEO_ROOT = DATA_ROOT / "videos_test_split_4971"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "worldatlas_statistics"


# Okabe–Ito inspired roles: blue and vermillion are the two focal categories;
# teal/gold are reserved for secondary structure and annotations.
INK = "#1F2933"
MUTED = "#667085"
GRID = "#D9E1E8"
PAPER = "#FFFFFF"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
TEAL = "#009E73"
GOLD = "#E69F00"
PURPLE = "#CC79A7"
GRAY = "#98A2B3"
LIGHT_GRAY = "#E7ECF0"
UNKNOWN = "#B7C0C8"

STYLE_COLORS = {"photorealistic": BLUE, "stylized": VERMILLION}
MODE_COLORS = {"static": BLUE, "dynamic": TEAL}
VIDEO_GROUP_COLORS = {
    "AAA Games": GOLD,
    "Static / photorealistic": BLUE,
    "Static / stylized": VERMILLION,
    "Dynamic / photorealistic": TEAL,
    "Dynamic / stylized": PURPLE,
}


def pretty(value: object) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def pct(value: float, denominator: float) -> str:
    return f"{100 * value / denominator:.1f}%" if denominator else "0.0%"


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.7,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=4)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)


def add_source_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.012, text, ha="left", va="bottom", fontsize=7.3, color=MUTED)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, source_note: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_source_note(fig, source_note)
    fig.tight_layout(rect=(0, 0.055, 1, 0.955))
    for ext, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 400}),
    ):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def image_taxonomy(rel: str) -> tuple[str, str, str]:
    parts = Path(rel).parts
    mode, style = parts[0], parts[1]
    if mode == "static":
        scope, category = parts[2], parts[3]
        return mode, style, f"{pretty(scope)} · {pretty(category)}"
    category = parts[3]
    return mode, style, pretty(category)


def _read_image_header(path: Path) -> tuple[int | None, int | None, str | None, str | None]:
    try:
        with Image.open(path) as image:
            return image.width, image.height, image.format, image.mode
    except Exception as exc:  # retained in the table instead of silently dropping a file
        return None, None, None, type(exc).__name__


def load_image_records() -> tuple[pd.DataFrame, dict]:
    metadata = load_json(IMAGE_ROOT / "image_annotations.json")
    items = metadata["items"]
    paths = [IMAGE_ROOT / item["relative_path"] for item in items]
    headers: list[tuple[int | None, int | None, str | None, str | None]] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        headers = list(executor.map(_read_image_header, paths))

    rows: list[dict] = []
    for item, header in zip(items, headers):
        rel = item["relative_path"]
        mode, style, taxonomy = image_taxonomy(rel)
        tags = item["caption"]["CategoryTags"]
        generation = item["generation_spec"]
        camera = generation.get("camera_path", [])
        target_objects = generation.get("objects", [])
        context_objects = generation.get("current_objects", [])
        width, height, fmt, image_mode = header
        rows.append(
            {
                "relative_path": rel,
                "mode": mode,
                "style": style,
                # Keep this machine-readable; presentation labels are added in plots.
                "split": f"{mode} / {style}",
                "taxonomy": taxonomy,
                "scope": taxonomy.split(" · ")[0] if " · " in taxonomy else "Dynamic",
                "category": taxonomy.split(" · ")[-1],
                "brightness": tags.get("brightness", "Unknown"),
                "time_of_day": tags.get("timeOfDay", "Unknown"),
                "weather": tags.get("weather", "Unknown"),
                "crowd_density": tags.get("crowdDensity", "Unknown"),
                "camera_path": " → ".join(camera),
                "camera_steps": len(camera),
                "camera_tokens": camera,
                "target_objects": len(target_objects),
                "context_objects": len(context_objects),
                "target_object_names": target_objects,
                "context_object_names": context_objects,
                "width": width,
                "height": height,
                "format": fmt or "Unreadable",
                "image_mode": image_mode or "Unreadable",
                "header_error": image_mode if width is None else "",
            }
        )
    return pd.DataFrame(rows), metadata


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_image_composition(df: pd.DataFrame, out: Path) -> None:
    counts = pd.crosstab(df["mode"], df["style"]).reindex(
        index=["static", "dynamic"], columns=["photorealistic", "stylized"], fill_value=0
    )
    total = counts.to_numpy().sum()
    fig, ax = plt.subplots(figsize=(6.8, 4.25))
    cmap = LinearSegmentedColormap.from_list("composition", ["#EAF2F6", "#0072B2"])
    image = ax.imshow(counts.to_numpy(), cmap=cmap, vmin=0, vmax=max(1, int(counts.to_numpy().max())))
    for row_index, mode in enumerate(counts.index):
        for col_index, style in enumerate(counts.columns):
            value = int(counts.loc[mode, style])
            color = PAPER if value > counts.to_numpy().max() * 0.58 else INK
            ax.text(col_index, row_index - 0.03, f"{value:,}", ha="center", va="center", fontsize=20, weight="bold", color=color)
            ax.text(col_index, row_index + 0.28, pct(value, total), ha="center", va="center", fontsize=8.5, color=color)
    ax.set_xticks(range(len(counts.columns)), ["Photorealistic", "Stylized"])
    ax.set_yticks(range(len(counts.index)), ["Static", "Dynamic"])
    ax.set_xlabel("Rendering style")
    ax.set_ylabel("Scene mode")
    ax.set_title("The 5,000-image annotation set is balanced across mode and style", loc="left", pad=12)
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04, label="Annotated images")
    save_figure(fig, out, "image_composition_overview", "Source: image/image_annotations.json; n = 5,000 annotated images.")


def _grouped_horizontal(ax: plt.Axes, table: pd.DataFrame, labels: list[str], value_columns: list[str], colors: list[str]) -> None:
    y = np.arange(len(labels))
    offsets = np.linspace(-0.18, 0.18, len(value_columns)) if len(value_columns) > 1 else [0]
    height = 0.32 if len(value_columns) > 1 else 0.52
    for offset, column, color in zip(offsets, value_columns, colors):
        values = table[column].to_numpy()
        bars = ax.barh(y + offset, values, height=height, color=color, label=pretty(column), alpha=0.93)
        for bar, value in zip(bars, values):
            if value:
                ax.text(value + max(table[value_columns].to_numpy().max() * 0.012, 1), bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, table[value_columns].to_numpy().max()) * 1.19)
    style_axis(ax)


def plot_image_taxonomy(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    static = df[df["mode"] == "static"].copy()
    static["label"] = static["scope"] + " · " + static["category"]
    static_table = static.pivot_table(index="label", columns="style", values="relative_path", aggfunc="count", fill_value=0)
    static_table["total"] = static_table.sum(axis=1)
    static_table = static_table.sort_values("total", ascending=True).drop(columns="total")
    _grouped_horizontal(axes[0], static_table, list(static_table.index), ["photorealistic", "stylized"], [BLUE, VERMILLION])
    axes[0].set_title("Static scenes", loc="left")
    axes[0].set_xlabel("Images")

    dynamic = df[df["mode"] == "dynamic"].copy()
    dynamic_table = dynamic.pivot_table(index="category", columns="style", values="relative_path", aggfunc="count", fill_value=0)
    dynamic_table["total"] = dynamic_table.sum(axis=1)
    dynamic_table = dynamic_table.sort_values("total", ascending=True).drop(columns="total")
    _grouped_horizontal(axes[1], dynamic_table, list(dynamic_table.index), ["photorealistic", "stylized"], [BLUE, VERMILLION])
    axes[1].set_title("Dynamic motion types", loc="left")
    axes[1].set_xlabel("Images")
    fig.legend(
        handles=[Patch(facecolor=BLUE, label="Photorealistic"), Patch(facecolor=VERMILLION, label="Stylized")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Coverage is deliberately regular at the taxonomy level", x=0.06, ha="left", y=0.985, fontsize=13, weight="semibold")
    save_figure(fig, out, "image_taxonomy_coverage", "Source: image/image_annotations.json; bars count annotated images per taxonomy cell.")


def _ordered_categories(series: pd.Series, preferred: list[str]) -> list[str]:
    values = set(series.dropna().unique())
    return [x for x in preferred if x in values] + sorted(values - set(preferred), key=lambda x: (-int((series == x).sum()), str(x)))


def plot_image_environment(df: pd.DataFrame, out: Path) -> None:
    specs = [
        ("brightness", "Brightness", ["Bright", "Dim/Dark", "Unknown"]),
        ("time_of_day", "Time of day", ["Daytime", "Dusk/Evening", "Night", "Unknown"]),
        ("weather", "Weather", ["Clear", "Cloudy", "Rainy", "Stormy", "Snowy", "Foggy", "Indoor/Unspecified", "Unknown"]),
        ("crowd_density", "Crowd density", ["Empty", "Sparse", "Moderate", "Dense", "Unknown"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.9))
    for ax, (column, title, preferred) in zip(axes.flat, specs):
        counts = df[column].value_counts()
        order = _ordered_categories(df[column], preferred)
        values = counts.reindex(order, fill_value=0).to_numpy()
        colors = [UNKNOWN if label == "Unknown" else BLUE for label in order]
        bars = ax.bar(np.arange(len(order)), values, color=colors, width=0.68)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, f"{value:,}\n{pct(value, len(df))}", ha="center", va="bottom", fontsize=7.5, color=INK, linespacing=1.1)
        ax.set_xticks(np.arange(len(order)), [pretty(x) for x in order], rotation=28, ha="right")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Images")
        ax.set_ylim(0, max(values) * 1.19)
        style_axis(ax, grid_axis="y")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Environmental labels expose both coverage and annotation uncertainty", x=0.06, ha="left", y=0.99, fontsize=13, weight="semibold")
    save_figure(fig, out, "image_environment_attributes", "Source: image/image_annotations.json; percentages use all n = 5,000 annotated images; Unknown is retained.")


def plot_image_camera(df: pd.DataFrame, out: Path) -> None:
    token_labels = {
        "fixed": "Fixed",
        "pan_left": "Pan left",
        "pan_right": "Pan right",
        "orbit_left": "Orbit left",
        "orbit_right": "Orbit right",
        "push_in": "Push in",
        "pull_out": "Pull out",
        "move_left": "Truck left",
        "move_right": "Truck right",
    }
    token_counts = Counter(token for tokens in df["camera_tokens"] for token in tokens)
    ordered_tokens = [x for x, _ in token_counts.most_common()]
    labels = [token_labels.get(x, pretty(x)) for x in ordered_tokens]
    values = [token_counts[x] for x in ordered_tokens]
    recipe = df["camera_steps"].map({1: "Single-step", 3: "Three-step"}).fillna("Other")
    recipe_by_mode = pd.crosstab(df["mode"], recipe).reindex(index=["static", "dynamic"], fill_value=0)
    recipe_by_mode = recipe_by_mode.reindex(columns=["Single-step", "Three-step", "Other"], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(labels))
    bars = axes[0].barh(y, values, color=TEAL, height=0.56)
    for bar, value in zip(bars, values):
        axes[0].text(value + max(values) * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=7.5)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Token occurrences")
    axes[0].set_title("Camera-operation vocabulary", loc="left")
    axes[0].set_xlim(0, max(values) * 1.18)
    style_axis(axes[0])

    x = np.arange(len(recipe_by_mode.index))
    bottoms = np.zeros(len(recipe_by_mode.index))
    recipe_colors = [TEAL, GOLD, UNKNOWN]
    for column, color in zip(recipe_by_mode.columns, recipe_colors):
        vals = recipe_by_mode[column].to_numpy()
        bars = axes[1].bar(x, vals, bottom=bottoms, color=color, width=0.58, label=column)
        for i, (bar, value) in enumerate(zip(bars, vals)):
            if value:
                axes[1].text(bar.get_x() + bar.get_width() / 2, bottoms[i] + value / 2, f"{value:,}\n{pct(value, recipe_by_mode.loc[recipe_by_mode.index[i]].sum())}", ha="center", va="center", fontsize=7.5, color=INK)
        bottoms += vals
    axes[1].set_xticks(x, ["Static", "Dynamic"])
    axes[1].set_ylabel("Images")
    axes[1].set_title("Most samples use a single camera step", loc="left")
    axes[1].legend(frameon=False, loc="upper right")
    style_axis(axes[1], grid_axis="y")
    axes[1].grid(axis="x", visible=False)
    fig.suptitle("Camera motion is represented as an explicit, composable recipe", x=0.06, ha="left", y=0.99, fontsize=13, weight="semibold")
    save_figure(fig, out, "image_camera_motion_recipes", "Source: image/image_annotations.json; camera recipes are parsed from generation_spec.camera_path.")


def plot_image_complexity(df: pd.DataFrame, out: Path) -> None:
    split_order = ["static / photorealistic", "static / stylized", "dynamic / photorealistic", "dynamic / stylized"]
    groups = [df.loc[df["split"] == split, "context_objects"].to_numpy() for split in split_order]
    all_objects = Counter(obj.lower() for values in df["context_object_names"] for obj in values)
    top_objects = all_objects.most_common(15)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.1), gridspec_kw={"width_ratios": [1, 1.2]})
    bp = axes[0].boxplot(groups, patch_artist=True, showfliers=False, widths=0.58, medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": MUTED, "linewidth": 0.8}, capprops={"color": MUTED, "linewidth": 0.8})
    for patch, split in zip(bp["boxes"], split_order):
        patch.set_facecolor(STYLE_COLORS["photorealistic"] if "photorealistic" in split else STYLE_COLORS["stylized"])
        patch.set_alpha(0.7)
        patch.set_edgecolor(INK)
    axes[0].set_xticks(range(1, len(split_order) + 1), ["Static\nphoto", "Static\nstylized", "Dynamic\nphoto", "Dynamic\nstylized"])
    axes[0].set_ylabel("Context objects per image")
    axes[0].set_title("Scene complexity", loc="left")
    axes[0].set_ylim(0, max(max(g) for g in groups) + 1)
    style_axis(axes[0], grid_axis="y")
    axes[0].grid(axis="x", visible=False)

    labels = [pretty(x[0]) for x in top_objects]
    values = [x[1] for x in top_objects]
    bars = axes[1].barh(np.arange(len(labels)), values, color=BLUE, alpha=0.9, height=0.62)
    for bar, value in zip(bars, values):
        axes[1].text(value + max(values) * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=7.5)
    axes[1].set_yticks(np.arange(len(labels)), labels)
    axes[1].set_xlabel("Images containing object token")
    axes[1].set_title("Most frequent scene elements", loc="left")
    axes[1].set_xlim(0, max(values) * 1.18)
    axes[1].invert_yaxis()
    style_axis(axes[1])
    fig.suptitle("The annotation schema captures both scene complexity and recurring elements", x=0.06, ha="left", y=0.99, fontsize=13, weight="semibold")
    save_figure(fig, out, "image_scene_complexity", "Source: image/image_annotations.json; object counts use generation_spec.current_objects; spelling variants are not merged beyond lowercase normalization.")


def aspect_class(width: float, height: float) -> str:
    ratio = width / height
    candidates = [("1:1", 1.0), ("4:3", 4 / 3), ("3:2", 3 / 2), ("16:9", 16 / 9)]
    label, target = min(candidates, key=lambda item: abs(ratio - item[1]))
    return label if abs(ratio - target) <= 0.055 else "Other"


def plot_image_geometry(df: pd.DataFrame, out: Path) -> None:
    valid = df.dropna(subset=["width", "height"]).copy()
    valid["aspect_class"] = [aspect_class(w, h) for w, h in zip(valid["width"], valid["height"])]
    valid["long_edge_bin"] = pd.cut(valid[["width", "height"]].max(axis=1), [-1, 1024, 2048, 4096, float("inf")], labels=["≤1K", "1–2K", "2–4K", ">4K"])
    aspect_order = ["1:1", "3:2", "4:3", "16:9", "Other"]
    aspect_table = pd.crosstab(valid["aspect_class"], valid["split"]).reindex(index=aspect_order, fill_value=0)
    aspect_table = aspect_table.reindex(columns=["static / photorealistic", "static / stylized", "dynamic / photorealistic", "dynamic / stylized"], fill_value=0)
    format_table = pd.crosstab(valid["format"], valid["split"]).reindex(columns=aspect_table.columns, fill_value=0)
    format_table["total"] = format_table.sum(axis=1)
    format_table = format_table.sort_values("total", ascending=True).drop(columns="total")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8))
    x = np.arange(len(aspect_order))
    offsets = np.linspace(-0.27, 0.27, len(aspect_table.columns))
    for offset, column in zip(offsets, aspect_table.columns):
        vals = aspect_table[column].to_numpy()
        is_static = column.startswith("static")
        bars = axes[0].bar(
            x + offset,
            vals,
            width=0.17,
            color=STYLE_COLORS["photorealistic"] if "photorealistic" in column else STYLE_COLORS["stylized"],
            alpha=0.95 if is_static else 0.58,
            hatch=None if is_static else "//",
            edgecolor=PAPER,
            linewidth=0.4,
        )
        for bar, value in zip(bars, vals):
            if value >= 50:
                axes[0].text(bar.get_x() + bar.get_width() / 2, value + 16, f"{value:,}", ha="center", va="bottom", fontsize=6.8, rotation=90)
    axes[0].set_xticks(x, aspect_order)
    axes[0].set_ylabel("Images")
    axes[0].set_title("Aspect-ratio families", loc="left")
    axes[0].set_ylim(0, aspect_table.to_numpy().max() * 1.24)
    style_axis(axes[0], grid_axis="y")
    axes[0].grid(axis="x", visible=False)

    y = np.arange(len(format_table.index))
    bars = axes[1].barh(y, format_table.sum(axis=1), color=[BLUE, TEAL, VERMILLION, GOLD][: len(y)])
    for bar, value in zip(bars, format_table.sum(axis=1)):
        axes[1].text(value + max(format_table.sum(axis=1)) * 0.015, bar.get_y() + bar.get_height() / 2, f"{int(value):,}  ({pct(value, len(valid))})", va="center", fontsize=7.5)
    axes[1].set_yticks(y, list(format_table.index))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Images")
    axes[1].set_title("File encoding", loc="left")
    axes[1].set_xlim(0, format_table.sum(axis=1).max() * 1.28)
    style_axis(axes[1])
    fig.legend(
        handles=[
            Patch(facecolor=BLUE, label="Photorealistic"),
            Patch(facecolor=VERMILLION, label="Stylized"),
            Patch(facecolor=LIGHT_GRAY, edgecolor=MUTED, label="Static"),
            Patch(facecolor=LIGHT_GRAY, edgecolor=MUTED, hatch="//", label="Dynamic"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        fontsize=7.5,
    )
    fig.suptitle("Image geometry is heterogeneous, with a strong square component", x=0.06, ha="left", y=0.99, fontsize=13, weight="semibold")
    save_figure(fig, out, "image_geometry_and_encoding", "Source: image files referenced by image_annotations.json; dimensions are read from image headers; n = 5,000.")


def summarize_image(df: pd.DataFrame, metadata: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_table(df.drop(columns=["camera_tokens", "target_object_names", "context_object_names"]), out / "image_records.csv")
    category = df.groupby(["mode", "style", "taxonomy"], as_index=False).size().rename(columns={"size": "count"})
    write_table(category, out / "image_taxonomy_counts.csv")
    environments = []
    for column in ["brightness", "time_of_day", "weather", "crowd_density"]:
        values = df[column].value_counts().rename_axis("value").reset_index(name="count")
        values.insert(0, "field", column)
        environments.append(values)
    write_table(pd.concat(environments, ignore_index=True), out / "image_environment_counts.csv")
    camera = pd.DataFrame({"camera_token": list(Counter(token for values in df["camera_tokens"] for token in values).keys())})
    camera["count"] = camera["camera_token"].map(Counter(token for values in df["camera_tokens"] for token in values))
    write_table(camera.sort_values("count", ascending=False), out / "image_camera_token_counts.csv")
    summary = {
        "source": str(IMAGE_ROOT),
        "annotation_format": metadata.get("format"),
        "annotation_version": metadata.get("version"),
        "annotated_images": int(len(df)),
        "header_errors": int((df["width"].isna()).sum()),
        "mode_counts": {str(k): int(v) for k, v in df["mode"].value_counts().items()},
        "style_counts": {str(k): int(v) for k, v in df["style"].value_counts().items()},
        "format_counts": {str(k): int(v) for k, v in df["format"].value_counts().items()},
        "camera_recipe_lengths": {str(k): int(v) for k, v in df["camera_steps"].value_counts().items()},
        "notes": [
            "Image counts are based on the 5,000 records in image_annotations.json.",
            "Environment and scene labels are annotation fields; Unknown is a reported category, not imputed.",
        ],
    }
    (out / "image_summary.json").write_text(json.dumps(summary, indent=2))


def run_image(output_root: Path) -> None:
    df, metadata = load_image_records()
    out = output_root / "image"
    summarize_image(df, metadata, out)
    plot_image_composition(df, out)
    plot_image_taxonomy(df, out)
    plot_image_environment(df, out)
    plot_image_camera(df, out)
    plot_image_complexity(df, out)
    plot_image_geometry(df, out)
    print(f"image: {len(df):,} records -> {out}")


def video_records_from_find() -> pd.DataFrame:
    raw = subprocess.check_output(
        ["/usr/bin/find", str(VIDEO_ROOT), "-type", "f", "-name", "*.mp4", "-printf", "%p\\t%s\\n"],
        text=True,
    )
    spec_pattern = re.compile(r"(?:^|[_-])(?:sd|hd)[_-](\d{3,5})[_-](\d{3,5})[_-](\d{1,3})fps(?:\.|[_-])", re.I)
    duration_pattern = re.compile(r"^(0-5s|5-10s|10-20s)$")
    rows: list[dict] = []
    for line in raw.splitlines():
        full_path, size = line.rsplit("\t", 1)
        path = Path(full_path)
        rel = path.relative_to(VIDEO_ROOT)
        parts = rel.parts
        record = {
            "relative_path": str(rel),
            "filename": path.name,
            "size_bytes": int(size),
            "size_mib": int(size) / 2**20,
            "branch": "Unknown",
            "style": "Unknown",
            "category": "Unknown",
            "category_label": "Unknown",
            "scene_scope": "Unknown",
            "duration_bucket": "Unknown",
        }
        if parts[0] == "AAA_Games":
            record.update(branch="AAA Games", style="AAA Games", category="AAA Games", category_label="AAA Games")
        elif parts[0] == "static":
            style, scope, category = parts[1], parts[2], parts[3]
            duration = parts[4] if len(parts) > 5 and duration_pattern.match(parts[4]) else "Unknown"
            record.update(
                branch="Static",
                style=style,
                category=category,
                category_label=f"{pretty(scope)} · {pretty(category)}",
                scene_scope=scope,
                duration_bucket=duration,
            )
        elif parts[0] == "dynamic":
            style, category = parts[1], parts[3]
            record.update(
                branch="Dynamic",
                style=style,
                category=category,
                category_label=pretty(category),
                scene_scope="Dynamic",
            )
        match = spec_pattern.search(path.name)
        if match:
            record.update(width=int(match.group(1)), height=int(match.group(2)), fps=int(match.group(3)), spec_source="filename")
        else:
            record.update(width=np.nan, height=np.nan, fps=np.nan, spec_source="unknown")
        stem = path.stem.lower()
        if "__pexels_" in stem or stem.startswith("pexels_"):
            source = "Pexels"
        elif "__pixabay_" in stem or stem.startswith("pixabay_"):
            source = "Pixabay"
        elif "__" in stem:
            source = "Named/other"
        else:
            source = "ID-only/other"
        record["filename_source"] = source
        rows.append(record)
    return pd.DataFrame(rows)


def video_group(row: pd.Series) -> str:
    if row["branch"] == "AAA Games":
        return "AAA Games"
    return f"{row['branch']} / {row['style']}"


def plot_video_composition(df: pd.DataFrame, out: Path) -> None:
    work = df.copy()
    work["group"] = work.apply(video_group, axis=1)
    order = [x for x in VIDEO_GROUP_COLORS if x in set(work["group"])]
    summary = work.groupby("group").agg(count=("relative_path", "size"), storage_gib=("size_bytes", lambda x: x.sum() / 2**30)).reindex(order).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.5))
    labels = [x.replace(" / ", "\n") for x in summary.index]
    colors = [VIDEO_GROUP_COLORS[x] for x in summary.index]
    y = np.arange(len(summary))
    for ax, column, title, xlabel in [
        (axes[0], "count", "Clips by subset", "MP4 files"),
        (axes[1], "storage_gib", "Storage is concentrated differently", "GiB on disk"),
    ]:
        bars = ax.barh(y, summary[column], color=colors, height=0.62)
        for bar, value in zip(bars, summary[column]):
            ax.text(value + max(summary[column]) * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:,.0f}" if column == "count" else f"{value:.1f}", va="center", fontsize=8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, summary[column].max() * 1.22)
        style_axis(ax)
    fig.suptitle("Video count and storage tell complementary stories", x=0.06, ha="left", y=0.99, fontsize=13, weight="semibold")
    save_figure(fig, out, "video_composition_and_storage", "Source: videos_test_split_4971; only .mp4 files are included; file sizes are read from the local filesystem.")


def plot_video_taxonomy(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.5), gridspec_kw={"width_ratios": [1.05, 1.45]})
    dynamic = df[df["branch"] == "Dynamic"]
    dynamic_table = dynamic.pivot_table(index="category", columns="style", values="relative_path", aggfunc="count", fill_value=0)
    dynamic_table["total"] = dynamic_table.sum(axis=1)
    dynamic_table = dynamic_table.sort_values("total", ascending=True).drop(columns="total")
    _grouped_horizontal(axes[0], dynamic_table, [pretty(x) for x in dynamic_table.index], ["photorealistic", "stylized"], [BLUE, VERMILLION])
    axes[0].set_title("Dynamic motion types", loc="left")
    axes[0].set_xlabel("Clips")

    static = df[df["branch"] == "Static"].copy()
    static_table = static.pivot_table(index="category_label", columns="style", values="relative_path", aggfunc="count", fill_value=0)
    static_table["total"] = static_table.sum(axis=1)
    static_table = static_table.sort_values("total", ascending=True).drop(columns="total")
    _grouped_horizontal(axes[1], static_table, list(static_table.index), ["photorealistic", "stylized"], [BLUE, VERMILLION])
    axes[1].set_title("Static scene taxonomy", loc="left")
    axes[1].set_xlabel("Clips")
    fig.legend(
        handles=[Patch(facecolor=BLUE, label="Photorealistic"), Patch(facecolor=VERMILLION, label="Stylized")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("The video split spans both motion primitives and scene contexts", x=0.06, ha="left", y=0.99, fontsize=13, weight="semibold")
    save_figure(fig, out, "video_taxonomy_coverage", "Source: videos_test_split_4971; counts are .mp4 files grouped by directory taxonomy.")


def plot_video_duration_size(df: pd.DataFrame, out: Path) -> None:
    static = df[df["branch"] == "Static"].copy()
    duration_order = ["0-5s", "5-10s", "10-20s", "Unknown"]
    duration_table = pd.crosstab(static["duration_bucket"], static["style"]).reindex(index=duration_order, fill_value=0)
    duration_table = duration_table.reindex(columns=["photorealistic", "stylized"], fill_value=0)
    work = df.copy()
    work["group"] = work.apply(video_group, axis=1)
    group_order = [x for x in VIDEO_GROUP_COLORS if x in set(work["group"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})
    x = np.arange(len(duration_order))
    width = 0.36
    for offset, style, color in [(-width / 2, "photorealistic", BLUE), (width / 2, "stylized", VERMILLION)]:
        vals = duration_table[style].to_numpy()
        bars = axes[0].bar(x + offset, vals, width=width, color=color, label=pretty(style))
        for bar, value in zip(bars, vals):
            if value: axes[0].text(bar.get_x() + bar.get_width() / 2, value + max(vals) * 0.02, f"{value:,}", ha="center", fontsize=7.5)
    axes[0].set_xticks(x, ["0–5 s", "5–10 s", "10–20 s", "Unknown"])
    axes[0].set_ylabel("Clips")
    axes[0].set_title("Static clip duration buckets", loc="left")
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0, duration_table.to_numpy().max() * 1.22)
    style_axis(axes[0], grid_axis="y")
    axes[0].grid(axis="x", visible=False)

    for group in group_order:
        values = np.sort(work.loc[work["group"] == group, "size_mib"].to_numpy())
        y = np.linspace(0, 1, len(values), endpoint=True)
        axes[1].plot(values, y, color=VIDEO_GROUP_COLORS[group], linewidth=2, label=group.replace(" / ", "\n"))
    axes[1].set_xscale("log")
    axes[1].set_xlabel("File size (MiB, log scale)")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("File-size distributions are long-tailed", loc="left")
    axes[1].set_ylim(0, 1)
    axes[1].xaxis.set_major_formatter(lambda x, pos: f"{x:g}")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    axes[1].legend(frameon=True, facecolor=PAPER, edgecolor="none", framealpha=0.9, fontsize=7.2, loc="lower right")
    style_axis(axes[1], grid_axis="both")
    fig.suptitle("Video duration and storage scale should be reported separately", x=0.06, ha="left", y=0.99, fontsize=13, weight="semibold")
    save_figure(fig, out, "video_duration_and_size_distributions", "Source: videos_test_split_4971; duration uses explicit static directory buckets; size is local filesystem size.")


def plot_video_specs(df: pd.DataFrame, out: Path) -> None:
    known = df[df["spec_source"] == "filename"].copy()
    resolution_all = known.apply(lambda row: f"{int(row['width'])}×{int(row['height'])}", axis=1).value_counts()
    resolution_top = resolution_all.sort_values(ascending=False).head(12)
    resolution_other = int(resolution_all.iloc[12:].sum())
    resolution = resolution_top.copy()
    if resolution_other:
        resolution.loc["Other encoded resolutions"] = resolution_other
    resolution = resolution.sort_values(ascending=True)
    fps = known["fps"].astype(int).value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(resolution))
    bars = axes[0].barh(y, resolution.to_numpy(), color=TEAL, height=0.58)
    for bar, value in zip(bars, resolution.to_numpy()):
        axes[0].text(value + max(resolution) * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=7.5)
    axes[0].set_yticks(y, list(resolution.index))
    axes[0].set_xlabel("Clips")
    axes[0].set_title("Filename-encoded resolutions", loc="left")
    axes[0].set_xlim(0, resolution.max() * 1.25)
    style_axis(axes[0])
    x = np.arange(len(fps))
    bars = axes[1].bar(x, fps.to_numpy(), color=GOLD, width=0.6)
    for bar, value in zip(bars, fps.to_numpy()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + max(fps) * 0.025, f"{value:,}", ha="center", fontsize=7.5)
    axes[1].set_xticks(x, [f"{int(v)}" for v in fps.index])
    axes[1].set_xlabel("Frames per second")
    axes[1].set_ylabel("Clips")
    axes[1].set_title("Filename-encoded frame rates", loc="left")
    axes[1].set_ylim(0, fps.max() * 1.22)
    style_axis(axes[1], grid_axis="y")
    axes[1].grid(axis="x", visible=False)
    fig.suptitle("Technical metadata is rich where filenames expose it—and incomplete elsewhere", x=0.06, ha="left", y=0.99, fontsize=13, weight="semibold")
    note = f"Source: videos_test_split_4971; {len(known):,}/{len(df):,} clips ({pct(len(known), len(df))}) have explicit resolution/FPS tokens in filenames; resolution shows top 12 plus Other; unknowns are not imputed."
    save_figure(fig, out, "video_filename_technical_specs", note)


def summarize_video(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_table(df, out / "video_records.csv")
    summary = df.copy()
    summary["group"] = summary.apply(video_group, axis=1)
    groups = summary.groupby("group").agg(count=("relative_path", "size"), storage_gib=("size_bytes", lambda x: x.sum() / 2**30)).reset_index()
    write_table(groups, out / "video_group_counts_and_storage.csv")
    duration = summary[summary["branch"] == "Static"].groupby(["style", "duration_bucket"], as_index=False).size().rename(columns={"size": "count"})
    write_table(duration, out / "video_static_duration_counts.csv")
    specs = summary[summary["spec_source"] == "filename"]
    write_table(specs.groupby(["width", "height", "fps"], as_index=False).size().rename(columns={"size": "count"}), out / "video_filename_specs.csv")
    info = {
        "source": str(VIDEO_ROOT),
        "mp4_files": int(len(summary)),
        "storage_gib": float(summary["size_bytes"].sum() / 2**30),
        "filename_spec_files": int(len(specs)),
        "filename_spec_fraction": float(len(specs) / len(summary)) if len(summary) else 0.0,
        "notes": [
            "Only MP4 files are counted as videos; masks, JSON, contact sheets, and review assets are excluded.",
            "Static duration values come from explicit directory buckets; resolution/FPS are parsed only when present in filenames.",
        ],
    }
    (out / "video_summary.json").write_text(json.dumps(info, indent=2))


def run_video(output_root: Path) -> None:
    df = video_records_from_find()
    out = output_root / "video"
    summarize_video(df, out)
    plot_video_composition(df, out)
    plot_video_taxonomy(df, out)
    plot_video_duration_size(df, out)
    plot_video_specs(df, out)
    print(f"video: {len(df):,} records -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["image", "video", "all"], default="image")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    set_paper_style()
    if args.section in {"image", "all"}:
        run_image(args.output_root)
    if args.section in {"video", "all"}:
        run_video(args.output_root)


if __name__ == "__main__":
    main()
