#!/usr/bin/env python3
"""Generate publication-quality WorldArena dataset statistics figures."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from wordcloud import WordCloud

ROOT = Path("/mnt/cpfs/yangboxue/visual_generation/juanxi/data/OpenEnvision/worldarena")
IMAGE_JSON = ROOT / "image_test_balanced_5000" / "image_annotations.json"
INDOOR_CSV = ROOT / "videos" / "_missing_annotations_indoor_metadata.csv"
OUTDOOR_CSV = ROOT / "videos" / "_missing_annotations_outdoor_metadata.csv"
DYNAMIC_CSV = ROOT / "videos" / "_missing_annotations_dynamic_metadata.csv"
OUT_DIR = Path(
    "/mnt/cpfs/yangboxue/visual_generation/juanxi/WorldArena/docs/dataset_statistics_figures"
)

# ── Palette (paper-friendly) ───────────────────────────────────────
C_INDOOR = "#4165A6"   # Elegant deep blue
C_OUTDOOR = "#5A9B6C"  # Elegant forest green
C_PHOTO = "#4A7C9A"    # Steel blue
C_STYLIZED = "#D17A5A" # Terracotta
C_DYNAMIC = "#C85A5A"  # Muted crimson
C_STATIC = "#83709B"   # Dusty purple
C_MOTION = ["#4165A6", "#5A9B6C", "#C85A5A", "#83709B", "#D1A354"]

SCENE_LABELS = {
    "dining_spaces": "Dining",
    "living_spaces": "Living",
    "passageways": "Passage",
    "public_spaces": "Public",
    "workspaces": "Work",
    "aquatic_landscape": "Aquatic",
    "city": "City",
    "suburb": "Suburb",
    "terrestrial_landscape": "Terrestrial",
    "verdant_landscape": "Verdant",
}

MOTION_LABELS = {
    "articulated": "Articulated",
    "rigid": "Rigid",
    "fluid": "Fluid",
    "deformable": "Deformable",
    "multi_motion": "Multi-motion",
}


def setup_style() -> None:
    # Use a professional academic style, leaning on Times New Roman / Serif
    sns.set_theme(style="whitegrid", rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "axes.titlepad": 12,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
    })


def label_panel(ax, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def normalize_object(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_image_items() -> list[dict]:
    payload = json.loads(IMAGE_JSON.read_text(encoding="utf-8"))
    return payload["items"]


def load_video_csvs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indoor = pd.read_csv(INDOOR_CSV)
    outdoor = pd.read_csv(OUTDOOR_CSV) if OUTDOOR_CSV.exists() else pd.DataFrame()
    dynamic = pd.read_csv(DYNAMIC_CSV)
    indoor["split"] = "Indoor Video"
    if not outdoor.empty:
        outdoor["split"] = "Outdoor Video"
    dynamic["split"] = "Dynamic Video"
    return indoor, outdoor, dynamic


def count_indoor_videos() -> pd.DataFrame:
    rows = []
    base = ROOT / "videos" / "indoor"
    for scene_dir in sorted(base.iterdir()):
        if not scene_dir.is_dir():
            continue
        for dur_dir in sorted(scene_dir.iterdir()):
            if not dur_dir.is_dir():
                continue
            n = len(list(dur_dir.glob("*.mp4")))
            rows.append(
                {
                    "scene": scene_dir.name,
                    "duration": dur_dir.name,
                    "count": n,
                }
            )
    return pd.DataFrame(rows)


def count_outdoor_videos() -> pd.DataFrame:
    rows = []
    base = ROOT / "videos" / "outdoor"
    if not base.exists():
        return pd.DataFrame()
    for scene_dir in sorted(base.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("_"):
            continue
        n = len(list(scene_dir.rglob("*.mp4")))
        rows.append(
            {
                "scene": scene_dir.name,
                "count": n,
            }
        )
    return pd.DataFrame(rows)

def count_dynamic_videos() -> pd.DataFrame:
    rows = []
    for style in ("photorealistic", "stylized"):
        vdir = ROOT / "videos" / "dynamic" / style / "videos"
        if not vdir.exists():
            continue
        for cat_dir in sorted(vdir.iterdir()):
            if not cat_dir.is_dir():
                continue
            rows.append(
                {
                    "style": style,
                    "motion_category": cat_dir.name,
                    "count": len(list(cat_dir.glob("*.mp4"))),
                }
            )
    return pd.DataFrame(rows)


def extract_instruction_labels(items: list[dict]) -> Counter:
    counter: Counter = Counter()
    for item in items:
        for labels in (item.get("instructions") or {}).values():
            for label in labels:
                counter[label.strip()] += 1
    return counter


def plot_modality_overview(ax, items: list[dict], indoor_n: int, outdoor_n: int, dynamic_n: int) -> None:
    static_n = sum(1 for x in items if x.get("motion_regime") == "static")
    dyn_img_n = sum(1 for x in items if x.get("motion_regime") == "dynamic")
    labels = [
        f"Static Images\n({static_n:,})",
        f"Dynamic Images\n({dyn_img_n:,})",
        f"Indoor Videos\n({indoor_n:,})",
        f"Outdoor Videos\n({outdoor_n:,})" if outdoor_n > 0 else "",
        f"Dynamic Videos\n({dynamic_n:,})",
    ]
    sizes = [static_n, dyn_img_n, indoor_n, outdoor_n, dynamic_n]
    colors = [C_STATIC, C_DYNAMIC, C_INDOOR, C_OUTDOOR, "#DD8452"]

    # Filter out empty items
    labels = [l for l, s in zip(labels, sizes) if s > 0]
    colors = [c for c, s in zip(colors, sizes) if s > 0]
    sizes = [s for s in sizes if s > 0]

    wedges, texts, autotexts =     ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.5, "alpha": 0.95},
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        pctdistance=0.78,
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("#FFFFFF")
        t.set_fontweight("bold")
    ax.set_title("Dataset Composition", pad=8)
    centre = plt.Circle((0, 0), 0.28, fc="white")
    ax.add_patch(centre)
    ax.text(0, 0.04, f"{sum(sizes):,}", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0, -0.15, "Total Samples", ha="center", va="center", fontsize=9, color="#555555")


def plot_scene_bar(ax, items: list[dict]) -> None:
    static = [x for x in items if x.get("motion_regime") == "static" and x.get("scene")]
    indoor_scenes = {
        "dining_spaces",
        "living_spaces",
        "passageways",
        "public_spaces",
        "workspaces",
    }
    counts: Counter = Counter()
    env_map: dict[str, str] = {}
    for item in static:
        scene = item["scene"]
        counts[scene] += 1
        env_map[scene] = "Indoor" if scene in indoor_scenes else "Outdoor"
    ordered = counts.most_common()
    names = [SCENE_LABELS.get(k, k.replace("_", " ").title()) for k, _ in ordered]
    vals = [v for _, v in ordered]
    colors = [C_INDOOR if env_map[k] == "Indoor" else C_OUTDOOR for k, _ in ordered]
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white", linewidth=1.0, alpha=0.9)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Count")
    ax.set_title("Static Image Scene Categories")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    y_offset = max(vals) * 0.03
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + y_offset, str(val), ha="center", va="bottom", fontsize=9)
    legend = [
        mpatches.Patch(color=C_INDOOR, label="Indoor", alpha=0.9),
        mpatches.Patch(color=C_OUTDOOR, label="Outdoor", alpha=0.9),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True, edgecolor="#E5E5E5")


def plot_environment_donut(ax, items: list[dict], no_text: bool = False) -> None:
    scenes = Counter()
    weather = Counter()
    tod = Counter()
    brightness = Counter()
    motion = Counter()

    for item in items:
        if "scene" in item and item["scene"]:
            scenes[item["scene"]] += 1
        tags = (item.get("caption") or {}).get("CategoryTags") or {}
        if w := tags.get("weather"): weather[str(w)] += 1
        if t := tags.get("timeOfDay"): tod[str(t)] += 1
        if b := tags.get("brightness"): brightness[str(b)] += 1
        if m := item.get("motion_category"): motion[m] += 1

    def get_dist(counter, ignore_keys=["Indoor/Unspecified", "Unknown", "None"]):
        total = sum(v for k,v in counter.items() if k not in ignore_keys)
        if total == 0: return {}
        return {k: v/total for k,v in counter.items() if k not in ignore_keys}

    dist_scene = get_dist(scenes)
    w_dist = get_dist(weather)
    dist_weather = {
        "Clear": w_dist.get("Clear", 0),
        "Cloudy": w_dist.get("Cloudy", 0),
        "Snowy": w_dist.get("Snowy", 0),
        "Rainy": w_dist.get("Rainy", 0),
        "Fog/Storm": w_dist.get("Foggy", 0) + w_dist.get("Stormy", 0)
    }
    dist_tod = get_dist(tod)
    dist_tod = {"Daytime": dist_tod.get("Daytime", 0), "Night": dist_tod.get("Night", 0), "Dusk/Eve": dist_tod.get("Dusk/Evening", 0)}
    dist_brightness = get_dist(brightness)
    dist_motion = get_dist(motion)

    # Reorder scene values slightly for neatness
    scene_keys_ordered = sorted(dist_scene.keys(), key=lambda k: dist_scene[k], reverse=True)
    dist_scene = {k: dist_scene[k] for k in scene_keys_ordered}

    dims = [
        ("10 Scene", dist_scene, sns.color_palette("Blues_r", 10)),
        ("5 Motion", dist_motion, sns.color_palette("Greens_r", 5)),
        ("2 Brightness", dist_brightness, sns.color_palette("Reds_r", 4)),
        ("3 Time", dist_tod, sns.color_palette("Purples_r", 4)),
        ("5 Weather", dist_weather, sns.color_palette("Oranges_r", 5))
    ]

    ax.axis("equal")
    inner_radius = 0.55
    outer_radius = 0.85
    angle_per_dim = 360 / len(dims)

    # Inner-most circle text
    center_circle = plt.Circle((0, 0), inner_radius * 0.45, facecolor="white", zorder=10)
    ax.add_patch(center_circle)
    ax.text(0, 0, "5\nDimensions\n(WorldArena)", ha="center", va="center", fontsize=10, fontweight="bold", zorder=11)

    # Initialize tracking variables OUTSIDE the sector loop to prevent cross-sector overlap
    last_text_ang = None
    stagger_level = 0

    for i, (dim_name, dist, palette) in enumerate(dims):
        start_angle = 90 - (i * angle_per_dim)
        end_angle = 90 - ((i + 1) * angle_per_dim)

        # Inner sector background
        wedge_inner = mpatches.Wedge((0, 0), inner_radius, end_angle, start_angle, facecolor="white", edgecolor="#E0E0E0", lw=1.5)
        ax.add_patch(wedge_inner)

        # Inner text
        mid_angle = (start_angle + end_angle) / 2
        mid_rad = inner_radius * 0.72
        x = mid_rad * np.cos(np.radians(mid_angle))
        y = mid_rad * np.sin(np.radians(mid_angle))

        rot = mid_angle
        if 90 < rot % 360 < 270:
            rot += 180

        ax.text(x, y, dim_name, ha="center", va="center", rotation=rot, fontsize=9, fontweight="bold", color="#333333")

        keys = list(dist.keys())
        vals = list(dist.values())

        current_angle = start_angle

        for j, (k, v) in enumerate(zip(keys, vals)):
            sweep = v * angle_per_dim
            color = palette[j % len(palette)]

            t1, t2 = sorted([current_angle, current_angle - sweep])
            wedge_outer = mpatches.Wedge((0, 0), outer_radius, t1, t2, width=outer_radius-inner_radius, facecolor=color, edgecolor="white", lw=1.5)
            ax.add_patch(wedge_outer)

            if not no_text and (sweep > 3 or len(keys) <= 3):
                text_ang = (t1 + t2) / 2

                # Check angular distance from the previously drawn text (handling 360 wrap if necessary)
                if last_text_ang is not None:
                    ang_diff = abs((text_ang - last_text_ang + 180) % 360 - 180)
                    if ang_diff < 14:
                        stagger_level = (stagger_level + 1) % 4
                    else:
                        stagger_level = 0
                else:
                    stagger_level = 0

                text_rad = outer_radius + 0.05 + stagger_level * 0.15
                last_text_ang = text_ang

                tx = text_rad * np.cos(np.radians(text_ang))
                ty = text_rad * np.sin(np.radians(text_ang))

                cos_val = np.cos(np.radians(text_ang))
                sin_val = np.sin(np.radians(text_ang))
                ha = "left" if cos_val > 0.01 else ("right" if cos_val < -0.01 else "center")
                va = "bottom" if sin_val > 0.01 else ("top" if sin_val < -0.01 else "center")

                label = k.replace("_", " ").title()
                if "Spaces" in label: label = label.replace("Spaces", "").strip()
                if "Landscape" in label: label = label.replace("Landscape", "").strip()
                if "Multi-Motion" in label: label = "Multi"

                ax.text(tx, ty, label, ha=ha, va=va, fontsize=9, fontweight="bold")

            current_angle -= sweep

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.set_title("WorldArena Data Attributes Distribution", pad=15)


def plot_object_wordcloud(ax, items: list[dict]) -> None:
    counter: Counter = Counter()

    # List of generic/boring words to completely suppress
    suppress_words = {
        "tree", "trees", "cloud", "clouds", "sky", "skies", "blue sky", "clear sky",
        "plant", "plants", "water", "people", "person", "man", "woman"
    }

    for item in items:
        cap = item.get("caption") or {}
        objs = cap.get("objects") or (cap.get("CategoryTags") or {}).get("objects") or []
        for obj in objs:
            obj_clean = normalize_object(str(obj))
            if obj_clean not in suppress_words:
                counter[obj_clean] += 1

    # Cap the maximum frequency to prevent any single word from dominating too much
    if counter:
        max_freq = counter.most_common(1)[0][1]
        cap_val = max_freq * 0.4  # Cap at 40% of the max freq
        for k in counter:
            if counter[k] > cap_val:
                counter[k] = cap_val

    # Use a high resolution and scale for crispness, and Times New Roman font
    import matplotlib.font_manager as fm
    font_path = None
    # Try to find a Times or serif font
    for font in fm.findSystemFonts():
        if "Times" in font or "LiberationSerif" in font or "DejaVuSerif" in font:
            font_path = font
            if "Times" in font: break # Prefer Times

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="crest",
        max_words=150,
        prefer_horizontal=0.9,
        min_font_size=12,
        relative_scaling=0.3, # Less size difference between ranks
        collocations=False,
        font_path=font_path
    ).generate_from_frequencies(counter)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Object Tag Frequency (Images)")


def plot_motion_category(ax, items: list[dict], dynamic_vid: pd.DataFrame) -> None:
    img_counts = Counter()
    for item in items:
        if item.get("motion_regime") == "dynamic" and item.get("motion_category"):
            img_counts[item["motion_category"]] += 1
    cats = list(MOTION_LABELS.keys())
    x = np.arange(len(cats))
    w = 0.35
    img_vals = [img_counts.get(c, 0) for c in cats]
    vid_vals = [
        int(dynamic_vid.loc[dynamic_vid["motion_category"] == c, "count"].sum())
        for c in cats
    ]
    bars1 = ax.bar(x - w / 2, img_vals, w, label="Dynamic Images", color=C_PHOTO, edgecolor="white", linewidth=1.0, alpha=0.9)
    bars2 = ax.bar(x + w / 2, vid_vals, w, label="Dynamic Videos", color=C_STYLIZED, edgecolor="white", linewidth=1.0, alpha=0.9)
    for bar in list(bars1) + list(bars2):
        val = bar.get_height()
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 5, str(int(val)), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(max(img_vals), max(vid_vals)) * 1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([MOTION_LABELS[c] for c in cats], rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Count")
    ax.set_title("Motion Category Distribution")
    ax.legend(frameon=True, edgecolor="#E5E5E5", loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def plot_indoor_duration_heatmap(ax, indoor_counts: pd.DataFrame) -> None:
    pivot = indoor_counts.pivot(index="scene", columns="duration", values="count").fillna(0)
    pivot = pivot.reindex(
        index=[
            "dining_spaces",
            "living_spaces",
            "passageways",
            "workspaces",
        ],
        columns=["0-5s", "5-10s", "10-20s"],
    )
    pivot.index = [SCENE_LABELS.get(i, i) for i in pivot.index]
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar_kws={"label": "Videos"},
        linewidths=0.8,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Indoor Videos: Scene × Duration")
    ax.set_xlabel("Duration Bucket")
    ax.set_ylabel("Scene")


def plot_style_split(ax, items: list[dict]) -> None:
    photo = sum(1 for x in items if x.get("style") == "photorealistic")
    stylized = sum(1 for x in items if x.get("style") == "stylized")
    bars = ax.bar(
        ["Photorealistic", "Stylized"],
        [photo, stylized],
        color=[C_PHOTO, C_STYLIZED],
        edgecolor="white",
        width=0.55,
        linewidth=1.0,
        alpha=0.9,
    )
    ax.set_ylim(0, max([photo, stylized]) * 1.25)
    ax.set_ylabel("Image Count")
    ax.set_title("Image Style Balance (n=5,000)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for i, v in enumerate([photo, stylized]):
        ax.text(i, v + 40, f"{v:,}\n({100*v/5000:.1f}%)", ha="center", va="bottom", fontsize=9)


def hist_kde(ax, series: pd.Series, label: str, color: str) -> None:
    clean = series.dropna()
    if clean.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    weights = np.ones_like(clean) * 100.0 / len(clean)
    ax.hist(clean, bins=30, weights=weights, alpha=0.6, color=color, edgecolor="white", linewidth=0.8)
    if len(clean) > 5:
        xs = np.linspace(clean.min(), clean.max(), 200)
        kde = stats.gaussian_kde(clean)
        # Slightly darker line for KDE
        kde_color = sns.set_hls_values(color, l=max(0, sns.color_palette([color])[0][1] - 0.1))
        ax.plot(xs, kde(xs) * 100 * (clean.max() - clean.min()) / 30, color=kde_color, lw=2.2)

    ymax = ax.get_ylim()[1]
    mean_v = clean.mean()
    ax.axvline(mean_v, color=color, ls="--", lw=1.5, alpha=0.9)
    if not np.isnan(mean_v):
        ax.text(
            mean_v,
            0.92,
            f" $\\mu={mean_v:.2f}$",
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
            transform=ax.get_xaxis_transform(),
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.0)
        )
    ax.set_ylim(0, ymax * 1.45)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(label)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def plot_video_quality_panels(indoor: pd.DataFrame, dynamic: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = ["a", "b", "c", "d", "e", "f"]
    metrics = [
        ("motion score", "Motion Score"),
        ("luminance score", "Luminance Score"),
        ("dynamicRatio", "Dynamic Ratio"),
        ("moveDist", "Move Distance"),
        ("rotAngle", "Rotation Angle"),
        ("trajTurns", "Trajectory Turns"),
    ]

    for ax, letter, (col, title) in zip(axes.flat, panels, metrics):
        if col in indoor.columns:
            hist_kde(ax, indoor[col], f"Indoor — {title}", C_INDOOR)
        if col in dynamic.columns:
            clean = dynamic[col].dropna()
            if not clean.empty and len(clean) > 1:
                weights = np.ones_like(clean) * 100.0 / len(clean)
                ax.hist(
                    clean,
                    bins=25,
                    weights=weights,
                    alpha=0.55,
                    color=C_STYLIZED,
                    edgecolor="white",
                    linewidth=0.8,
                )
                xs = np.linspace(clean.min(), clean.max(), 200)
                kde = stats.gaussian_kde(clean)
                ymax = ax.get_ylim()[1]
                kde_color = sns.set_hls_values(C_STYLIZED, l=max(0, sns.color_palette([C_STYLIZED])[0][1] - 0.1))
                ax.plot(
                    xs,
                    kde(xs) * 100 * max(clean.max() - clean.min(), 1e-6) / 25,
                    color=kde_color,
                    lw=2.2,
                    label="Dynamic Video",
                )
                mean_v = clean.mean()
                ax.axvline(mean_v, color=C_STYLIZED, ls=":", lw=1.5, alpha=0.9)
                ax.text(
                    mean_v,
                    0.80,
                    f" $\\mu={mean_v:.2f}$",
                    color=C_STYLIZED,
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.0)
                )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        label_panel(ax, letter)
        ax.set_title(title, pad=8)
        if letter == "a":
            ax.legend(
                [
                    mpatches.Patch(color=C_INDOOR, alpha=0.6),
                    mpatches.Patch(color=C_STYLIZED, alpha=0.55),
                ],
                ["Indoor Video", "Dynamic Video"],
                frameon=True,
                edgecolor="#E5E5E5",
                loc="upper right",
                fontsize=9,
            )

    fig.suptitle("WorldArena Video Quality & Camera Motion Statistics", fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.5)
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

    # Save individual video panels
    panels_dir = out_path.parent / "individual_panels"
    panels_dir.mkdir(exist_ok=True)
    for letter, (col, title) in zip(panels, metrics):
        fig_ind, ax_ind = plt.subplots(figsize=(6, 4.5))
        if col in indoor.columns:
            hist_kde(ax_ind, indoor[col], f"Indoor — {title}", C_INDOOR)
        if col in dynamic.columns:
            clean = dynamic[col].dropna()
            if not clean.empty and len(clean) > 1:
                weights = np.ones_like(clean) * 100.0 / len(clean)
                ax_ind.hist(clean, bins=25, weights=weights, alpha=0.55, color=C_STYLIZED, edgecolor="white", linewidth=0.8)
                xs = np.linspace(clean.min(), clean.max(), 200)
                kde = stats.gaussian_kde(clean)
                kde_color = sns.set_hls_values(C_STYLIZED, l=max(0, sns.color_palette([C_STYLIZED])[0][1] - 0.1))
                ax_ind.plot(xs, kde(xs) * 100 * max(clean.max() - clean.min(), 1e-6) / 25, color=kde_color, lw=2.2, label="Dynamic Video")
                mean_v = clean.mean()
                ax_ind.axvline(mean_v, color=C_STYLIZED, ls=":", lw=1.5, alpha=0.9)
                ax_ind.text(
                    mean_v,
                    0.80,
                    f" $\\mu={mean_v:.2f}$",
                    color=C_STYLIZED,
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="top",
                    transform=ax_ind.get_xaxis_transform(),
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.0)
                )
        ax_ind.set_ylim(0, ax_ind.get_ylim()[1] * 1.45)
        ax_ind.set_title(title, pad=8)
        if letter == "a":
            ax_ind.legend(
                [mpatches.Patch(color=C_INDOOR, alpha=0.6), mpatches.Patch(color=C_STYLIZED, alpha=0.55)],
                ["Indoor Video", "Dynamic Video"],
                frameon=True, edgecolor="#E5E5E5", loc="upper right", fontsize=9
            )
        fig_ind.tight_layout()
        safe_title = title.replace(" ", "_").lower()
        fig_ind.savefig(panels_dir / f"video_panel_{letter}_{safe_title}.pdf", bbox_inches="tight", dpi=300)
        fig_ind.savefig(panels_dir / f"video_panel_{letter}_{safe_title}.png", bbox_inches="tight", dpi=300)
        plt.close(fig_ind)


def plot_instruction_bar(ax, items: list[dict], indoor_df: pd.DataFrame) -> None:
    # Prefer annotated indoor video instructions from filesystem scan
    instr_counts: Counter = Counter()
    ann_base = ROOT / "videos" / "indoor"
    for jpath in ann_base.rglob("instructions.json"):
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for labels in data.values():
            for label in labels:
                instr_counts[label.strip()] += 1
    top = instr_counts.most_common(10)
    labels = [k for k, _ in top]
    vals = [v for _, v in top]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color=C_INDOOR, edgecolor="white", linewidth=1.0, alpha=0.9)
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title("Indoor Video Camera Instructions")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(v + 10, i, str(v), va="center", fontsize=8)


def plot_caption_tokens(ax, items: list[dict]) -> None:
    lengths = []
    for item in items:
        cap = item.get("caption") or {}
        text = " ".join(
            str(cap.get(k, ""))
            for k in ("SceneSummary", "SceneDescription", "CameraMotion", "ShotImmersion")
        )
        lengths.append(len(text.split()))
    hist_kde(ax, pd.Series(lengths), "Image Caption Length (tokens)", C_STATIC)


def save_individual_subplots(items, indoor_counts, dynamic_vid, indoor_n, outdoor_n, dynamic_n, out_dir: Path) -> None:
    subplots_dir = out_dir / "individual_panels"
    subplots_dir.mkdir(exist_ok=True)

    def _save_one(func, args, filename, figsize=(6, 4.5)):
        fig, ax = plt.subplots(figsize=figsize)
        func(ax, *args)
        fig.tight_layout()
        fig.savefig(subplots_dir / f"{filename}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(subplots_dir / f"{filename}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    _save_one(plot_modality_overview, (items, indoor_n, outdoor_n, dynamic_n), "panel_a_modality_overview")
    _save_one(plot_scene_bar, (items,), "panel_b_scene_bar", figsize=(8, 4.5))
    _save_one(plot_style_split, (items,), "panel_c_style_split")
    _save_one(plot_motion_category, (items, dynamic_vid), "panel_d_motion_category", figsize=(7, 4.5))
    _save_one(plot_indoor_duration_heatmap, (indoor_counts,), "panel_e_indoor_duration_heatmap")
    _save_one(plot_environment_donut, (items,), "panel_f_environment_donut")
    _save_one(plot_environment_donut, (items, True), "panel_f_environment_donut_no_text")
    _save_one(plot_caption_tokens, (items,), "panel_g_caption_tokens")
    _save_one(plot_instruction_bar, (items, pd.DataFrame()), "panel_h_instruction_bar", figsize=(7, 5))

def build_figure1(items, indoor_counts, dynamic_vid, indoor_n, outdoor_n, dynamic_n, out_path: Path) -> None:
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_modality_overview(ax_a, items, indoor_n, outdoor_n, dynamic_n)
    label_panel(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1:])
    plot_scene_bar(ax_b, items)
    label_panel(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    plot_style_split(ax_c, items)
    label_panel(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    plot_motion_category(ax_d, items, dynamic_vid)
    label_panel(ax_d, "d")

    ax_e = fig.add_subplot(gs[1, 2])
    plot_indoor_duration_heatmap(ax_e, indoor_counts)
    label_panel(ax_e, "e")

    ax_f = fig.add_subplot(gs[2, 0])
    plot_environment_donut(ax_f, items)
    label_panel(ax_f, "f")

    ax_g = fig.add_subplot(gs[2, 1])
    plot_caption_tokens(ax_g, items)
    label_panel(ax_g, "g")

    ax_h = fig.add_subplot(gs[2, 2])
    plot_instruction_bar(ax_h, items, pd.DataFrame())
    label_panel(ax_h, "h")

    # Word cloud as separate wide figure panel - use inset on row 2 spanning?
    # Add wordcloud as separate figure for clarity
    fig.suptitle(
        "WorldArena Dataset Statistics — Images & Videos",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def build_wordcloud_figure(items, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_object_wordcloud(ax, items)
    label_panel(ax, "i")
    fig.suptitle("WorldArena Object Tag Word Cloud", fontsize=14, y=1.02)
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_summary_table(
    items,
    indoor_counts,
    outdoor_counts,
    dynamic_vid,
    indoor_df,
    dynamic_df,
    indoor_n,
    outdoor_n,
    out_path: Path,
) -> None:
    static_n = sum(1 for x in items if x.get("motion_regime") == "static")
    dynamic_img_n = sum(1 for x in items if x.get("motion_regime") == "dynamic")
    rows = [
        ("Total images", 5000),
        ("Static images", static_n),
        ("Dynamic images", dynamic_img_n),
        ("Photorealistic images", sum(1 for x in items if x.get("style") == "photorealistic")),
        ("Stylized images", sum(1 for x in items if x.get("style") == "stylized")),
        ("Indoor videos", indoor_n),
        ("Outdoor videos", outdoor_n),
        ("Dynamic videos", int(dynamic_vid["count"].sum())),
        ("Indoor video scenes", indoor_counts["scene"].nunique() if not indoor_counts.empty else 0),
        ("Outdoor video scenes", outdoor_counts["scene"].nunique() if 'outdoor_counts' in locals() and not outdoor_counts.empty else 0),
        ("Dynamic motion categories", dynamic_vid["motion_category"].nunique()),
    ]
    if "motion score" in indoor_df.columns:
        rows.append(("Indoor motion score (mean)", round(indoor_df["motion score"].mean(), 2)))
    if "luminance score" in indoor_df.columns:
        rows.append(("Indoor luminance (mean)", round(indoor_df["luminance score"].mean(), 2)))
    if "motion score" in dynamic_df.columns:
        rows.append(("Dynamic motion score (mean)", round(dynamic_df["motion score"].mean(), 2)))
    if "dynamicRatio" in dynamic_df.columns:
        rows.append(("Dynamic ratio (mean)", round(dynamic_df["dynamicRatio"].mean(), 3)))
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    df.to_csv(out_path, index=False)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = load_image_items()
    indoor_df, outdoor_df, dynamic_df = load_video_csvs()
    indoor_counts = count_indoor_videos()
    outdoor_counts = count_outdoor_videos()
    dynamic_vid = count_dynamic_videos()
    indoor_n = int(indoor_counts["count"].sum()) if not indoor_counts.empty else 0
    outdoor_n = int(outdoor_counts["count"].sum()) if not outdoor_counts.empty else 0
    dynamic_n = int(dynamic_vid["count"].sum()) if not dynamic_vid.empty else 0

    build_figure1(
        items,
        indoor_counts,
        dynamic_vid,
        indoor_n,
        outdoor_n,
        dynamic_n,
        OUT_DIR / "worldarena_dataset_overview",
    )
    build_wordcloud_figure(items, OUT_DIR / "worldarena_object_wordcloud")
    plot_video_quality_panels(indoor_df, dynamic_df, OUT_DIR / "worldarena_video_quality")
    write_summary_table(
        items,
        indoor_counts,
        outdoor_counts,
        dynamic_vid,
        indoor_df,
        dynamic_df,
        indoor_n,
        outdoor_n,
        OUT_DIR / "summary_statistics.csv",
    )
    save_individual_subplots(items, indoor_counts, dynamic_vid, indoor_n, outdoor_n, dynamic_n, OUT_DIR)

    print(f"Figures saved to: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
