#!/usr/bin/env python3
"""Create publication-ready statistical tables for the local WorldArena data.

The table atlas complements ``plot_worldatlas_dataset_statistics.py``.  It
uses the already-audited record CSV files, enriches image records with the
annotation JSON, and exports each table as PDF/SVG/400-dpi PNG plus CSV and
booktabs-style LaTeX.  A multipage PDF collects all tables in reading order.

Usage:
    python scripts/plot_worldatlas_dataset_tables.py
    python scripts/plot_worldatlas_dataset_tables.py --output-dir outputs/worldatlas_statistics/paper_tables
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
STATS_ROOT = REPO_ROOT / "outputs" / "worldatlas_statistics"
IMAGE_RECORDS = STATS_ROOT / "image" / "image_records.csv"
VIDEO_RECORDS = STATS_ROOT / "video" / "video_records.csv"
IMAGE_ANNOTATIONS = REPO_ROOT / "data_external" / "OpenEnvision" / "worldarena" / "image" / "image_annotations.json"
REFERENCE_SUMMARY = REPO_ROOT / "docs" / "dataset_statistics_figures" / "summary_statistics.csv"
DEFAULT_OUTPUT = STATS_ROOT / "paper_tables"


INK = "#1F2933"
NAVY = "#17324D"
MUTED = "#667085"
GRID = "#D9E1E8"
PAPER = "#FFFFFF"
ROW_ALT = "#F6F8FA"
SECTION_BG = "#EAF2F8"
TOTAL_BG = "#FFF6E6"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
TEAL = "#009E73"
GOLD = "#E69F00"
PURPLE = "#CC79A7"
UNKNOWN = "#98A2B3"
PALE_BLUE = "#DCECF5"
PALE_ORANGE = "#F8E5D8"
PALE_TEAL = "#DDF2EB"
PALE_PURPLE = "#F3E5EF"


VIDEO_COLORS = {
    "AAA Games": GOLD,
    "Static / Photorealistic": BLUE,
    "Static / Stylized": VERMILLION,
    "Dynamic / Photorealistic": TEAL,
    "Dynamic / Stylized": PURPLE,
}


@dataclass
class DisplayRow:
    cells: list[str]
    kind: str = "data"
    emphasis: bool = False
    fill: str | None = None
    accent: str | None = None
    bars: dict[int, tuple[float, str]] = field(default_factory=dict)
    cell_colors: dict[int, str] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class TableSpec:
    stem: str
    kicker: str
    title: str
    subtitle: str
    headers: list[str]
    widths: list[float]
    aligns: list[str]
    rows: list[DisplayRow]
    data: pd.DataFrame
    note: str


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def pretty(value: object) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def pct(value: float, denominator: float) -> str:
    return f"{100 * value / denominator:.1f}%" if denominator else "0.0%"


def fmt_int(value: float | int) -> str:
    return f"{int(round(value)):,}"


def fmt_float(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text or ""))


def ensure_base_records() -> None:
    missing = [path for path in (IMAGE_RECORDS, VIDEO_RECORDS) if not path.exists()]
    if not missing:
        return
    script = REPO_ROOT / "scripts" / "plot_worldatlas_dataset_statistics.py"
    subprocess.run([sys.executable, str(script), "--section", "all"], cwd=REPO_ROOT, check=True)


def load_annotation_records() -> pd.DataFrame:
    payload = json.loads(IMAGE_ANNOTATIONS.read_text())
    rows: list[dict[str, object]] = []
    for item in payload["items"]:
        rel = item["relative_path"]
        path_parts = Path(rel).parts
        generation = item.get("generation_spec") or {}
        tags = ((item.get("caption") or {}).get("CategoryTags") or {})
        scene = tags.get("sceneType") or {}
        prompts = generation.get("prompt_list") or []
        segment_prompts = generation.get("segment_prompts") or []
        instructions = item.get("instructions") or {}
        instruction_tokens = [token for value in instructions.values() if isinstance(value, list) for token in value]
        camera_path = generation.get("camera_path") or []
        rows.append(
            {
                "relative_path": rel,
                "mode": path_parts[0],
                "style": path_parts[1],
                "scene_first": scene.get("first"),
                "scene_second": scene.get("second"),
                "brightness": tags.get("brightness"),
                "time_of_day": tags.get("timeOfDay"),
                "weather": tags.get("weather"),
                "crowd_density": tags.get("crowdDensity"),
                "camera_recipe": " → ".join(camera_path),
                "camera_tokens": camera_path,
                "context_list": generation.get("current_objects") or [],
                "target_list": generation.get("objects") or [],
                "prompt_list": prompts,
                "segment_prompt_list": segment_prompts,
                "instruction_tokens": instruction_tokens,
                "context_objects": len(generation.get("current_objects") or []),
                "target_objects": len(generation.get("objects") or []),
                "prompt_words": words(" ".join(prompts)),
                "segment_prompt_words": words(" ".join(segment_prompts)),
                "instruction_ops": len(instruction_tokens),
            }
        )
    return pd.DataFrame(rows)


def video_group(row: pd.Series) -> str:
    if row["branch"] == "AAA Games":
        return "AAA Games"
    return f"{row['branch']} / {pretty(row['style'])}"


def render_table(spec: TableSpec) -> plt.Figure:
    widths = np.asarray(spec.widths, dtype=float)
    widths = widths / widths.sum()
    total_weight = 1.0 + sum(row.weight for row in spec.rows)
    wrapped_title = textwrap.fill(spec.title, width=72, break_long_words=False, break_on_hyphens=False)
    wrapped_subtitle = textwrap.fill(spec.subtitle, width=104, break_long_words=False, break_on_hyphens=False)
    title_lines = wrapped_title.count("\n") + 1
    subtitle_lines = wrapped_subtitle.count("\n") + 1
    top_block_inches = 0.56 + title_lines * 0.29 + 0.07 + subtitle_lines * 0.17 + 0.15
    note_block_inches = 0.50
    table_height_inches = total_weight * 0.33
    figure_height = max(3.8, top_block_inches + table_height_inches + note_block_inches)
    fig, ax = plt.subplots(figsize=(8.2, figure_height))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, right = 0.045, 0.955
    table_top = 1 - top_block_inches / figure_height
    table_bottom = note_block_inches / figure_height
    table_width = right - left
    row_unit = (table_top - table_bottom) / total_weight
    xs = [left]
    for width in widths:
        xs.append(xs[-1] + table_width * width)

    accent_y = 1 - 0.18 / figure_height
    kicker_y = 1 - 0.33 / figure_height
    title_y = 1 - 0.56 / figure_height
    subtitle_y = 1 - (0.56 + title_lines * 0.29 + 0.07) / figure_height
    ax.add_patch(Rectangle((left, accent_y), 0.095, 0.035 / figure_height, color=BLUE, linewidth=0))
    ax.text(left, kicker_y, spec.kicker.upper(), ha="left", va="top", fontsize=7.2, color=BLUE, weight="bold")
    ax.text(left, title_y, wrapped_title, ha="left", va="top", fontsize=14.0, color=INK, weight="semibold", linespacing=1.08)
    ax.text(left, subtitle_y, wrapped_subtitle, ha="left", va="top", fontsize=8.4, color=MUTED, linespacing=1.20)

    header_height = row_unit
    y = table_top
    ax.add_patch(Rectangle((left, y - header_height), table_width, header_height, facecolor=NAVY, edgecolor="none"))
    for index, header in enumerate(spec.headers):
        align = "left" if spec.aligns[index] == "left" else "center"
        x = xs[index] + 0.009 if align == "left" else (xs[index] + xs[index + 1]) / 2
        ax.text(x, y - header_height / 2, header, ha=align, va="center", fontsize=7.65, color=PAPER, weight="semibold", linespacing=1.05)
    y -= header_height

    data_index = 0
    for row in spec.rows:
        height = row_unit * row.weight
        if row.kind == "section":
            fill = row.fill or SECTION_BG
            ax.add_patch(Rectangle((left, y - height), table_width, height, facecolor=fill, edgecolor="none"))
            ax.text(left + 0.01, y - height / 2, row.cells[0], ha="left", va="center", fontsize=8.1, color=NAVY, weight="bold")
            ax.plot([left, right], [y - height, y - height], color=GRID, linewidth=0.7)
            y -= height
            continue

        fill = row.fill or (ROW_ALT if data_index % 2 else PAPER)
        ax.add_patch(Rectangle((left, y - height), table_width, height, facecolor=fill, edgecolor="none"))
        if row.accent:
            ax.add_patch(Rectangle((left, y - height), 0.0045, height, facecolor=row.accent, edgecolor="none"))

        for col_index, cell in enumerate(row.cells):
            x0, x1 = xs[col_index], xs[col_index + 1]
            if col_index in row.bars:
                fraction, color = row.bars[col_index]
                fraction = float(np.clip(fraction, 0, 1))
                pad = 0.007
                ax.add_patch(
                    Rectangle(
                        (x0 + pad, y - height * 0.68),
                        max(0, x1 - x0 - 2 * pad) * fraction,
                        height * 0.36,
                        facecolor=color,
                        edgecolor="none",
                    )
                )
            align = spec.aligns[col_index]
            if align == "left":
                x, ha = x0 + 0.009, "left"
            elif align == "center":
                x, ha = (x0 + x1) / 2, "center"
            else:
                x, ha = x1 - 0.009, "right"
            color = row.cell_colors.get(col_index, INK)
            ax.text(
                x,
                y - height / 2,
                str(cell),
                ha=ha,
                va="center",
                fontsize=7.7,
                color=color,
                weight="semibold" if row.emphasis else "normal",
                linespacing=1.12,
            )
        ax.plot([left, right], [y - height, y - height], color=GRID, linewidth=0.55)
        data_index += 1
        y -= height

    ax.plot([left, right], [table_top, table_top], color=NAVY, linewidth=0.8)
    ax.text(left, 0.18 / figure_height, textwrap.fill(spec.note, width=150, break_long_words=False, break_on_hyphens=False), ha="left", va="center", fontsize=6.9, color=MUTED, linespacing=1.15)
    return fig


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex(spec: TableSpec, output_path: Path) -> None:
    tabular = spec.data.to_latex(index=False, escape=True)
    label = "tab:" + re.sub(r"[^a-z0-9]+", "_", spec.stem.lower()).strip("_")
    text = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{latex_escape(spec.title)}}}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{tabular}"
        "}\n"
        f"\\parbox{{\\textwidth}}{{\\footnotesize \\textit{{Note:}} {latex_escape(spec.note)}}}\n"
        "\\end{table*}\n"
    )
    output_path.write_text(text)


def save_table(spec: TableSpec, output_dir: Path, book: PdfPages) -> None:
    fig = render_table(spec)
    for ext, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 400})):
        fig.savefig(output_dir / f"{spec.stem}.{ext}", bbox_inches="tight", **kwargs)
    book.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    spec.data.to_csv(output_dir / f"{spec.stem}.csv", index=False)
    write_latex(spec, output_dir / f"{spec.stem}.tex")


def section(label: str, *, fill: str | None = None) -> DisplayRow:
    return DisplayRow([label], kind="section", fill=fill, weight=0.78)


def total_row(cells: Iterable[object], bars: dict[int, tuple[float, str]] | None = None) -> DisplayRow:
    return DisplayRow([str(value) for value in cells], emphasis=True, fill=TOTAL_BG, accent=GOLD, bars=bars or {})


def build_scope_table(image: pd.DataFrame, video: pd.DataFrame) -> TableSpec:
    current_images = len(image)
    current_videos = len(video)
    current_total = current_images + current_videos
    ref_images = np.nan
    ref_videos = np.nan
    if REFERENCE_SUMMARY.exists():
        ref = pd.read_csv(REFERENCE_SUMMARY).set_index("Metric")["Value"]
        ref_images = float(ref.get("Total images", np.nan))
        ref_videos = float(ref.get("Indoor videos", 0)) + float(ref.get("Outdoor videos", 0)) + float(ref.get("Dynamic videos", 0))
    ref_total = ref_images + ref_videos
    spec_count = int((video["spec_source"] == "filename").sum())
    storage = float(video["size_bytes"].sum() / 2**30)

    records = [
        {"Metric": "Images", "Current local scan": current_images, "Reference figures": int(ref_images), "Difference": int(current_images - ref_images), "Status": "Matched"},
        {"Metric": "Videos", "Current local scan": current_videos, "Reference figures": int(ref_videos), "Difference": int(current_videos - ref_videos), "Status": "Different scope"},
        {"Metric": "Primary visual assets", "Current local scan": current_total, "Reference figures": int(ref_total), "Difference": int(current_total - ref_total), "Status": "Different scope"},
        {"Metric": "Video storage", "Current local scan": f"{storage:.2f} GiB", "Reference figures": "Not reported", "Difference": "—", "Status": "Current scan only"},
        {"Metric": "Filename technical specs", "Current local scan": f"{spec_count:,} ({pct(spec_count, current_videos)})", "Reference figures": "Not reported", "Difference": "—", "Status": "Current scan only"},
    ]
    rows: list[DisplayRow] = []
    for record in records:
        delta = record["Difference"]
        delta_text = f"{delta:+,}".replace("-", "−") if isinstance(delta, int) else str(delta)
        mismatch = record["Status"] == "Different scope"
        rows.append(
            DisplayRow(
                [record["Metric"], fmt_int(record["Current local scan"]) if isinstance(record["Current local scan"], int) else record["Current local scan"], fmt_int(record["Reference figures"]) if isinstance(record["Reference figures"], int) else record["Reference figures"], delta_text, record["Status"]],
                emphasis=record["Metric"] in {"Images", "Videos", "Primary visual assets"},
                fill=PALE_ORANGE if mismatch else None,
                accent=VERMILLION if mismatch else TEAL if record["Status"] == "Matched" else BLUE,
                cell_colors={4: VERMILLION if mismatch else TEAL if record["Status"] == "Matched" else BLUE},
            )
        )
    data = pd.DataFrame(records)
    return TableSpec(
        stem="table_01_scope_reconciliation",
        kicker="Table 01 · Scope audit",
        title="The image count matches the reference, but the video counting universe does not",
        subtitle="Current statistics use the files presently available in the local WorldArena tree; reference values come only from docs/dataset_statistics_figures/summary_statistics.csv.",
        headers=["Metric", "Current\nlocal scan", "Reference\nfigures", "Difference", "Status"],
        widths=[0.28, 0.18, 0.18, 0.13, 0.23],
        aligns=["left", "right", "right", "right", "left"],
        rows=rows,
        data=data,
        note="Use one counting universe consistently in the paper. The current image universe is 5,000 annotated images; the current video universe is 5,426 MP4 files. The historical reference total of 6,388 videos should not be mixed with current-scan statistics.",
    )


def build_image_subset_table(image: pd.DataFrame, annotations: pd.DataFrame) -> TableSpec:
    work = image.merge(
        annotations[["relative_path", "context_objects", "prompt_words"]],
        on="relative_path",
        how="left",
        suffixes=("", "_annotation"),
    )
    work["group"] = work["mode"].map(pretty) + " / " + work["style"].map(pretty)
    work["megapixels"] = work["width"] * work["height"] / 1e6
    work["square"] = (work["width"] / work["height"]).between(0.95, 1.05)
    work["rgba"] = work["image_mode"].eq("RGBA")
    order = ["Static / Photorealistic", "Static / Stylized", "Dynamic / Photorealistic", "Dynamic / Stylized"]

    records: list[dict[str, object]] = []
    rows: list[DisplayRow] = []
    for group in order + ["All images"]:
        part = work if group == "All images" else work[work["group"] == group]
        record = {
            "Subset": group,
            "Images": len(part),
            "Share": len(part) / len(work),
            "Median megapixels": float(part["megapixels"].median()),
            "P90 megapixels": float(part["megapixels"].quantile(0.9)),
            "Square share": float(part["square"].mean()),
            "RGBA share": float(part["rgba"].mean()),
            "Median context objects": float(part["context_objects_annotation"].median()),
            "Median prompt words": float(part["prompt_words"].median()),
        }
        records.append(record)
        cells = [
            group,
            fmt_int(record["Images"]),
            pct(record["Images"], len(work)),
            fmt_float(record["Median megapixels"], 2),
            fmt_float(record["P90 megapixels"], 2),
            pct(record["Square share"], 1),
            pct(record["RGBA share"], 1),
            fmt_float(record["Median context objects"], 1),
            fmt_float(record["Median prompt words"], 1),
        ]
        bars = {2: (float(record["Share"]), PALE_BLUE), 5: (float(record["Square share"]), PALE_TEAL)}
        if group == "All images":
            rows.append(total_row(cells, bars))
        else:
            accent = VERMILLION if group.endswith("Stylized") else BLUE
            fill = PALE_TEAL if group == "Dynamic / Photorealistic" else None
            rows.append(DisplayRow(cells, accent=accent, fill=fill, bars=bars))

    export = pd.DataFrame(records)
    for column in ["Share", "Square share", "RGBA share"]:
        export[column] = export[column].map(lambda value: f"{100 * value:.1f}%")
    return TableSpec(
        stem="table_02_image_subset_profile",
        kicker="Table 02 · Image profile",
        title="Image styles are count-balanced, but their geometry and encoding profiles differ sharply",
        subtitle="Photorealistic dynamic images are much larger, while both stylized subsets are almost entirely square. Object and prompt statistics come from the annotation generation specification.",
        headers=["Subset", "n", "Share", "Median\nMP", "P90\nMP", "Square", "RGBA", "Context\nobjects†", "Prompt\nwords†"],
        widths=[0.22, 0.07, 0.085, 0.095, 0.095, 0.09, 0.08, 0.13, 0.135],
        aligns=["left", "right", "right", "right", "right", "right", "right", "right", "right"],
        rows=rows,
        data=export,
        note="Source: 5,000 image annotation records and decoded image headers. Square means aspect ratio in [0.95, 1.05]. MP denotes megapixels. †Context objects and prompt words are medians; prompt length is whitespace/word-pattern based, not tokenizer-specific.",
    )


def taxonomy_rows(
    table: pd.DataFrame,
    section_name: str,
    order: list[str],
    *,
    imbalance_threshold: float,
    photo_column: str = "photorealistic",
    stylized_column: str = "stylized",
) -> tuple[list[DisplayRow], list[dict[str, object]]]:
    rows = [section(section_name)]
    records: list[dict[str, object]] = []
    for label in order:
        values = table.loc[label] if label in table.index else pd.Series(dtype=float)
        photo = int(values.get(photo_column, 0))
        stylized = int(values.get(stylized_column, 0))
        total = photo + stylized
        photo_share = photo / total if total else 0.0
        gap = abs(photo - stylized)
        imbalance = gap / total if total else 0.0
        record = {
            "Section": section_name,
            "Taxonomy": label,
            "Photorealistic": photo,
            "Stylized": stylized,
            "Total": total,
            "Photorealistic share": photo_share,
            "Absolute style gap": gap,
            "Relative style gap": imbalance,
        }
        records.append(record)
        rows.append(
            DisplayRow(
                [label, fmt_int(photo), fmt_int(stylized), fmt_int(total), pct(photo, total), f"{gap:,} ({pct(gap, total)})"],
                emphasis=imbalance >= imbalance_threshold,
                fill=PALE_ORANGE if imbalance >= imbalance_threshold else None,
                accent=VERMILLION if imbalance >= imbalance_threshold else None,
                bars={4: (photo_share, PALE_BLUE)},
            )
        )
    photo_total = sum(int(table.loc[label].get(photo_column, 0)) for label in order if label in table.index)
    stylized_total = sum(int(table.loc[label].get(stylized_column, 0)) for label in order if label in table.index)
    subtotal = photo_total + stylized_total
    gap = abs(photo_total - stylized_total)
    rows.append(total_row([f"{section_name} subtotal", fmt_int(photo_total), fmt_int(stylized_total), fmt_int(subtotal), pct(photo_total, subtotal), f"{gap:,} ({pct(gap, subtotal)})"], {4: (photo_total / subtotal if subtotal else 0, PALE_BLUE)}))
    records.append(
        {
            "Section": section_name,
            "Taxonomy": f"{section_name} subtotal",
            "Photorealistic": photo_total,
            "Stylized": stylized_total,
            "Total": subtotal,
            "Photorealistic share": photo_total / subtotal if subtotal else 0,
            "Absolute style gap": gap,
            "Relative style gap": gap / subtotal if subtotal else 0,
        }
    )
    return rows, records


def build_image_taxonomy_table(image: pd.DataFrame) -> TableSpec:
    table = image.pivot_table(index="taxonomy", columns="style", values="relative_path", aggfunc="count", fill_value=0)
    static_order = [
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
    ]
    dynamic_order = ["Articulated", "Rigid", "Fluid", "Deformable", "Multi Motion"]
    static_rows, static_records = taxonomy_rows(table, "Static scenes", static_order, imbalance_threshold=0.10)
    dynamic_rows, dynamic_records = taxonomy_rows(table, "Dynamic motion types", dynamic_order, imbalance_threshold=0.10)
    export = pd.DataFrame(static_records + dynamic_records)
    export["Photorealistic share"] = export["Photorealistic share"].map(lambda value: f"{100 * value:.1f}%")
    export["Relative style gap"] = export["Relative style gap"].map(lambda value: f"{100 * value:.1f}%")
    return TableSpec(
        stem="table_03_image_taxonomy_matrix",
        kicker="Table 03 · Image taxonomy",
        title="Image taxonomy coverage is nearly uniform, with one visible photorealistic rigid-motion deficit",
        subtitle="Counts are split by rendering style so category coverage and style balance can be inspected simultaneously.",
        headers=["Taxonomy", "Photo.", "Stylized", "Total", "Photo.\nshare", "Style gap\n|P−S|"],
        widths=[0.37, 0.115, 0.115, 0.105, 0.135, 0.15],
        aligns=["left", "right", "right", "right", "right", "right"],
        rows=static_rows + dynamic_rows,
        data=export,
        note="Source: image/image_annotations.json; n = 5,000. Style gap is the absolute count difference and its percentage of the category total. The highlighted Dynamic · Rigid row contains 108 photorealistic versus 169 stylized images.",
    )


def _categorical_audit(
    annotations: pd.DataFrame,
    field_name: str,
    label: str,
    mask: pd.Series | None = None,
) -> tuple[dict[str, object], DisplayRow]:
    part = annotations if mask is None else annotations[mask]
    series = part[field_name]
    populated = series.notna() & series.astype(str).ne("")
    unknown = series.astype(str).str.casefold().eq("unknown")
    explicit = series[populated & ~unknown]
    top_value = explicit.value_counts().index[0] if len(explicit) else "—"
    top_count = int((series == top_value).sum()) if top_value != "—" else 0
    record = {
        "Section": "Semantic labels",
        "Field": label,
        "Applicable n": len(part),
        "Populated": int(populated.sum()),
        "Explicit unknown": int(unknown.sum()),
        "Leading value/statistic": str(top_value),
        "Leading share": top_count / len(part) if len(part) else 0,
        "Distinct values": int(series[populated].nunique()),
    }
    row = DisplayRow(
        [label, fmt_int(len(part)), pct(populated.sum(), len(part)), f"{unknown.sum():,} ({pct(unknown.sum(), len(part))})", str(top_value), pct(top_count, len(part)), fmt_int(record["Distinct values"])],
        fill=PALE_ORANGE if unknown.mean() >= 0.20 else None,
        accent=VERMILLION if unknown.mean() >= 0.20 else None,
        bars={2: (float(populated.mean()), PALE_TEAL), 3: (float(unknown.mean()), PALE_ORANGE)},
    )
    return record, row


def _list_audit_record(
    annotations: pd.DataFrame,
    label: str,
    list_column: str,
    statistic_column: str,
    mask: pd.Series,
    statistic_label: str,
    section_name: str,
) -> tuple[dict[str, object], DisplayRow]:
    part = annotations[mask]
    populated = part[list_column].map(bool)
    vocabulary = sorted({str(value) for values in part[list_column] for value in values})
    statistic = float(part.loc[populated, statistic_column].median()) if populated.any() else 0.0
    record = {
        "Section": section_name,
        "Field": label,
        "Applicable n": len(part),
        "Populated": int(populated.sum()),
        "Explicit unknown": 0,
        "Leading value/statistic": f"Median {statistic:g} {statistic_label}",
        "Leading share": np.nan,
        "Distinct values": len(vocabulary),
    }
    row = DisplayRow(
        [label, fmt_int(len(part)), pct(populated.sum(), len(part)), "—", f"Median {statistic:g} {statistic_label}", "—", fmt_int(len(vocabulary))],
        bars={2: (float(populated.mean()), PALE_TEAL)},
    )
    return record, row


def build_image_annotation_table(annotations: pd.DataFrame) -> TableSpec:
    rows: list[DisplayRow] = [section("Semantic labels")]
    records: list[dict[str, object]] = []
    for field_name, label in [
        ("scene_first", "Scene type · level 1"),
        ("scene_second", "Scene type · level 2"),
        ("brightness", "Brightness"),
        ("time_of_day", "Time of day"),
        ("weather", "Weather"),
        ("crowd_density", "Crowd density"),
        ("camera_recipe", "Camera recipe"),
    ]:
        record, row = _categorical_audit(annotations, field_name, label)
        records.append(record)
        rows.append(row)

    rows.append(section("Generation and motion metadata", fill=PALE_TEAL))
    all_mask = pd.Series(True, index=annotations.index)
    dynamic_mask = annotations["mode"].eq("dynamic")
    list_specs = [
        ("Context object tags", "context_list", "context_objects", all_mask, "objects"),
        ("Target object tags", "target_list", "target_objects", dynamic_mask, "objects"),
        ("Prompt list", "prompt_list", "prompt_words", all_mask, "words"),
        ("Segment prompts", "segment_prompt_list", "segment_prompt_words", dynamic_mask, "words"),
        ("Transition instructions", "instruction_tokens", "instruction_ops", all_mask, "operations"),
    ]
    for label, list_column, stat_column, mask, unit in list_specs:
        record, row = _list_audit_record(annotations, label, list_column, stat_column, mask, unit, "Generation and motion metadata")
        records.append(record)
        rows.append(row)

    export = pd.DataFrame(records)
    export["Populated share"] = export["Populated"] / export["Applicable n"]
    export["Explicit unknown share"] = export["Explicit unknown"] / export["Applicable n"]
    export["Leading share"] = export["Leading share"].map(lambda value: "—" if pd.isna(value) else f"{100 * value:.1f}%")
    export["Populated share"] = export["Populated share"].map(lambda value: f"{100 * value:.1f}%")
    export["Explicit unknown share"] = export["Explicit unknown share"].map(lambda value: f"{100 * value:.1f}%")
    return TableSpec(
        stem="table_04_image_annotation_audit",
        kicker="Table 04 · Annotation audit",
        title="Image annotations are structurally complete; uncertainty is concentrated in brightness and time labels",
        subtitle="Dynamic-only fields use the 1,628 dynamic images as their applicable population, so intentional absence from static images is not counted as missingness.",
        headers=["Field", "Applicable\nn", "Populated", "Explicit\nunknown", "Leading value / statistic", "Lead\nshare", "Distinct"],
        widths=[0.225, 0.105, 0.11, 0.14, 0.25, 0.095, 0.075],
        aligns=["left", "right", "right", "right", "left", "right", "right"],
        rows=rows,
        data=export,
        note="Source: image/image_annotations.json; n = 5,000. Explicit Unknown is reported rather than imputed. Distinct values denote categorical values, full camera recipes, or vocabulary items depending on field. Target tags and segment prompts are evaluated only on dynamic images.",
    )


def build_video_subset_table(video: pd.DataFrame) -> TableSpec:
    work = video.copy()
    work["group"] = work.apply(video_group, axis=1)
    order = ["AAA Games", "Static / Photorealistic", "Static / Stylized", "Dynamic / Photorealistic", "Dynamic / Stylized"]
    total_count = len(work)
    total_storage = float(work["size_bytes"].sum())
    records: list[dict[str, object]] = []
    rows: list[DisplayRow] = []
    for group in order + ["All videos"]:
        part = work if group == "All videos" else work[work["group"] == group]
        known = int(part["spec_source"].eq("filename").sum())
        storage_bytes = float(part["size_bytes"].sum())
        record = {
            "Subset": group,
            "Clips": len(part),
            "Clip share": len(part) / total_count,
            "Storage GiB": storage_bytes / 2**30,
            "Storage share": storage_bytes / total_storage,
            "Median MiB": float(part["size_mib"].median()),
            "P90 MiB": float(part["size_mib"].quantile(0.9)),
            "P99 MiB": float(part["size_mib"].quantile(0.99)),
            "Filename specs": known,
            "Filename spec share": known / len(part) if len(part) else 0,
        }
        records.append(record)
        cells = [
            group,
            fmt_int(len(part)),
            pct(len(part), total_count),
            fmt_float(storage_bytes / 2**30, 2),
            pct(storage_bytes, total_storage),
            fmt_float(record["Median MiB"], 2),
            fmt_float(record["P90 MiB"], 2),
            fmt_float(record["P99 MiB"], 2),
            f"{known:,} ({pct(known, len(part))})",
        ]
        bars = {2: (len(part) / total_count, PALE_BLUE), 4: (storage_bytes / total_storage, PALE_TEAL)}
        if group == "All videos":
            rows.append(total_row(cells, bars))
        else:
            highlight = group == "Dynamic / Photorealistic"
            rows.append(
                DisplayRow(
                    cells,
                    emphasis=highlight,
                    fill=PALE_TEAL if highlight else None,
                    accent=VIDEO_COLORS[group],
                    bars=bars,
                )
            )
    export = pd.DataFrame(records)
    for column in ["Clip share", "Storage share", "Filename spec share"]:
        export[column] = export[column].map(lambda value: f"{100 * value:.1f}%")
    return TableSpec(
        stem="table_05_video_subset_profile",
        kicker="Table 05 · Video profile",
        title="Dynamic photorealistic clips dominate storage despite representing only one sixth of the videos",
        subtitle="Count share, storage share, file-size quantiles, and filename-derived specification coverage are shown together to expose the long tail.",
        headers=["Subset", "Clips", "Clip\nshare", "Storage\nGiB", "Storage\nshare", "Median\nMiB", "P90\nMiB", "P99\nMiB", "Filename\nspecs"],
        widths=[0.23, 0.075, 0.085, 0.095, 0.095, 0.095, 0.085, 0.085, 0.155],
        aligns=["left", "right", "right", "right", "right", "right", "right", "right", "right"],
        rows=rows,
        data=export,
        note="Source: current local videos_test_split_4971 scan; 5,426 MP4 files, 62.52 GiB. File-size quantiles use local filesystem sizes. Filename specs are counted only when both resolution and FPS tokens are explicitly present; unknown values are not imputed.",
    )


def build_video_taxonomy_table(video: pd.DataFrame) -> TableSpec:
    static = video[video["branch"] == "Static"]
    dynamic = video[video["branch"] == "Dynamic"]
    static_table = static.pivot_table(index="category_label", columns="style", values="relative_path", aggfunc="count", fill_value=0)
    dynamic_table = dynamic.pivot_table(index="category_label", columns="style", values="relative_path", aggfunc="count", fill_value=0)
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
    static_rows, static_records = taxonomy_rows(static_table, "Static scenes", static_order, imbalance_threshold=0.35)
    dynamic_rows, dynamic_records = taxonomy_rows(dynamic_table, "Dynamic motion types", dynamic_order, imbalance_threshold=0.35)
    export = pd.DataFrame(static_records + dynamic_records)
    export["Photorealistic share"] = export["Photorealistic share"].map(lambda value: f"{100 * value:.1f}%")
    export["Relative style gap"] = export["Relative style gap"].map(lambda value: f"{100 * value:.1f}%")
    return TableSpec(
        stem="table_06_video_taxonomy_matrix",
        kicker="Table 06 · Video taxonomy",
        title="Video taxonomy spans 15 categories, but stylized coverage remains highly asymmetric",
        subtitle="AAA Games are excluded because they do not follow the photorealistic/stylized scene taxonomy. Highlighted rows have a style gap above 35% of the category total.",
        headers=["Taxonomy", "Photo.", "Stylized", "Total", "Photo.\nshare", "Style gap\n|P−S|"],
        widths=[0.37, 0.115, 0.115, 0.105, 0.135, 0.15],
        aligns=["left", "right", "right", "right", "right", "right"],
        rows=static_rows + dynamic_rows,
        data=export,
        note="Source: current local MP4 paths. Static taxonomy covers 4,000 clips and dynamic motion taxonomy covers 961 clips; 465 AAA Games clips are outside this matrix. Style gap is the absolute count difference and its category-relative percentage.",
    )


def build_video_duration_table(video: pd.DataFrame) -> TableSpec:
    total = len(video)
    rows: list[DisplayRow] = [section("Explicit static duration buckets · n = 4,000")]
    records: list[dict[str, object]] = []

    def add_record(label: str, part: pd.DataFrame, section_name: str, *, is_total: bool = False) -> None:
        photo = int(part["style"].eq("photorealistic").sum())
        stylized = int(part["style"].eq("stylized").sum())
        aaa = int(part["branch"].eq("AAA Games").sum())
        record = {
            "Section": section_name,
            "Population": label,
            "Photorealistic": photo,
            "Stylized": stylized,
            "AAA Games": aaa,
            "Total": len(part),
            "Share of all clips": len(part) / total,
            "Median MiB": float(part["size_mib"].median()),
            "P90 MiB": float(part["size_mib"].quantile(0.9)),
        }
        records.append(record)
        cells = [label, fmt_int(photo), fmt_int(stylized), fmt_int(aaa), fmt_int(len(part)), pct(len(part), total), fmt_float(record["Median MiB"], 2), fmt_float(record["P90 MiB"], 2)]
        bars = {5: (len(part) / total, PALE_BLUE)}
        if is_total:
            rows.append(total_row(cells, bars))
        else:
            rows.append(DisplayRow(cells, bars=bars))

    static = video[video["branch"] == "Static"]
    duration_order = [("0–5 s", "0-5s"), ("5–10 s", "5-10s"), ("10–20 s", "10-20s")]
    for display, token in duration_order:
        add_record(display, static[static["duration_bucket"] == token], "Explicit static duration buckets")
    add_record("Bucketed static subtotal", static, "Explicit static duration buckets", is_total=True)

    rows.append(section("No explicit directory duration bucket · n = 1,426", fill=PALE_PURPLE))
    dynamic = video[video["branch"] == "Dynamic"]
    aaa = video[video["branch"] == "AAA Games"]
    add_record("Dynamic clips", dynamic, "No explicit duration bucket")
    add_record("AAA Games", aaa, "No explicit duration bucket")
    add_record("Unbucketed subtotal", pd.concat([dynamic, aaa]), "No explicit duration bucket", is_total=True)
    add_record("All videos", video, "All", is_total=True)

    export = pd.DataFrame(records)
    export["Share of all clips"] = export["Share of all clips"].map(lambda value: f"{100 * value:.1f}%")
    return TableSpec(
        stem="table_07_video_duration_coverage",
        kicker="Table 07 · Duration coverage",
        title="Directory duration labels cover 73.7% of videos; dynamic and AAA clips remain unbucketed",
        subtitle="The table separates explicit static duration buckets from clips with no directory-level duration label and reports their file-size distributions without inferring duration.",
        headers=["Population", "Photo.", "Stylized", "AAA", "Total", "All clips", "Median\nMiB", "P90\nMiB"],
        widths=[0.25, 0.10, 0.10, 0.085, 0.10, 0.12, 0.115, 0.13],
        aligns=["left", "right", "right", "right", "right", "right", "right", "right"],
        rows=rows,
        data=export,
        note="Source: current local MP4 paths. Duration buckets are directory labels, not durations decoded from video containers. The 1,426 unbucketed clips consist of 961 dynamic videos and 465 AAA Games videos; no duration values are imputed.",
    )


def _top_values(series: pd.Series, top_n: int = 3, denominator: int | None = None) -> str:
    counts = series.value_counts().head(top_n)
    denom = denominator or int(series.notna().sum())
    return "; ".join(f"{label}: {count:,} ({pct(count, denom)})" for label, count in counts.items())


def build_video_metadata_table(video: pd.DataFrame) -> TableSpec:
    known = video[video["spec_source"] == "filename"].copy()
    known["resolution"] = known.apply(lambda row: f"{int(row['width'])}×{int(row['height'])}", axis=1)
    known["fps_label"] = known["fps"].astype(int).astype(str) + " fps"
    source_counts = video["filename_source"].value_counts()
    group_counts = known.apply(video_group, axis=1).value_counts()
    total = len(video)

    records: list[dict[str, object]] = []
    rows: list[DisplayRow] = [section("Filename-encoded technical tokens")]

    def append(label: str, count: int, dominant: str, interpretation: str, *, highlight: bool = False) -> None:
        record = {"Section": "Filename-encoded technical tokens", "Dimension": label, "Count": count, "Share of all clips": count / total, "Dominant values / split": dominant, "Interpretation": interpretation}
        records.append(record)
        line_count = max(dominant.count("\n") + 1, interpretation.count("\n") + 1)
        rows.append(
            DisplayRow(
                [label, fmt_int(count), pct(count, total), dominant, interpretation],
                emphasis=highlight,
                fill=PALE_ORANGE if highlight else None,
                accent=VERMILLION if highlight else BLUE,
                bars={2: (count / total, PALE_ORANGE if highlight else PALE_BLUE)},
                weight=max(1.0, 0.85 + 0.45 * (line_count - 1)),
            )
        )

    append("Resolution token", len(known), _top_values(known["resolution"], 3, len(known)).replace("; ", "\n"), f"{known['resolution'].nunique()} distinct encoded resolutions", highlight=True)
    append("Frame-rate token", len(known), _top_values(known["fps_label"], 3, len(known)).replace("; ", "\n"), f"{known['fps_label'].nunique()} distinct encoded frame rates", highlight=True)
    append("Spec-bearing subset", len(known), "\n".join(f"{name}: {count:,}" for name, count in group_counts.items()), "659/665 spec-bearing clips are\ndynamic photorealistic", highlight=True)

    rows.append(section("Filename provenance", fill=PALE_TEAL))
    provenance_notes = {
        "Named/other": "Descriptive filename; no recognized source token",
        "ID-only/other": "Identifier-like or otherwise opaque filename",
        "Pexels": "Pexels source token is visible in filename",
        "Pixabay": "Pixabay source token is visible in filename",
    }
    for label in ["Named/other", "ID-only/other", "Pexels", "Pixabay"]:
        count = int(source_counts.get(label, 0))
        record = {"Section": "Filename provenance", "Dimension": label, "Count": count, "Share of all clips": count / total, "Dominant values / split": "—", "Interpretation": provenance_notes[label]}
        records.append(record)
        rows.append(DisplayRow([label, fmt_int(count), pct(count, total), "—", provenance_notes[label]], accent=TEAL, bars={2: (count / total, PALE_TEAL)}))

    export = pd.DataFrame(records)
    export["Share of all clips"] = export["Share of all clips"].map(lambda value: f"{100 * value:.1f}%")
    return TableSpec(
        stem="table_08_video_technical_metadata",
        kicker="Table 08 · Technical metadata",
        title="Only 12.3% of video filenames expose resolution and frame-rate specifications",
        subtitle="Technical tokens are strongly concentrated in the dynamic photorealistic subset; filename provenance is shown separately because it is not a quality metric.",
        headers=["Dimension", "Count", "All clips", "Dominant values / split", "Interpretation"],
        widths=[0.20, 0.10, 0.12, 0.32, 0.26],
        aligns=["left", "right", "right", "left", "left"],
        rows=rows,
        data=export,
        note="Source: current local MP4 filenames; n = 5,426. Resolution and FPS are reported only when explicit tokens are present in the filename. The remaining 4,761 clips are Unknown for these fields; container properties are not inferred or imputed.",
    )


def write_readme(output_dir: Path, specs: list[TableSpec]) -> None:
    lines = [
        "# WorldArena paper-ready statistical tables",
        "",
        "Each table is available as 400-dpi PNG, vector PDF/SVG, machine-readable CSV, and LaTeX. The combined multipage PDF is `worldarena_statistical_tables.pdf`.",
        "",
        "## Tables",
        "",
    ]
    for spec in specs:
        lines.append(f"- `{spec.stem}` — {spec.title}")
    lines.extend(
        [
            "",
            "## Counting rules",
            "",
            "- Images: 5,000 records in `image/image_annotations.json`; masks are excluded.",
            "- Videos: 5,426 `.mp4` files in `videos_test_split_4971`; masks, JSON, review assets, and contact sheets are excluded.",
            "- Video duration uses explicit static directory buckets only.",
            "- Video resolution/FPS values are parsed only from explicit filename tokens; unknowns are not imputed.",
            "- The reference figures under `docs/dataset_statistics_figures` contain a historical 6,388-video scope and should not be mixed with the current local scan.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    set_style()
    ensure_base_records()
    image = pd.read_csv(IMAGE_RECORDS)
    video = pd.read_csv(VIDEO_RECORDS)
    annotations = load_annotation_records()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        build_scope_table(image, video),
        build_image_subset_table(image, annotations),
        build_image_taxonomy_table(image),
        build_image_annotation_table(annotations),
        build_video_subset_table(video),
        build_video_taxonomy_table(video),
        build_video_duration_table(video),
        build_video_metadata_table(video),
    ]

    book_path = output_dir / "worldarena_statistical_tables.pdf"
    with PdfPages(book_path) as book:
        for spec in specs:
            save_table(spec, output_dir, book)

    manifest = {
        "image_records": len(image),
        "video_records": len(video),
        "video_storage_gib": float(video["size_bytes"].sum() / 2**30),
        "table_count": len(specs),
        "tables": [spec.stem for spec in specs],
        "formats": ["png (400 dpi)", "pdf", "svg", "csv", "tex"],
        "combined_pdf": book_path.name,
    }
    (output_dir / "table_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_readme(output_dir, specs)
    print(f"tables: {len(specs)} -> {output_dir}")


if __name__ == "__main__":
    main()
