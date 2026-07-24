"""Build MkDocs documentation pages from scanned dataset inventory."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import plotly.express as px
import plotly.io as pio
from PIL import Image, ImageOps

from worldarena.common.serialization import ensure_dir
from worldarena.datasets.config import load_project_config
from worldarena.datasets.summarize import (
    CAPTION_DIMENSION_LABELS,
    build_dataset_frame,
    build_summary_tables,
    load_cached_dataset_frame,
    merge_collection_snapshot,
    write_artifacts,
)


def _format_number(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "0"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{int(value):,}"


def _metric_cards(metrics: list[tuple[str, str, str]]) -> str:
    cards = []
    for label, value, note in metrics:
        cards.append(
            f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}</div>
  <div class="metric-note">{note}</div>
</div>
""".strip()
        )
    return '<div class="metric-grid">\n' + "\n".join(cards) + "\n</div>"


def _overview_hero_cards(metrics: list[tuple[str, str, str, str]]) -> str:
    cards = []
    for label, value, note, accent in metrics:
        cards.append(
            f"""
<div class="overview-hero-card overview-hero-card--{accent}">
  <div class="overview-hero-label">{label}</div>
  <div class="overview-hero-value">{value}</div>
  <div class="overview-hero-note">{note}</div>
</div>
""".strip()
        )
    return '<div class="overview-hero-grid">\n' + "\n".join(cards) + "\n</div>"


def _overview_regime_cards(regimes: list[tuple[str, str, str, str]]) -> str:
    cards = []
    for label, value, share, note in regimes:
        cards.append(
            f"""
<div class="overview-regime-card">
  <div class="overview-regime-top">
    <span class="overview-regime-label">{label}</span>
    <span class="overview-regime-share">{share}</span>
  </div>
  <div class="overview-regime-value">{value}</div>
  <div class="overview-regime-note">{note}</div>
</div>
""".strip()
        )
    return '<div class="overview-regime-grid">\n' + "\n".join(cards) + "\n</div>"


def _overview_section(label: str, note: str, body: str) -> str:
    note_html = f'\n  <p class="overview-section-note">{note}</p>' if note else ""
    return f"""<div class="overview-section">
  <p class="overview-section-label">{label}</p>{note_html}
{body}
</div>"""


def _overview_scale_page(intro: str, sections: list[tuple[str, str, str]]) -> str:
    section_blocks = "\n\n".join(
        _overview_section(label, note, body) for label, note, body in sections
    )
    return f"""<p class="overview-section-lede">{intro}</p>

<div class="overview-page">

{section_blocks}

</div>"""


def _overview_dimension_bars(rows: list[tuple[str, str, int, int]]) -> str:
    if not rows:
        return "_No dimension rows available._"
    max_count = max(count for _, _, count, _ in rows) or 1
    bars = []
    for key, label, count, official in rows:
        width = max(8, round(100 * count / max_count))
        bars.append(
            f"""
<div class="overview-dimension-row">
  <div class="overview-dimension-meta">
    <span class="overview-dimension-label">{label}</span>
    <span class="overview-dimension-count">{count} metrics · {official} official</span>
  </div>
  <div class="overview-dimension-track">
    <div class="overview-dimension-fill overview-dimension-fill--{key}" style="width: {width}%;"></div>
  </div>
</div>
""".strip()
        )
    return '<div class="overview-dimension-bars">\n' + "\n".join(bars) + "\n</div>"


def _overview_metric_split(official: int, diagnostic: int, experimental: int) -> str:
    return f"""<div class="overview-split-grid">
<div class="overview-split-card overview-split-card--official">
  <div class="overview-split-label">Official surface</div>
  <div class="overview-split-value">{_format_number(official)}</div>
  <div class="overview-split-note">Headline or conditionally official metrics in the public inventory</div>
</div>
<div class="overview-split-card overview-split-card--diagnostic">
  <div class="overview-split-label">Diagnostic probes</div>
  <div class="overview-split-value">{_format_number(diagnostic)}</div>
  <div class="overview-split-note">Supplemental geometry, reconstruction, and long-horizon probes</div>
</div>
<div class="overview-split-card overview-split-card--experimental">
  <div class="overview-split-label">Experimental suite</div>
  <div class="overview-split-value">{_format_number(experimental)}</div>
  <div class="overview-split-note">Non-headline slice declared separately in `config/benchmark.yaml`</div>
</div>
</div>"""


def _parse_metric_inventory(project_root: Path) -> dict[str, Any]:
    index_path = project_root / "docs" / "en" / "benchmark" / "metrics" / "index.md"
    empty = {
        "total": 0,
        "official": 0,
        "diagnostic": 0,
        "dimensions": [],
        "dimension_counts": {},
    }
    if not index_path.exists():
        return empty

    content = index_path.read_text(encoding="utf-8")
    table_rows = _parse_metric_inventory_table(content)
    if table_rows:
        rows = table_rows
    else:
        rows = _parse_metric_inventory_lists(content)
        rows.extend(_parse_metric_inventory_dimension_docs(project_root))

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["dimension"], row["slug"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    dimension_counts = Counter(row["dimension"] for row in deduped)
    official_counts = Counter(
        row["dimension"]
        for row in deduped
        if not row["aggregation"].lower().startswith("diagnostic")
    )
    official = sum(
        1 for row in deduped if not row["aggregation"].lower().startswith("diagnostic")
    )
    diagnostic = len(deduped) - official
    dimension_labels = {
        "long_sequence": "Long Sequence",
        "action_control": "Action Control",
        "consistency_3d_4d": "3D/4D Consistency",
        "physics": "Physics",
        "quality": "Quality",
        "real_time": "Real Time",
    }
    dimension_rows = []
    for key, label in dimension_labels.items():
        count = int(dimension_counts.get(key, 0))
        if count:
            dimension_rows.append((key, label, count, int(official_counts.get(key, 0))))
    return {
        "total": len(deduped),
        "official": official,
        "diagnostic": diagnostic,
        "dimensions": dimension_rows,
        "dimension_counts": dict(dimension_counts),
    }


_METRIC_DIMENSION_SECTIONS = {
    "action control": "action_control",
    "3d/4d consistency": "consistency_3d_4d",
    "physics": "physics",
    "quality": "quality",
    "real time": "real_time",
    "long sequence (memory)": "long_sequence",
    "long sequence": "long_sequence",
}


def _metric_aggregation_from_status(status: str) -> str:
    lowered = status.lower()
    has_official = bool(re.search(r"\bofficial\b", lowered))
    has_diagnostic = bool(re.search(r"\bdiagnostic\b", lowered))
    if has_diagnostic and not has_official:
        return "diagnostic"
    return "official"


def _metric_slug_from_link(link: str) -> str:
    stem = Path(link.split("#", 1)[0]).stem
    return stem or link.strip("/").replace("/", "-")


def _extract_markdown_links(text: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for match in re.finditer(r"\[`([^`]+)`\]\(([^)]+)\)", text):
        links.append((match.group(1), match.group(2)))
    for match in re.finditer(r"(?<![`])\[(?![`])([^\]]+)\]\(([^)]+)\)", text):
        links.append((match.group(1), match.group(2)))
    return links


def _parse_metric_inventory_table(content: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in content.splitlines():
        if not line.startswith("| [`"):
            continue
        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) < 6:
            continue
        link = parts[0].strip("`[]() ")
        match = re.search(r"\[`([^`]+)`\]\(([^)]+)\)", parts[0])
        if match is None:
            plain = _extract_markdown_links(parts[0])
            slug = _metric_slug_from_link(plain[0][1]) if plain else parts[0]
        else:
            slug = _metric_slug_from_link(match.group(2))
        rows.append(
            {
                "slug": slug,
                "dimension": parts[1].strip("`"),
                "aggregation": parts[5],
            }
        )
    return rows


def _parse_metric_inventory_lists(content: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_dimension: str | None = None
    skip_simulator_block = False

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            current_dimension = _METRIC_DIMENSION_SECTIONS.get(stripped[4:].strip().lower())
            skip_simulator_block = False
            continue

        lowered = stripped.lower()
        if "simulator-backed physics" in lowered or "outside the four-suite" in lowered:
            skip_simulator_block = True
            continue
        if skip_simulator_block:
            continue
        if not stripped.startswith("- "):
            continue

        status = ""
        for separator in ("—", "--", " - "):
            if separator in stripped:
                status = stripped.split(separator, 1)[1].strip()
                break

        metric_id_match = re.match(r"-\s+`([^`]+)`", stripped)
        if metric_id_match:
            rows.append(
                {
                    "slug": metric_id_match.group(1),
                    "dimension": current_dimension or "physics",
                    "aggregation": _metric_aggregation_from_status(status or stripped),
                }
            )
            continue

        link_matches = _extract_markdown_links(stripped)
        if not link_matches:
            continue

        aggregation = _metric_aggregation_from_status(status or stripped)
        for _, link in link_matches:
            dimension = current_dimension
            if dimension is None and "/physics/" in link:
                dimension = "physics"
            rows.append(
                {
                    "slug": _metric_slug_from_link(link),
                    "dimension": dimension or "quality",
                    "aggregation": aggregation,
                }
            )
    return rows


def _parse_metric_inventory_dimension_docs(project_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    dimensions_dir = project_root / "docs" / "en" / "benchmark" / "metrics" / "dimensions"
    if not dimensions_dir.is_dir():
        return rows

    for path in sorted(dimensions_dir.glob("*.md")):
        dimension = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ["):
                continue
            links = _extract_markdown_links(stripped)
            if not links:
                continue
            status = ""
            for separator in ("—", "--", " - "):
                if separator in stripped:
                    status = stripped.split(separator, 1)[1].strip()
                    break
            for _, link in links:
                rows.append(
                    {
                        "slug": _metric_slug_from_link(link),
                        "dimension": dimension,
                        "aggregation": _metric_aggregation_from_status(status or stripped),
                    }
                )
    return rows


def _load_benchmark_overview_stats(project_root: Path) -> dict[str, Any]:
    benchmark_path = project_root / "config" / "benchmark.yaml"
    payload = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    metrics_by_suite = payload.get("benchmark", {}).get("metrics_by_suite", {})
    formal_suites = [
        "image_static",
        "image_dynamic",
        "video_static",
        "video_dynamic",
    ]
    suite_metric_counts = {
        suite: len(metrics_by_suite.get(suite, []))
        for suite in formal_suites
    }
    unique_formal_metrics = {
        metric
        for suite in formal_suites
        for metric in metrics_by_suite.get(suite, [])
    }
    model_count = len(list((project_root / "config" / "models").glob("*.yaml")))
    inventory = _parse_metric_inventory(project_root)
    return {
        "formal_suites": len(formal_suites),
        "suite_metric_counts": suite_metric_counts,
        "unique_formal_metrics": len(unique_formal_metrics),
        "experimental_metrics": len(metrics_by_suite.get("experimental", [])),
        "model_count": model_count,
        **inventory,
    }


def _frame_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    if frame.empty:
        return "_No rows available._"
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return "_No rows available._"
    table = frame.loc[:, available_columns].head(limit).copy()
    return table.to_markdown(index=False)


def _styled_figure(fig: Any, title: str) -> Any:
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="",
        height=500,
    )
    return fig


class ChartWriter:
    def __init__(self, chart_dir: Path, src_prefix: str) -> None:
        self.chart_dir = ensure_dir(chart_dir)
        self.src_prefix = src_prefix.rstrip("/")

    def reset(self) -> None:
        """Reset."""
        for path in self.chart_dir.iterdir():
            if path.is_file():
                path.unlink()

    def render(
        self,
        page_slug: str,
        chart_slug: str,
        heading: str,
        figure: Any,
        *,
        academic: bool = False,
        caption: str | None = None,
    ) -> str:
        if figure is None:
            return ""
        filename = f"{page_slug}-{chart_slug}.html"
        output_path = self.chart_dir / filename
        pio.write_html(
            figure,
            file=output_path,
            full_html=True,
            include_plotlyjs="directory",
            config={"displaylogo": False, "responsive": True},
            auto_open=False,
        )
        iframe = (
            f'<iframe class="chart-frame" src="{self.src_prefix}/{filename}" '
            f'title="{heading}" loading="lazy"></iframe>'
        )
        if not academic:
            return f"## {heading}\n\n{iframe}"
        figure_caption = caption or heading
        return (
            f'<figure class="analytics-exhibit">\n'
            f"{iframe}\n"
            f'<figcaption class="analytics-exhibit-caption">{figure_caption}</figcaption>\n'
            f"</figure>"
        )


def _artifact_path(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "sample"


def _sample_rows(frame: pd.DataFrame, columns: list[str], limit: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    samples = frame.copy().sort_values([column for column in ["collection", "path"] if column in frame.columns])
    key_columns = [column for column in columns if column in samples.columns]
    if key_columns:
        samples["_sample_key"] = samples[key_columns].fillna("Unspecified").astype(str).agg("|".join, axis=1)
        if "collection" in samples.columns:
            base = samples.drop_duplicates("collection").head(limit)
            remaining = samples.loc[~samples.index.isin(base.index)]
            remaining = remaining.loc[~remaining["_sample_key"].isin(set(base["_sample_key"]))]
            diverse = remaining.drop_duplicates("_sample_key")
            chosen = pd.concat([base, diverse]).head(limit)
        else:
            chosen = samples.drop_duplicates("_sample_key").head(limit)
        if chosen.shape[0] < limit:
            extra = samples.loc[~samples.index.isin(chosen.index)].head(limit - chosen.shape[0])
            chosen = pd.concat([chosen, extra])
    else:
        chosen = samples.head(limit)
    return chosen.drop(columns=["_sample_key"], errors="ignore")


def _write_thumbnail(source: Path, target: Path, max_size: tuple[int, int] = (720, 480)) -> bool:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail(max_size)
            image.save(target, "JPEG", quality=84, optimize=True)
        return True
    except OSError:
        return False


def _run_ffmpeg(command: list[str]) -> bool:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=45,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _write_video_assets(source: Path, poster: Path, preview: Path) -> tuple[bool, bool]:
    poster.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    poster_ok = _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale=w='min(720,iw)':h=-2",
            str(poster),
        ]
    )
    preview_ok = _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-t",
            "3",
            "-i",
            str(source),
            "-vf",
            "scale=w='min(720,iw)':h=-2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(preview),
        ]
    )
    return poster_ok, preview_ok


def _sample_title(row: pd.Series, columns: list[str]) -> str:
    values = []
    for column in columns:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        values.append(str(value).replace("_", " ").title())
    return " / ".join(values[:3]) or str(row.get("filename") or "Sample")


def _build_sample_assets(summary: dict[str, Any], output_dir: Path) -> dict[str, list[dict[str, str]]]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets: dict[str, list[dict[str, str]]] = {"image": [], "video": [], "physics": []}
    sample_specs = [
        ("image", summary["images"], ["collection", "motion_regime", "style", "scene", "motion_category"], 36),
        ("video", summary["videos"], ["collection", "motion_regime", "scene", "motion_category"], 24),
        (
            "physics",
            summary["physics"].sort_values("has_physics_metadata", ascending=False),
            ["physics_group", "physics_dimension"],
            32,
        ),
    ]

    for kind, frame, key_columns, limit in sample_specs:
        for index, (_, row) in enumerate(_sample_rows(frame, key_columns, limit).iterrows(), start=1):
            source = Path(str(row["path"]))
            slug = _slugify(f"{index:02d}-{row.get('collection', kind)}-{row.get('stem', source.stem)}")
            title_columns = [column for column in key_columns if column != "collection"]
            item = {
                "title": _sample_title(row, title_columns or key_columns),
                "subtitle": str(row.get("relative_path") or row.get("filename") or source.name),
                "poster": "",
                "preview": "",
                "kind": kind,
            }
            if kind == "image":
                target = output_dir / "images" / f"{slug}.jpg"
                if not _write_thumbnail(source, target):
                    continue
                item["poster"] = f"samples/images/{target.name}"
            else:
                poster = output_dir / "posters" / f"{slug}.jpg"
                preview = output_dir / "previews" / f"{slug}.mp4"
                first_frame = row.get("first_frame_path")
                poster_ok = _write_thumbnail(Path(str(first_frame)), poster) if first_frame else False
                if not poster_ok:
                    poster_ok, preview_ok = _write_video_assets(source, poster, preview)
                else:
                    _, preview_ok = _write_video_assets(source, poster, preview)
                if not poster_ok and not preview_ok:
                    continue
                item["poster"] = f"samples/posters/{poster.name}" if poster.exists() else ""
                item["preview"] = f"samples/previews/{preview.name}" if preview.exists() else ""
            assets[kind].append(item)
    return assets


def _sample_gallery(
    title: str,
    items: list[dict[str, str]],
    asset_prefix: str,
    *,
    academic: bool = False,
) -> str:
    if not items:
        return ""
    cards: list[str] = []
    for item in items:
        poster = f"{asset_prefix.rstrip('/')}/{item['poster']}" if item.get("poster") else ""
        preview = f"{asset_prefix.rstrip('/')}/{item['preview']}" if item.get("preview") else ""
        if preview:
            media = (
                f'<video class="sample-media" controls muted preload="metadata" poster="{poster}">'
                f'<source src="{preview}" type="video/mp4"></video>'
            )
        elif poster:
            media = f'<img class="sample-media" src="{poster}" alt="{item["title"]}" loading="lazy">'
        else:
            continue
        if academic:
            cards.append(
                f"""
<figure class="sample-exhibit">
  <div class="sample-exhibit-frame">{media}</div>
  <figcaption>
    <span class="sample-exhibit-title">{item["title"]}</span>
    <span class="sample-exhibit-path">{item["subtitle"]}</span>
  </figcaption>
</figure>
""".strip()
            )
        else:
            cards.append(
                f"""
<figure class="sample-card">
  {media}
  <figcaption>
    <strong>{item["title"]}</strong>
    <span>{item["subtitle"]}</span>
  </figcaption>
</figure>
""".strip()
            )
    if not cards:
        return ""
    gallery_kind = items[0].get("kind", "mixed") if items else "mixed"
    if academic:
        return (
            f"## {title}\n\n"
            f'<p class="section-lede">Representative clips drawn from static, dynamic, and gameplay collections.</p>\n\n'
            f'<div class="sample-exhibit-grid sample-exhibit-grid--{gallery_kind}">\n'
            + "\n\n".join(cards)
            + "\n</div>"
        )
    return (
        f"## {title}\n\n"
        + f'<div class="sample-grid sample-grid--{gallery_kind}">\n'
        + "\n".join(cards)
        + "\n</div>"
    )


ZH_REPLACEMENTS = {
    "# Dataset Overview": "# 数据概览",
    "# Image Analytics": "# 图像统计",
    "# Video Analytics": "# 视频统计",
    "# Physics Analytics": "# Physics 统计",
    "# Caption Analytics": "# Caption 统计",
    "# Collection Details": "# 集合明细",
    "!!! info \"Generated\"": "!!! info \"生成内容\"",
    "Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets.": "当你新增、删除或重组源数据资产后，请运行 `worldarena build-docs --config config/datasets.yaml`。",
    "Benchmark Items": "基准条目数",
    "Image Rows": "图像条目",
    "Video Rows": "视频条目",
    "Physics Rows": "Physics 条目",
    "Static Images": "静态图像",
    "Dynamic Images": "动态图像",
    "Benchmark Videos": "基准视频",
    "AAA Game Videos": "AAA 游戏视频",
    "Annotated": "已标注",
    "Exact Duration Available": "可读取时长",
    "Physics Metadata": "Physics 元数据",
    "Physics Groups": "Physics 组",
    "Physics Dimensions": "Physics 维度",
    "Captioned Media": "带 Caption 的媒体",
    "Topology": "拓扑",
    "Collection Summary": "集合汇总",
    "Sample Image Gallery": "图像样例",
    "Sample Video Gallery": "视频样例",
    "Representative Samples": "代表样例",
    "Statistical Exhibits": "统计图表",
    "Tabular Summaries": "表格汇总",
    "Annotation Coverage": "标注覆盖",
    "Resolution Distribution": "分辨率分布",
    "Sample Physics Gallery": "Physics 样例",
    "Static Scene Summary": "静态场景汇总",
    "Dynamic Mask Summary": "动态 Mask 汇总",
    "Annotation Summary": "标注汇总",
    "Resolution Summary": "分辨率汇总",
    "Physics Dimension Summary": "Physics 维度汇总",
    "Quality And Motion Summary": "质量与运动汇总",
    "Coverage Summary": "覆盖率汇总",
    "WorldAtlas Arena combines a multi-regime benchmark corpus, a six-axis evaluation contract, and a registry of world-model adapters. The panels below summarize scale from complementary angles—media inventory, task regime, metric surface, and model coverage—before the collection tables and sample galleries.": "WorldAtlas Arena 将多任务基准语料、六维评估契约与世界模型适配器注册表整合在同一工作区。下列面板从媒体规模、任务范式、指标面与模型覆盖等多个角度汇总统计，再进入集合明细与样例画廊。",
    "Formal Suites": "正式评测套件",
    "Public benchmark contracts in `config/benchmark.yaml`": "公开基准契约，定义于 `config/benchmark.yaml`",
    "Documented Metrics": "文档化指标",
    "Public metric inventory in the benchmark docs": "基准文档中的公开指标清单",
    "Evaluation Dimensions": "评估维度",
    "Axes aggregated in `per_dimension_summary.json`": "聚合写入 `per_dimension_summary.json` 的六个主轴",
    "Registered Models": "已注册模型",
    "Adapter configs under `config/models/`": "`config/models/` 下的适配器配置",
    "Annotated Rows": "已标注条目",
    "of benchmark rows carry supervision sidecars": "的基准条目附带监督侧车数据",
    "Benchmark Scale": "基准规模",
    "Headline totals, media inventory, task regimes, and benchmark contract surface.": "总量指标、媒体库存、任务范式与基准契约面。",
    "Primary inventory counts, suites, metrics, and model coverage.": "基准条目、套件、指标与模型覆盖的主统计。",
    "Media inventory": "媒体库存",
    "Image, video, and physics assets in the benchmark corpus.": "基准语料中的图像、视频与 physics 资产。",
    "How benchmark rows split across static, dynamic, interaction, and physics tracks.": "基准条目在 static、dynamic、interaction 与 physics 轨道上的分布。",
    "## Summary": "## 概述",
    "Masked Dynamics": "带 Mask 的动态样本",
    "Dynamic image and gameplay rows with segmentation masks": "含分割 mask 的动态图像与 gameplay 条目",
    "Task Regimes": "任务范式",
    "Static world generation": "静态世界生成",
    "Camera or layout change with scene preservation": "相机或布局变化下的场景一致性扩展",
    "Dynamic world generation": "动态世界生成",
    "In-scene temporal evolution under controlled viewpoints": "固定或受控视角下的场景内时序演化",
    "Interaction-conditioned": "交互条件生成",
    "Gameplay and action-conditioned rollouts": "Gameplay 与动作条件 rollout",
    "Physics diagnostics": "物理诊断",
    "Reference motion and simulator-backed physics tracks": "参考运动与仿真器支持的物理评测轨道",
    "Evaluation Framework": "评估框架",
    "Formal suites separate static, dynamic, and interaction-conditioned generation while sharing a common reporting contract. Each suite declares an applicable metric vector in `config/benchmark.yaml`.": "正式套件区分静态、动态与交互条件生成，并共享统一的报告契约；各套件在 `config/benchmark.yaml` 中声明适用的指标向量。",
    "Metrics by Dimension": "按维度划分的指标",
    "Official vs Diagnostic Surface": "官方指标 vs 诊断探针",
    "Official surface": "官方指标面",
    "Headline or conditionally official metrics in the public inventory": "公开清单中进入 headline 或条件性 official 聚合的指标",
    "Diagnostic probes": "诊断探针",
    "Supplemental geometry, reconstruction, and long-horizon probes": "补充几何、重建与长序列诊断探针",
    "Experimental suite": "实验套件",
    "Non-headline slice declared separately in `config/benchmark.yaml`": "在 `config/benchmark.yaml` 中单独声明、不计入 headline 的切片",
    "See the [Metrics Overview](benchmark/metrics/index.md) for eligibility rules, normalization, and aggregation semantics.": "指标适用性、归一化与聚合语义见 [指标概览](benchmark/metrics/index.md)。",
    "Long Sequence": "长序列",
    "Action Control": "动作控制",
    "3D/4D Consistency": "3D/4D 一致性",
    "Real Time": "实时性",
    "Primary rows across image, video, and physics suites": "图像、视频与 physics 套件的主基准行",
    " metrics · ": " 项指标 · ",
    " official</span>": " 项 official</span>",
    "The image corpus spans static scene frames and motion-labelled dynamic stills used by the `image_static` and `image_dynamic` benchmark suites. Panels below summarize inventory scale, taxonomy coverage, mask supervision, and applicable metrics before the detailed tables and interactive charts.": "图像语料覆盖静态场景帧与带运动标签的动态静帧，分别服务于 `image_static` 与 `image_dynamic` 基准套件。下列面板从库存规模、分类覆盖、mask 监督与适用指标等角度汇总统计，再进入明细表与交互图表。",
    "The video corpus covers camera-motion static clips, in-scene dynamic evolution, and gameplay rollouts evaluated under `video_static`, `video_dynamic`, and auxiliary gameplay tracks. Panels below break down collection mix, temporal metadata, annotation coverage, and benchmark metric surface.": "视频语料涵盖相机运动静态片段、场景内动态演化与 gameplay rollout，对应 `video_static`、`video_dynamic` 及辅助 gameplay 轨道。下列面板从集合构成、时序元数据、标注覆盖与基准指标面等角度展开统计。",
    "Collection Scale": "集合规模",
    "Static vs Dynamic": "静态 vs 动态",
    "Benchmark Suites": "基准套件",
    "Image suites declare applicable metrics in `config/benchmark.yaml`. Static rows stress layout and camera consistency; dynamic rows add motion and mask-aware diagnostics.": "图像套件在 `config/benchmark.yaml` 中声明适用指标：静态样本侧重布局与相机一致性，动态样本额外覆盖运动与 mask 感知诊断。",
    "Video suites span camera-motion static clips, dynamic temporal rollouts, and fully annotated gameplay sequences. Metric vectors differ by regime but share the public reporting contract.": "视频套件覆盖相机运动静态片段、动态时序 rollout 与完整标注的 gameplay 序列；各范式指标向量不同，但共享公开报告契约。",
    "Taxonomy": "分类拓扑",
    "Visual Styles": "视觉风格",
    "Scene Environments": "场景环境",
    "Motion Categories": "运动类别",
    "Unique Resolutions": "分辨率种类",
    "Mask Coverage": "Mask 覆盖",
    "Formal Image Suites": "正式图像套件",
    "Applicable Image Metrics": "适用图像指标",
    "Unique metrics across `image_static` and `image_dynamic`": "`image_static` 与 `image_dynamic` 的去重指标数",
    "Photorealistic + stylized still frames": "写实与风格化静帧",
    "of image rows carry segmentation masks": "的图像条目附带分割 mask",
    "Indoor + outdoor scene buckets": "室内与室外场景桶",
    "Articulated, rigid, fluid, and multi-motion labels": "关节、刚体、流体与复合运动标签",
    "Distinct resolution labels in inventory": "库存中的不同分辨率标签",
    "Formal Video Suites": "正式视频套件",
    "Applicable Video Metrics": "适用视频指标",
    "Unique metrics across `video_static` and `video_dynamic`": "`video_static` 与 `video_dynamic` 的去重指标数",
    "Static, dynamic, and gameplay collections": "静态、动态与 gameplay 集合",
    "Rows with ffprobe duration metadata": "含 ffprobe 时长元数据的条目",
    "Fully annotated gameplay subset": "完整标注的 gameplay 子集",
    "Camera-motion scene clips (0–5s buckets)": "相机运动场景片段（0–5s 桶）",
    "In-scene temporal evolution clips": "场景内时序演化片段",
    "Action-conditioned gameplay rollouts": "动作条件 gameplay rollout",
    "Collection Mix": "集合构成",
    "Temporal Metadata": "时序元数据",
    "Annotation Surface": "标注面",
    "Camera Instructions": "相机指令标签",
    "Distinct instruction labels": "不同指令标签",
    "Primary + secondary scene tags": "主/次场景标签",
    "All scanned image benchmark rows": "全部扫描到的图像基准行",
    "All scanned video benchmark rows": "全部扫描到的视频基准行",
    "Public contracts: `image_static`, `image_dynamic`": "公开契约：`image_static`、`image_dynamic`",
    "Public contracts: `video_static`, `video_dynamic`": "公开契约：`video_static`、`video_dynamic`",
    "Scene layout and camera-consistency evaluation": "场景布局与相机一致性评测",
    "Motion, mask, and temporal-structure diagnostics": "运动、mask 与时序结构诊断",
    "rows with ffprobe duration metadata": "的条目含 ffprobe 时长元数据",
    "rows with supervision sidecars": "的条目附带监督侧车数据",
    "See the [Metrics Overview](benchmark/metrics/index.md) for eligibility rules and aggregation semantics.": "指标适用性与聚合语义见 [指标概览](benchmark/metrics/index.md)。",
    "Caption sidecars attach weather, time-of-day, brightness, crowd-density, and scene-type labels to video and physics rows. Panels below summarize caption coverage by collection before the detailed tables and interactive charts.": "Caption 侧车为视频与 physics 条目附加 weather、time-of-day、brightness、crowd-density 与 scene-type 等标签。下列面板按集合汇总 caption 覆盖情况，再进入明细表与交互图表。",
    "Headline totals": "总量指标",
    "Headline caption counts and normalized label coverage across video and physics collections.": "视频与 physics 集合的 headline caption 计数与归一化标签覆盖。",
    "Primary caption inventory and normalized label coverage.": "主 caption 库存与归一化标签覆盖。",
    "Label breadth": "标签广度",
    "Weather Tagged": "Weather 标注",
    "Time-Of-Day Tagged": "Time-Of-Day 标注",
    "Primary Scene Types": "主场景类型",
    "Video and physics rows with captions": "带 caption 的视频与 physics 条目",
    "Videos with normalized weather labels": "含归一化 weather 标签的视频",
    "Videos with normalized time-of-day labels": "含归一化 time-of-day 标签的视频",
    "Unique normalized primary scene labels": "不同归一化主场景标签",
    "% captioned": "% 含 caption",
    "Distinct weather, time-of-day, brightness, and crowd-density labels in inventory.": "库存中不同的 weather、time-of-day、brightness 与 crowd-density 标签。",
    "Weather Labels": "Weather 标签",
    "Distinct normalized weather labels": "不同归一化 weather 标签",
    "Time-Of-Day Labels": "Time-Of-Day 标签",
    "Distinct normalized time-of-day labels": "不同归一化 time-of-day 标签",
    "Brightness Labels": "Brightness 标签",
    "Distinct normalized brightness labels": "不同归一化 brightness 标签",
    "Crowd Density Labels": "Crowd Density 标签",
    "Distinct normalized crowd-density labels": "不同归一化 crowd-density 标签",
    "Caption coverage by collection": "按集合的 caption 覆盖",
    "How caption sidecars distribute across static, physics, dynamic, and gameplay collections.": "caption 侧车在 static、physics、dynamic 与 gameplay 集合中的分布。",
    "Static Videos": "静态视频",
    "Physics Videos": "Physics 视频",
    "Dynamic Videos": "动态视频",
    "Camera-motion clips with sparse caption sidecars": "caption 侧车稀疏的相机运动片段",
    "Simulator-backed physics clips with partial caption coverage": "部分 caption 覆盖的仿真 physics 片段",
    "In-scene temporal evolution clips without caption sidecars": "尚无 caption 侧车的场景内时序演化片段",
    "Fully annotated gameplay rollouts": "完整标注的 gameplay rollout",
    "captioned rows": "条含 caption",
    "## Caption Coverage": "## Caption 覆盖",
    "## Weather Distribution": "## Weather 分布",
    "## Time-Of-Day Distribution": "## Time-Of-Day 分布",
    "## Brightness And Crowd Density": "## Brightness 与 Crowd Density",
    "## Primary Scene Types": "## 主场景类型分布",
    "## Weather Versus Time Of Day": "## Weather 与 Time-Of-Day 交叉",
    "## Brightness Versus Crowd Density": "## Brightness 与 Crowd Density 交叉",
    'title="Caption Coverage"': 'title="Caption 覆盖"',
    'title="Weather Distribution"': 'title="Weather 分布"',
    'title="Time-Of-Day Distribution"': 'title="Time-Of-Day 分布"',
    'title="Brightness And Crowd Density"': 'title="Brightness 与 Crowd Density"',
    'title="Primary Scene Types"': 'title="主场景类型分布"',
    'title="Weather Versus Time Of Day"': 'title="Weather 与 Time-Of-Day 交叉"',
    'title="Brightness Versus Crowd Density"': 'title="Brightness 与 Crowd Density 交叉"',
    "## Collection Volume": "## 集合规模分布",
    "## Benchmark Hierarchy": "## 基准分类层级",
    'title="Collection Volume"': 'title="集合规模分布"',
    'title="Benchmark Hierarchy"': 'title="基准分类层级"',
    "## Image Distribution": "## 图像分布",
    "## Static Scene Hierarchy": "## 静态场景层级",
    "## Mask Coverage": "## Mask 覆盖",
    'title="Image Distribution"': 'title="图像分布"',
    'title="Static Scene Hierarchy"': 'title="静态场景层级"',
    'title="Mask Coverage"': 'title="Mask 覆盖"',
    "## Video Collection Mix": "## 视频集合构成",
    "## Duration Mix": "## 时长分布",
    "## Static Video Hierarchy": "## 静态视频层级",
    "## Scene Tag Mix": "## 场景标签分布",
    "## Camera Motion Labels": "## 相机运动标签",
    "## Resolution Mix": "## 分辨率分布",
    'title="Video Collection Mix"': 'title="视频集合构成"',
    'title="Duration Mix"': 'title="时长分布"',
    'title="Static Video Hierarchy"': 'title="静态视频层级"',
    'title="Scene Tag Mix"': 'title="场景标签分布"',
    'title="Camera Motion Labels"': 'title="相机运动标签"',
    'title="Resolution Mix"': 'title="分辨率分布"',
    '<p class="doc-provenance">Snapshot note: panels and charts below use cached `docs/generated/` artifacts in this checkout because live dataset roots were unavailable during the last `worldarena build-docs` run. Re-run against local dataset roots when a live rescan is available.</p>': '<p class="doc-provenance">快照说明：由于最近一次 `worldarena build-docs` 运行时本地数据根目录不可用，下列面板与图表使用本 checkout 中缓存的 `docs/generated/` 产物。当本地数据根目录可用时，请重新运行该命令以进行实时扫描。</p>',
    '<p class="section-lede">Interactive plots summarizing collection mix, temporal metadata, annotation coverage, and label distributions.</p>': '<p class="section-lede">交互式图表汇总集合构成、时序元数据、标注覆盖与标签分布。</p>',
    "Distribution of video rows across static, dynamic, and gameplay collections.": "静态、动态与 gameplay 集合的视频条目分布。",
    "Duration bucket counts grouped by collection.": "按集合分组的时长桶计数。",
    "Sunburst hierarchy over style, environment, scene, and duration for static clips.": "静态片段在风格、环境、场景与时长维度上的 sunburst 层级。",
    "Fraction of rows with complete annotation sidecars per collection.": "各集合中带完整标注侧车的条目占比。",
    "Top resolution labels stacked by collection.": "按集合堆叠的主要分辨率标签。",
    "Primary and secondary scene-type tags in caption sidecars.": "Caption 侧车中的主/次场景类型标签。",
    "Most frequent camera-instruction labels in static video rows.": "静态视频条目中最常见的相机指令标签。",
    'Representative clips drawn from static, dynamic, and gameplay collections.': "取自静态、动态与 gameplay 集合的代表性片段。",
}


def _localize_zh(content: str) -> str:
    for source, target in ZH_REPLACEMENTS.items():
        content = content.replace(source, target)
    return content


def _taxonomy_diagram(collection_summary: pd.DataFrame) -> str:
    counts = {
        row["collection"]: int(row["items"])
        for row in collection_summary.to_dict(orient="records")
    }
    return f"""```mermaid
graph TD
  root[WorldAtlas Arena]
  root --> images[Image]
  root --> videos[Video]
  root --> physics[Physics]
  images --> static_images[Static Images • {counts.get('static_images', 0)}]
  images --> dynamic_images[Dynamic Images • {counts.get('dynamic_images', 0)}]
  videos --> static_videos[Static Videos • {counts.get('static_videos', 0)}]
  videos --> dynamic_videos[Dynamic Videos • {counts.get('dynamic_videos', 0)}]
  videos --> aaa_games[AAA Games • {counts.get('aaa_games', 0)}]
  physics --> physics_videos[Physics Videos • {counts.get('physics_videos', 0)}]
```"""


def _overview_page(
    summary: dict[str, Any],
    charts: ChartWriter,
    data_prefix: str,
    sample_assets: dict[str, list[dict[str, str]]],
    asset_prefix: str,
) -> str:
    frame = summary["frame"]
    images = summary["images"]
    videos = summary["videos"]
    physics = summary["physics"]
    collection_summary = summary["collection_summary"]
    project = summary["project"]
    project_root = project.config_path.parent.parent
    benchmark_stats = _load_benchmark_overview_stats(project_root)

    benchmark_frame = frame[frame["group"] == "benchmark"] if not frame.empty else frame
    collection_counts = {
        row["collection"]: int(row["items"])
        for row in collection_summary.to_dict(orient="records")
    }
    benchmark_total = max(
        int(benchmark_frame.shape[0]),
        int(collection_summary.loc[collection_summary["group"] == "benchmark", "items"].sum())
        if not collection_summary.empty
        else 0,
    )
    annotated_total = int(collection_summary["annotated"].fillna(0).sum())
    masked_total = int(collection_summary["masked"].fillna(0).sum())
    image_total = max(
        int(images.shape[0]),
        collection_counts.get("static_images", 0) + collection_counts.get("dynamic_images", 0),
    )
    video_total = max(
        int(videos.shape[0]),
        collection_counts.get("static_videos", 0)
        + collection_counts.get("dynamic_videos", 0)
        + collection_counts.get("aaa_games", 0),
    )
    physics_total = max(
        int(physics.shape[0]),
        collection_counts.get("physics_videos", 0),
    )
    static_total = collection_counts.get("static_images", 0) + collection_counts.get("static_videos", 0)
    dynamic_total = collection_counts.get("dynamic_images", 0) + collection_counts.get("dynamic_videos", 0)
    interaction_total = collection_counts.get("aaa_games", 0)
    physics_total = collection_counts.get("physics_videos", 0)

    def _share(value: int) -> str:
        if benchmark_total <= 0:
            return "0%"
        return f"{100 * value / benchmark_total:.1f}%"

    hero_cards = _overview_hero_cards(
        [
            ("Benchmark Items", _format_number(benchmark_total), "Primary rows across image, video, and physics suites", "primary"),
            ("Formal Suites", _format_number(benchmark_stats["formal_suites"]), "Public benchmark contracts in `config/benchmark.yaml`", "green"),
            ("Documented Metrics", _format_number(benchmark_stats["total"]), "Public metric inventory in the benchmark docs", "warm"),
            ("Evaluation Dimensions", "6", "Axes aggregated in `per_dimension_summary.json`", "blue"),
            ("Registered Models", _format_number(benchmark_stats["model_count"]), "Adapter configs under `config/models/`", "ink"),
            ("Annotated Rows", _format_number(annotated_total), f"{_share(annotated_total)} of benchmark rows carry supervision sidecars", "green"),
        ]
    )

    modality_cards = _metric_cards(
        [
            ("Images", _format_number(image_total), "Static and dynamic image assets"),
            ("Videos", _format_number(video_total), "Static, dynamic, and gameplay videos"),
            ("Physics", _format_number(physics_total), "Physics-focused video assets"),
            ("Masked Dynamics", _format_number(masked_total), "Dynamic image and gameplay rows with segmentation masks"),
        ]
    )

    regime_cards = _overview_regime_cards(
        [
            ("Static world generation", _format_number(static_total), _share(static_total), "Camera or layout change with scene preservation"),
            ("Dynamic world generation", _format_number(dynamic_total), _share(dynamic_total), "In-scene temporal evolution under controlled viewpoints"),
            ("Interaction-conditioned", _format_number(interaction_total), _share(interaction_total), "Gameplay and action-conditioned rollouts"),
            ("Physics diagnostics", _format_number(physics_total), _share(physics_total), "Reference motion and simulator-backed physics tracks"),
        ]
    )

    suite_rows = [
        {
            "suite": suite,
            "modality": "image → video" if suite.startswith("image_") else "video → video",
            "regime": suite.split("_", 1)[1],
            "metrics": benchmark_stats["suite_metric_counts"][suite],
        }
        for suite in benchmark_stats["suite_metric_counts"]
    ]
    suite_table = pd.DataFrame(suite_rows).to_markdown(index=False)

    dimension_bars = _overview_dimension_bars(benchmark_stats["dimensions"])
    metric_split = _overview_metric_split(
        benchmark_stats["official"],
        benchmark_stats["diagnostic"],
        benchmark_stats["experimental_metrics"],
    )

    collection_fig = _styled_figure(
        px.bar(
            collection_summary,
            x="collection",
            y="items",
            color="modality",
            barmode="group",
        ),
        "Collection Sizes",
    )

    hierarchy_frame = benchmark_frame.copy()
    hierarchy_frame["style"] = hierarchy_frame["style"].fillna("Unspecified")
    hierarchy_frame["motion_regime"] = hierarchy_frame["motion_regime"].fillna("Unspecified")
    hierarchy_fig = _styled_figure(
        px.sunburst(
            hierarchy_frame,
            path=["modality", "motion_regime", "style"],
        ),
        "Benchmark Taxonomy",
    )

    chart_sections = "\n\n".join(
        filter(
            None,
            [
                charts.render("overview", "collection-volume", "Collection Volume", collection_fig),
                charts.render("overview", "benchmark-hierarchy", "Benchmark Hierarchy", hierarchy_fig),
            ],
        )
    )

    provenance = ""
    if summary.get("collection_snapshot") or summary.get("data_source") == "cached":
        provenance = (
            '<p class="doc-provenance">Snapshot note: panels and charts below use cached '
            "`docs/generated/` artifacts in this checkout because live dataset roots were "
            "unavailable during the last `worldarena build-docs` run. Re-run against local "
            "dataset roots when a live rescan is available.</p>\n\n"
        )

    return f"""# Dataset Overview

!!! info "Generated"
    Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets.

{provenance}## Summary

<p class="overview-lede">WorldAtlas Arena combines a multi-regime benchmark corpus, a six-axis evaluation contract, and a registry of world-model adapters. The panels below summarize scale from complementary angles—media inventory, task regime, metric surface, and model coverage—before the collection tables and sample galleries.</p>

## Benchmark Scale

{_overview_scale_page(
    "Headline totals, media inventory, task regimes, and benchmark contract surface.",
    [
        (
            "Headline totals",
            "Primary inventory counts, suites, metrics, and model coverage.",
            hero_cards,
        ),
        (
            "Media inventory",
            "Image, video, and physics assets in the benchmark corpus.",
            modality_cards,
        ),
        (
            "Task regimes",
            "How benchmark rows split across static, dynamic, interaction, and physics tracks.",
            regime_cards,
        ),
    ],
)}

## Evaluation Framework

Formal suites separate static, dynamic, and interaction-conditioned generation while sharing a common reporting contract. Each suite declares an applicable metric vector in `config/benchmark.yaml`.

{suite_table}

### Metrics by Dimension

{dimension_bars}

### Official vs Diagnostic Surface

{metric_split}

See the [Metrics Overview](benchmark/metrics/index.md) for eligibility rules, normalization, and aggregation semantics.

## Topology

{_taxonomy_diagram(collection_summary)}

## Collection Summary

{_frame_table(collection_summary, ["group", "collection", "modality", "items", "annotated", "masked"])}

{chart_sections}

{_sample_gallery("Sample Image Gallery", sample_assets.get("image", [])[:12], asset_prefix)}

{_sample_gallery("Sample Video Gallery", sample_assets.get("video", [])[:8], asset_prefix)}

{_sample_gallery("Sample Physics Gallery", sample_assets.get("physics", [])[:10], asset_prefix)}
"""


def _image_taxonomy_diagram(static_count: int, dynamic_count: int) -> str:
    return f"""```mermaid
graph TD
  root[Image Corpus • {static_count + dynamic_count}]
  root --> static[Static Images • {static_count}]
  root --> dynamic[Dynamic Images • {dynamic_count}]
  static --> indoor[Indoor Scenes]
  static --> outdoor[Outdoor Scenes]
  dynamic --> motion[Motion Categories]
  dynamic --> masks[Segmentation Masks • {dynamic_count}]
```"""


def _video_taxonomy_diagram(
    static_count: int,
    dynamic_count: int,
    gameplay_count: int,
) -> str:
    total = static_count + dynamic_count + gameplay_count
    return f"""```mermaid
graph TD
  root[Video Corpus • {total}]
  root --> static_videos[Static Videos • {static_count}]
  root --> dynamic_videos[Dynamic Videos • {dynamic_count}]
  root --> aaa_games[AAA Games • {gameplay_count}]
  static_videos --> camera[Camera-motion Clips]
  dynamic_videos --> temporal[In-scene Evolution]
  aaa_games --> action[Action-conditioned Rollouts]
```"""


def _suite_metric_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No suite rows available._"
    return pd.DataFrame(rows).to_markdown(index=False)


def _images_page(
    summary: dict[str, Any],
    charts: ChartWriter,
    sample_assets: dict[str, list[dict[str, str]]],
    asset_prefix: str,
    data_prefix: str = "../generated/data",
) -> str:
    images = summary["images"].copy()
    if images.empty:
        return "# Images\n\n_No image records found._"

    project = summary["project"]
    project_root = project.config_path.parent.parent
    benchmark_stats = _load_benchmark_overview_stats(project_root)
    image_suites = ["image_static", "image_dynamic"]
    image_suite_metrics = {
        suite: benchmark_stats["suite_metric_counts"].get(suite, 0) for suite in image_suites
    }
    unique_image_metrics = len(
        {
            metric
            for suite in image_suites
            for metric in yaml.safe_load(
                (project_root / "config" / "benchmark.yaml").read_text(encoding="utf-8")
            )["benchmark"]["metrics_by_suite"].get(suite, [])
        }
    )

    total_images = int(images.shape[0])
    static_count = int((images["motion_regime"] == "static").sum())
    dynamic_count = int((images["motion_regime"] == "dynamic").sum())
    masked_count = int(images["has_mask"].sum())
    style_count = int(images["style"].nunique(dropna=True))
    environment_count = int(images["environment"].nunique(dropna=True))
    motion_category_count = int(images["motion_category"].nunique(dropna=True))
    resolution_count = int(images["resolution_label"].nunique(dropna=True))

    def _share(value: int) -> str:
        if total_images <= 0:
            return "0%"
        return f"{100 * value / total_images:.1f}%"

    hero_cards = _overview_hero_cards(
        [
            ("Image Rows", _format_number(total_images), "All scanned image benchmark rows", "primary"),
            ("Mask Coverage", _format_number(masked_count), f"{_share(masked_count)} of image rows carry segmentation masks", "blue"),
            ("Formal Image Suites", _format_number(len(image_suites)), "Public contracts: `image_static`, `image_dynamic`", "ink"),
            ("Applicable Image Metrics", _format_number(unique_image_metrics), "Unique metrics across `image_static` and `image_dynamic`", "green"),
        ]
    )

    scale_cards = _metric_cards(
        [
            ("Visual Styles", _format_number(style_count), "Photorealistic + stylized still frames"),
            ("Scene Environments", _format_number(environment_count), "Indoor + outdoor scene buckets"),
            ("Motion Categories", _format_number(motion_category_count), "Articulated, rigid, fluid, and multi-motion labels"),
            ("Unique Resolutions", _format_number(resolution_count), "Distinct resolution labels in inventory"),
        ]
    )

    regime_cards = _overview_regime_cards(
        [
            ("Static Images", _format_number(static_count), _share(static_count), "Scene layout and camera-consistency evaluation"),
            ("Dynamic Images", _format_number(dynamic_count), _share(dynamic_count), "Motion, mask, and temporal-structure diagnostics"),
        ]
    )

    suite_table = _suite_metric_table(
        [
            {
                "suite": suite,
                "regime": "static" if suite.endswith("_static") else "dynamic",
                "items": static_count if suite == "image_static" else dynamic_count,
                "metrics": image_suite_metrics[suite],
            }
            for suite in image_suites
        ]
    )

    image_style = (
        images.groupby(["motion_regime", "style"], dropna=False)
        .size()
        .reset_index(name="items")
    )
    style_fig = _styled_figure(
        px.bar(
            image_style,
            x="style",
            y="items",
            color="motion_regime",
            barmode="group",
        ),
        "Image Counts by Style and Motion Regime",
    )

    static_images = images[images["motion_regime"] == "static"].copy()
    static_images["scene"] = static_images["scene"].fillna("Unspecified")
    static_tree = _styled_figure(
        px.sunburst(
            static_images,
            path=["style", "environment", "scene"],
        ),
        "Static Image Scene Distribution",
    ) if not static_images.empty else None

    dynamic_images = images[images["motion_regime"] == "dynamic"].copy()
    mask_coverage = (
        dynamic_images.groupby(["style", "motion_category"], dropna=False)
        .agg(items=("path", "count"), masked=("has_mask", "sum"))
        .reset_index()
    )
    if not mask_coverage.empty:
        mask_coverage["mask_rate"] = (mask_coverage["masked"] / mask_coverage["items"]).round(4)
    mask_fig = _styled_figure(
        px.bar(
            mask_coverage,
            x="motion_category",
            y="mask_rate",
            color="style",
            barmode="group",
            range_y=[0, 1],
        ),
        "Dynamic Image Mask Coverage",
    ) if not mask_coverage.empty else None

    resolution_counts = (
        images.groupby(["resolution_label"], dropna=False)
        .size()
        .reset_index(name="items")
        .sort_values("items", ascending=False)
        .head(15)
    )
    resolution_fig = _styled_figure(
        px.bar(resolution_counts, x="resolution_label", y="items"),
        "Top Image Resolutions",
    )

    top_scenes = (
        static_images.groupby(["style", "environment", "scene"], dropna=False)
        .size()
        .reset_index(name="items")
        .sort_values("items", ascending=False)
    )

    chart_sections = "\n\n".join(
        filter(
            None,
            [
                charts.render("images", "distribution", "Image Distribution", style_fig),
                charts.render("images", "static-scene-hierarchy", "Static Scene Hierarchy", static_tree),
                charts.render("images", "mask-coverage", "Mask Coverage", mask_fig),
                charts.render("images", "resolution-mix", "Resolution Mix", resolution_fig),
            ],
        )
    )

    return f"""# Image Analytics

!!! info "Generated"
    Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets.

## Summary

<p class="overview-lede">The image corpus spans static scene frames and motion-labelled dynamic stills used by the `image_static` and `image_dynamic` benchmark suites. Panels below summarize inventory scale, taxonomy coverage, mask supervision, and applicable metrics before the detailed tables and interactive charts.</p>

## Collection Scale

{_overview_scale_page(
    "Headline totals, taxonomy breadth, and static/dynamic regime mix for the image corpus.",
    [
        ("Headline totals", "Primary inventory counts and benchmark contract surface.", hero_cards),
        ("Taxonomy breadth", "Distinct style, environment, motion, and resolution labels in inventory.", scale_cards),
        ("Regime mix", "How rows split between static layout evaluation and dynamic motion diagnostics.", regime_cards),
    ],
)}

## Benchmark Suites

Image suites declare applicable metrics in `config/benchmark.yaml`. Static rows stress layout and camera consistency; dynamic rows add motion and mask-aware diagnostics.

{suite_table}

See the [Metrics Overview](benchmark/metrics/index.md) for eligibility rules and aggregation semantics.

## Taxonomy

{_image_taxonomy_diagram(static_count, dynamic_count)}

## Static Scene Summary

{_frame_table(top_scenes, ["style", "environment", "scene", "items"], limit=15)}

## Dynamic Mask Summary

{_frame_table(mask_coverage, ["style", "motion_category", "items", "masked", "mask_rate"], limit=15)}

{_sample_gallery("Sample Image Gallery", sample_assets.get("image", []), asset_prefix)}

{chart_sections}
"""


def _videos_page(
    summary: dict[str, Any],
    charts: ChartWriter,
    sample_assets: dict[str, list[dict[str, str]]],
    asset_prefix: str,
    data_prefix: str = "../generated/data",
) -> str:
    videos = summary["videos"].copy()
    if videos.empty:
        return "# Videos\n\n_No video records found._"

    project = summary["project"]
    project_root = project.config_path.parent.parent
    benchmark_stats = _load_benchmark_overview_stats(project_root)
    video_suites = ["video_static", "video_dynamic"]
    video_suite_metrics = {
        suite: benchmark_stats["suite_metric_counts"].get(suite, 0) for suite in video_suites
    }
    unique_video_metrics = len(
        {
            metric
            for suite in video_suites
            for metric in yaml.safe_load(
                (project_root / "config" / "benchmark.yaml").read_text(encoding="utf-8")
            )["benchmark"]["metrics_by_suite"].get(suite, [])
        }
    )

    total_videos = int(videos.shape[0])
    benchmark_videos = videos[videos["group"] == "benchmark"].copy()
    static_video_count = int((videos["collection"] == "static_videos").sum())
    dynamic_video_count = int((videos["collection"] == "dynamic_videos").sum())
    gameplay_count = int((videos["collection"] == "aaa_games").sum())
    duration_available = int(videos["duration_seconds"].notna().sum())
    resolution_count = int(videos["resolution_label"].nunique(dropna=True))
    instruction_summary = summary["instruction_summary"].copy()
    instruction_count = int(instruction_summary.shape[0]) if not instruction_summary.empty else 0
    scene_tag_summary = summary["scene_tag_summary"].copy()
    scene_tag_count = int(scene_tag_summary.shape[0]) if not scene_tag_summary.empty else 0
    annotation_summary = summary["annotation_summary"].copy()
    annotated_total = int(annotation_summary["annotated"].fillna(0).sum()) if not annotation_summary.empty else 0

    def _share(value: int) -> str:
        if total_videos <= 0:
            return "0%"
        return f"{100 * value / total_videos:.1f}%"

    hero_cards = _overview_hero_cards(
        [
            ("Video Rows", _format_number(total_videos), "All scanned video benchmark rows", "primary"),
            ("Exact Duration Available", _format_number(duration_available), f"{_share(duration_available)} of rows with ffprobe duration metadata", "blue"),
            ("Formal Video Suites", _format_number(len(video_suites)), "Public contracts: `video_static`, `video_dynamic`", "ink"),
            ("Applicable Video Metrics", _format_number(unique_video_metrics), "Unique metrics across `video_static` and `video_dynamic`", "green"),
        ]
    )

    scale_cards = _metric_cards(
        [
            ("Collection Mix", "3", "Static, dynamic, and gameplay collections"),
            ("Unique Resolutions", _format_number(resolution_count), "Distinct resolution labels in inventory"),
            ("Annotated Rows", _format_number(annotated_total), f"{_share(annotated_total)} of rows with supervision sidecars"),
            ("Camera Instructions", _format_number(instruction_count), "Distinct instruction labels"),
        ]
    )

    regime_cards = _overview_regime_cards(
        [
            ("Static Videos", _format_number(static_video_count), _share(static_video_count), "Camera-motion scene clips (0–5s buckets)"),
            ("Dynamic Videos", _format_number(dynamic_video_count), _share(dynamic_video_count), "In-scene temporal evolution clips"),
            ("AAA Games", _format_number(gameplay_count), _share(gameplay_count), "Action-conditioned gameplay rollouts"),
        ]
    )

    suite_table = _suite_metric_table(
        [
            {
                "suite": suite,
                "regime": "static" if suite.endswith("_static") else "dynamic",
                "items": static_video_count if suite == "video_static" else dynamic_video_count,
                "metrics": video_suite_metrics[suite],
            }
            for suite in video_suites
        ]
    )

    collection_counts = (
        videos.groupby(["collection"], dropna=False).size().reset_index(name="items")
    )
    collection_fig = _styled_figure(
        px.bar(collection_counts, x="collection", y="items"),
        "Video Counts by Collection",
    )

    duration_counts = (
        benchmark_videos.groupby(["collection", "duration_bucket_effective"], dropna=False)
        .size()
        .reset_index(name="items")
    )
    duration_fig = _styled_figure(
        px.bar(
            duration_counts,
            x="duration_bucket_effective",
            y="items",
            color="collection",
            barmode="group",
        ),
        "Duration Bucket Distribution",
    )

    static_videos = benchmark_videos[benchmark_videos["motion_regime"] == "static"].copy()
    static_videos["scene"] = static_videos["scene"].fillna("Unspecified")
    static_tree = _styled_figure(
        px.sunburst(
            static_videos,
            path=["style", "environment", "scene", "duration_bucket_effective"],
        ),
        "Static Video Taxonomy",
    ) if not static_videos.empty else None

    annotation_fig = _styled_figure(
        px.bar(
            annotation_summary,
            x="collection",
            y="annotation_rate",
            range_y=[0, 1],
        ),
        "Annotation Coverage by Collection",
    ) if not annotation_summary.empty else None

    scene_tags = scene_tag_summary.head(12).copy()
    scene_tag_fig = _styled_figure(
        px.bar(
            scene_tags,
            x="caption_scene_type_primary",
            y="items",
            color="caption_scene_type_secondary",
        ),
        "Top Scene Tags",
    ) if not scene_tags.empty else None

    instruction_top = instruction_summary.head(12).copy()
    instruction_fig = _styled_figure(
        px.bar(
            instruction_top,
            x="instruction_label",
            y="items",
        ),
        "Top Camera Instruction Labels",
    ) if not instruction_top.empty else None

    top_resolutions = (
        videos.groupby(["collection", "resolution_label"], dropna=False)
        .size()
        .reset_index(name="items")
        .sort_values("items", ascending=False)
    )

    top_resolutions_chart = top_resolutions.head(15).copy()
    resolution_mix_fig = _styled_figure(
        px.bar(
            top_resolutions_chart,
            x="resolution_label",
            y="items",
            color="collection",
            barmode="stack",
        ),
        "Top Video Resolutions",
    ) if not top_resolutions_chart.empty else None

    chart_sections = "\n\n".join(
        filter(
            None,
            [
                charts.render(
                    "videos",
                    "collection-mix",
                    "Video Collection Mix",
                    collection_fig,
                    academic=True,
                    caption="Distribution of video rows across static, dynamic, and gameplay collections.",
                ),
                charts.render(
                    "videos",
                    "duration-mix",
                    "Duration Mix",
                    duration_fig,
                    academic=True,
                    caption="Duration bucket counts grouped by collection.",
                ),
                charts.render(
                    "videos",
                    "static-video-hierarchy",
                    "Static Video Hierarchy",
                    static_tree,
                    academic=True,
                    caption="Sunburst hierarchy over style, environment, scene, and duration for static clips.",
                ),
                charts.render(
                    "videos",
                    "annotation-coverage",
                    "Annotation Coverage",
                    annotation_fig,
                    academic=True,
                    caption="Fraction of rows with complete annotation sidecars per collection.",
                ),
                charts.render(
                    "videos",
                    "resolution-mix",
                    "Resolution Mix",
                    resolution_mix_fig,
                    academic=True,
                    caption="Top resolution labels stacked by collection.",
                ),
                charts.render(
                    "videos",
                    "scene-tag-mix",
                    "Scene Tag Mix",
                    scene_tag_fig,
                    academic=True,
                    caption="Primary and secondary scene-type tags in caption sidecars.",
                ),
                charts.render(
                    "videos",
                    "camera-motion-labels",
                    "Camera Motion Labels",
                    instruction_fig,
                    academic=True,
                    caption="Most frequent camera-instruction labels in static video rows.",
                ),
            ],
        )
    )

    return f"""# Video Analytics

<p class="doc-provenance">Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets. Snapshot note: panels and charts below use cached `docs/generated/` artifacts in this checkout; re-run against local dataset roots when a live rescan is available.</p>

## Summary

<p class="overview-lede">The video corpus covers camera-motion static clips, in-scene dynamic evolution, and gameplay rollouts evaluated under `video_static`, `video_dynamic`, and auxiliary gameplay tracks. Panels below break down collection mix, temporal metadata, annotation coverage, and benchmark metric surface.</p>

## Collection Scale

{_overview_scale_page(
    "Headline totals, metadata coverage, and collection mix across static, dynamic, and gameplay video tracks.",
    [
        ("Headline totals", "Primary inventory counts and benchmark contract surface.", hero_cards),
        ("Metadata coverage", "Resolution diversity, annotation sidecars, and camera-instruction labels.", scale_cards),
        ("Collection mix", "How rows split across static camera-motion clips, dynamic evolution, and gameplay rollouts.", regime_cards),
    ],
)}

## Benchmark Suites

Video suites span camera-motion static clips, dynamic temporal rollouts, and fully annotated gameplay sequences. Metric vectors differ by regime but share the public reporting contract.

{suite_table}

See the [Metrics Overview](benchmark/metrics/index.md) for eligibility rules and aggregation semantics.

## Taxonomy

{_video_taxonomy_diagram(static_video_count, dynamic_video_count, gameplay_count)}

## Tabular Summaries

### Annotation Coverage

{_frame_table(annotation_summary, ["collection", "items", "annotated", "complete_annotations", "annotation_rate"], limit=15)}

### Resolution Distribution

{_frame_table(top_resolutions, ["collection", "resolution_label", "items"], limit=15)}

## Statistical Exhibits

<p class="section-lede">Interactive plots summarizing collection mix, temporal metadata, annotation coverage, and label distributions.</p>

<div class="analytics-exhibit-grid">

{chart_sections}

</div>

{_sample_gallery("Representative Samples", sample_assets.get("video", [])[:8], asset_prefix, academic=True)}
"""


def _physics_page(
    summary: dict[str, Any],
    charts: ChartWriter,
    sample_assets: dict[str, list[dict[str, str]]],
    asset_prefix: str,
    data_prefix: str,
) -> str:
    physics = summary["physics"].copy()
    if physics.empty:
        return "# Physics Analytics\n\n_No physics records found._"

    dimension_summary = (
        physics.groupby(["physics_group", "physics_dimension"], dropna=False)
        .agg(
            items=("path", "count"),
            metadata=("has_physics_metadata", "sum"),
            avg_duration_seconds=("duration_seconds", "mean"),
            avg_visual_quality=("visual_quality_score", "mean"),
            avg_motion_score=("motion_score_v2", "mean"),
        )
        .reset_index()
        .sort_values(["physics_group", "items"], ascending=[True, False])
    )

    group_summary = (
        physics.groupby("physics_group", dropna=False)
        .agg(
            items=("path", "count"),
            dimensions=("physics_dimension", "nunique"),
            metadata=("has_physics_metadata", "sum"),
        )
        .reset_index()
        .sort_values("items", ascending=False)
    )

    group_fig = _styled_figure(
        px.bar(group_summary, x="physics_group", y="items", color="physics_group"),
        "Physics Videos by Group",
    )
    dimension_fig = _styled_figure(
        px.bar(
            dimension_summary,
            x="physics_dimension",
            y="items",
            color="physics_group",
            barmode="group",
        ),
        "Physics Dimension Coverage",
    )
    duration_frame = physics.dropna(subset=["duration_seconds"]).copy()
    duration_fig = _styled_figure(
        px.histogram(
            duration_frame,
            x="duration_seconds",
            color="physics_group",
            nbins=30,
        ),
        "Physics Duration Distribution",
    ) if not duration_frame.empty else None

    quality_frame = physics.dropna(subset=["visual_quality_score", "motion_score_v2"]).copy()
    quality_fig = _styled_figure(
        px.scatter(
            quality_frame,
            x="motion_score_v2",
            y="visual_quality_score",
            color="physics_group",
            hover_data=["physics_dimension", "physics_label"],
        ),
        "Visual Quality vs Motion Score",
    ) if not quality_frame.empty else None

    cards = _metric_cards(
        [
            ("Physics Rows", _format_number(int(physics.shape[0])), "All scanned physics videos"),
            (
                "Physics Metadata",
                _format_number(int(physics["has_physics_metadata"].sum())),
                "Rows matched to the curated subset CSV",
            ),
            ("Physics Groups", _format_number(int(physics["physics_group"].nunique())), "Top-level physics families"),
            ("Physics Dimensions", _format_number(int(physics["physics_dimension"].nunique())), "Fine-grained physics labels"),
        ]
    )

    quality_summary = (
        physics.dropna(subset=["visual_quality_score", "motion_score_v2"])
        .groupby(["physics_group"], dropna=False)
        .agg(
            items=("path", "count"),
            avg_visual_quality=("visual_quality_score", "mean"),
            avg_motion_score=("motion_score_v2", "mean"),
            avg_selection_score=("selection_score", "mean"),
        )
        .reset_index()
        .sort_values("items", ascending=False)
    )

    chart_sections = "\n\n".join(
        filter(
            None,
            [
                charts.render("physics", "group-mix", "Physics Group Mix", group_fig),
                charts.render("physics", "dimension-coverage", "Dimension Coverage", dimension_fig),
                charts.render("physics", "duration", "Duration Distribution", duration_fig),
                charts.render("physics", "quality-motion", "Quality And Motion Scores", quality_fig),
            ],
        )
    )

    return f"""# Physics Analytics

{cards}

## Physics Dimension Summary

{_frame_table(
    dimension_summary,
    [
        "physics_group",
        "physics_dimension",
        "items",
        "metadata",
        "avg_duration_seconds",
        "avg_visual_quality",
        "avg_motion_score",
    ],
    limit=30,
)}

## Quality And Motion Summary

{_frame_table(
    quality_summary,
    ["physics_group", "items", "avg_visual_quality", "avg_motion_score", "avg_selection_score"],
    limit=10,
)}

{_sample_gallery("Sample Physics Gallery", sample_assets.get("physics", []), asset_prefix)}

{chart_sections}
"""


def _caption_distribution_table(
    distribution_summary: pd.DataFrame,
    dimension: str,
    limit: int = 12,
) -> str:
    subset = distribution_summary[distribution_summary["dimension"] == dimension].copy()
    if subset.empty:
        return "_No rows available._"
    subset["share"] = subset["share"].map(lambda value: f"{value:.2%}")
    return _frame_table(subset, ["value", "items", "share"], limit=limit)


CAPTION_COLLECTION_LABELS = {
    "static_videos": ("Static Videos", "Camera-motion clips with sparse caption sidecars"),
    "physics_videos": ("Physics Videos", "Simulator-backed physics clips with partial caption coverage"),
    "dynamic_videos": ("Dynamic Videos", "In-scene temporal evolution clips without caption sidecars"),
    "aaa_games": ("AAA Games", "Fully annotated gameplay rollouts"),
}


def _caption_dimension_count(distribution_summary: pd.DataFrame, dimension: str) -> int:
    subset = distribution_summary[distribution_summary["dimension"] == dimension]
    if subset.empty:
        return 0
    return int(subset["value"].nunique())


def _captions_page(summary: dict[str, Any], charts: ChartWriter, data_prefix: str) -> str:
    caption_videos = summary["caption_videos"].copy()
    caption_distribution_summary = summary["caption_distribution_summary"].copy()
    caption_collection_summary = summary["caption_collection_summary"].copy()
    caption_coverage_summary = summary["caption_coverage_summary"].copy()
    caption_coverage_long = summary["caption_coverage_long"].copy()
    caption_weather_time_summary = summary["caption_weather_time_summary"].copy()
    caption_brightness_crowd_summary = summary["caption_brightness_crowd_summary"].copy()

    hero_cards = _overview_hero_cards(
        [
            (
                "Captioned Media",
                _format_number(int(caption_videos.shape[0])),
                "Video and physics rows with captions",
                "primary",
            ),
            (
                "Weather Tagged",
                _format_number(int(caption_videos["caption_weather"].notna().sum())),
                "Videos with normalized weather labels",
                "blue",
            ),
            (
                "Time-Of-Day Tagged",
                _format_number(int(caption_videos["caption_time_of_day"].notna().sum())),
                "Videos with normalized time-of-day labels",
                "warm",
            ),
            (
                "Primary Scene Types",
                _format_number(int(caption_videos["caption_scene_type_primary"].dropna().nunique())),
                "Unique normalized primary scene labels",
                "ink",
            ),
        ]
    )

    label_cards = _metric_cards(
        [
            (
                "Weather Labels",
                _format_number(_caption_dimension_count(caption_distribution_summary, "weather")),
                "Distinct normalized weather labels",
            ),
            (
                "Time-Of-Day Labels",
                _format_number(_caption_dimension_count(caption_distribution_summary, "time_of_day")),
                "Distinct normalized time-of-day labels",
            ),
            (
                "Brightness Labels",
                _format_number(_caption_dimension_count(caption_distribution_summary, "brightness")),
                "Distinct normalized brightness labels",
            ),
            (
                "Crowd Density Labels",
                _format_number(_caption_dimension_count(caption_distribution_summary, "crowd_density")),
                "Distinct normalized crowd-density labels",
            ),
        ]
    )

    collection_cards = []
    for row in caption_coverage_summary.itertuples(index=False):
        label, note = CAPTION_COLLECTION_LABELS.get(
            row.collection,
            (str(row.collection).replace("_", " ").title(), "Collection caption coverage"),
        )
        caption_rate = float(row.caption_rate)
        collection_cards.append(
            (
                label,
                _format_number(int(row.captions)),
                f"{caption_rate:.1%} captioned",
                note,
            )
        )
    regime_cards = _overview_regime_cards(collection_cards)

    coverage_fig = _styled_figure(
        px.bar(
            caption_coverage_long,
            x="collection",
            y="rate",
            color="dimension",
            barmode="group",
            range_y=[0, 1],
        ),
        "Caption Coverage By Collection",
    ) if not caption_coverage_long.empty else None

    weather_by_collection = caption_collection_summary[
        caption_collection_summary["dimension"] == "weather"
    ].copy()
    weather_fig = _styled_figure(
        px.bar(
            weather_by_collection,
            x="collection",
            y="items",
            color="value",
            barmode="stack",
        ),
        "Weather Distribution By Collection",
    ) if not weather_by_collection.empty else None

    time_of_day_by_collection = caption_collection_summary[
        caption_collection_summary["dimension"] == "time_of_day"
    ].copy()
    time_of_day_fig = _styled_figure(
        px.bar(
            time_of_day_by_collection,
            x="collection",
            y="items",
            color="value",
            barmode="stack",
        ),
        "Time-Of-Day Distribution By Collection",
    ) if not time_of_day_by_collection.empty else None

    attribute_mix = caption_distribution_summary[
        caption_distribution_summary["dimension"].isin(["brightness", "crowd_density"])
    ].copy()
    attribute_mix["dimension_label"] = attribute_mix["dimension"].map(CAPTION_DIMENSION_LABELS)
    attribute_fig = _styled_figure(
        px.bar(
            attribute_mix,
            x="value",
            y="items",
            color="dimension_label",
            facet_row="dimension_label",
        ),
        "Brightness And Crowd Density Mix",
    ) if not attribute_mix.empty else None

    scene_distribution = caption_distribution_summary[
        caption_distribution_summary["dimension"] == "scene_type_primary"
    ].copy()
    top_scene_values = scene_distribution.head(25)["value"].tolist()
    scene_by_collection = caption_collection_summary[
        (caption_collection_summary["dimension"] == "scene_type_primary")
        & (caption_collection_summary["value"].isin(top_scene_values))
    ].copy()
    scene_fig = _styled_figure(
        px.treemap(
            scene_by_collection,
            path=["collection", "value"],
            values="items",
            color="collection",
        ),
        "Primary Scene Types By Collection",
    ) if not scene_by_collection.empty else None

    weather_time_fig = _styled_figure(
        px.density_heatmap(
            caption_weather_time_summary,
            x="caption_weather",
            y="caption_time_of_day",
            z="items",
            histfunc="sum",
            color_continuous_scale="Teal",
        ),
        "Weather Versus Time Of Day",
    ) if not caption_weather_time_summary.empty else None

    brightness_crowd_fig = _styled_figure(
        px.density_heatmap(
            caption_brightness_crowd_summary,
            x="caption_brightness",
            y="caption_crowd_density",
            z="items",
            histfunc="sum",
            color_continuous_scale="Sunsetdark",
        ),
        "Brightness Versus Crowd Density",
    ) if not caption_brightness_crowd_summary.empty else None

    chart_sections = "\n\n".join(
        filter(
            None,
            [
                charts.render("captions", "coverage", "Caption Coverage", coverage_fig),
                charts.render("captions", "weather-by-collection", "Weather Distribution", weather_fig),
                charts.render("captions", "time-of-day-by-collection", "Time-Of-Day Distribution", time_of_day_fig),
                charts.render("captions", "attribute-mix", "Brightness And Crowd Density", attribute_fig),
                charts.render("captions", "scene-types", "Primary Scene Types", scene_fig),
                charts.render("captions", "weather-time", "Weather Versus Time Of Day", weather_time_fig),
                charts.render("captions", "brightness-crowd", "Brightness Versus Crowd Density", brightness_crowd_fig),
            ],
        )
    )

    coverage_table = caption_coverage_summary.copy()
    for column in [
        "caption_rate",
        "weather_rate",
        "time_of_day_rate",
        "brightness_rate",
        "crowd_density_rate",
        "scene_primary_rate",
    ]:
        if column in coverage_table.columns:
            coverage_table[column] = coverage_table[column].map(lambda value: f"{value:.2%}")

    return f"""# Caption Analytics

!!! info "Generated"
    Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets.

## Summary

<p class="overview-lede">Caption sidecars attach weather, time-of-day, brightness, crowd-density, and scene-type labels to video and physics rows. Panels below summarize caption coverage by collection before the detailed tables and interactive charts.</p>

## Collection Scale

{_overview_scale_page(
    "Headline caption counts and normalized label coverage across video and physics collections.",
    [
        ("Headline totals", "Primary caption inventory and normalized label coverage.", hero_cards),
        ("Label breadth", "Distinct weather, time-of-day, brightness, and crowd-density labels in inventory.", label_cards),
        ("Caption coverage by collection", "How caption sidecars distribute across static, physics, dynamic, and gameplay collections.", regime_cards),
    ],
)}

## Coverage Summary

{_frame_table(
    coverage_table,
    [
        "collection",
        "items",
        "captions",
        "caption_rate",
        "weather_rate",
        "time_of_day_rate",
        "brightness_rate",
        "crowd_density_rate",
        "scene_primary_rate",
    ],
    limit=20,
)}

{chart_sections}
"""


def _collections_page(summary: dict[str, Any], data_prefix: str) -> str:
    collection_summary = summary["collection_summary"]
    mask_summary = summary["mask_summary"]
    frame = summary["frame"]

    dimension_summary = (
        frame.groupby("collection", dropna=False)
        .agg(
            items=("path", "count"),
            styles=("style", lambda values: values.dropna().nunique()),
            scenes=("scene", lambda values: values.dropna().nunique()),
            motion_categories=("motion_category", lambda values: values.dropna().nunique()),
            physics_dimensions=("physics_dimension", lambda values: values.dropna().nunique()),
            avg_size_mb=("size_mb", "mean"),
        )
        .reset_index()
    )

    return f"""# Collection Details

## Collection Summary

{_frame_table(collection_summary, ["group", "collection", "modality", "items", "annotated", "masked"], limit=20)}

## Dimension Cardinality

    {_frame_table(dimension_summary, ["collection", "items", "styles", "scenes", "motion_categories", "physics_dimensions", "avg_size_mb"], limit=20)}

## Mask Summary

{_frame_table(mask_summary, ["collection", "modality", "items", "masked", "mask_files", "mask_rate"], limit=20)}
"""


def build_docs_site(
    config_path: Path,
    max_workers: int = 8,
    write_docs: bool = True,
) -> dict[str, Any]:
    project = load_project_config(config_path)
    frame = build_dataset_frame(project, max_workers=max_workers)
    data_source = "live"
    if frame.empty:
        frame = load_cached_dataset_frame(project)
        if not frame.empty:
            data_source = "cached"

    summary = build_summary_tables(frame)
    collection_summary, used_snapshot = merge_collection_snapshot(summary["collection_summary"])
    summary["collection_summary"] = collection_summary
    summary["collection_snapshot"] = used_snapshot
    summary["data_source"] = "snapshot" if used_snapshot else data_source
    summary["project"] = project

    if not frame.empty:
        write_artifacts(
            summary=summary,
            output_dir=project.paths.artifacts_dir,
            docs_data_dir=project.paths.docs_data_dir,
        )

    if write_docs:
        ensure_dir(project.paths.docs_generated_dir)
        docs_root = project.paths.docs_generated_dir.parent
        sample_assets = _build_sample_assets(summary, project.paths.docs_generated_dir / "samples")

        chart_writer_en = ChartWriter(
            project.paths.docs_generated_dir / "charts",
            src_prefix="../generated/charts",
        )
        chart_writer_en.reset()
        chart_writer_zh = ChartWriter(
            project.paths.docs_generated_dir / "charts",
            src_prefix="../../generated/charts",
        )
        for path in project.paths.docs_generated_dir.glob("*.md"):
            path.unlink()

        pages_en = {
            "overview.md": _overview_page(
                summary,
                chart_writer_en,
                "../generated/data",
                sample_assets,
                "../generated",
            ),
            "images.md": _images_page(summary, chart_writer_en, sample_assets, "../generated"),
            "videos.md": _videos_page(summary, chart_writer_en, sample_assets, "../generated"),
            "physics.md": _physics_page(
                summary,
                chart_writer_en,
                sample_assets,
                "../generated",
                "../generated/data",
            ),
            "captions.md": _captions_page(summary, chart_writer_en, "../generated/data"),
            "collections.md": _collections_page(summary, "../generated/data"),
        }
        pages_zh = {
            "overview.md": _localize_zh(
                _overview_page(
                    summary,
                    chart_writer_zh,
                    "../generated/data",
                    sample_assets,
                    "../../generated",
                )
            ),
            "images.md": _localize_zh(
                _images_page(summary, chart_writer_zh, sample_assets, "../../generated")
            ),
            "videos.md": _localize_zh(
                _videos_page(summary, chart_writer_zh, sample_assets, "../../generated")
            ),
            "physics.md": _localize_zh(
                _physics_page(
                    summary,
                    chart_writer_zh,
                    sample_assets,
                    "../../generated",
                    "../generated/data",
                )
            ),
            "captions.md": _localize_zh(_captions_page(summary, chart_writer_zh, "../generated/data")),
            "collections.md": _localize_zh(_collections_page(summary, "../generated/data")),
        }
        for locale, pages in [("en", pages_en), ("zh", pages_zh)]:
            locale_dir = ensure_dir(docs_root / locale)
            for filename, content in pages.items():
                (locale_dir / filename).write_text(content, encoding="utf-8")

    return summary
