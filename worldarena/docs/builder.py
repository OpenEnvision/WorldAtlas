from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio

from worldarena.common.serialization import ensure_dir
from worldarena.datasets.config import load_project_config
from worldarena.datasets.summarize import (
    CAPTION_DIMENSION_LABELS,
    build_dataset_frame,
    build_summary_tables,
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
        for path in self.chart_dir.iterdir():
            if path.is_file():
                path.unlink()

    def render(self, page_slug: str, chart_slug: str, heading: str, figure: Any) -> str:
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
        return (
            f"## {heading}\n\n"
            f'<iframe class="chart-frame" src="{self.src_prefix}/{filename}" title="{heading}" loading="lazy"></iframe>'
        )


def _artifact_path(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def _taxonomy_diagram(collection_summary: pd.DataFrame) -> str:
    counts = {
        row["collection"]: int(row["items"])
        for row in collection_summary.to_dict(orient="records")
    }
    return f"""```mermaid
graph TD
  root[WorldArena]
  root --> benchmark[Benchmark]
  root --> auxiliary[Auxiliary]
  benchmark --> images[Images]
  benchmark --> videos[Videos]
  images --> static_images[Static Images • {counts.get('static_images', 0)}]
  images --> dynamic_images[Dynamic Images • {counts.get('dynamic_images', 0)}]
  videos --> static_videos[Static Videos • {counts.get('static_videos', 0)}]
  videos --> dynamic_videos[Dynamic Videos • {counts.get('dynamic_videos', 0)}]
  videos --> aaa_games[AAA Games • {counts.get('aaa_games', 0)}]
  auxiliary --> review_assets[Review Assets • {counts.get('review_assets', 0)}]
```"""


def _overview_page(summary: dict[str, Any], charts: ChartWriter, data_prefix: str) -> str:
    frame = summary["frame"]
    images = summary["images"]
    videos = summary["videos"]
    collection_summary = summary["collection_summary"]

    benchmark_frame = frame[frame["group"] == "benchmark"]
    cards = _metric_cards(
        [
            ("Benchmark Items", _format_number(int(benchmark_frame.shape[0])), "Primary benchmark rows"),
            ("Images", _format_number(int(images.shape[0])), "Static and dynamic image assets"),
            ("Videos", _format_number(int(videos.shape[0])), "Benchmark and auxiliary video assets"),
            (
                "Annotated Videos",
                _format_number(int(videos["has_annotation"].sum())),
                "Rows with SpatialVID-style metadata",
            ),
        ]
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

    return f"""# Dataset Overview

!!! info "Generated"
    Run `worldarena build-docs --config config/datasets.yaml` after adding, deleting, or reorganizing source assets.

{cards}

## Topology

{_taxonomy_diagram(collection_summary)}

## Collection Summary

{_frame_table(collection_summary, ["group", "collection", "modality", "items", "annotated", "masked"])}

{chart_sections}

## Downloadable Artifacts

- [`summary.json`]({_artifact_path(data_prefix, "summary.json")})
- [`all_records.csv`]({_artifact_path(data_prefix, "all_records.csv")})
- [`image_records.csv`]({_artifact_path(data_prefix, "image_records.csv")})
- [`video_records.csv`]({_artifact_path(data_prefix, "video_records.csv")})
"""


def _images_page(summary: dict[str, Any], charts: ChartWriter) -> str:
    images = summary["images"].copy()
    if images.empty:
        return "# Images\n\n_No image records found._"

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

    cards = _metric_cards(
        [
            ("Image Rows", _format_number(int(images.shape[0])), "All scanned images"),
            ("Static Images", _format_number(int((images["motion_regime"] == "static").sum())), "Scene-centric still frames"),
            ("Dynamic Images", _format_number(int((images["motion_regime"] == "dynamic").sum())), "Motion-labelled still frames"),
            ("Rows With Masks", _format_number(int(images["has_mask"].sum())), "Images linked to segmentation masks"),
        ]
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

{cards}

## Static Scene Summary

{_frame_table(top_scenes, ["style", "environment", "scene", "items"], limit=15)}

## Dynamic Mask Summary

{_frame_table(mask_coverage, ["style", "motion_category", "items", "masked", "mask_rate"], limit=15)}

{chart_sections}
"""


def _videos_page(summary: dict[str, Any], charts: ChartWriter) -> str:
    videos = summary["videos"].copy()
    if videos.empty:
        return "# Videos\n\n_No video records found._"

    collection_counts = (
        videos.groupby(["collection"], dropna=False).size().reset_index(name="items")
    )
    collection_fig = _styled_figure(
        px.bar(collection_counts, x="collection", y="items"),
        "Video Counts by Collection",
    )

    benchmark_videos = videos[videos["group"] == "benchmark"].copy()
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

    annotation_summary = summary["annotation_summary"].copy()
    annotation_fig = _styled_figure(
        px.bar(
            annotation_summary,
            x="collection",
            y="annotation_rate",
            range_y=[0, 1],
        ),
        "Annotation Coverage by Collection",
    ) if not annotation_summary.empty else None

    scene_tags = summary["scene_tag_summary"].head(12).copy()
    scene_tag_fig = _styled_figure(
        px.bar(
            scene_tags,
            x="caption_scene_type_primary",
            y="items",
            color="caption_scene_type_secondary",
        ),
        "Top Scene Tags",
    ) if not scene_tags.empty else None

    instruction_summary = summary["instruction_summary"].head(12).copy()
    instruction_fig = _styled_figure(
        px.bar(
            instruction_summary,
            x="instruction_label",
            y="items",
        ),
        "Top Camera Instruction Labels",
    ) if not instruction_summary.empty else None

    cards = _metric_cards(
        [
            ("Video Rows", _format_number(int(videos.shape[0])), "All scanned videos"),
            ("Benchmark Videos", _format_number(int(benchmark_videos.shape[0])), "Primary benchmark video rows"),
            ("Annotated", _format_number(int(videos["has_annotation"].sum())), "Rows with annotation bundles"),
            ("Exact Duration Available", _format_number(int(videos["duration_seconds"].notna().sum())), "Rows with ffprobe duration"),
        ]
    )

    top_resolutions = (
        videos.groupby(["collection", "resolution_label"], dropna=False)
        .size()
        .reset_index(name="items")
        .sort_values("items", ascending=False)
    )

    chart_sections = "\n\n".join(
        filter(
            None,
            [
                charts.render("videos", "collection-mix", "Video Collection Mix", collection_fig),
                charts.render("videos", "duration-mix", "Duration Mix", duration_fig),
                charts.render("videos", "static-video-hierarchy", "Static Video Hierarchy", static_tree),
                charts.render("videos", "annotation-coverage", "Annotation Coverage", annotation_fig),
                charts.render("videos", "scene-tag-mix", "Scene Tag Mix", scene_tag_fig),
                charts.render("videos", "camera-motion-labels", "Camera Motion Labels", instruction_fig),
            ],
        )
    )

    return f"""# Video Analytics

{cards}

## Annotation Summary

{_frame_table(annotation_summary, ["collection", "items", "annotated", "complete_annotations", "annotation_rate"], limit=15)}

## Resolution Summary

{_frame_table(top_resolutions, ["collection", "resolution_label", "items"], limit=15)}

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


def _captions_page(summary: dict[str, Any], charts: ChartWriter, data_prefix: str) -> str:
    caption_videos = summary["caption_videos"].copy()
    caption_distribution_summary = summary["caption_distribution_summary"].copy()
    caption_collection_summary = summary["caption_collection_summary"].copy()
    caption_coverage_summary = summary["caption_coverage_summary"].copy()
    caption_coverage_long = summary["caption_coverage_long"].copy()
    caption_weather_time_summary = summary["caption_weather_time_summary"].copy()
    caption_brightness_crowd_summary = summary["caption_brightness_crowd_summary"].copy()

    cards = _metric_cards(
        [
            ("Captioned Videos", _format_number(int(caption_videos.shape[0])), "Videos with caption metadata"),
            (
                "Weather Tagged",
                _format_number(int(caption_videos["caption_weather"].notna().sum())),
                "Videos with normalized weather labels",
            ),
            (
                "Time-Of-Day Tagged",
                _format_number(int(caption_videos["caption_time_of_day"].notna().sum())),
                "Videos with normalized time-of-day labels",
            ),
            (
                "Primary Scene Types",
                _format_number(int(caption_videos["caption_scene_type_primary"].dropna().nunique())),
                "Unique normalized primary scene labels",
            ),
        ]
    )

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

{cards}

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

## Top Weather Labels

{_caption_distribution_table(caption_distribution_summary, "weather", limit=12)}

## Top Time-Of-Day Labels

{_caption_distribution_table(caption_distribution_summary, "time_of_day", limit=12)}

## Top Primary Scene Types

{_caption_distribution_table(caption_distribution_summary, "scene_type_primary", limit=15)}

{chart_sections}

## Artifacts

- [`caption_distribution_summary.csv`]({_artifact_path(data_prefix, "caption_distribution_summary.csv")})
- [`caption_collection_summary.csv`]({_artifact_path(data_prefix, "caption_collection_summary.csv")})
- [`caption_coverage_summary.csv`]({_artifact_path(data_prefix, "caption_coverage_summary.csv")})
- [`caption_weather_time_summary.csv`]({_artifact_path(data_prefix, "caption_weather_time_summary.csv")})
- [`caption_brightness_crowd_summary.csv`]({_artifact_path(data_prefix, "caption_brightness_crowd_summary.csv")})
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
            avg_size_mb=("size_mb", "mean"),
        )
        .reset_index()
    )

    return f"""# Collection Details

## Collection Summary

{_frame_table(collection_summary, ["group", "collection", "modality", "items", "annotated", "masked"], limit=20)}

## Dimension Cardinality

{_frame_table(dimension_summary, ["collection", "items", "styles", "scenes", "motion_categories", "avg_size_mb"], limit=20)}

## Mask Summary

{_frame_table(mask_summary, ["collection", "modality", "items", "masked", "mask_files", "mask_rate"], limit=20)}

## Artifacts

- [`collection_summary.csv`]({_artifact_path(data_prefix, "collection_summary.csv")})
- [`annotation_summary.csv`]({_artifact_path(data_prefix, "annotation_summary.csv")})
- [`mask_summary.csv`]({_artifact_path(data_prefix, "mask_summary.csv")})
"""


def build_docs_site(
    config_path: Path,
    max_workers: int = 8,
    write_docs: bool = True,
) -> dict[str, Any]:
    project = load_project_config(config_path)
    frame = build_dataset_frame(project, max_workers=max_workers)
    summary = build_summary_tables(frame)
    summary["project"] = project

    write_artifacts(
        summary=summary,
        output_dir=project.paths.artifacts_dir,
        docs_data_dir=project.paths.docs_data_dir,
    )

    if write_docs:
        ensure_dir(project.paths.docs_generated_dir)
        docs_root = project.paths.docs_generated_dir.parent
        chart_writer = ChartWriter(
            project.paths.docs_generated_dir / "charts",
            src_prefix="../generated/charts",
        )
        chart_writer.reset()
        for path in project.paths.docs_generated_dir.glob("*.md"):
            path.unlink()
        pages = {
            "overview.md": _overview_page(summary, chart_writer, "generated/data"),
            "images.md": _images_page(summary, chart_writer),
            "videos.md": _videos_page(summary, chart_writer),
            "captions.md": _captions_page(summary, chart_writer, "generated/data"),
            "collections.md": _collections_page(summary, "generated/data"),
        }
        for filename, content in pages.items():
            (docs_root / filename).write_text(content, encoding="utf-8")

    return summary
