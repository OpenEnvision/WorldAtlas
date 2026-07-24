"""Summarize scanned dataset inventory into statistics tables and reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worldarena.common.media import orientation, probe_media
from worldarena.common.serialization import ensure_dir, write_json
from worldarena.datasets.config import ProjectConfig
from worldarena.datasets.scanners import bucketize_duration, scan_collections

CAPTION_DIMENSION_LABELS = {
    "weather": "Weather",
    "time_of_day": "Time Of Day",
    "brightness": "Brightness",
    "crowd_density": "Crowd Density",
    "scene_type_primary": "Primary Scene Type",
    "scene_type_secondary": "Secondary Scene Type",
}

# Last-known-good collection counts when dataset roots are unavailable offline.
COLLECTION_SNAPSHOT_ROWS: list[dict[str, Any]] = [
    {
        "group": "benchmark",
        "collection": "dynamic_images",
        "modality": "image",
        "items": 1628,
        "annotated": 1628,
        "complete_annotations": 0,
        "masked": 1628,
    },
    {
        "group": "benchmark",
        "collection": "static_images",
        "modality": "image",
        "items": 3372,
        "annotated": 3372,
        "complete_annotations": 0,
        "masked": 0,
    },
    {
        "group": "benchmark",
        "collection": "physics_videos",
        "modality": "physics",
        "items": 1400,
        "annotated": 0,
        "complete_annotations": 0,
        "masked": 0,
    },
    {
        "group": "benchmark",
        "collection": "aaa_games",
        "modality": "video",
        "items": 465,
        "annotated": 465,
        "complete_annotations": 465,
        "masked": 465,
    },
    {
        "group": "benchmark",
        "collection": "dynamic_videos",
        "modality": "video",
        "items": 788,
        "annotated": 0,
        "complete_annotations": 0,
        "masked": 0,
    },
    {
        "group": "benchmark",
        "collection": "static_videos",
        "modality": "video",
        "items": 4000,
        "annotated": 31,
        "complete_annotations": 31,
        "masked": 31,
    },
]


def _frame_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(index=frame.index, dtype="object")


def _clean_tag_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _titleize_label(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized.title()


def _normalize_weather(value: Any) -> str | None:
    text = _clean_tag_value(value)
    if not text:
        return None

    lowered = text.lower()
    if "unknown" in lowered:
        return "Unknown"
    if "indoor" in lowered:
        return "Indoor/Unspecified"
    for label in ["clear", "sunny", "cloudy", "rainy", "snowy", "foggy", "stormy"]:
        if label in lowered:
            return label.title()
    return _titleize_label(text)


def _normalize_time_of_day(value: Any) -> str | None:
    text = _clean_tag_value(value)
    if not text:
        return None

    lowered = text.lower()
    if "unknown" in lowered:
        return "Unknown"
    if "night" in lowered:
        return "Night"
    if ("dawn" in lowered and "evening" in lowered) or ("dawn" in lowered and "dusk" in lowered):
        return "Transition (Dawn/Dusk)"
    if "dusk" in lowered or "evening" in lowered:
        return "Dusk/Evening"
    if "dawn" in lowered:
        return "Dawn"
    if "morning" in lowered:
        return "Morning"
    if any(token in lowered for token in ["daytime", "day", "afternoon", "noon", "midday"]):
        return "Daytime"
    return _titleize_label(text)


def _normalize_brightness(value: Any) -> str | None:
    text = _clean_tag_value(value)
    if not text:
        return None

    lowered = text.lower()
    if "bright" in lowered:
        return "Bright"
    if "dim" in lowered or "dark" in lowered:
        return "Dim/Dark"
    if "soft" in lowered:
        return "Soft"
    if "unknown" in lowered:
        return "Unknown"
    return _titleize_label(text)


def _normalize_crowd_density(value: Any) -> str | None:
    text = _clean_tag_value(value)
    if not text:
        return None

    lowered = text.lower()
    if "deserted" in lowered:
        return "Deserted"
    if "sparse" in lowered:
        return "Sparse"
    if "moderate" in lowered:
        return "Moderate"
    if "crowded" in lowered:
        return "Crowded"
    if "unknown" in lowered:
        return "Unknown"
    return _titleize_label(text)


def _normalize_scene_label(value: Any) -> str | None:
    return _titleize_label(_clean_tag_value(value))


def _build_caption_distribution(
    frame: pd.DataFrame,
    dimension_columns: dict[str, str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for dimension, column in dimension_columns.items():
        subset = frame.dropna(subset=[column]).groupby(column, dropna=False).size().reset_index(name="items")
        if subset.empty:
            continue
        subset["dimension"] = dimension
        subset.rename(columns={column: "value"}, inplace=True)
        subset["share"] = (subset["items"] / subset["items"].sum()).round(4)
        parts.append(subset)

    if not parts:
        return pd.DataFrame(columns=["dimension", "value", "items", "share"])

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["dimension", "items", "value"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def _build_caption_distribution_by_collection(
    frame: pd.DataFrame,
    dimension_columns: dict[str, str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for dimension, column in dimension_columns.items():
        subset = (
            frame.dropna(subset=[column])
            .groupby(["collection", column], dropna=False)
            .size()
            .reset_index(name="items")
        )
        if subset.empty:
            continue
        subset["dimension"] = dimension
        subset.rename(columns={column: "value"}, inplace=True)
        subset["share_within_collection"] = (
            subset["items"] / subset.groupby("collection")["items"].transform("sum")
        ).round(4)
        parts.append(subset)

    if not parts:
        return pd.DataFrame(columns=["collection", "dimension", "value", "items", "share_within_collection"])

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["dimension", "collection", "items"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    for column in ["has_annotation", "annotation_complete", "has_mask", "has_caption"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna(False).astype(bool)

    for column in ["size_bytes", "mask_count", "annotation_present_count", "pixel_count"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in [
        "width",
        "height",
        "fps",
        "duration_seconds",
        "aspect_ratio",
        "metadata_width",
        "metadata_height",
        "metadata_duration_seconds",
        "metadata_fps",
        "visual_quality_score",
        "motion_score_v2",
        "text_bbox_ratio",
        "selection_score",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["size_mb"] = (frame["size_bytes"] / (1024 * 1024)).round(4)
    frame["orientation"] = frame.apply(
        lambda row: orientation(row.get("width"), row.get("height")),
        axis=1,
    )
    frame["resolution_label"] = frame.apply(
        lambda row: (
            f"{int(row['width'])}x{int(row['height'])}"
            if pd.notna(row.get("width")) and pd.notna(row.get("height"))
            else "Unknown"
        ),
        axis=1,
    )
    frame["duration_bucket_inferred"] = frame["duration_seconds"].apply(bucketize_duration)
    if "duration_bucket" in frame.columns:
        frame["duration_bucket_effective"] = frame["duration_bucket"].combine_first(
            frame["duration_bucket_inferred"]
        )
    else:
        frame["duration_bucket_effective"] = frame["duration_bucket_inferred"]

    frame["caption_weather"] = _frame_column(frame, "tag_weather").apply(_normalize_weather)
    frame["caption_time_of_day"] = _frame_column(frame, "tag_time_of_day").apply(_normalize_time_of_day)
    frame["caption_brightness"] = _frame_column(frame, "tag_brightness").apply(_normalize_brightness)
    frame["caption_crowd_density"] = _frame_column(frame, "tag_crowd_density").apply(_normalize_crowd_density)
    frame["caption_scene_type_primary"] = _frame_column(frame, "tag_scene_type_primary").apply(
        _normalize_scene_label
    )
    frame["caption_scene_type_secondary"] = _frame_column(frame, "tag_scene_type_secondary").apply(
        _normalize_scene_label
    )

    caption_columns = [
        "caption_weather",
        "caption_time_of_day",
        "caption_brightness",
        "caption_crowd_density",
        "caption_scene_type_primary",
        "caption_scene_type_secondary",
    ]
    tag_counts = pd.Series(0, index=frame.index, dtype="int64")
    for col in caption_columns:
        tag_counts = tag_counts + frame[col].notna().astype("int64")
    frame["caption_tag_count"] = tag_counts

    return frame


def _cached_records_csv(*directories: Path) -> Path | None:
    for directory in directories:
        candidate = directory / "all_records.csv"
        if candidate.is_file() and candidate.stat().st_size > 512:
            return candidate
    return None


def _cached_summary_totals(summary_path: Path) -> int:
    if not summary_path.is_file():
        return 0
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    totals = payload.get("totals", {})
    return int(totals.get("records") or 0)


def load_cached_dataset_frame(project: ProjectConfig) -> pd.DataFrame:
    """Load the most recent non-empty inventory CSV when live scanning finds nothing."""
    candidate = _cached_records_csv(project.paths.docs_data_dir, project.paths.artifacts_dir)
    if candidate is None:
        return pd.DataFrame()
    frame = pd.read_csv(candidate)
    if frame.empty:
        return pd.DataFrame()
    return frame


def merge_collection_snapshot(collection_summary: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Fill missing benchmark collections from the last-known-good snapshot."""
    if collection_summary.empty:
        merged = pd.DataFrame(COLLECTION_SNAPSHOT_ROWS)
        return merged, True

    existing = {
        str(row["collection"])
        for row in collection_summary.to_dict(orient="records")
        if row.get("collection")
    }
    missing_rows = [
        row for row in COLLECTION_SNAPSHOT_ROWS if row["collection"] not in existing
    ]
    if not missing_rows:
        return collection_summary, False

    merged = pd.concat(
        [collection_summary, pd.DataFrame(missing_rows)],
        ignore_index=True,
    )
    return merged, True


def build_dataset_frame(project: ProjectConfig, max_workers: int = 8) -> pd.DataFrame:
    records = scan_collections(project.collections)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame

    image_paths = [Path(path) for path in frame.loc[frame["modality"] == "image", "path"].tolist()]
    video_paths = [
        Path(path)
        for path in frame.loc[frame["modality"].isin(["video", "physics"]), "path"].tolist()
    ]
    media_lookup = probe_media(
        image_paths=image_paths,
        video_paths=video_paths,
        cache_path=project.paths.cache_path,
        max_workers=max_workers,
    )
    media_frame = pd.DataFrame(
        [{"path": path, **payload} for path, payload in media_lookup.items()]
    )
    frame = frame.merge(media_frame, on="path", how="left")
    return _normalize_frame(frame)


def build_summary_tables(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.copy()
    defaults: dict[str, Any] = {
        "path": None,
        "size_bytes": 0,
        "size_mb": 0.0,
        "group": None,
        "collection": None,
        "modality": None,
        "has_annotation": False,
        "annotation_complete": False,
        "has_mask": False,
        "mask_count": 0,
        "duration_seconds": pd.NA,
        "duration_bucket": None,
        "width": pd.NA,
        "height": pd.NA,
        "source_type": None,
        "motion_regime": None,
        "style": None,
        "environment": None,
        "scene": None,
        "instruction_labels": None,
        "motion_category": None,
        "physics_group": None,
        "physics_dimension": None,
        "physics_label": None,
        "has_physics_metadata": False,
        "visual_quality_score": pd.NA,
        "motion_score_v2": pd.NA,
        "selection_score": pd.NA,
        "resolution_label": "Unknown",
        "has_caption": False,
        "caption_weather": None,
        "caption_time_of_day": None,
        "caption_brightness": None,
        "caption_crowd_density": None,
        "caption_scene_type_primary": None,
        "caption_scene_type_secondary": None,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    images = frame[frame["modality"] == "image"].copy()
    videos = frame[frame["modality"] == "video"].copy()
    physics = frame[frame["modality"] == "physics"].copy()
    video_like = frame[frame["modality"].isin(["video", "physics"])].copy()
    has_caption = (
        video_like["has_caption"]
        if "has_caption" in video_like.columns
        else pd.Series(False, index=video_like.index)
    )
    caption_videos = video_like[has_caption.fillna(False)].copy()

    collection_summary = (
        frame.groupby(["group", "collection", "modality"], dropna=False)
        .agg(
            items=("path", "count"),
            annotated=("has_annotation", "sum"),
            complete_annotations=("annotation_complete", "sum"),
            masked=("has_mask", "sum"),
            avg_duration_seconds=("duration_seconds", "mean"),
            avg_width=("width", "mean"),
            avg_height=("height", "mean"),
        )
        .reset_index()
        .sort_values(["group", "modality", "collection"])
    )

    annotation_summary = (
        videos.groupby(["collection"], dropna=False)
        .agg(
            items=("path", "count"),
            annotated=("has_annotation", "sum"),
            complete_annotations=("annotation_complete", "sum"),
            with_mask=("has_mask", "sum"),
        )
        .reset_index()
        .sort_values("items", ascending=False)
    )
    if not annotation_summary.empty:
        annotation_summary["annotation_rate"] = (
            annotation_summary["annotated"] / annotation_summary["items"]
        ).round(4)
        annotation_summary["complete_annotation_rate"] = (
            annotation_summary["complete_annotations"] / annotation_summary["items"]
        ).round(4)

    mask_summary = (
        frame.groupby(["collection", "modality"], dropna=False)
        .agg(items=("path", "count"), masked=("has_mask", "sum"), mask_files=("mask_count", "sum"))
        .reset_index()
        .sort_values(["modality", "collection"])
    )
    if not mask_summary.empty:
        mask_summary["mask_rate"] = (mask_summary["masked"] / mask_summary["items"]).round(4)

    if {"caption_scene_type_primary", "caption_scene_type_secondary"}.issubset(video_like.columns):
        scene_tag_summary = (
            video_like.dropna(subset=["caption_scene_type_primary"])
            .groupby(["caption_scene_type_primary", "caption_scene_type_secondary"], dropna=False)
            .size()
            .reset_index(name="items")
            .sort_values("items", ascending=False)
        )
    else:
        scene_tag_summary = pd.DataFrame(
            columns=["caption_scene_type_primary", "caption_scene_type_secondary", "items"]
        )

    if "instruction_labels" in videos.columns:
        instruction_summary = (
            videos.dropna(subset=["instruction_labels"])
            .assign(
                instruction_label=lambda data_frame: data_frame["instruction_labels"].str.split(r"\s+\|\s+")
            )
            .explode("instruction_label")
            .groupby("instruction_label", dropna=False)
            .size()
            .reset_index(name="items")
            .sort_values("items", ascending=False)
        )
    else:
        instruction_summary = pd.DataFrame(columns=["instruction_label", "items"])

    resolution_summary = (
        frame.groupby(["modality", "resolution_label"], dropna=False)
        .size()
        .reset_index(name="items")
        .sort_values(["modality", "items"], ascending=[True, False])
    )

    caption_dimension_columns = {
        "weather": "caption_weather",
        "time_of_day": "caption_time_of_day",
        "brightness": "caption_brightness",
        "crowd_density": "caption_crowd_density",
        "scene_type_primary": "caption_scene_type_primary",
        "scene_type_secondary": "caption_scene_type_secondary",
    }

    caption_distribution_summary = _build_caption_distribution(
        caption_videos,
        caption_dimension_columns,
    )
    caption_collection_summary = _build_caption_distribution_by_collection(
        caption_videos,
        caption_dimension_columns,
    )

    if video_like.empty:
        caption_coverage_summary = pd.DataFrame(
            columns=[
                "collection",
                "items",
                "captions",
                "weather_labeled",
                "time_of_day_labeled",
                "brightness_labeled",
                "crowd_density_labeled",
                "scene_primary_labeled",
            ]
        )
    else:
        caption_coverage_summary = (
            video_like.groupby(["collection"], dropna=False)
            .agg(
                items=("path", "count"),
                captions=("has_caption", "sum"),
                weather_labeled=("caption_weather", lambda values: values.notna().sum()),
                time_of_day_labeled=("caption_time_of_day", lambda values: values.notna().sum()),
                brightness_labeled=("caption_brightness", lambda values: values.notna().sum()),
                crowd_density_labeled=("caption_crowd_density", lambda values: values.notna().sum()),
                scene_primary_labeled=("caption_scene_type_primary", lambda values: values.notna().sum()),
            )
            .reset_index()
            .sort_values("items", ascending=False)
        )
    if not caption_coverage_summary.empty:
        for source, target in [
            ("captions", "caption_rate"),
            ("weather_labeled", "weather_rate"),
            ("time_of_day_labeled", "time_of_day_rate"),
            ("brightness_labeled", "brightness_rate"),
            ("crowd_density_labeled", "crowd_density_rate"),
            ("scene_primary_labeled", "scene_primary_rate"),
        ]:
            caption_coverage_summary[target] = (
                caption_coverage_summary[source] / caption_coverage_summary["items"]
            ).round(4)

    if caption_coverage_summary.empty:
        caption_coverage_long = pd.DataFrame(columns=["collection", "dimension", "rate"])
    else:
        caption_coverage_long = caption_coverage_summary.melt(
            id_vars=["collection"],
            value_vars=[
                "caption_rate",
                "weather_rate",
                "time_of_day_rate",
                "brightness_rate",
                "crowd_density_rate",
                "scene_primary_rate",
            ],
            var_name="dimension",
            value_name="rate",
        )
        caption_coverage_long["dimension"] = caption_coverage_long["dimension"].map(
            {
                "caption_rate": "Caption Coverage",
                "weather_rate": "Weather",
                "time_of_day_rate": "Time Of Day",
                "brightness_rate": "Brightness",
                "crowd_density_rate": "Crowd Density",
                "scene_primary_rate": "Primary Scene Type",
            }
        )

    if {"caption_weather", "caption_time_of_day"}.issubset(caption_videos.columns):
        caption_weather_time_summary = (
            caption_videos.dropna(subset=["caption_weather", "caption_time_of_day"])
            .groupby(["caption_weather", "caption_time_of_day"], dropna=False)
            .size()
            .reset_index(name="items")
            .sort_values("items", ascending=False)
        )
    else:
        caption_weather_time_summary = pd.DataFrame(
            columns=["caption_weather", "caption_time_of_day", "items"]
        )

    if {"caption_brightness", "caption_crowd_density"}.issubset(caption_videos.columns):
        caption_brightness_crowd_summary = (
            caption_videos.dropna(subset=["caption_brightness", "caption_crowd_density"])
            .groupby(["caption_brightness", "caption_crowd_density"], dropna=False)
            .size()
            .reset_index(name="items")
            .sort_values("items", ascending=False)
        )
    else:
        caption_brightness_crowd_summary = pd.DataFrame(
            columns=["caption_brightness", "caption_crowd_density", "items"]
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame": frame,
        "images": images,
        "videos": videos,
        "physics": physics,
        "video_like": video_like,
        "caption_videos": caption_videos,
        "collection_summary": collection_summary,
        "annotation_summary": annotation_summary,
        "mask_summary": mask_summary,
        "scene_tag_summary": scene_tag_summary,
        "instruction_summary": instruction_summary,
        "resolution_summary": resolution_summary,
        "caption_distribution_summary": caption_distribution_summary,
        "caption_collection_summary": caption_collection_summary,
        "caption_coverage_summary": caption_coverage_summary,
        "caption_coverage_long": caption_coverage_long,
        "caption_weather_time_summary": caption_weather_time_summary,
        "caption_brightness_crowd_summary": caption_brightness_crowd_summary,
    }


def _round_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    serializable = frame.copy()
    float_columns = serializable.select_dtypes(include=["float64", "float32"]).columns
    serializable[float_columns] = serializable[float_columns].round(4)
    return json.loads(serializable.to_json(orient="records"))


def write_artifacts(summary: dict[str, Any], output_dir: Path, docs_data_dir: Path) -> None:
    ensure_dir(output_dir)
    ensure_dir(docs_data_dir)

    frame = summary["frame"]
    if frame.empty:
        cached_records = _cached_summary_totals(docs_data_dir / "summary.json")
        cached_records = max(cached_records, _cached_summary_totals(output_dir / "summary.json"))
        if cached_records > 0:
            return
    images = summary["images"]
    videos = summary["videos"]
    physics = summary["physics"]
    collection_summary = summary["collection_summary"]
    annotation_summary = summary["annotation_summary"]
    mask_summary = summary["mask_summary"]
    caption_distribution_summary = summary["caption_distribution_summary"]
    caption_collection_summary = summary["caption_collection_summary"]
    caption_coverage_summary = summary["caption_coverage_summary"]
    caption_weather_time_summary = summary["caption_weather_time_summary"]
    caption_brightness_crowd_summary = summary["caption_brightness_crowd_summary"]

    artifacts = {
        "all_records.csv": frame,
        "image_records.csv": images,
        "video_records.csv": videos,
        "physics_records.csv": physics,
        "collection_summary.csv": collection_summary,
        "annotation_summary.csv": annotation_summary,
        "mask_summary.csv": mask_summary,
        "caption_distribution_summary.csv": caption_distribution_summary,
        "caption_collection_summary.csv": caption_collection_summary,
        "caption_coverage_summary.csv": caption_coverage_summary,
        "caption_weather_time_summary.csv": caption_weather_time_summary,
        "caption_brightness_crowd_summary.csv": caption_brightness_crowd_summary,
    }

    for filename, data_frame in artifacts.items():
        target = output_dir / filename
        data_frame.to_csv(target, index=False)
        data_frame.to_csv(docs_data_dir / filename, index=False)

    write_json(
        output_dir / "summary.json",
        {
            "generated_at": summary["generated_at"],
            "totals": {
                "records": int(frame.shape[0]),
                "images": int(images.shape[0]),
                "videos": int(videos.shape[0]),
                "physics": int(physics.shape[0]),
            },
            "collections": _round_records(collection_summary),
            "annotations": _round_records(annotation_summary),
            "masks": _round_records(mask_summary),
            "caption_coverage": _round_records(caption_coverage_summary),
        },
    )
    write_json(output_dir / "all_records.json", _round_records(frame))
    write_json(
        docs_data_dir / "summary.json",
        {
            "generated_at": summary["generated_at"],
            "totals": {
                "records": int(frame.shape[0]),
                "images": int(images.shape[0]),
                "videos": int(videos.shape[0]),
                "physics": int(physics.shape[0]),
            },
        },
    )
