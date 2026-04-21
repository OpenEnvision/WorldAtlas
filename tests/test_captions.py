import pandas as pd

from worldarena.datasets.scanners import _extract_caption_metadata
from worldarena.datasets.summarize import (
    _normalize_brightness,
    _normalize_crowd_density,
    _normalize_frame,
    _normalize_scene_label,
    _normalize_time_of_day,
    _normalize_weather,
    build_summary_tables,
)


def test_extract_caption_metadata_handles_non_mapping_tags() -> None:
    payload = {
        "CategoryTags": "invalid",
    }

    result = _extract_caption_metadata(payload)

    assert result["has_caption"] is True
    assert result["tag_weather"] is None
    assert result["tag_scene_type_primary"] is None


def test_caption_normalizers_collapse_common_variants() -> None:
    assert _normalize_weather("sunny") == "Sunny"
    assert _normalize_weather("Indoor") == "Indoor/Unspecified"
    assert _normalize_weather(pd.NA) is None
    assert _normalize_time_of_day("Daytime (Midday/Noon/Afternoon)") == "Daytime"
    assert _normalize_time_of_day("Dawn/Morning") == "Dawn"
    assert _normalize_time_of_day("Dawn/Evening") == "Transition (Dawn/Dusk)"
    assert _normalize_brightness("Dim") == "Dim/Dark"
    assert _normalize_crowd_density("deserted") == "Deserted"
    assert _normalize_scene_label("dining_spaces") == "Dining Spaces"


def test_caption_tag_count_can_be_derived_after_normalization() -> None:
    frame = pd.DataFrame(
        [
            {
                "tag_weather": "Sunny",
                "tag_time_of_day": "Daytime",
                "tag_brightness": "Bright",
                "tag_crowd_density": None,
                "tag_scene_type_primary": "indoor",
                "tag_scene_type_secondary": "dining_spaces",
            }
        ]
    )

    frame["caption_weather"] = frame["tag_weather"].apply(_normalize_weather)
    frame["caption_time_of_day"] = frame["tag_time_of_day"].apply(_normalize_time_of_day)
    frame["caption_brightness"] = frame["tag_brightness"].apply(_normalize_brightness)
    frame["caption_crowd_density"] = frame["tag_crowd_density"].apply(_normalize_crowd_density)
    frame["caption_scene_type_primary"] = frame["tag_scene_type_primary"].apply(_normalize_scene_label)
    frame["caption_scene_type_secondary"] = frame["tag_scene_type_secondary"].apply(_normalize_scene_label)

    caption_columns = [
        "caption_weather",
        "caption_time_of_day",
        "caption_brightness",
        "caption_crowd_density",
        "caption_scene_type_primary",
        "caption_scene_type_secondary",
    ]
    frame["caption_tag_count"] = frame[caption_columns].notna().sum(axis=1)

    assert frame.loc[0, "caption_tag_count"] == 5


def test_normalize_frame_handles_missing_caption_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "path": "example.mp4",
                "modality": "video",
                "size_bytes": 1024,
                "width": 1280,
                "height": 720,
                "duration_seconds": 3.5,
            }
        ]
    )

    normalized = _normalize_frame(frame)

    assert normalized.loc[0, "caption_weather"] is None
    assert normalized.loc[0, "caption_scene_type_primary"] is None
    assert normalized.loc[0, "caption_tag_count"] == 0


def test_build_summary_tables_handles_missing_instruction_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "path": "example.mp4",
                "group": "benchmark",
                "collection": "static_videos",
                "modality": "video",
                "has_caption": True,
                "caption_weather": "Sunny",
                "caption_time_of_day": "Daytime",
                "caption_brightness": "Bright",
                "caption_crowd_density": "Sparse",
                "caption_scene_type_primary": "Indoor",
                "caption_scene_type_secondary": "Dining Space",
                "has_annotation": False,
                "annotation_complete": False,
                "has_mask": False,
                "mask_count": 0,
                "resolution_label": "1280x720",
            }
        ]
    )

    summary = build_summary_tables(frame)

    assert summary["instruction_summary"].empty
    assert int(summary["caption_videos"].shape[0]) == 1
    assert int(summary["caption_distribution_summary"]["items"].sum()) == 6
