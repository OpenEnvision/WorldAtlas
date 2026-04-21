from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SuiteDefinition:
    name: str
    display_name: str
    official: bool
    modality: str
    default_metrics: tuple[str, ...]


SUITE_DEFINITIONS: dict[str, SuiteDefinition] = {
    "image_static": SuiteDefinition(
        name="image_static",
        display_name="WorldArena-Image-Static",
        official=True,
        modality="image",
        default_metrics=("scene_alignment", "style_consistency", "subjective_quality"),
    ),
    "image_dynamic": SuiteDefinition(
        name="image_dynamic",
        display_name="WorldArena-Image-Dynamic",
        official=True,
        modality="image",
        default_metrics=("dynamic_alignment", "style_consistency", "subjective_quality"),
    ),
    "video_static": SuiteDefinition(
        name="video_static",
        display_name="WorldArena-Video-Static",
        official=True,
        modality="video",
        default_metrics=("scene_alignment", "style_consistency", "subjective_quality", "rollout_quality"),
    ),
    "video_dynamic": SuiteDefinition(
        name="video_dynamic",
        display_name="WorldArena-Video-Dynamic",
        official=True,
        modality="video",
        default_metrics=(
            "dynamic_alignment",
            "style_consistency",
            "subjective_quality",
            "motion_magnitude",
            "motion_smoothness",
            "rollout_quality",
        ),
    ),
    "auxiliary": SuiteDefinition(
        name="auxiliary",
        display_name="WorldArena-Auxiliary",
        official=False,
        modality="mixed",
        default_metrics=("dynamic_alignment", "subjective_quality"),
    ),
    "experimental": SuiteDefinition(
        name="experimental",
        display_name="WorldArena-Experimental",
        official=False,
        modality="mixed",
        default_metrics=("dynamic_alignment", "subjective_quality", "motion_magnitude", "motion_smoothness"),
    ),
}


OFFICIAL_SUITES = ("image_static", "image_dynamic", "video_static", "video_dynamic")
