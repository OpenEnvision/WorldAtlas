"""Benchmark suite definitions and metric groupings."""

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
        display_name="WorldAtlas Arena-Image-Static",
        official=True,
        modality="video",
        default_metrics=(
            "prompt_alignment",
            "style_consistency",
            "perceptual_quality",
            "brightness_consistency",
            "color_temperature_consistency",
            "sharpness_retention",
        ),
    ),
    "image_dynamic": SuiteDefinition(
        name="image_dynamic",
        display_name="WorldAtlas Arena-Image-Dynamic",
        official=True,
        modality="video",
        default_metrics=(
            "prompt_alignment",
            "style_consistency",
            "perceptual_quality",
            "brightness_consistency",
            "color_temperature_consistency",
            "sharpness_retention",
        ),
    ),
    "video_static": SuiteDefinition(
        name="video_static",
        display_name="WorldAtlas Arena-Video-Static",
        official=True,
        modality="video",
        default_metrics=(
            "prompt_alignment",
            "semantic_alignment",
            "style_consistency",
            "perceptual_quality",
            "brightness_consistency",
            "color_temperature_consistency",
            "sharpness_retention",
            "camera_error",
            "seen_region_preservation",
            "object_detection",
            "reprojection_error",
            "optical_flow_aepe",
            "depth_accuracy",
        ),
    ),
    "video_dynamic": SuiteDefinition(
        name="video_dynamic",
        display_name="WorldAtlas Arena-Video-Dynamic",
        official=True,
        modality="video",
        default_metrics=(
            "prompt_alignment",
            "semantic_alignment",
            "style_consistency",
            "perceptual_quality",
            "brightness_consistency",
            "color_temperature_consistency",
            "sharpness_retention",
            "object_permanence",
            "pmf",
            "motion_magnitude",
            "motion_smoothness",
            "motion_accuracy",
            "depth_accuracy",
            "physical_fps_error",
            "physical_fps_pct_error",
            "physical_fps_intra_video_cv",
        ),
    ),
    "auxiliary": SuiteDefinition(
        name="auxiliary",
        display_name="WorldAtlas Arena-Auxiliary",
        official=False,
        modality="mixed",
        default_metrics=(),
    ),
    "experimental": SuiteDefinition(
        name="experimental",
        display_name="WorldAtlas Arena-Experimental",
        official=False,
        modality="mixed",
        default_metrics=(
            "motion_magnitude",
            "motion_smoothness",
        ),
    ),
}


OFFICIAL_SUITES = ("image_static", "image_dynamic", "video_static", "video_dynamic")
