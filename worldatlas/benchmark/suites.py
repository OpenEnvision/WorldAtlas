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
        display_name="WorldAtlas-Image-Static",
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
        display_name="WorldAtlas-Image-Dynamic",
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
        display_name="WorldAtlas-Video-Static",
        official=True,
        modality="video",
        default_metrics=(
            "prompt_alignment",
            "style_consistency",
            "perceptual_quality",
            "brightness_consistency",
            "color_temperature_consistency",
            "sharpness_retention",
            "camera_error",
            "reprojection_error",
            "optical_flow_aepe",
        ),
    ),
    "video_dynamic": SuiteDefinition(
        name="video_dynamic",
        display_name="WorldAtlas-Video-Dynamic",
        official=True,
        modality="video",
        default_metrics=(
            "prompt_alignment",
            "style_consistency",
            "perceptual_quality",
            "brightness_consistency",
            "color_temperature_consistency",
            "sharpness_retention",
            "motion_magnitude",
            "motion_smoothness",
            "motion_accuracy",
        ),
    ),
    "memory_loop": SuiteDefinition(
        name="memory_loop",
        display_name="WorldAtlas-Memory-Loop",
        official=True,
        modality="video",
        default_metrics=(
            "memory_return_gate",
            "memory_scene_f1",
            "memory_revisit_fidelity",
            "memory_geometric_closure",
            "memory_rendering_recovery",
            "memory_half_life",
        ),
    ),
    "auxiliary": SuiteDefinition(
        name="auxiliary",
        display_name="WorldAtlas-Auxiliary",
        official=False,
        modality="mixed",
        default_metrics=(),
    ),
    "experimental": SuiteDefinition(
        name="experimental",
        display_name="WorldAtlas-Experimental",
        official=False,
        modality="mixed",
        default_metrics=(
            "motion_magnitude",
            "motion_smoothness",
        ),
    ),
}


OFFICIAL_SUITES = ("image_static", "image_dynamic", "video_static", "video_dynamic")
# Closed-loop memory is scored on its own track: it only accepts interactive world
# models and requires a 60-second rollout, so it is reported separately rather than
# folded into the four generation suites.
MEMORY_SUITES = ("memory_loop",)
