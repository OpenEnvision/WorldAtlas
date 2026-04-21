from __future__ import annotations


METRIC_DIMENSIONS: dict[str, str] = {
    "prompt_alignment": "content_alignment",
    "style_consistency": "style_consistency",
    "perceptual_quality": "perceptual_quality",
    "reference_appearance_alignment": "appearance_alignment",
    "anchor_consistency": "appearance_alignment",
    "camera_error": "camera_control",
    "object_detection": "object_control",
    "reprojection_error": "geometry_consistency",
    "optical_flow_aepe": "photometric_consistency",
    "motion_accuracy": "motion_control",
    "motion_magnitude": "motion_dynamics",
    "motion_smoothness": "motion_dynamics",
}


def group_metrics_by_dimension(metric_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for metric_name in metric_names:
        dimension = METRIC_DIMENSIONS.get(metric_name, "uncategorized")
        grouped.setdefault(dimension, []).append(metric_name)
    return grouped


__all__ = ["METRIC_DIMENSIONS", "group_metrics_by_dimension"]
