"""Metric catalog: dimensions, normalization specs, and headline groupings."""

from __future__ import annotations


TOP_LEVEL_DIMENSIONS: tuple[str, ...] = (
    "long_sequence",
    "action_control",
    "consistency_3d_4d",
    "physics",
    "quality",
)

DIMENSION_LABELS: dict[str, str] = {
    "long_sequence": "Long Sequence / Memory",
    "action_control": "Action Control",
    "consistency_3d_4d": "3D/4D Consistency",
    "physics": "Physics",
    "quality": "Quality",
}

# Closed-loop memory metrics that carry the ``long_sequence`` headline score.
# Persistence probes on this dimension stay diagnostic unless a suite marks them official.
MEMORY_OFFICIAL_METRICS: tuple[str, ...] = (
    "memory_scene_f1",
    "memory_revisit_fidelity",
    "memory_geometric_closure",
    "memory_rendering_recovery",
    "memory_half_life",
)
# ``memory_return_gate`` diagnoses departure/return without blocking memory scoring
# or entering the dimension average. The rest predate the closed-loop protocol: they compare
# fixed frame indices, which conflates control error with memory decay.
MEMORY_GATE_METRIC = "memory_return_gate"
MEMORY_DIAGNOSTIC_METRICS: tuple[str, ...] = (
    "memory_revisit_consistency",
    "memory_symmetry",
    "entity_reappearance_consistency",
    "revisit_return_gate",
)


METRIC_DIMENSIONS: dict[str, str] = {
    # Long sequence / memory
    "long_sequence_motion_smoothness": "long_sequence",
    "memory_return_gate": "long_sequence",
    "memory_scene_f1": "long_sequence",
    "memory_revisit_fidelity": "long_sequence",
    "memory_geometric_closure": "long_sequence",
    "memory_rendering_recovery": "long_sequence",
    "memory_half_life": "long_sequence",
    "memory_entity_recall": "long_sequence",
    "scene_memory_f1": "long_sequence",
    "memory_revisit_consistency": "long_sequence",
    "revisit_return_gate": "long_sequence",
    "entity_reappearance_consistency": "long_sequence",
    "memory_symmetry": "long_sequence",
    # Action control
    "prompt_alignment": "action_control",
    "camera_error": "action_control",
    "motion_accuracy": "action_control",
    "keystroke_strict_action_accuracy": "action_control",
    "keystroke_partial_action_accuracy": "action_control",
    "keystroke_traj_score": "action_control",
    "keystroke_natet": "action_control",
    "keystroke_nater": "action_control",
    # 3D/4D consistency
    "met3r_consistency": "consistency_3d_4d",
    "reprojection_error": "consistency_3d_4d",
    "epipolar_error": "consistency_3d_4d",
    "epipolar_inlier_rate": "consistency_3d_4d",
    "feature_match_coverage": "consistency_3d_4d",
    "optical_flow_aepe": "consistency_3d_4d",
    "reconstruction_consistency": "consistency_3d_4d",
    "geometric_consistency": "consistency_3d_4d",
    "photometric_consistency": "consistency_3d_4d",
    # Physics
    "physical_fps_error": "physics",
    "physical_fps_pct_error": "physics",
    "physical_fps_intra_video_cv": "physics",
    "fvmd": "physics",
    "trajan": "physics",
    "jedi": "quality",
    "motion_magnitude": "physics",
    "motion_smoothness": "physics",
    "long_sequence_dynamic_degree": "physics",
    # Quality
    "style_consistency": "quality",
    "perceptual_quality": "quality",
    "image_quality": "quality",
    "hpsv3_norm": "quality",
    "brightness_consistency": "quality",
    "brightness_distribution_consistency": "quality",
    "color_temperature_consistency": "quality",
    "color_temperature_constraint": "quality",
    "sharpness_retention": "quality",
    "sharpness_vector_retention": "quality",
    "temporal_flickering": "quality",
    "background_consistency": "quality",
    "segment_continuity": "quality",
    "long_sequence_aesthetic_quality": "quality",
    "long_sequence_imaging_quality": "quality",
    "rollout_aesthetic_drift": "quality",
    "rollout_imaging_drift": "quality",
}


def group_metrics_by_dimension(metric_names: list[str]) -> dict[str, list[str]]:
    """Bucket metric names under their headline evaluation dimension."""
    grouped: dict[str, list[str]] = {}
    for metric_name in metric_names:
        dimension = METRIC_DIMENSIONS.get(metric_name, "uncategorized")
        grouped.setdefault(dimension, []).append(metric_name)
    return grouped


__all__ = [
    "DIMENSION_LABELS",
    "MEMORY_DIAGNOSTIC_METRICS",
    "MEMORY_GATE_METRIC",
    "MEMORY_OFFICIAL_METRICS",
    "METRIC_DIMENSIONS",
    "TOP_LEVEL_DIMENSIONS",
    "group_metrics_by_dimension",
]
