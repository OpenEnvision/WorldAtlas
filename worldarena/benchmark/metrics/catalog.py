from __future__ import annotations


TOP_LEVEL_DIMENSIONS: tuple[str, ...] = (
    "long_sequence",
    "action_control",
    "consistency_3d_4d",
    "physics",
    "quality",
    "real_time",
)

DIMENSION_LABELS: dict[str, str] = {
    "long_sequence": "Long Sequence",
    "action_control": "Action Control",
    "consistency_3d_4d": "3D/4D Consistency",
    "physics": "Physics",
    "quality": "Quality",
    "real_time": "Real Time",
}


METRIC_DIMENSIONS: dict[str, str] = {
    # Long sequence (memory)
    "seen_region_preservation": "long_sequence",
    "object_permanence": "long_sequence",
    "memory_revisit_consistency": "long_sequence",
    "memory_symmetry": "long_sequence",
    "long_sequence_motion_smoothness": "long_sequence",
    # Action control
    "prompt_alignment": "action_control",
    "semantic_alignment": "action_control",
    "camera_error": "action_control",
    "object_detection": "action_control",
    "motion_accuracy": "action_control",
    # 3D/4D consistency
    "novel_region_fidelity": "consistency_3d_4d",
    "met3r_consistency": "consistency_3d_4d",
    "reprojection_error": "consistency_3d_4d",
    "epipolar_error": "consistency_3d_4d",
    "epipolar_inlier_rate": "consistency_3d_4d",
    "feature_match_coverage": "consistency_3d_4d",
    "depth_accuracy": "consistency_3d_4d",
    "depth_collision": "consistency_3d_4d",
    "optical_flow_aepe": "consistency_3d_4d",
    "reconstruction_consistency": "consistency_3d_4d",
    # Physics
    "pmf": "physics",
    "fvmd": "physics",
    "jedi": "quality",
    "motion_magnitude": "physics",
    "motion_smoothness": "physics",
    "long_sequence_dynamic_degree": "physics",
    "mse": "physics",
    "st_iou": "physics",
    "s_iou": "physics",
    "ws_iou": "physics",
    "abs_rel": "physics",
    "rmse": "physics",
    "delta1": "physics",
    "delta2": "physics",
    "delta3": "physics",
    "depth_warp_l1": "physics",
    "depth_warp_charb": "physics",
    "rgb_warp_charb": "physics",
    "epe": "physics",
    "fl_all": "physics",
    "one_px_out": "physics",
    "chamfer_4d": "physics",
    "worldline_l2_error": "physics",
    "worldline_mean_drift": "physics",
    "worldline_final_drift": "physics",
    "worldline_fail_rate": "physics",
    "worldline_length": "physics",
    "trajectory_rmse": "physics",
    "final_position_error": "physics",
    "speed_similarity": "physics",
    "acceleration_similarity": "physics",
    "directional_consistency": "physics",
    "final_state_accuracy": "physics",
    "rgb_warp_lpips": "physics",
    "novel_time_depth_l1": "physics",
    "novel_time_depth_charb": "physics",
    "novel_time_rgb_l1": "physics",
    "novel_time_rgb_charb": "physics",
    "physics_judge_overall": "physics",
    "physics_judge_phenomenon": "physics",
    "physics_judge_causality": "physics",
    "physics_judge_material": "physics",
    "physics_judge_temporal": "physics",
    "physics_judge_prompt": "physics",
    # Quality
    "style_consistency": "quality",
    "perceptual_quality": "quality",
    "image_quality": "quality",
    "brightness_consistency": "quality",
    "brightness_distribution_consistency": "quality",
    "color_temperature_consistency": "quality",
    "color_temperature_constraint": "quality",
    "sharpness_retention": "quality",
    "sharpness_vector_retention": "quality",
    "long_sequence_aesthetic_quality": "quality",
    "long_sequence_imaging_quality": "quality",
    "psnr": "quality",
    "ssim": "quality",
    # Real time
    "physical_fps_error": "real_time",
    "physical_fps_pct_error": "real_time",
    "physical_fps_intra_video_cv": "real_time",
    "effective_generation_fps": "real_time",
    "realtime_factor": "real_time",
    "action_response_latency": "real_time",
    "latency_budget_pass_rate": "real_time",
}


def group_metrics_by_dimension(metric_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for metric_name in metric_names:
        dimension = METRIC_DIMENSIONS.get(metric_name, "uncategorized")
        grouped.setdefault(dimension, []).append(metric_name)
    return grouped


__all__ = [
    "TOP_LEVEL_DIMENSIONS",
    "DIMENSION_LABELS",
    "METRIC_DIMENSIONS",
    "group_metrics_by_dimension",
]
