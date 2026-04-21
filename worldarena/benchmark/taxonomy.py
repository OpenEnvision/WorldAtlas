from __future__ import annotations

import hashlib


def humanize_label(value: str | None) -> str:
    if not value:
        return "scene"
    return value.replace("_", " ").replace("-", " ").strip()


def build_prompts(
    asset_level: str,
    style: str | None,
    environment: str | None,
    scene: str | None,
    motion_category: str | None,
) -> tuple[str, str]:
    style_text = humanize_label(style)
    if asset_level in {"image_static", "video_static"}:
        environment_text = humanize_label(environment)
        scene_text = humanize_label(scene)
        current = f"A {style_text} {environment_text} {scene_text} scene."
        if asset_level == "image_static":
            target = f"Generate a {style_text} still image of a {environment_text} {scene_text} scene."
        else:
            target = f"Generate a {style_text} video of a coherent {environment_text} {scene_text} scene."
        return current, target

    motion_text = humanize_label(motion_category)
    current = f"A {style_text} scene with {motion_text} motion."
    if asset_level in {"image_dynamic", "image_nonformal"}:
        target = f"Generate a {style_text} image that clearly conveys {motion_text} motion."
    else:
        target = f"Generate a {style_text} video with {motion_text} motion and coherent appearance."
    return current, target


def assign_split(source_group_id: str, split_ratios: dict[str, float]) -> str:
    token = hashlib.md5(source_group_id.encode("utf-8")).hexdigest()[:8]
    value = int(token, 16) / 0xFFFFFFFF
    cumulative = 0.0
    for split_name in ("dev_public", "dev_private", "test_hidden"):
        cumulative += split_ratios.get(split_name, 0.0)
        if value <= cumulative or split_name == "test_hidden":
            return split_name
    return "test_hidden"


def suite_for_asset(
    asset_level: str,
    is_formal: bool,
    has_annotation: bool,
    has_instruction: bool,
    has_pose: bool,
    has_mask: bool,
) -> str:
    if is_formal:
        return asset_level
    if has_annotation or has_instruction or has_pose or has_mask:
        return "experimental"
    return "auxiliary"


def prediction_extensions(modality: str) -> tuple[str, ...]:
    if modality == "video":
        return (".mp4", ".mov", ".avi", ".mkv", ".webm")
    return (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def build_sample_id(asset_level: str, style: str | None, fine_class: str | None, index: int) -> str:
    style_token = (style or "unknown").replace("-", "_")
    class_token = (fine_class or "unknown").replace("-", "_")
    return f"{asset_level}_{style_token}_{class_token}_{index:04d}"
