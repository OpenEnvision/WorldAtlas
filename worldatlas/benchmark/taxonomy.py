"""Benchmark taxonomy: suites, task families, modalities, and sample ID conventions."""

from __future__ import annotations


def artifact_type_for_modality(modality: str) -> str:
    return "video" if modality == "video" else "image"


BENCHMARK_SPLIT = "test"

MEMORY_SUITE = "memory_loop"
VIDEO_PREDICTION_SUITES = (
    "image_static",
    "image_dynamic",
    "video_static",
    "video_dynamic",
    MEMORY_SUITE,
)
# Suites whose camera trajectory is synthesized from ``camera_path`` tokens rather than
# recovered from a reference video. Metrics use this to decide whether a nominal
# trajectory exists for a sample at all.
CAMERA_PATH_SUITES = ("image_static", MEMORY_SUITE)


def benchmark_split() -> str:
    """Benchmark split -> str."""
    return BENCHMARK_SPLIT


def is_camera_path_suite(suite: str | None) -> bool:
    """Return True when a suite drives generation from synthesized camera poses."""
    return str(suite or "").strip().lower() in CAMERA_PATH_SUITES


def is_memory_suite(suite: str | None) -> bool:
    """Return True for the closed-loop memory evaluation track."""
    return str(suite or "").strip().lower() == MEMORY_SUITE


def prediction_modality_for_task(
    *,
    suite: str | None,
    source_modality: str | None,
    task_family: str | None = None,
    artifact_type: str | None = None,
) -> str:
    suite_token = str(suite or "").strip().lower()
    task_token = str(task_family or "").strip().lower()
    artifact_token = str(artifact_type or "").strip().lower()
    source_token = str(source_modality or "").strip().lower()

    if task_token in {"video_generation", "world_model"}:
        return "video"
    if suite_token in VIDEO_PREDICTION_SUITES:
        return "video"
    if suite_token.startswith("image_") or suite_token.startswith("video_"):
        return "video"
    if artifact_token == "video":
        return "video"
    if artifact_token == "image":
        return "image"
    return "video" if source_token == "video" else "image"


def task_family_for_suite(suite: str, *, track: str | None = None) -> str:
    if track and track.startswith("physics"):
        return "world_model"
    if is_memory_suite(suite):
        return "world_model"
    return "video_generation"


def control_signals_for_sample(
    *,
    modality: str,
    conditioning_strategy: str,
    has_pose: bool,
) -> list[str]:
    control_signals = ["text"]
    if conditioning_strategy in {"first_frame", "reference_image", "external_image"}:
        control_signals.append("image")
    elif conditioning_strategy in {"reference_video", "external_video"}:
        control_signals.append("video")
    elif modality == "video":
        control_signals.append("image")

    if has_pose and "camera_pose" not in control_signals:
        control_signals.append("camera_pose")

    return control_signals


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
            target = (
                f"Generate a {style_text} video rollout from the reference image of a "
                f"{environment_text} {scene_text} scene with stable scene identity."
            )
        else:
            target = f"Generate a {style_text} video of a coherent {environment_text} {scene_text} scene."
        return current, target

    motion_text = humanize_label(motion_category)
    motion_control = motion_text if motion_text.endswith("motion") else f"{motion_text} motion"
    current = f"A {style_text} scene with {motion_control}."
    if asset_level in {"image_dynamic", "image_nonformal"}:
        target = (
            f"Generate a {style_text} video rollout from the reference image with "
            f"clear {motion_control} and coherent appearance."
        )
    else:
        target = f"Generate a {style_text} video with {motion_control} and coherent appearance."
    return current, target


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


__all__ = [
    "artifact_type_for_modality",
    "BENCHMARK_SPLIT",
    "benchmark_split",
    "build_prompts",
    "CAMERA_PATH_SUITES",
    "MEMORY_SUITE",
    "VIDEO_PREDICTION_SUITES",
    "build_sample_id",
    "control_signals_for_sample",
    "humanize_label",
    "is_camera_path_suite",
    "is_memory_suite",
    "prediction_extensions",
    "prediction_modality_for_task",
    "suite_for_asset",
    "task_family_for_suite",
]
