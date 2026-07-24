"""Shared taxonomy normalization for model configs and benchmark metadata."""

from __future__ import annotations

from typing import Iterable


TASK_FAMILIES = (
    "video_generation",
    "scene_3d_generation",
    "scene_4d_generation",
    "world_model",
    "reconstruction_3d",
    "foundation_oracle",
)
BACKEND_TYPES = (
    "local_repo",
    "python_module",
    "sdk_api",
    "http_api",
)
ARTIFACT_TYPES = (
    "image",
    "video",
    "mesh",
    "gaussian_splat",
    "point_cloud",
    "multiview",
    "camera_trajectory",
    "depth_map",
    "feature_map",
)
CONTROL_SIGNALS = (
    "text",
    "image",
    "video",
    "camera_pose",
    "actions",
    "trajectory",
    "multi_view",
    "depth",
    "mask",
)
MODEL_ROLES = (
    "candidate",
    "oracle",
    "metric_backend",
    "reconstruction",
    "foundation",
)

_TASK_FAMILY_ALIASES = {
    "3d_generation": "scene_3d_generation",
    "4d_generation": "scene_4d_generation",
    "foundation_model": "foundation_oracle",
    "foundation_models": "foundation_oracle",
    "reconstruction": "reconstruction_3d",
    "reconstruction3d": "reconstruction_3d",
    "scene3d_generation": "scene_3d_generation",
    "scene4d_generation": "scene_4d_generation",
    "world_model_rollout": "world_model",
}
_BACKEND_TYPE_ALIASES = {
    "api_http": "http_api",
    "api_sdk": "sdk_api",
    "local": "local_repo",
    "python": "python_module",
}
_ARTIFACT_TYPE_ALIASES = {
    "camera_pose": "camera_trajectory",
    "camera_poses": "camera_trajectory",
    "depth": "depth_map",
    "feature": "feature_map",
    "features": "feature_map",
    "gs": "gaussian_splat",
    "pointcloud": "point_cloud",
}
_CONTROL_SIGNAL_ALIASES = {
    "action": "actions",
    "camera_poses": "camera_pose",
    "conditioning_image": "image",
    "conditioning_video": "video",
    "multiview": "multi_view",
    "pose": "camera_pose",
    "prompt": "text",
    "reference_frame": "image",
}
_MODEL_ROLE_ALIASES = {
    "eval_model": "candidate",
    "foundation_model": "foundation",
    "metric": "metric_backend",
    "oracle_model": "oracle",
}


def normalize_token(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_list(values: str | Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = values.strip()
        return [raw] if raw else []
    return [str(value) for value in values if str(value).strip()]


def _normalize_choices(
    values: str | Iterable[str] | None,
    *,
    aliases: dict[str, str],
    allowed: tuple[str, ...],
    field_name: str,
) -> list[str]:
    normalized: list[str] = []
    for value in _coerce_list(values):
        token = normalize_token(value)
        canonical = aliases.get(token, token)
        if canonical not in allowed:
            allowed_values = ", ".join(allowed)
            raise ValueError(
                f"unsupported value {value!r} for {field_name}; supported values: {allowed_values}"
            )
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def normalize_task_families(values: str | Iterable[str] | None) -> list[str]:
    return _normalize_choices(
        values,
        aliases=_TASK_FAMILY_ALIASES,
        allowed=TASK_FAMILIES,
        field_name="task_families",
    )


def normalize_backend_type(value: str | None) -> str:
    token = normalize_token(value or "local_repo")
    canonical = _BACKEND_TYPE_ALIASES.get(token, token)
    if canonical not in BACKEND_TYPES:
        allowed_values = ", ".join(BACKEND_TYPES)
        raise ValueError(
            f"unsupported value {value!r} for backend_type; supported values: {allowed_values}"
        )
    return canonical


def normalize_artifact_types(values: str | Iterable[str] | None) -> list[str]:
    return _normalize_choices(
        values,
        aliases=_ARTIFACT_TYPE_ALIASES,
        allowed=ARTIFACT_TYPES,
        field_name="artifact_types",
    )


def normalize_control_signals(values: str | Iterable[str] | None) -> list[str]:
    return _normalize_choices(
        values,
        aliases=_CONTROL_SIGNAL_ALIASES,
        allowed=CONTROL_SIGNALS,
        field_name="control_signals",
    )


def normalize_model_roles(values: str | Iterable[str] | None) -> list[str]:
    return _normalize_choices(
        values,
        aliases=_MODEL_ROLE_ALIASES,
        allowed=MODEL_ROLES,
        field_name="roles",
    )


def default_task_families_for_suites(supported_suites: Iterable[str]) -> list[str]:
    suite_tokens = {normalize_token(value) for value in supported_suites if str(value).strip()}
    if not suite_tokens:
        return []
    if any(token.startswith("image_") or token.startswith("video_") for token in suite_tokens):
        return ["video_generation"]
    return []


def default_artifact_types_for_suites(supported_suites: Iterable[str]) -> list[str]:
    suite_tokens = {normalize_token(value) for value in supported_suites if str(value).strip()}
    artifact_types: list[str] = []
    if any(token.startswith("image_") or token.startswith("video_") for token in suite_tokens):
        artifact_types.append("video")
    return artifact_types


def default_control_signals(
    *,
    prompt_mode: str,
    reference_mode: str,
) -> list[str]:
    control_signals: list[str] = []
    if normalize_token(prompt_mode) != "none":
        control_signals.append("text")

    reference_token = normalize_token(reference_mode)
    if reference_token in {"first_frame", "reference_image", "external_image"}:
        control_signals.append("image")
    elif reference_token in {"reference_video", "external_video"}:
        control_signals.append("video")
    elif reference_token in {"camera_pose", "pose"}:
        control_signals.append("camera_pose")

    return control_signals


__all__ = [
    "ARTIFACT_TYPES",
    "BACKEND_TYPES",
    "CONTROL_SIGNALS",
    "MODEL_ROLES",
    "TASK_FAMILIES",
    "default_artifact_types_for_suites",
    "default_control_signals",
    "default_task_families_for_suites",
    "normalize_artifact_types",
    "normalize_backend_type",
    "normalize_control_signals",
    "normalize_model_roles",
    "normalize_task_families",
    "normalize_token",
]
