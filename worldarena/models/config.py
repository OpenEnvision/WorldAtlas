"""Model runtime configuration loaded from YAML model config files."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml

from worldarena.common.checkpoints import (
    filter_checkpoint_env_overrides,
    project_root as worldarena_project_root,
    resolve_checkpoint_path,
    resolve_project_path,
)
from worldarena.common.taxonomy import (
    default_artifact_types_for_suites,
    default_control_signals,
    default_task_families_for_suites,
    normalize_artifact_types,
    normalize_backend_type,
    normalize_control_signals,
    normalize_model_roles,
    normalize_task_families,
)


@dataclass(slots=True)
class ModelRuntimeConfig:
    """Resolved settings for one benchmarked world model."""

    config_path: Path
    name: str
    family: str
    python_bin: str
    repo_root: Path | None
    checkpoint_dir: Path | None
    entrypoint: str | None
    output_ext: str
    supported_suites: list[str]
    prompt_mode: str
    reference_mode: str
    backend_type: str = "local_repo"
    task_families: list[str] = field(default_factory=list)
    artifact_types: list[str] = field(default_factory=list)
    control_signals: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=lambda: ["candidate"])
    env: dict[str, str] = field(default_factory=dict)
    metric_overrides: dict[str, list[str]] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    api: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_python_bin(project_root: Path, value: str | None) -> str:
    raw = str(value or "python")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if raw.startswith(".") or "/" in raw:
        return os.path.abspath(project_root / path)
    return raw


def _resolve_model_checkpoint_dir(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return resolve_project_path(path)
    return resolve_checkpoint_path(path)


def load_model_config(path: Path) -> ModelRuntimeConfig:
    config_path = path.resolve()
    project_root = worldarena_project_root()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_payload = payload.get("model", {})
    supported_suites = [str(value) for value in model_payload.get("supported_suites", [])]
    prompt_mode = str(model_payload.get("prompt_mode", "prompt_target"))
    reference_mode = str(model_payload.get("reference_mode", "first_frame"))
    backend_type = normalize_backend_type(model_payload.get("backend_type", "local_repo"))
    repo_root = resolve_project_path(model_payload.get("repo_root"))
    entrypoint = model_payload.get("entrypoint")
    if entrypoint is not None:
        entrypoint = str(entrypoint)
    if backend_type in {"local_repo", "python_module"}:
        if repo_root is None:
            raise ValueError(
                f"model {model_payload.get('name', '<unknown>')} requires repo_root for backend_type={backend_type}"
            )
        if not entrypoint:
            entrypoint = "generate.py"

    default_task_families = default_task_families_for_suites(supported_suites)
    default_artifact_types = default_artifact_types_for_suites(supported_suites)
    default_controls = default_control_signals(
        prompt_mode=prompt_mode,
        reference_mode=reference_mode,
    )

    return ModelRuntimeConfig(
        config_path=config_path,
        name=str(model_payload["name"]),
        family=str(model_payload["family"]),
        backend_type=backend_type,
        task_families=normalize_task_families(
            model_payload.get("task_families", default_task_families)
        ),
        artifact_types=normalize_artifact_types(
            model_payload.get("artifact_types", default_artifact_types)
        ),
        control_signals=normalize_control_signals(
            model_payload.get("control_signals", default_controls)
        ),
        roles=normalize_model_roles(model_payload.get("roles", ["candidate"])),
        python_bin=_resolve_python_bin(project_root, model_payload.get("python_bin", "python")),
        repo_root=repo_root,
        checkpoint_dir=_resolve_model_checkpoint_dir(model_payload.get("checkpoint_dir")),
        entrypoint=entrypoint,
        output_ext=str(model_payload.get("output_ext", ".mp4")),
        supported_suites=supported_suites,
        prompt_mode=prompt_mode,
        reference_mode=reference_mode,
        env=filter_checkpoint_env_overrides(dict(model_payload.get("env", {}))),
        metric_overrides={
            str(key): [str(metric) for metric in value]
            for key, value in model_payload.get("metric_overrides", {}).items()
        },
        generation=dict(model_payload.get("generation", {})),
        api=dict(model_payload.get("api", {})),
        metadata=dict(model_payload.get("metadata", {})),
    )
