from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelRuntimeConfig:
    config_path: Path
    name: str
    family: str
    python_bin: str
    repo_root: Path
    checkpoint_dir: Path | None
    entrypoint: str
    output_ext: str
    supported_suites: list[str]
    prompt_mode: str
    reference_mode: str
    metric_overrides: dict[str, list[str]] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)


def _resolve(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_python_bin(project_root: Path, value: str | None) -> str:
    raw = str(value or "python")
    if raw == "auto":
        for candidate in (
            project_root / ".runtime312" / "bin" / "python",
            project_root / ".runtime310" / "bin" / "python",
            project_root / ".runtime" / "bin" / "python",
        ):
            if candidate.exists():
                return os.path.abspath(candidate)
        return "python"

    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if raw.startswith(".") or "/" in raw:
        return os.path.abspath(project_root / path)
    return raw


def load_model_config(path: Path) -> ModelRuntimeConfig:
    config_path = path.resolve()
    project_root = config_path.parent.parent.parent
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_payload = payload.get("model", {})

    return ModelRuntimeConfig(
        config_path=config_path,
        name=str(model_payload["name"]),
        family=str(model_payload["family"]),
        python_bin=_resolve_python_bin(project_root, model_payload.get("python_bin", "python")),
        repo_root=_resolve(project_root, model_payload["repo_root"]),
        checkpoint_dir=_resolve(project_root, model_payload.get("checkpoint_dir")),
        entrypoint=str(model_payload.get("entrypoint", "generate.py")),
        output_ext=str(model_payload.get("output_ext", ".mp4")),
        supported_suites=[str(value) for value in model_payload.get("supported_suites", [])],
        prompt_mode=str(model_payload.get("prompt_mode", "prompt_target")),
        reference_mode=str(model_payload.get("reference_mode", "first_frame")),
        metric_overrides={
            str(key): [str(metric) for metric in value]
            for key, value in model_payload.get("metric_overrides", {}).items()
        },
        generation=dict(model_payload.get("generation", {})),
    )
