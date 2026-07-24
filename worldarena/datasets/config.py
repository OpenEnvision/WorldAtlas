"""Dataset collection configuration loaded from ``config/datasets.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class CollectionConfig:
    name: str
    scanner: str
    modality: str
    group: str
    root: Path
    description: str = ""
    enabled: bool = True
    exclude_dir_prefixes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProjectPaths:
    artifacts_dir: Path
    docs_generated_dir: Path
    docs_data_dir: Path
    cache_path: Path


@dataclass(slots=True)
class ProjectConfig:
    config_path: Path
    paths: ProjectPaths
    collections: list[CollectionConfig]


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_project_config(path: Path) -> ProjectConfig:
    config_path = path.resolve()
    config_root = config_path.parent
    project_root = config_root.parent
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    project_paths = ProjectPaths(
        artifacts_dir=_resolve(project_root, payload["paths"]["artifacts_dir"]),
        docs_generated_dir=_resolve(project_root, payload["paths"]["docs_generated_dir"]),
        docs_data_dir=_resolve(project_root, payload["paths"]["docs_data_dir"]),
        cache_path=_resolve(project_root, payload["paths"]["cache_path"]),
    )

    collections = [
        CollectionConfig(
            name=item["name"],
            scanner=item["scanner"],
            modality=item["modality"],
            group=item["group"],
            root=_resolve(project_root, item["root"]),
            description=item.get("description", ""),
            enabled=item.get("enabled", True),
            exclude_dir_prefixes=item.get("exclude_dir_prefixes", []),
        )
        for item in payload.get("collections", [])
        if item.get("enabled", True)
    ]

    return ProjectConfig(
        config_path=config_path,
        paths=project_paths,
        collections=collections,
    )
