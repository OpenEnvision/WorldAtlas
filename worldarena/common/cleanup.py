"""Remove generated artifacts and cache directories from the project tree."""

from __future__ import annotations

import shutil
from pathlib import Path


def collect_cleanup_targets(project_root: Path, include_generated: bool = False) -> list[Path]:
    targets: list[Path] = []

    for relative in [".cache", ".pytest_cache", ".ruff_cache", "site", "artifacts/dataset_stats/cache"]:
        path = project_root / relative
        if path.exists():
            targets.append(path)

    for path in project_root.glob("*.egg-info"):
        if path.exists():
            targets.append(path)

    for package_dir in ["worldarena", "tests", "scripts"]:
        root = project_root / package_dir
        if not root.exists():
            continue
        for path in root.rglob("__pycache__"):
            if path.is_dir():
                targets.append(path)

    if include_generated:
        for relative in ["docs/generated", "artifacts/dataset_stats"]:
            path = project_root / relative
            if path.exists():
                targets.append(path)

    unique_targets = sorted({path.resolve() for path in targets}, key=lambda path: len(path.parts))
    return unique_targets


def clean_project(project_root: Path, include_generated: bool = False) -> list[str]:
    removed: list[str] = []
    for path in collect_cleanup_targets(project_root, include_generated=include_generated):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        removed.append(str(path.relative_to(project_root)))
    return removed
