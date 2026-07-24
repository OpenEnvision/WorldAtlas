"""Index and resolve annotation sidecars referenced by ``annotation_path::`` syntax."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from worldarena.common.checkpoints import rehome_legacy_workspace_path


ANNOTATION_INDEX_SEPARATOR = "::"


def normalize_relative_annotation_path(relative_path: str | Path) -> str:
    return str(relative_path).replace("\\", "/")


def build_annotation_reference(index_path: Path, relative_path: str | Path) -> str:
    return (
        f"{index_path.expanduser().resolve()}"
        f"{ANNOTATION_INDEX_SEPARATOR}"
        f"{normalize_relative_annotation_path(relative_path)}"
    )


def split_annotation_reference(annotation_path: str | None) -> tuple[Path, str] | None:
    if not annotation_path or ANNOTATION_INDEX_SEPARATOR not in annotation_path:
        return None
    index_path_text, relative_path = annotation_path.split(ANNOTATION_INDEX_SEPARATOR, 1)
    normalized_relative_path = normalize_relative_annotation_path(relative_path)
    if not index_path_text or not normalized_relative_path:
        return None
    remapped = rehome_legacy_workspace_path(index_path_text)
    if remapped is None:
        return None
    return Path(remapped).expanduser(), normalized_relative_path


def _normalize_entry(relative_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["relative_path"] = normalize_relative_annotation_path(
        normalized.get("relative_path", relative_path)
    )
    return normalized


@lru_cache(maxsize=8)
def _load_annotation_index_cached(
    index_path_str: str,
    mtime_ns: int,
    size_bytes: int,
) -> dict[str, dict[str, Any]]:
    del mtime_ns
    del size_bytes

    payload = json.loads(Path(index_path_str).read_text(encoding="utf-8"))
    items: Any
    if isinstance(payload, dict):
        items = payload.get("items", [])
    else:
        items = payload

    entries: dict[str, dict[str, Any]] = {}
    if isinstance(items, dict):
        iterator = items.items()
        for relative_path, entry in iterator:
            if isinstance(relative_path, str) and isinstance(entry, dict):
                normalized_relative_path = normalize_relative_annotation_path(relative_path)
                entries[normalized_relative_path] = _normalize_entry(normalized_relative_path, entry)
        return entries

    if not isinstance(items, list):
        return entries

    for item in items:
        if not isinstance(item, dict):
            continue
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            continue
        normalized_relative_path = normalize_relative_annotation_path(relative_path)
        entries[normalized_relative_path] = _normalize_entry(normalized_relative_path, item)
    return entries


def load_annotation_index(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.exists() or not index_path.is_file():
        return {}
    stat = index_path.stat()
    return _load_annotation_index_cached(
        str(index_path.expanduser().resolve()),
        stat.st_mtime_ns,
        stat.st_size,
    )


def load_annotation_entry(index_path: Path, relative_path: str | Path) -> dict[str, Any] | None:
    return load_annotation_index(index_path).get(normalize_relative_annotation_path(relative_path))


def load_annotation_entry_from_reference(annotation_path: str | None) -> dict[str, Any] | None:
    parsed = split_annotation_reference(annotation_path)
    if parsed is None:
        return None
    index_path, relative_path = parsed
    return load_annotation_entry(index_path, relative_path)


__all__ = [
    "ANNOTATION_INDEX_SEPARATOR",
    "build_annotation_reference",
    "load_annotation_entry",
    "load_annotation_entry_from_reference",
    "load_annotation_index",
    "normalize_relative_annotation_path",
    "split_annotation_reference",
]
