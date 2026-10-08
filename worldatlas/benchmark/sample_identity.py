"""Stable image IDs and explicit, content-pinned legacy prediction bindings."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import re
from typing import Any

from worldatlas.benchmark.taxonomy import build_sample_id

SOURCE_FILENAME_POLICY = "source_filename_v1"
IMAGE_LEVELS = {"image_static", "image_dynamic"}
_FILE_DIGESTS: ContextVar[dict | None] = ContextVar("file_digests", default=None)


@contextmanager
def cache_file_digests():
    """Share digests for fixed inputs within one generation or evaluation run."""
    if _FILE_DIGESTS.get() is not None:
        yield
        return
    token = _FILE_DIGESTS.set({})
    try:
        yield
    finally:
        _FILE_DIGESTS.reset(token)


def file_sha256(path: Path) -> str:
    """Reuse unchanged files within an active run; read fresh outside it."""
    path = path.resolve()
    cache = _FILE_DIGESTS.get()
    if cache is not None:
        stat = path.stat()
        key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key in cache:
            return cache[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    if cache is not None:
        cache[key] = result
    return result


def prediction_relative_path(binding: dict[str, Any]) -> Path:
    """Resolve both relative bindings and existing absolute-path manifests."""
    if not binding.get("path"):
        raise ValueError("Prediction binding has no path")
    path = Path(binding["path"])
    if path.is_absolute():
        path = path.relative_to(Path(binding["predictions_root"]))
    return path


def image_sample_id(asset_level: str, style: str | None, fine_class: str | None,
                    source_path: str) -> str:
    """2.jpg and 002.png mean image 2, never the second sorted entry."""
    stem = Path(source_path).stem
    if asset_level not in IMAGE_LEVELS or not re.fullmatch(r"[0-9]+", stem):
        raise ValueError(f"{SOURCE_FILENAME_POLICY} requires a numeric image filename: {source_path}")
    return build_sample_id(asset_level, style, fine_class, int(stem))


def validate_source_identity(sample: Any) -> str:
    if sample.sample_id_policy != SOURCE_FILENAME_POLICY:
        raise ValueError(f"Unknown image identity policy: {sample.sample_id_policy}")
    category = sample.motion_category if sample.asset_level == "image_dynamic" else sample.scene
    expected = image_sample_id(sample.asset_level, sample.style, category, sample.path)
    if sample.sample_id != expected or sample.prediction_stem != expected:
        raise ValueError(f"Image ID must follow its filename: {sample.sample_id} != {expected}")
    source_hash = file_sha256(Path(sample.path))
    if Path(sample.reference_path).resolve() != Path(sample.path).resolve():
        raise ValueError(f"Image reference_path differs from verified source: {sample.sample_id}")
    if sample.source_sha256 and sample.source_sha256 != source_hash:
        raise ValueError(f"Source image changed: {sample.path}")
    return source_hash


def resolve_bound_prediction(sample: Any, predictions_root: Path) -> Path:
    """Never infer a migrated video's identity from its filename alone.

    A binding is supplied by a separately verified per-model manifest. Hashes
    pin the verified source and video; they do NOT themselves establish that
    the video was generated from that image. The evidence must establish that.
    """
    source_hash = validate_source_identity(sample)
    binding = sample.prediction_binding
    if not isinstance(binding, dict) or not binding.get("evidence"):
        raise ValueError(
            f"{sample.sample_id}: {SOURCE_FILENAME_POLICY} requires a verified per-video "
            "prediction_binding; same-name legacy videos are not trusted. "
            "Do not evaluate the unbound canonical manifest directly."
        )
    if binding.get("sample_id") != sample.sample_id or binding.get("source_sha256") != source_hash:
        raise ValueError(f"Prediction binding belongs to a different image: {sample.sample_id}")
    root = predictions_root.resolve()
    path = root / prediction_relative_path(binding)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Bound prediction is missing or empty: {path}")
    if file_sha256(path) != binding.get("prediction_sha256"):
        raise ValueError(f"Bound prediction changed: {path}")
    return path
