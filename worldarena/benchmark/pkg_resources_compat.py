from __future__ import annotations

import importlib
import sys
import types
from functools import wraps
from pathlib import Path

from worldarena.common.checkpoints import checkpoint_path, checkpoint_root

def _packaging_namespace() -> object:
    import packaging
    from packaging import version as packaging_version

    if not hasattr(packaging, "version"):
        packaging.version = packaging_version
    return packaging


def ensure_pkg_resources_compat() -> None:
    try:
        import pkg_resources  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        from pip._vendor import pkg_resources as vendored_pkg_resources
    except Exception:
        shim = types.ModuleType("pkg_resources")
        shim.packaging = _packaging_namespace()
        shim.parse_version = shim.packaging.version.parse
        shim.declare_namespace = lambda _name: None
        sys.modules["pkg_resources"] = shim
        return

    if not hasattr(vendored_pkg_resources, "packaging"):
        vendored_pkg_resources.packaging = _packaging_namespace()
    sys.modules["pkg_resources"] = vendored_pkg_resources


def _candidate_openai_clip_cache_dirs() -> list[Path]:
    return [
        checkpoint_path("torch", "hub", "clip"),
        checkpoint_root() / "clip",
    ]


def _resolve_openai_clip_cache_dir() -> Path | None:
    for path in _candidate_openai_clip_cache_dirs():
        try:
            resolved = path.expanduser().resolve()
        except FileNotFoundError:
            continue
        if resolved.is_dir() and any(resolved.glob("*.pt")):
            return resolved
    return None


def resolve_openai_clip_cache_dir() -> Path | None:
    return _resolve_openai_clip_cache_dir()


def ensure_openai_clip_cache_compat() -> None:
    ensure_pkg_resources_compat()

    clip = importlib.import_module("clip")
    clip_impl = importlib.import_module("clip.clip")
    cache_dir = _resolve_openai_clip_cache_dir()
    if cache_dir is None:
        return

    original_load = getattr(clip_impl, "load", None)
    if original_load is None or getattr(original_load, "__worldarena_cache_patched__", False):
        return

    @wraps(original_load)
    def patched_load(*args, **kwargs):
        args_list = list(args)
        if "download_root" in kwargs:
            if kwargs["download_root"] is None:
                kwargs["download_root"] = str(cache_dir)
        elif len(args_list) >= 4:
            if args_list[3] is None:
                args_list[3] = str(cache_dir)
        else:
            kwargs["download_root"] = str(cache_dir)
        return original_load(*tuple(args_list), **kwargs)

    patched_load.__worldarena_cache_patched__ = True
    clip_impl.load = patched_load
    clip.load = patched_load
