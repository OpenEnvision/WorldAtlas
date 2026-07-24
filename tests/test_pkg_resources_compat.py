from __future__ import annotations

import sys
import types
from pathlib import Path

from worldarena.benchmark.pkg_resources_compat import (
    _resolve_openai_clip_cache_dir,
    ensure_openai_clip_cache_compat,
)
from worldarena.common import checkpoints


def _patch_checkpoint_root(monkeypatch, tmp_path: Path) -> Path:
    repo_root = tmp_path / "WorldAtlas Arena"
    ckpt_root = repo_root / "ckpt"
    monkeypatch.setattr(checkpoints, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(checkpoints, "CANONICAL_CHECKPOINT_ROOT", ckpt_root)
    return ckpt_root


def test_resolve_openai_clip_cache_dir_prefers_ckpt_torch_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ckpt_root = _patch_checkpoint_root(monkeypatch, tmp_path)
    cache_dir = ckpt_root / "torch" / "hub" / "clip"
    cache_dir.mkdir(parents=True)
    (cache_dir / "ViT-L-14.pt").write_bytes(b"stub")

    resolved = _resolve_openai_clip_cache_dir()

    assert resolved == cache_dir


def test_ensure_openai_clip_cache_compat_only_overrides_download_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ckpt_root = _patch_checkpoint_root(monkeypatch, tmp_path)
    cache_dir = ckpt_root / "torch" / "hub" / "clip"
    cache_dir.mkdir(parents=True)
    (cache_dir / "ViT-B-32.pt").write_bytes(b"stub")

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    clip_module = types.ModuleType("clip")
    clip_impl_module = types.ModuleType("clip.clip")
    clip_module.load = fake_load
    clip_impl_module.load = fake_load
    monkeypatch.setitem(sys.modules, "clip", clip_module)
    monkeypatch.setitem(sys.modules, "clip.clip", clip_impl_module)

    ensure_openai_clip_cache_compat()

    assert clip_module.load is clip_impl_module.load
    assert clip_impl_module.load is not fake_load

    clip_impl_module.load("ViT-L/14", "cuda:3", True)
    clip_impl_module.load("ViT-L/14", "cuda:7", False, "/tmp/manual-clip-cache")
    clip_impl_module.load("ViT-L/14", download_root="/tmp/explicit-cache")

    assert calls == [
        (
            ("ViT-L/14", "cuda:3", True),
            {"download_root": str(cache_dir)},
        ),
        (
            ("ViT-L/14", "cuda:7", False, "/tmp/manual-clip-cache"),
            {},
        ),
        (
            ("ViT-L/14",),
            {"download_root": "/tmp/explicit-cache"},
        ),
    ]
