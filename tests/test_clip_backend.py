from __future__ import annotations

from pathlib import Path

from worldarena.benchmark.clip_backend import (
    DEFAULT_CLIP_MODEL_NAME,
    load_clip_backend,
    resolve_cached_clip_artifacts,
)
from worldarena.common import checkpoints


def _clear_clip_caches() -> None:
    resolve_cached_clip_artifacts.cache_clear()
    load_clip_backend.cache_clear()


def _write_file(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _populate_processor_snapshot(snapshot_dir: Path) -> None:
    _write_file(snapshot_dir / "config.json", "{}")
    _write_file(snapshot_dir / "preprocessor_config.json", "{}")
    _write_file(snapshot_dir / "tokenizer_config.json", "{}")
    _write_file(snapshot_dir / "tokenizer.json", "{}")


def _patch_checkpoint_root(monkeypatch, tmp_path: Path) -> Path:
    repo_root = tmp_path / "WorldArena"
    ckpt_root = repo_root / "ckpt"
    monkeypatch.setattr(checkpoints, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(checkpoints, "CANONICAL_CHECKPOINT_ROOT", ckpt_root)
    return ckpt_root


def test_resolve_cached_clip_artifacts_uses_canonical_ckpt_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ckpt_root = _patch_checkpoint_root(monkeypatch, tmp_path)
    model_dir = ckpt_root / Path(*checkpoints.HF_REPO_SUBPATHS[DEFAULT_CLIP_MODEL_NAME])

    _populate_processor_snapshot(model_dir)
    (model_dir / "model.safetensors").write_bytes(b"stub")
    _clear_clip_caches()

    artifacts = resolve_cached_clip_artifacts()

    assert artifacts is not None
    assert artifacts.config_dir == model_dir
    assert artifacts.weights_path == model_dir / "model.safetensors"


def test_resolve_cached_clip_artifacts_supports_explicit_repo_id_mapping(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ckpt_root = _patch_checkpoint_root(monkeypatch, tmp_path)
    model_name = "org/custom-clip"
    model_dir = ckpt_root / model_name.replace("/", "--")

    _populate_processor_snapshot(model_dir)
    (model_dir / "model.safetensors").write_bytes(b"stub")
    _clear_clip_caches()

    artifacts = resolve_cached_clip_artifacts(model_name)

    assert artifacts is not None
    assert artifacts.config_dir == model_dir
    assert artifacts.weights_path == model_dir / "model.safetensors"
