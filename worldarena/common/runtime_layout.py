from __future__ import annotations

import shutil
from pathlib import Path


DEFAULT_MODEL_LAYOUT = {
    "lingbot-world": {
        "repo_root": "third_party/lingbot-world",
        "checkpoint_dir": "cache/hfd/robbyant--lingbot-world-base-cam",
    },
    "matrix-game-1": {
        "repo_root": "third_party/Matrix-Game-1",
        "checkpoint_dir": "cache/hfd/Matrix-Game-1.0",
    },
    "matrix-game-2": {
        "repo_root": "third_party/Matrix-Game-2",
        "checkpoint_dir": "cache/hfd/Skywork--Matrix-Game-2.0",
    },
    "matrix-game": {
        "repo_root": "third_party/Matrix-Game-3.0",
        "checkpoint_dir": "cache/hfd/Matrix-Game-3.0",
    },
    "lyra-1": {
        "repo_root": "third_party/Lyra-1",
        "checkpoint_dir": "cache/hfd/Lyra",
    },
    "lyra-2": {
        "repo_root": "third_party/Lyra-2",
        "checkpoint_dir": "cache/hfd/Lyra-2.0",
    },
    "infinite-world": {
        "repo_root": "third_party/Infinite-World",
        "checkpoint_dir": "cache/hfd/MeiGen-AI--Infinite-World",
    },
    "worldcam": {
        "repo_root": "third_party/WorldCam",
        "checkpoint_dir": "cache/hfd/worldcam--worldcam",
    },
    "worldfm": {
        "repo_root": "third_party/worldfm",
        "checkpoint_dir": "cache/hfd/inspatio--worldfm",
    },
    "HunyuanWorld-1.0": {
        "repo_root": "third_party/HunyuanWorld-1.0",
        "checkpoint_dir": "cache/hfd/tencent--HunyuanWorld-1",
    },
}
DEFAULT_RUNTIME_PYTHONS = {
    "managed": ".venv/bin/python",
    "runtime312": ".runtime312/bin/python",
    "runtime310": ".runtime310/bin/python",
    "runtime": ".runtime/bin/python",
}


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _ensure_symlink(link_path: Path, target_path: Path, force: bool) -> None:
    if link_path.exists() or link_path.is_symlink():
        same_target = link_path.is_symlink() and link_path.resolve() == target_path
        if same_target:
            return
        if not force:
            raise ValueError(
                f"{link_path} already exists. Pass force=True to replace it or reuse the current layout."
            )
        _remove_path(link_path)
    link_path.symlink_to(target_path, target_is_directory=True)


def _recommended_env(cache_root: Path) -> dict[str, str]:
    return {
        "HF_HOME": str(cache_root),
        "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
    }


def _expected_model_layout(root: Path) -> dict[str, dict[str, str]]:
    return {
        name: {
            "repo_root": str((root / values["repo_root"]).resolve()),
            "checkpoint_dir": str((root / values["checkpoint_dir"]).resolve()),
        }
        for name, values in DEFAULT_MODEL_LAYOUT.items()
    }


def _runtime_python_layout(root: Path) -> dict[str, dict[str, str | bool | None]]:
    layout: dict[str, dict[str, str | bool | None]] = {}
    for name, relative_path in DEFAULT_RUNTIME_PYTHONS.items():
        path = root / relative_path
        layout[name] = {
            "path": str(path.resolve(strict=False)),
            "exists": path.exists(),
            "resolved_path": str(path.resolve()) if path.exists() else None,
        }
    return layout


def inspect_runtime_layout(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    cache_root = root / "cache" / "hfd"
    third_party_root = root / "third_party"

    if cache_root.is_symlink():
        cache_mode = "symlink"
        cache_storage_root: Path | None = cache_root.resolve()
    elif cache_root.is_dir():
        cache_mode = "directory"
        cache_storage_root = cache_root
    elif cache_root.exists():
        cache_mode = "file"
        cache_storage_root = cache_root
    else:
        cache_mode = "missing"
        cache_storage_root = None

    linked_repositories: dict[str, dict[str, str | bool]] = {}
    if third_party_root.exists():
        for child in sorted(third_party_root.iterdir()):
            if child.name.startswith("."):
                continue
            if not child.is_dir() and not child.is_symlink():
                continue
            linked_repositories[child.name] = {
                "link_path": str(child),
                "source_path": str(child.resolve()) if child.exists() or child.is_symlink() else str(child),
                "is_symlink": child.is_symlink(),
            }

    return {
        "project_root": str(root),
        "cache_root": str(cache_root),
        "cache_storage_root": str(cache_storage_root) if cache_storage_root is not None else None,
        "cache_mode": cache_mode,
        "third_party_root": str(third_party_root),
        "third_party_exists": third_party_root.exists(),
        "recommended_env": _recommended_env(cache_root),
        "linked_repositories": linked_repositories,
        "expected_model_layout": _expected_model_layout(root),
        "runtime_pythons": _runtime_python_layout(root),
    }


def prepare_runtime_layout(
    project_root: Path,
    shared_hf_cache: str | None = None,
    repo_links: dict[str, str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    root = project_root.resolve()
    cache_parent = _ensure_directory(root / "cache")
    third_party_root = _ensure_directory(root / "third_party")
    cache_root = cache_parent / "hfd"

    if shared_hf_cache:
        storage_root = Path(shared_hf_cache).expanduser().resolve()
        _ensure_directory(storage_root)
        _ensure_symlink(cache_root, storage_root, force=force)
        cache_mode = "symlink"
    elif cache_root.is_symlink():
        storage_root = cache_root.resolve()
        _ensure_directory(storage_root)
        cache_mode = "symlink"
    else:
        storage_root = _ensure_directory(cache_root)
        cache_mode = "directory"

    _ensure_directory(storage_root / "huggingface" / "hub")
    _ensure_directory(storage_root / "transformers")

    linked_repositories: dict[str, dict[str, str]] = {}
    for name, source in sorted((repo_links or {}).items()):
        target = Path(source).expanduser().resolve()
        if not target.exists():
            raise ValueError(f"Third-party source path does not exist: {target}")
        link_path = third_party_root / name
        _ensure_symlink(link_path, target, force=force)
        linked_repositories[name] = {
            "link_path": str(link_path),
            "source_path": str(target),
        }

    return {
        "project_root": str(root),
        "cache_root": str(cache_root),
        "cache_storage_root": str(storage_root),
        "cache_mode": cache_mode,
        "third_party_root": str(third_party_root),
        "recommended_env": _recommended_env(cache_root),
        "linked_repositories": linked_repositories,
        "expected_model_layout": _expected_model_layout(root),
        "runtime_pythons": _runtime_python_layout(root),
    }
