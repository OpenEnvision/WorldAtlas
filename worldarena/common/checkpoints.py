from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


CHECKPOINT_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORKSPACE_ROOT = Path("/bench-workspace/yesliu/hjd")
CANONICAL_CHECKPOINT_ROOT = PROJECT_ROOT / "ckpt"
HF_REPO_SUBPATHS: dict[str, tuple[str, ...]] = {
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": ("depth-anything--DA3NESTED-GIANT-LARGE",),
    "depth-anything/Depth-Anything-V2-Small-hf": ("models--depth-anything--Depth-Anything-V2-Small-hf",),
    "facebook/dinov2-base": ("models--facebook--dinov2-base",),
    "google-t5/t5-11b": ("t5-11b",),
    "google/umt5-xxl": ("google", "umt5-xxl"),
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p": ("Cosmos-Tokenize1-CV8x8x8-720p",),
    "openai/clip-vit-base-patch16": ("models--openai--clip-vit-base-patch16",),
}


@dataclass(frozen=True, slots=True)
class CheckpointCacheEnv:
    hf_home: Path
    hf_hub_cache: Path
    transformers_cache: Path
    torch_home: Path


def project_root() -> Path:
    """Return the WorldArena repository root used for all relative path resolution."""
    return PROJECT_ROOT


def workspace_root() -> Path:
    """Return the workspace root that contains this WorldArena checkout."""
    return PROJECT_ROOT.parent.parent


def rehome_legacy_workspace_path(value: str | os.PathLike[str] | None) -> Path | None:
    """Map stale /bench-workspace paths onto the active workspace when possible."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        return path
    try:
        relative = path.relative_to(LEGACY_WORKSPACE_ROOT)
    except ValueError:
        return path
    remapped = (workspace_root() / relative).resolve()
    if remapped.exists():
        return remapped
    return path


def checkpoint_root(*, create: bool = True) -> Path:
    """Return the single checkpoint root under the WorldArena repository."""
    root = canonical_checkpoint_root()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def canonical_checkpoint_root() -> Path:
    """Return the user-facing canonical checkpoint root path."""
    return CANONICAL_CHECKPOINT_ROOT


def resolve_project_path(value: str | os.PathLike[str] | None) -> Path | None:
    """Resolve a project-relative or absolute path against the WorldArena root."""
    if value is None:
        return None
    path = rehome_legacy_workspace_path(value)
    if path is None:
        return None
    target = path if path.is_absolute() else project_root() / path
    return Path(os.path.abspath(str(target)))


def _validate_existing_path_kind(path: Path, kind: str) -> None:
    if kind == "any":
        return
    if kind == "dir" and path.exists() and not path.is_dir():
        raise FileNotFoundError(f"expected checkpoint directory, found non-directory path: {path}")
    if kind == "file" and path.exists() and not path.is_file():
        raise FileNotFoundError(f"expected checkpoint file, found non-file path: {path}")


def resolve_checkpoint_path(
    value: str | os.PathLike[str] | None,
    *,
    kind: str = "any",
    required: bool = False,
) -> Path | None:
    """Resolve a path and require it to stay under the single ckpt root."""
    path = resolve_project_path(value)
    if path is None:
        return None
    root = checkpoint_root()
    path_resolved = path.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    if (
        path != root
        and not path.is_relative_to(root)
        and path_resolved != root_resolved
        and not path_resolved.is_relative_to(root_resolved)
    ):
        raise ValueError(
            f"checkpoint path must stay under {canonical_checkpoint_root()}, got {path}"
        )
    _validate_existing_path_kind(path, kind)
    if required:
        if kind == "dir" and not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {path}")
        if kind == "file" and not path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {path}")
        if kind == "any" and not path.exists():
            raise FileNotFoundError(f"checkpoint path not found: {path}")
    return path


def checkpoint_path(*parts: str, kind: str = "any", required: bool = False) -> Path:
    """Build a deterministic path under the single ckpt root."""
    path = Path(os.path.abspath(str(checkpoint_root() / Path(*parts))))
    _validate_existing_path_kind(path, kind)
    if required:
        if kind == "dir" and not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {path}")
        if kind == "file" and not path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {path}")
        if kind == "any" and not path.exists():
            raise FileNotFoundError(f"checkpoint path not found: {path}")
    return path


def _hf_repo_root(repo_id: str) -> Path:
    relative_parts = HF_REPO_SUBPATHS.get(repo_id)
    if relative_parts is None:
        relative_parts = (repo_id.replace("/", "--"),)
    return checkpoint_path(*relative_parts, kind="dir", required=False)


def _hf_snapshot_dir(cache_root: Path) -> Path | None:
    refs_main = cache_root / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = cache_root / "snapshots" / revision
        if snapshot_dir.is_dir():
            return snapshot_dir
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.is_dir():
        return None
    snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
    if len(snapshots) == 1:
        return snapshots[0]
    return None


def hf_local_dir(repo_id: str, *, required: bool = False) -> Path:
    """Return the canonical direct-download directory for a Hugging Face repo."""
    root = _hf_repo_root(repo_id)
    resolved = _hf_snapshot_dir(root) or root
    if required and not resolved.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {resolved}")
    return resolved


def checkpoint_cache_env(*, create: bool = True) -> CheckpointCacheEnv:
    """Return the fixed cache directories that keep HF/Torch downloads under ckpt."""
    root = checkpoint_root(create=create)
    hf_home = (root / "huggingface").resolve()
    hf_hub_cache = (hf_home / "hub").resolve()
    transformers_cache = (root / "transformers").resolve()
    torch_home = (root / "torch").resolve()
    if create:
        for path in (hf_home, hf_hub_cache, transformers_cache, torch_home):
            path.mkdir(parents=True, exist_ok=True)
    return CheckpointCacheEnv(
        hf_home=hf_home,
        hf_hub_cache=hf_hub_cache,
        transformers_cache=transformers_cache,
        torch_home=torch_home,
    )


def checkpoint_env(*, create: bool = True) -> dict[str, str]:
    """Return fixed env vars that pin runtime caches under the ckpt root."""
    cache_env = checkpoint_cache_env(create=create)
    return {
        "HF_HOME": str(cache_env.hf_home),
        "HF_HUB_CACHE": str(cache_env.hf_hub_cache),
        "TRANSFORMERS_CACHE": str(cache_env.transformers_cache),
        "TORCH_HOME": str(cache_env.torch_home),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def filter_checkpoint_env_overrides(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    """Drop env overrides that would try to redirect checkpoint caches elsewhere."""
    if not overrides:
        return {}
    return {
        str(key): str(value)
        for key, value in overrides.items()
        if str(key) not in CHECKPOINT_ENV_KEYS
    }


def apply_checkpoint_env(
    env: Mapping[str, str] | None = None,
    *,
    create: bool = True,
) -> dict[str, str]:
    """Overlay the fixed checkpoint cache env onto an existing environment."""
    payload = dict(env or os.environ.copy())
    payload.update(checkpoint_env(create=create))
    return payload


__all__ = [
    "CANONICAL_CHECKPOINT_ROOT",
    "CHECKPOINT_ENV_KEYS",
    "CheckpointCacheEnv",
    "HF_REPO_SUBPATHS",
    "apply_checkpoint_env",
    "canonical_checkpoint_root",
    "checkpoint_cache_env",
    "checkpoint_env",
    "checkpoint_path",
    "checkpoint_root",
    "filter_checkpoint_env_overrides",
    "hf_local_dir",
    "project_root",
    "rehome_legacy_workspace_path",
    "resolve_checkpoint_path",
    "resolve_project_path",
    "workspace_root",
]
