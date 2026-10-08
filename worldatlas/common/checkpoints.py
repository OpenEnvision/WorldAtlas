"""Checkpoint and HuggingFace cache path resolution for models and metrics."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


CHECKPOINT_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)
CHECKPOINT_ROOT_ENV = "WORLDATLAS_CHECKPOINT_ROOT"
CHECKPOINT_CACHE_ROOT_ENV = "WORLDATLAS_CHECKPOINT_CACHE_ROOT"
LEGACY_WORKSPACE_ROOT_ENV = "WORLDATLAS_LEGACY_WORKSPACE_ROOT"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CHECKPOINT_ROOT = PROJECT_ROOT / "ckpt"
HF_REPO_SUBPATHS: dict[str, tuple[str, ...]] = {
    "depth-anything/DA3-GIANT": ("DA3-GIANT",),
    "depth-anything/DA3-GIANT-1.1": ("DA3-GIANT",),
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": ("depth-anything--DA3NESTED-GIANT-LARGE",),
    "facebook/dinov2-base": ("models--facebook--dinov2-base",),
    "google-t5/t5-11b": ("t5-11b",),
    "google/umt5-xxl": ("google", "umt5-xxl"),
    "mhamilton723/FeatUp": ("mhamilton723--FeatUp",),
    "nvidia/Cosmos-Tokenize1-CV8x8x8-720p": ("Cosmos-Tokenize1-CV8x8x8-720p",),
    # Hub snapshot under ckpt/huggingface/hub/. The old flat mapping
    # (ckpt/models--openai--clip-vit-base-patch16) never existed on disk.
    "openai/clip-vit-base-patch16": (
        "huggingface",
        "hub",
        "models--openai--clip-vit-base-patch16",
    ),
    # Direct-download tree used by background_consistency. Keep this first
    # so bgcons keeps resolving the working ViT-B/32 copy.
    "openai/clip-vit-base-patch32": ("openai--clip-vit-base-patch32",),
    "Wan-AI/Wan2.1-T2V-1.3B": ("Wan2.1-T2V-1.3B",),
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": ("Wan2.2-TI2V-5B-Diffusers",),
}


@dataclass(frozen=True, slots=True)
class CheckpointCacheEnv:
    hf_home: Path
    hf_hub_cache: Path
    transformers_cache: Path
    torch_home: Path


def project_root() -> Path:
    """Return the WorldAtlas repository root used for all relative path resolution."""
    return PROJECT_ROOT


def workspace_root() -> Path:
    """Return the workspace root that contains this WorldAtlas checkout."""
    return PROJECT_ROOT.parent.parent


def legacy_workspace_root() -> Path | None:
    """Return an optional old workspace root used for local manifest migration."""
    raw = os.environ.get(LEGACY_WORKSPACE_ROOT_ENV, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _lexical_abspath(path: Path) -> Path:
    """Return an absolute path without following symlinks."""
    return Path(os.path.abspath(str(path.expanduser())))


def rehome_legacy_workspace_path(value: str | os.PathLike[str] | None) -> Path | None:
    """Map stale absolute workspace paths onto the active workspace when requested.

    Existing paths stay at their lexical location. Following symlinks here would
    turn ``ckpt/<model>`` links into ``world_models/...`` real paths, which then
    fail the checkpoint-root check used by inference runners.
    """
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return _lexical_abspath(path)
    if not path.is_absolute():
        return path
    legacy_root = legacy_workspace_root()
    if legacy_root is None:
        return path
    try:
        relative = path.relative_to(legacy_root)
    except ValueError:
        return path
    remapped = _lexical_abspath(workspace_root() / relative)
    if remapped.exists():
        return remapped
    return path


def checkpoint_root(*, create: bool = True) -> Path:
    """Return the active checkpoint root.

    WorldAtlas keeps `ckpt/` as the canonical in-repo layout. Some shared
    workspaces keep the actual large checkpoint tree next to the checkout and
    point this helper at it with WORLDATLAS_CHECKPOINT_ROOT.
    """
    root = _active_checkpoint_root()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def canonical_checkpoint_root() -> Path:
    """Return the user-facing canonical checkpoint root path."""
    return CANONICAL_CHECKPOINT_ROOT


def _configured_checkpoint_root() -> Path | None:
    """Configured checkpoint root -> Path | None."""
    raw = os.environ.get(CHECKPOINT_ROOT_ENV)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return Path(os.path.abspath(str(path)))


def _active_checkpoint_root() -> Path:
    """Active checkpoint root -> Path."""
    return _configured_checkpoint_root() or CANONICAL_CHECKPOINT_ROOT


def _allowed_checkpoint_roots() -> tuple[Path, ...]:
    """Allowed checkpoint roots -> tuple[Path, ...]."""
    roots: list[Path] = []
    for candidate in (
        _configured_checkpoint_root(),
        CANONICAL_CHECKPOINT_ROOT,
        PROJECT_ROOT.parent / "ckpt",
        PROJECT_ROOT.parent / "ckpts",
    ):
        if candidate is None:
            continue
        root = Path(os.path.abspath(str(candidate)))
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def resolve_project_path(value: str | os.PathLike[str] | None) -> Path | None:
    """Resolve a project-relative or absolute path against the WorldAtlas root."""
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
    """Resolve a path and require it to stay under an approved ckpt root."""
    path = resolve_project_path(value)
    if path is None:
        return None
    path_resolved = path.resolve(strict=False)
    allowed = False
    for root in _allowed_checkpoint_roots():
        root_resolved = root.resolve(strict=False)
        if (
            path == root
            or path.is_relative_to(root)
            or path_resolved == root_resolved
            or path_resolved.is_relative_to(root_resolved)
        ):
            allowed = True
            break
    if not allowed:
        raise ValueError(
            "checkpoint path must stay under one of "
            f"{', '.join(str(root) for root in _allowed_checkpoint_roots())}, got {path}"
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


def _hf_hub_cache_name(repo_id: str) -> str:
    """Return the Hugging Face hub cache folder name for ``org/name``."""
    return "models--" + repo_id.replace("/", "--")


def _hf_repo_root_candidates(repo_id: str) -> list[Path]:
    """Return ckpt lookup order for a Hugging Face repo.

    Mapped ``HF_REPO_SUBPATHS`` stay first so existing direct-download trees
    (CLIP ViT-B/32, Cosmos, T5, …) keep winning. ``huggingface/hub/<cache>``
    is always tried next because some weights only exist in the hub layout.
    """
    mapped = HF_REPO_SUBPATHS.get(repo_id)
    candidates: list[Path] = []
    if mapped is not None:
        candidates.append(checkpoint_path(*mapped, kind="dir", required=False))
    else:
        candidates.append(checkpoint_path(repo_id.replace("/", "--"), kind="dir", required=False))
    hub_layout = checkpoint_path("huggingface", "hub", _hf_hub_cache_name(repo_id), kind="dir", required=False)
    if hub_layout not in candidates:
        candidates.append(hub_layout)
    return candidates


def _hf_repo_root(repo_id: str) -> Path:
    """Return the first existing candidate, else the mapped/canonical path."""
    candidates = _hf_repo_root_candidates(repo_id)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


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
    """Return the canonical local directory for a Hugging Face repo.

    Prefers a mapped direct-download tree, then ``ckpt/huggingface/hub``.
    Hub snapshots resolve to ``snapshots/<rev>`` when ``refs/main`` exists.
    """
    root = _hf_repo_root(repo_id)
    resolved = _hf_snapshot_dir(root) or root
    if required and not resolved.is_dir():
        looked = ", ".join(str(path) for path in _hf_repo_root_candidates(repo_id))
        raise FileNotFoundError(
            f"checkpoint directory not found: {resolved} (looked in {looked})"
        )
    return resolved


def checkpoint_cache_env(*, create: bool = True) -> CheckpointCacheEnv:
    """Return the fixed cache directories that keep HF/Torch downloads under ckpt."""
    cache_root = os.environ.get(CHECKPOINT_CACHE_ROOT_ENV, "").strip()
    root = Path(cache_root) if cache_root else checkpoint_root(create=create)
    return _checkpoint_cache_env_for_root(root, create=create)


def _checkpoint_cache_env_for_root(root: Path, *, create: bool = True) -> CheckpointCacheEnv:
    root = root.expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = Path(os.path.abspath(str(root)))
    if create:
        root.mkdir(parents=True, exist_ok=True)
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
    return _checkpoint_env_values(checkpoint_cache_env(create=create))


def _checkpoint_env_values(cache_env: CheckpointCacheEnv) -> dict[str, str]:
    return {
        "HF_HOME": str(cache_env.hf_home),
        "HF_HUB_CACHE": str(cache_env.hf_hub_cache),
        "HUGGINGFACE_HUB_CACHE": str(cache_env.hf_hub_cache),
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
    payload = dict(os.environ if env is None else env)
    root_override = payload.get(CHECKPOINT_ROOT_ENV)
    cache_root_override = payload.get(CHECKPOINT_CACHE_ROOT_ENV)
    if cache_root_override or root_override:
        cache_root = cache_root_override or root_override
        cache_env = _checkpoint_cache_env_for_root(Path(str(cache_root)), create=create)
        payload.update(_checkpoint_env_values(cache_env))
    else:
        payload.update(checkpoint_env(create=create))
    return payload


__all__ = [
    "CANONICAL_CHECKPOINT_ROOT",
    "CHECKPOINT_CACHE_ROOT_ENV",
    "CHECKPOINT_ENV_KEYS",
    "CHECKPOINT_ROOT_ENV",
    "CheckpointCacheEnv",
    "HF_REPO_SUBPATHS",
    "LEGACY_WORKSPACE_ROOT_ENV",
    "apply_checkpoint_env",
    "canonical_checkpoint_root",
    "checkpoint_cache_env",
    "checkpoint_env",
    "checkpoint_path",
    "checkpoint_root",
    "filter_checkpoint_env_overrides",
    "hf_local_dir",
    "legacy_workspace_root",
    "project_root",
    "rehome_legacy_workspace_path",
    "resolve_checkpoint_path",
    "resolve_project_path",
    "workspace_root",
]
