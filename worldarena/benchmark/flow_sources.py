"""Resolve optical-flow sidecars and GT flow sources for metric evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_amt_root() -> Path:
    """Resolve amt root -> Path."""
    return (PROJECT_ROOT / "thirdparty" / "AMT").resolve()


def resolve_raft_root() -> Path:
    """Resolve raft root -> Path."""
    return (PROJECT_ROOT / "thirdparty" / "RAFT").resolve()


def inspect_flow_source_layout() -> dict[str, Any]:
    """Inspect flow source layout -> dict[str, Any]."""
    amt_root = resolve_amt_root()
    raft_root = resolve_raft_root()

    amt_config = amt_root / "cfgs" / "AMT-S.yaml"
    amt_build_utils = amt_root / "utils" / "build_utils.py"
    amt_utils = amt_root / "utils" / "utils.py"
    amt_model = amt_root / "networks" / "AMT-S.py"
    amt_ready = all(path.exists() for path in (amt_config, amt_build_utils, amt_utils, amt_model))

    raft_core_root = raft_root / "core"
    raft_model = raft_core_root / "raft.py"
    raft_utils = raft_core_root / "utils" / "utils.py"
    raft_ready = all(path.exists() for path in (raft_core_root, raft_model, raft_utils))

    ready = amt_ready and raft_ready
    error = None
    if not amt_ready:
        error = f"AMT checkout is incomplete under {amt_root}"
    elif not raft_ready:
        error = f"RAFT checkout is incomplete under {raft_root}"

    return {
        "amt_root": str(amt_root),
        "amt_config": str(amt_config),
        "amt_model": str(amt_model),
        "amt_ready": amt_ready,
        "raft_root": str(raft_root),
        "raft_core_root": str(raft_core_root),
        "raft_model": str(raft_model),
        "raft_ready": raft_ready,
        "ready": ready,
        "error": error,
    }


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def _module_under_roots(module: object, roots: Sequence[Path]) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        resolved = Path(str(module_file)).resolve()
    except OSError:
        return False
    return any(resolved.is_relative_to(root) for root in roots)


def _is_generic_upstream_module(name: str) -> bool:
    return name == "utils" or name.startswith("utils.")


@contextmanager
def isolated_upstream_import_context(*paths: Path) -> Iterator[None]:
    roots = _dedupe_paths(paths)
    existing_modules = set(sys.modules)
    inserted_paths: list[str] = []
    saved_modules: dict[str, object] = {}
    try:
        for name, module in list(sys.modules.items()):
            if not _is_generic_upstream_module(name):
                continue
            if _module_under_roots(module, roots):
                continue
            saved_modules[name] = module
            del sys.modules[name]
        for root in reversed(roots):
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
                inserted_paths.append(root_str)
        yield
    finally:
        for root_str in inserted_paths:
            while root_str in sys.path:
                sys.path.remove(root_str)
        for name, module in list(sys.modules.items()):
            if name in saved_modules:
                continue
            if name not in existing_modules and _module_under_roots(module, roots):
                del sys.modules[name]
            elif _is_generic_upstream_module(name) and _module_under_roots(module, roots):
                del sys.modules[name]
        sys.modules.update(saved_modules)


@contextmanager
def amt_runtime_context() -> Iterator[Path]:
    """Amt runtime context -> Iterator[Path]."""
    root = resolve_amt_root()
    with isolated_upstream_import_context(root):
        yield root


@contextmanager
def raft_runtime_context() -> Iterator[Path]:
    """Raft runtime context -> Iterator[Path]."""
    root = resolve_raft_root()
    with isolated_upstream_import_context(root / "core", root):
        yield root


__all__ = [
    "amt_runtime_context",
    "isolated_upstream_import_context",
    "inspect_flow_source_layout",
    "raft_runtime_context",
    "resolve_amt_root",
    "resolve_raft_root",
]
