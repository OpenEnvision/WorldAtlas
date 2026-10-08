"""Subprocess runtime for MeT3R reconstruction-consistency metrics."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image

from worldatlas.benchmark.met3r_protocol import (
    PROTOCOL, RESIZE_POLICY, UPSTREAM_COMMIT, native_video_pair_batches, paper_input_shape, verify_upstream,
)

from worldatlas.common.checkpoints import (
    apply_checkpoint_env,
    checkpoint_path,
    filter_checkpoint_env_overrides,
    hf_local_dir,
)
from worldatlas.common.progress import run_progress_subprocess

MET3R_RUNTIME_WORKER_ENV = "WORLDATLAS_MET3R_RUNTIME_WORKER"
FEATUP_NATIVE_DIR_ENV = "WORLDATLAS_FEATUP_NATIVE_DIR"
FEATUP_FAULT_LOG_ENV = "WORLDATLAS_EVAL_FAULT_LOG"
FEATUP_NATIVE_MODULES = ("adaptive_conv_cuda_impl", "adaptive_conv_cpp_impl")
FEATUP_EXT_KIND = "featup_ext"
_FAULT_LOG_HANDLES: list[Any] = []
MET3R_DISTANCE_DIRECTIONS = {
    "cosine": "lower_is_better",
    "lpips": "lower_is_better",
    "rmse": "lower_is_better",
    "mse": "lower_is_better",
    "psnr": "higher_is_better",
    "ssim": "higher_is_better",
}
MET3R_BACKBONE_REPOS = {
    "mast3r": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    "dust3r": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
}


def _project_root() -> Path:
    """Project root -> Path."""
    return Path(__file__).resolve().parents[2]


def _runtime_cache_dir(name: str) -> Path:
    path = _project_root() / ".cache" / "benchmark" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_met3r_root() -> Path:
    """Resolve met3r root -> Path."""
    return (_project_root() / "thirdparty" / "met3r").resolve()


def _resolve_runtime_python_bin(runtime: dict[str, Any] | None) -> str | None:
    payload = dict(runtime or {})
    raw = str(payload.get("python_bin") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if raw.startswith(".") or "/" in raw:
        return str((_project_root() / path).absolute())
    return raw


def resolve_met3r_runtime_python(runtime: dict[str, Any] | None = None) -> str | None:
    return _resolve_runtime_python_bin(runtime)


def inspect_met3r_layout() -> dict[str, Any]:
    """Inspect met3r layout -> dict[str, Any]."""
    root = resolve_met3r_root()
    package_dir = root / "met3r"
    init_path = package_dir / "__init__.py"
    metric_path = package_dir / "met3r.py"
    mast3r_root = root / "mast3r"
    mast3r_package = mast3r_root / "mast3r"
    dust3r_package = mast3r_root / "dust3r"
    source_ready = init_path.exists() and metric_path.exists()
    submodule_ready = mast3r_package.is_dir() and dust3r_package.is_dir()
    ready = source_ready and submodule_ready
    error = None
    if not source_ready:
        error = f"MEt3R checkout is missing package files under {package_dir}"
    elif not submodule_ready:
        error = (
            "MEt3R checkout is missing initialized MASt3R/DUSt3R submodules; "
            "run `git submodule update --init --recursive` inside the met3r repo"
        )
    return {
        "root": str(root),
        "package_dir": str(package_dir),
        "metric_path": str(metric_path),
        "mast3r_root": str(mast3r_root),
        "mast3r_package": str(mast3r_package),
        "dust3r_package": str(dust3r_package),
        "source_ready": source_ready,
        "submodule_ready": submodule_ready,
        "ready": ready,
        "error": error,
    }


def _coerce_runtime_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _default_device() -> str:
    """Default device -> str."""
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_img_size(value: Any, default: int | None = 256) -> int | None:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return int(value)


def _metric_options(runtime: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(runtime or {})
    fixed = dict(protocol=PROTOCOL, backbone="dust3r", feature_backbone="dino16",
                 upsampler="featup", distance="cosine", use_norm=True, freeze=True,
                 direction="symmetric", frame_source="native_consecutive", img_size=None)
    unknown = set(payload) - set(fixed) - {"device", "batch_size", "feature_backbone_weights", "python_bin", "env", "keep_ipc"}
    if unknown:
        raise ValueError(f"Unknown MEt3R paper runtime options: {sorted(unknown)}")
    for key, expected in fixed.items():
        if key in payload and payload[key] != expected:
            raise ValueError(f"MEt3R paper protocol requires {key}={expected!r}; legacy/alternative settings cannot use this metric")
    if int(payload.get("batch_size", 1)) < 1:
        raise ValueError("MEt3R batch_size must be positive")
    return {
        **fixed,
        "device": str(payload.get("device") or _default_device()),
        "batch_size": int(payload.get("batch_size", 1)),
        "feature_backbone_weights": str(
            payload.get("feature_backbone_weights") or hf_local_dir("mhamilton723/FeatUp")
        ),
    }


def _snapshot_dir(cache_root: Path) -> Path | None:
    refs_main = cache_root / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = cache_root / "snapshots" / revision
        if snapshot.is_dir():
            return snapshot
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.is_dir():
        return None
    snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
    return snapshots[0] if len(snapshots) == 1 else None


def _ensure_backbone_weights_available(backbone: str) -> None:
    repo_id = MET3R_BACKBONE_REPOS.get(backbone)
    if repo_id is None:
        return
    cache_root = checkpoint_path(
        "huggingface",
        "hub",
        f"models--{repo_id.replace('/', '--')}",
        kind="dir",
        required=False,
    )
    snapshot = _snapshot_dir(cache_root)
    if snapshot is None:
        raise FileNotFoundError(f"MEt3R {backbone} weights are not cached under {cache_root}")
    if not any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(
            f"MEt3R {backbone} weights are incomplete under {snapshot}; "
            "expected model.safetensors or pytorch_model.bin"
        )


def _ensure_featup_weights_available() -> None:
    # Official FeatUp hubconf.py (use_norm=True, dino16) loads this JBU stack
    # and its DINO base. Require the actual cached files before any hub call.
    for name in ("dino16_jbu_stack_cocostuff.ckpt", "dino_deitsmall16_pretrain.pth"):
        checkpoint_path("torch", "hub", "checkpoints", name, required=True)
    checkpoint_path("torch", "hub", "facebookresearch_dino_main", "hubconf.py", required=True)


def resolve_featup_native_extension_files() -> dict[str, Path]:
    """Locate installed FeatUp ``.so`` files without importing/dlopening them."""
    found: dict[str, Path] = {}
    for name in FEATUP_NATIVE_MODULES:
        spec = importlib.util.find_spec(name)
        origin = getattr(spec, "origin", None) if spec is not None else None
        if not origin:
            raise FileNotFoundError(f"FeatUp compiled module {name} is not installed")
        path = Path(origin)
        if not path.is_file():
            raise FileNotFoundError(f"FeatUp compiled module {name} path is missing: {path}")
        found[name] = path
    return found


def _unescape_mount_path(raw: str) -> str:
    return (
        raw.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def mount_flags_for(path: Path) -> set[str]:
    """Return /proc/self/mounts flags for the longest mount covering ``path``."""
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)
    best_mp = ""
    best_flags: set[str] = set()
    try:
        lines = Path("/proc/self/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        mountpoint = _unescape_mount_path(parts[1])
        flags = set(parts[3].split(","))
        covered = (
            resolved == mountpoint
            or resolved.startswith(mountpoint.rstrip("/") + "/")
            or mountpoint == "/"
        )
        if covered and len(mountpoint) >= len(best_mp):
            best_mp = mountpoint
            best_flags = flags
    return best_flags


def probe_exec_mmap(path: Path) -> tuple[bool, str]:
    """True when the kernel allows ``PROT_EXEC`` mmap of ``path`` (what dlopen needs)."""
    import ctypes

    prot_read = 1
    prot_exec = 4
    map_private = 2
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        return False, str(exc)
    try:
        size = max(int(os.fstat(fd).st_size), 1)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        addr = libc.mmap(None, size, prot_read | prot_exec, map_private, fd, 0)
        failed = ctypes.c_void_p(-1).value
        if addr in {None, failed}:
            err = ctypes.get_errno()
            return False, os.strerror(err) if err else "mmap PROT_EXEC failed"
        libc.munmap(ctypes.c_void_p(addr), size)
        return True, "ok"
    except OSError as exc:
        return False, str(exc)
    finally:
        os.close(fd)


def path_allows_exec_mapping(directory: Path) -> tuple[bool, str]:
    """Reject noexec mounts; confirm with a PROT_EXEC mmap probe file."""
    flags = mount_flags_for(directory if directory.exists() else directory.parent)
    flag_text = ",".join(sorted(flags)) if flags else "unknown"
    if "noexec" in flags:
        return False, f"mount_noexec flags={flag_text}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"mkdir_failed:{exc}"
    probe = directory / f".exec_probe.{os.getpid()}"
    try:
        probe.write_bytes(b"\x00" * 4096)
        os.chmod(probe, 0o755)
        ok, detail = probe_exec_mmap(probe)
        if not ok:
            return False, f"mmap_prot_exec_failed {detail} flags={flag_text}"
        return True, f"ok flags={flag_text}"
    except OSError as exc:
        return False, f"probe_error:{exc}"
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


DOLPHINFS_PRELOAD_MARKERS = (
    "dolphinfs",
    "dolphin_preload",
    "dfs_preload",
    "libdfs",
    "libdfs_client",
)
FEATUP_RUNTIME_DEP_NAMES = (
    "libc10.so",
    "libc10_cuda.so",
    "libstdc++.so.6",
)


def _featup_stage_candidates() -> list[Path]:
    from worldatlas.benchmark.local_checkpoints import local_cache_candidates

    uid = os.getuid()
    extras = [
        Path(os.environ.get("PWD") or os.getcwd()) / f"worldatlas-{FEATUP_EXT_KIND}-{uid}",
        Path("/tmp/hope_nogit") / f"worldatlas-{FEATUP_EXT_KIND}-{uid}",
        Path("/var/tmp") / f"worldatlas-{FEATUP_EXT_KIND}-{uid}",
    ]
    hope_workdir = os.environ.get("HOPE_WORKDIR", "").strip()
    if hope_workdir:
        extras.insert(0, Path(hope_workdir) / f"worldatlas-{FEATUP_EXT_KIND}-{uid}")
    ordered: list[Path] = []
    for candidate in [*local_cache_candidates(FEATUP_EXT_KIND), *extras]:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def strip_dolphinfs_ld_preload(value: str) -> tuple[str, list[str]]:
    """Drop dolphinfs interposers; keep unrelated ``LD_PRELOAD`` entries."""
    kept: list[str] = []
    stripped: list[str] = []
    for item in value.split(os.pathsep):
        if not item:
            continue
        lowered = item.lower()
        name = Path(item).name.lower()
        if any(marker in name or marker in lowered for marker in DOLPHINFS_PRELOAD_MARKERS):
            stripped.append(item)
        else:
            kept.append(item)
    return os.pathsep.join(kept), stripped


def _is_cv2_library_dir(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    return "/cv2/" in f"/{lowered}/" or lowered.rstrip("/").endswith("/cv2") or "/site-packages/cv2" in lowered


def featup_runtime_library_dirs(python_executable: str | None = None) -> list[str]:
    """torch/lib (libc10) plus the env lib (libstdc++ GLIBCXX_3.4.29)."""
    dirs: list[str] = []
    exe = Path(python_executable or sys.executable).resolve()
    env_root = exe.parent.parent
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    conda = os.environ.get("CONDA_PREFIX") or os.environ.get("WORLDATLAS_EVAL_ENV")
    candidates = [
        env_root / "lib" / version / "site-packages" / "torch" / "lib",
        env_root / "lib",
    ]
    if conda:
        conda_root = Path(conda)
        candidates.extend(
            [
                conda_root / "lib" / version / "site-packages" / "torch" / "lib",
                conda_root / "lib",
            ]
        )
    for candidate in candidates:
        resolved = str(candidate)
        if candidate.is_dir() and resolved not in dirs:
            dirs.append(resolved)
    return dirs


def _featup_runtime_dep_sources() -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for directory in featup_runtime_library_dirs():
        for name in FEATUP_RUNTIME_DEP_NAMES:
            if name in seen:
                continue
            path = Path(directory) / name
            if path.is_file():
                found.append(path)
                seen.add(name)
    return found


def _copy_featup_runtime_deps(dest_root: Path) -> list[str]:
    from worldatlas.benchmark.local_checkpoints import exclusive_file_lock
    from worldatlas.common.progress import log_progress

    dest_deps = dest_root / "deps"
    dest_deps.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    lock_path = dest_root / ".stage.lock"
    with exclusive_file_lock(lock_path):
        for src in _featup_runtime_dep_sources():
            dest = dest_deps / src.name
            if dest.is_file() and dest.stat().st_size == src.stat().st_size:
                copied.append(src.name)
                continue
            temporary = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
            if temporary.exists():
                temporary.unlink()
            try:
                shutil.copyfile(src, temporary)
                os.chmod(temporary, 0o755)
                os.replace(temporary, dest)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise
            copied.append(src.name)
    if copied:
        log_progress(
            "met3r_model_load",
            status="featup_native_deps_ok",
            dest=str(dest_deps),
            libs=",".join(copied),
        )
    return copied


@contextmanager
def featup_dlopen_environment(dest_root: Path) -> Iterator[dict[str, str]]:
    """Sanitize linker env for FeatUp ``dlopen``, then restore dolphinfs preload.

    OpenCV prepends its own ``lib64`` onto ``LD_LIBRARY_PATH`` during
    ``import met3r``. Combined with a dolphinfs ``LD_PRELOAD``, that is how
    a PROT_EXEC-good local ``.so`` still SIGSEGV's. Torch stays first after
    this returns; only ``LD_PRELOAD`` is restored so later NFS reads work.
    """
    previous_preload = os.environ.get("LD_PRELOAD")
    kept, stripped = strip_dolphinfs_ld_preload(previous_preload or "")
    if stripped:
        if kept:
            os.environ["LD_PRELOAD"] = kept
        else:
            os.environ.pop("LD_PRELOAD", None)
    dest_str = str(dest_root.resolve())
    deps_dir = dest_root / "deps"
    lib_dirs = [str(deps_dir.resolve())] if deps_dir.is_dir() else []
    lib_dirs.extend(featup_runtime_library_dirs())
    existing = [
        item
        for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if item and not _is_cv2_library_dir(item) and item not in lib_dirs
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*lib_dirs, *existing])
    if dest_str in sys.path:
        sys.path.remove(dest_str)
    sys.path.insert(0, dest_str)
    info = {
        "ld_preload_stripped": ";".join(stripped) if stripped else "none",
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "")[:240],
        "mount": ",".join(sorted(mount_flags_for(dest_root))) or "unknown",
        "dest": dest_str,
    }
    try:
        yield info
    finally:
        if previous_preload is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = previous_preload


def import_featup_native_extensions(dest_root: Path) -> list[str]:
    """``dlopen`` the staged copies by exact path so torch.hub cannot see NFS."""
    loaded: list[str] = []
    for name in FEATUP_NATIVE_MODULES:
        if name in sys.modules:
            origin = getattr(sys.modules[name], "__file__", None)
            loaded.append(f"{name}=already:{origin or 'in-memory'}")
            continue
        matches = sorted(dest_root.glob(f"{name}*.so"))
        if not matches:
            raise FileNotFoundError(f"staged FeatUp module missing: {name} under {dest_root}")
        path = matches[0]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded.append(f"{name}={path}")
    return loaded


def _ldd_not_found(paths: Sequence[Path]) -> list[str]:
    missing: list[str] = []
    for path in paths:
        try:
            result = subprocess.run(
                ["ldd", str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in (result.stdout or "").splitlines():
            if "not found" in line:
                missing.append(f"{path.name}: {line.strip()}")
    return missing


def _prepend_process_path(key: str, *parts: str) -> None:
    existing = [item for item in os.environ.get(key, "").split(os.pathsep) if item]
    unique_prefix: list[str] = []
    for part in parts:
        if part and part not in unique_prefix:
            unique_prefix.append(part)
    rest = [item for item in existing if item not in unique_prefix]
    os.environ[key] = os.pathsep.join(unique_prefix + rest)


def _enable_featup_faulthandler() -> None:
    import faulthandler

    raw = os.environ.get(FEATUP_FAULT_LOG_ENV, "").strip()
    if raw:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")
        _FAULT_LOG_HANDLES.append(handle)
        faulthandler.enable(file=handle, all_threads=True)
        return
    faulthandler.enable(all_threads=True)


def _copy_featup_native_modules(sources: dict[str, Path], dest_root: Path) -> None:
    from worldatlas.benchmark.local_checkpoints import exclusive_file_lock
    from worldatlas.common.progress import log_progress

    dest_root.mkdir(parents=True, exist_ok=True)
    lock_path = dest_root / ".stage.lock"
    with exclusive_file_lock(lock_path):
        for name, src in sources.items():
            dest = dest_root / src.name
            if dest.is_file() and dest.stat().st_size == src.stat().st_size:
                os.chmod(dest, 0o755)
                continue
            log_progress(
                "met3r_model_load",
                status="featup_native_copy_start",
                module=name,
                src=str(src),
                dest=str(dest),
                bytes=src.stat().st_size,
            )
            temporary = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
            if temporary.exists():
                temporary.unlink()
            try:
                shutil.copyfile(src, temporary)
                copied = temporary.stat().st_size
                if copied != src.stat().st_size:
                    raise OSError(f"FeatUp native copy size mismatch: {temporary}")
                os.chmod(temporary, 0o755)
                os.replace(temporary, dest)
                os.chmod(dest, 0o755)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise
            log_progress(
                "met3r_model_load",
                status="featup_native_copy_ok",
                module=name,
                dest=str(dest),
            )


def _verify_featup_native_dir(dest_root: Path) -> tuple[bool, str]:
    ok, reason = path_allows_exec_mapping(dest_root)
    if not ok:
        return False, reason
    details: list[str] = [reason]
    for path in sorted(dest_root.glob("*.so")):
        mapped, map_reason = probe_exec_mmap(path)
        if not mapped:
            return False, f"{path.name}: {map_reason}"
        details.append(f"{path.name}=exec_ok")
    return True, "; ".join(details)


def stage_featup_native_extensions(*, apply_sys_path: bool = True) -> dict[str, str]:
    """Copy FeatUp CUDA/C++ extensions onto an exec-capable local dir before dlopen.

    The NFS-built ``adaptive_conv_*_impl*.so`` files have no RPATH. mmap/dlopen
    from dolphinfs still SIGSEGV's after LD_LIBRARY_PATH harden. A /dev/shm copy
    still fails ``dlopen`` when tmpfs is ``noexec`` ("failed to map segment from
    shared object"). Import must see a local *executable* copy; scoring/protocol
    is unchanged.
    """
    from worldatlas.benchmark.local_checkpoints import is_already_materialized
    from worldatlas.common.progress import log_progress

    sources = resolve_featup_native_extension_files()
    dest_root: Path | None = None
    if os.environ.get("WORLDATLAS_SKIP_LOCAL_CHECKPOINT_COPY") == "1":
        log_progress("met3r_model_load", status="featup_native_skip", reason="WORLDATLAS_SKIP_LOCAL_CHECKPOINT_COPY")
        dest_root = Path(next(iter(sources.values()))).resolve().parent
    elif all(is_already_materialized(path, FEATUP_EXT_KIND) for path in sources.values()):
        reuse = Path(next(iter(sources.values()))).resolve().parent
        ok, reason = _verify_featup_native_dir(reuse)
        if ok:
            dest_root = reuse
            log_progress(
                "met3r_model_load",
                status="featup_native_reuse",
                dest=str(dest_root),
                exec_probe=reason,
            )
        else:
            log_progress(
                "met3r_model_load",
                status="featup_native_reuse_rejected",
                dest=str(reuse),
                reason=reason,
            )
    if dest_root is None:
        needed = sum(path.stat().st_size for path in sources.values()) + 64 * 1024**2
        for candidate in _featup_stage_candidates():
            try:
                candidate.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            try:
                if shutil.disk_usage(str(candidate)).free < needed:
                    continue
            except OSError:
                continue
            ok, reason = path_allows_exec_mapping(candidate)
            if not ok:
                log_progress(
                    "met3r_model_load",
                    status="featup_native_skip_noexec",
                    dest=str(candidate),
                    reason=reason,
                )
                continue
            dest_root = candidate / "modules"
            dest_root.mkdir(parents=True, exist_ok=True)
            _copy_featup_native_modules(sources, dest_root)
            verified, verify_reason = _verify_featup_native_dir(dest_root)
            if verified:
                log_progress(
                    "met3r_model_load",
                    status="featup_native_exec_ok",
                    dest=str(dest_root),
                    reason=verify_reason,
                )
                break
            log_progress(
                "met3r_model_load",
                status="featup_native_exec_fail",
                dest=str(dest_root),
                reason=verify_reason,
            )
            dest_root = None
        if dest_root is None:
            dest_root = Path("/tmp") / f"worldatlas-featup-ext-{os.getuid()}" / "modules"
            dest_root.mkdir(parents=True, exist_ok=True)
            log_progress(
                "met3r_model_load",
                status="featup_native_fallback_tmp",
                dest=str(dest_root),
            )
            _copy_featup_native_modules(sources, dest_root)
            verified, verify_reason = _verify_featup_native_dir(dest_root)
            log_progress(
                "met3r_model_load",
                status="featup_native_exec_ok" if verified else "featup_native_exec_fail",
                dest=str(dest_root),
                reason=verify_reason,
            )
            if not verified:
                raise OSError(
                    f"FeatUp native extensions cannot be mmap'd PROT_EXEC under {dest_root}: {verify_reason}"
                )
    dest_str = str(dest_root.resolve())
    os.environ[FEATUP_NATIVE_DIR_ENV] = dest_str
    _prepend_process_path("PYTHONPATH", dest_str)
    if apply_sys_path:
        if dest_str in sys.path:
            sys.path.remove(dest_str)
        sys.path.insert(0, dest_str)
    _copy_featup_runtime_deps(dest_root)
    staged = sorted(path for path in dest_root.glob("*.so") if path.parent == dest_root)
    unresolved = _ldd_not_found(staged)
    log_progress(
        "met3r_model_load",
        status="featup_native_ready",
        dest=dest_str,
        modules=",".join(path.name for path in staged) or "none",
        unresolved_libs="; ".join(unresolved) if unresolved else "none",
        exec_ok="1",
        mount=",".join(sorted(mount_flags_for(dest_root))) or "unknown",
    )
    return {FEATUP_NATIVE_DIR_ENV: dest_str}


def _distance_direction(distance: str) -> str:
    return MET3R_DISTANCE_DIRECTIONS.get(distance.lower(), "lower_is_better")


def _subprocess_runtime_env(runtime: dict[str, Any] | None) -> dict[str, str]:
    env = apply_checkpoint_env()
    env[MET3R_RUNTIME_WORKER_ENV] = "1"
    pythonpath_entries = [str(_project_root())]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.setdefault("PYTHONUNBUFFERED", "1")
    for key, value in filter_checkpoint_env_overrides(dict((runtime or {}).get("env", {}))).items():
        env[str(key)] = str(value)
    return env


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype == np.uint8:
        contiguous = np.ascontiguousarray(array)
        return contiguous if contiguous.flags.writeable else contiguous.copy()
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _frame_pairs(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not len(reference_frames) or not len(prediction_frames):
        raise ValueError("MEt3R requires at least one reference/prediction frame pair")
    if len(reference_frames) != len(prediction_frames):
        raise ValueError("MEt3R explicit pair lists differ; refusing broadcast or silent truncation")
    return list(zip(reference_frames, prediction_frames))


def _serialize_frame_pairs(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    pair_dir: Path,
) -> list[dict[str, str]]:
    pair_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, str]] = []
    for index, (reference_frame, prediction_frame) in enumerate(
        _frame_pairs(reference_frames, prediction_frames)
    ):
        reference_path = pair_dir / f"{index:04d}.reference.png"
        prediction_path = pair_dir / f"{index:04d}.prediction.png"
        Image.fromarray(_to_uint8(reference_frame)).save(reference_path)
        Image.fromarray(_to_uint8(prediction_frame)).save(prediction_path)
        payload.append(
            {
                "reference_path": str(reference_path),
                "prediction_path": str(prediction_path),
            }
        )
    return payload


def _cleanup_ipc_paths(
    request_path: Path,
    response_path: Path,
    pair_dir: Path,
    *,
    keep_ipc: bool,
) -> None:
    if keep_ipc:
        return
    request_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    shutil.rmtree(pair_dir, ignore_errors=True)


def _maybe_run_in_subprocess(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    *,
    runtime: dict[str, Any] | None,
    video_path: Path | None = None,
) -> dict[str, Any] | None:
    if os.getenv(MET3R_RUNTIME_WORKER_ENV) == "1":
        return None
    python_bin = _resolve_runtime_python_bin(runtime)
    if python_bin is None:
        return None

    ipc_dir = _runtime_cache_dir("met3r_runtime_ipc")
    token = uuid.uuid4().hex
    pair_dir = ipc_dir / f"pairs.{token}"
    request_path = ipc_dir / f"met3r.{token}.request.json"
    response_path = ipc_dir / f"met3r.{token}.response.json"
    payload = {"protocol": PROTOCOL, "runtime": dict(runtime or {})}
    if video_path is None:
        payload["pairs"] = _serialize_frame_pairs(reference_frames, prediction_frames, pair_dir)
    else:
        payload["video_path"] = str(video_path.absolute())
    request_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    command = [
        python_bin,
        "-u",
        "-m",
        "worldatlas.benchmark.met3r_runtime_worker",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    keep_ipc = _coerce_runtime_bool((runtime or {}).get("keep_ipc"), default=False)
    try:
        completed = run_progress_subprocess(
            command,
            cwd=str(_project_root()),
            env=_subprocess_runtime_env(runtime),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = " | ".join(part for part in (stderr, stdout) if part)
        raise RuntimeError(f"MEt3R runtime subprocess failed: {details}") from exc

    if not response_path.exists():
        logs = " | ".join(
            part for part in ((completed.stderr or "").strip(), (completed.stdout or "").strip()) if part
        )
        raise RuntimeError(f"MEt3R runtime subprocess did not write a response: {logs}")

    response = json.loads(response_path.read_text(encoding="utf-8"))
    try:
        if not bool(response.get("ok")):
            error = str(response.get("error") or "MEt3R runtime subprocess failed")
            traceback_text = str(response.get("traceback") or "").strip()
            stdout = str(response.get("stdout") or "").strip()
            stderr = str(response.get("stderr") or "").strip()
            raise RuntimeError(
                " | ".join(part for part in (error, traceback_text, stderr, stdout) if part)
            )
        return dict(response["result"])
    finally:
        _cleanup_ipc_paths(request_path, response_path, pair_dir, keep_ipc=keep_ipc)


@contextmanager
def _met3r_import_context(root: Path) -> Iterator[None]:
    root_str = str(root)
    sys.path.insert(0, root_str)
    try:
        yield
    finally:
        while root_str in sys.path:
            sys.path.remove(root_str)


def _clear_met3r_modules() -> None:
    """Clear met3r modules."""
    for name in list(sys.modules):
        if name == "met3r" or name.startswith("met3r."):
            sys.modules.pop(name, None)


@contextmanager
def _met3r_torch_hub_context() -> Iterator[None]:
    """Met3r torch hub context -> Iterator[None]."""
    import torch

    original_load = torch.hub.load

    def patched_load(repo_or_dir: Any, *args: Any, **kwargs: Any) -> Any:
        raw_repo = str(repo_or_dir)
        repo_path = Path(raw_repo).expanduser()
        if not repo_path.is_absolute() and (raw_repo.startswith(".") or "/" in raw_repo):
            repo_path = _project_root() / repo_path
        if repo_path.exists():
            kwargs.setdefault("source", "local")
            return original_load(str(repo_path), *args, **kwargs)
        if raw_repo in {"facebookresearch/dino:main", "facebookresearch/dino"}:
            cached = checkpoint_path("torch", "hub", "facebookresearch_dino_main", kind="dir", required=True)
            kwargs["source"] = "local"
            return original_load(str(cached), *args, **kwargs)
        raise FileNotFoundError(f"Unprepared remote torch-hub dependency in paper MEt3R: {raw_repo}")

    torch.hub.load = patched_load
    try:
        yield
    finally:
        torch.hub.load = original_load




@lru_cache(maxsize=4)
def _load_met3r_model(
    root: str,
    device: str,
    img_size: int | None,
    use_norm: bool,
    backbone: str,
    feature_backbone: str,
    feature_backbone_weights: str,
    upsampler: str,
    distance: str,
    freeze: bool,
):
    os.environ.update(apply_checkpoint_env())
    layout = inspect_met3r_layout()
    if not layout["ready"]:
        raise RuntimeError(str(layout["error"]))
    verify_upstream(Path(root))
    if importlib.util.find_spec("featup") is None:
        raise RuntimeError("MEt3R paper protocol requires the official FeatUp package; no bilinear fallback is allowed")
    featup_root = Path(feature_backbone_weights)
    if not (featup_root.is_dir() and (featup_root / "hubconf.py").is_file()):
        raise FileNotFoundError("MEt3R requires a prepared local official FeatUp hub checkout and weights; remote fallback is disabled")
    _ensure_backbone_weights_available(backbone)
    _ensure_featup_weights_available()

    from worldatlas.common.progress import log_progress

    _enable_featup_faulthandler()
    log_progress(
        "met3r_model_load",
        status="start",
        device=device,
        backbone=backbone,
        upsampler=upsampler,
        feature_backbone=feature_backbone,
    )
    log_progress("met3r_model_load", status="staging_featup_native")
    staged = stage_featup_native_extensions(apply_sys_path=True)
    dest_root = Path(staged[FEATUP_NATIVE_DIR_ENV])
    _clear_met3r_modules()
    try:
        log_progress("met3r_model_load", status="import_met3r")
        with _met3r_import_context(Path(root)):
            module = importlib.import_module("met3r")
            metric_class = getattr(module, "MEt3R")
            with featup_dlopen_environment(dest_root) as linker:
                log_progress(
                    "met3r_model_load",
                    status="loading_featup",
                    feature_backbone_weights=feature_backbone_weights,
                    native_dir=str(dest_root),
                    dest=linker["dest"],
                    mount=linker["mount"],
                    ld_preload_stripped=linker["ld_preload_stripped"],
                    ld_library_path=linker["ld_library_path"],
                )
                imported = import_featup_native_extensions(dest_root)
                log_progress(
                    "met3r_model_load",
                    status="featup_native_dlopen_ok",
                    dest=str(dest_root),
                    modules=",".join(imported),
                )
                with _met3r_torch_hub_context():
                    metric = metric_class(
                        img_size=img_size,
                        use_norm=use_norm,
                        backbone=backbone,
                        feature_backbone=feature_backbone,
                        feature_backbone_weights=feature_backbone_weights,
                        upsampler=upsampler,
                        distance=distance,
                        freeze=freeze,
                    )
    except Exception:
        log_progress("met3r_model_load", status="fail")
        _clear_met3r_modules()
        raise
    if hasattr(metric, "to"):
        metric = metric.to(device)
    if hasattr(metric, "eval"):
        metric.eval()
    log_progress("met3r_model_ready", device=device, backbone=backbone, upsampler=upsampler)
    return metric


def _pair_batch_to_tensor(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    device: str,
    img_size: int | None,
):
    import torch
    from torch.nn import functional as F

    def resize_image(image: Any, target_size: tuple[int, int]) -> Any:
        if tuple(image.shape[-2:]) == target_size:
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    batch: list[Any] = []
    for reference_frame, prediction_frame in pairs:
        if reference_frame.shape != prediction_frame.shape:
            raise ValueError("MEt3R pair dimensions differ; do not stretch unrelated views to match")
        reference_tensor = torch.from_numpy(_to_uint8(reference_frame)).permute(2, 0, 1).float()
        prediction_tensor = torch.from_numpy(_to_uint8(prediction_frame)).permute(2, 0, 1).float()
        if img_size is not None:
            raise ValueError("MEt3R paper video adapter uses aspect-preserving input and dynamic rasterization")
        target_size = paper_input_shape(*prediction_tensor.shape[-2:])
        reference_tensor = resize_image(reference_tensor, target_size)
        prediction_tensor = resize_image(prediction_tensor, target_size)
        batch.append(torch.stack([reference_tensor, prediction_tensor], dim=0))
    tensor = torch.stack(batch, dim=0) / 127.5 - 1.0
    return tensor.to(device)


def _tensor_to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(item) for item in array.tolist()]


def _score_pair_batches(batches, options: dict[str, Any]) -> dict[str, Any]:
    """Use unchanged author scoring in both directions, then paper Eq. (1)."""
    layout = inspect_met3r_layout()
    if not layout["ready"]:
        raise RuntimeError(str(layout["error"]))
    metric = _load_met3r_model(
        str(resolve_met3r_root()),
        str(options["device"]),
        options["img_size"],
        bool(options["use_norm"]),
        str(options["backbone"]),
        str(options["feature_backbone"]),
        str(options["feature_backbone_weights"]),
        str(options["upsampler"]),
        str(options["distance"]),
        bool(options["freeze"]),
    )

    import torch

    frame_scores: list[float] = []
    overlap_ratios: list[float] = []
    directional_scores: list[list[float]] = []
    directional_overlaps: list[list[float]] = []
    empty_overlap_pair_indices: list[int] = []
    with torch.inference_mode():
        for batch_pairs in batches:
            if not batch_pairs:
                raise ValueError("Empty MEt3R pair batch")
            batch = _pair_batch_to_tensor(
                batch_pairs,
                str(options["device"]),
                options["img_size"],
            )
            directions = []
            for oriented in (batch, batch.flip(1)):
                outputs = metric(
                    images=oriented, return_overlap_mask=True,
                    return_score_map=False, return_projections=False,
                )
                if not isinstance(outputs, tuple) or len(outputs) < 2:
                    raise RuntimeError("MEt3R did not return the requested overlap mask")
                scores, masks = outputs[:2]
                if tuple(scores.shape) != (len(batch_pairs),) or masks.ndim != 3 or masks.shape[0] != len(batch_pairs):
                    raise RuntimeError("MEt3R returned incomplete/mismatched score and overlap-mask batch sizes")
                # Validate every pixel, not just its mean. Never repair NaN/Inf,
                # and never replace upstream's finite zero-overlap score.
                if not bool(torch.isfinite(masks).all()) or bool(((masks < 0) | (masks > 1)).any()):
                    raise ValueError("MEt3R returned invalid overlap mask")
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("MEt3R produced a non-finite score")
                directions.append((_tensor_to_float_list(scores), _tensor_to_float_list(masks.float().flatten(1).mean(1))))
            for offset in range(len(batch_pairs)):
                scores = [direction[0][offset] for direction in directions]
                overlaps = [direction[1][offset] for direction in directions]
                if 0.0 in overlaps:
                    empty_overlap_pair_indices.append(len(frame_scores))
                frame_scores.append(float(np.mean(scores)))
                overlap_ratios.append(float(np.mean(overlaps)))
                directional_scores.append(scores)
                directional_overlaps.append(overlaps)
    if not frame_scores:
        raise ValueError("MEt3R requires at least one consecutive frame pair")
    raw = float(np.mean(frame_scores))
    return {
        "raw": raw,
        "backend": "met3r",
        "details": {
            "pairing_protocol": PROTOCOL,
            "implementation_revision": PROTOCOL,
            "upstream_commit": UPSTREAM_COMMIT,
            "frame_scores": frame_scores,
            "directional_frame_scores": directional_scores,
            "overlap_ratios": overlap_ratios,
            "directional_overlap_ratios": directional_overlaps,
            "mean_overlap_ratio": float(np.mean(overlap_ratios)),
            "pair_count": len(frame_scores),
            "valid_pair_count": len(frame_scores),
            "empty_overlap_pair_count": len(empty_overlap_pair_indices),
            "empty_overlap_pair_indices": empty_overlap_pair_indices,
            "empty_overlap_policy": "upstream_unmodified",
            "direction": "symmetric_paper_equation_1",
            "raw_direction": "lower_is_better",
            "resize_policy": RESIZE_POLICY,
            "source_root": str(resolve_met3r_root()),
            "device": str(options["device"]),
            "batch_size": int(options["batch_size"]),
            "img_size": options["img_size"],
            "use_norm": bool(options["use_norm"]),
            "backbone": str(options["backbone"]),
            "feature_backbone": str(options["feature_backbone"]),
            "feature_backbone_weights": str(options["feature_backbone_weights"]),
            "upsampler": str(options["upsampler"]),
            "distance": str(options["distance"]),
            "freeze": bool(options["freeze"]),
            "native_direction": _distance_direction(str(options["distance"])),
        },
    }


def compute_met3r_consistency(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score explicitly supplied pairs; no broadcasting or temporal sampling.

    The benchmark uses compute_met3r_video_consistency instead. This pair-level
    API supports an independent official-call numerical comparison.
    """
    options = _metric_options(runtime)
    pairs = _frame_pairs(reference_frames, prediction_frames)
    delegated = _maybe_run_in_subprocess(reference_frames, prediction_frames, runtime=runtime)
    if delegated is not None:
        return delegated
    size = options["batch_size"]
    return _score_pair_batches((pairs[start:start + size] for start in range(0, len(pairs), size)), options)


def compute_met3r_video_consistency(path: Path, *, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score the supplied generated video, never the source image or anchors."""
    options = _metric_options(runtime)
    delegated = _maybe_run_in_subprocess((), (), runtime=runtime, video_path=Path(path))
    if delegated is not None:
        return delegated
    metadata: dict[str, Any] = {}
    batches = native_video_pair_batches(Path(path), options["batch_size"], metadata)
    try:
        result = _score_pair_batches(batches, options)
    finally:
        batches.close()
    if result["details"]["pair_count"] != metadata["expected_pair_count"]:
        raise ValueError("MEt3R did not score every native consecutive pair")
    result["details"].update(metadata)
    return result


def compute_met3r_consistency_from_paths(
    pairs: Sequence[dict[str, str]],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_frames: list[np.ndarray] = []
    prediction_frames: list[np.ndarray] = []
    for pair in pairs:
        reference_frames.append(
            np.asarray(Image.open(pair["reference_path"]).convert("RGB"))
        )
        prediction_frames.append(
            np.asarray(Image.open(pair["prediction_path"]).convert("RGB"))
        )
    return compute_met3r_consistency(
        reference_frames,
        prediction_frames,
        runtime=runtime,
    )


__all__ = [
    "MET3R_RUNTIME_WORKER_ENV",
    "FEATUP_NATIVE_DIR_ENV",
    "FEATUP_NATIVE_MODULES",
    "compute_met3r_consistency",
    "compute_met3r_video_consistency",
    "compute_met3r_consistency_from_paths",
    "featup_dlopen_environment",
    "featup_runtime_library_dirs",
    "import_featup_native_extensions",
    "inspect_met3r_layout",
    "mount_flags_for",
    "path_allows_exec_mapping",
    "probe_exec_mmap",
    "resolve_featup_native_extension_files",
    "resolve_met3r_root",
    "resolve_met3r_runtime_python",
    "stage_featup_native_extensions",
    "strip_dolphinfs_ld_preload",
]
