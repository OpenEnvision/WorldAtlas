"""Resolve metric weights and stage shared checkpoints on node-local storage."""

from __future__ import annotations

import argparse
import fcntl
import os
import shlex
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.common.checkpoints import (
    checkpoint_cache_env,
    checkpoint_path,
    checkpoint_root,
    hf_local_dir,
    project_root,
)
from worldatlas.common.progress import log_progress


SEA_RAFT_KIND = "sea_raft"
MEGASAM_KIND = "megasam"
TRAJAN_KIND = "trajan"
LPIPS_KIND = "lpips"
ALEXNET_WEIGHTS_NAME = "alexnet-owt-7be5be79.pth"
ALEXNET_MIN_BYTES = 1_000_000
CLIP_MIN_WEIGHT_BYTES = 1_000_000
CLIP_VIT_B16_REPO = "openai/clip-vit-base-patch16"
CLIP_VIT_B32_REPO = "openai/clip-vit-base-patch32"
CLIP_REQUIRED_CONFIG_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)
CLIP_TOKENIZER_FILE_GROUPS = (
    ("tokenizer.json",),
    ("vocab.json", "merges.txt"),
)
UNIDEPTH_HF_REPO = "models--lpiccinelli--unidepth-v2-vitl14"
UNIDEPTH_REVISION = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"


@contextmanager
def exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def local_cache_candidates(kind: str) -> list[Path]:
    override = os.environ.get("WORLDATLAS_LOCAL_CHECKPOINT_CACHE", "").strip()
    if override:
        return [Path(override).expanduser() / kind]
    kind_override = os.environ.get(f"WORLDATLAS_{kind.upper()}_LOCAL_CACHE", "").strip()
    if kind_override:
        return [Path(kind_override).expanduser()]
    uid = os.getuid()
    return [
        Path("/dev/shm") / f"worldatlas-{kind}-{uid}",
        Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / f"worldatlas-{kind}-{uid}",
    ]


def _path_has_free_bytes(path: Path, needed: int) -> bool:
    probe = path if path.exists() else path.parent
    try:
        return shutil.disk_usage(str(probe)).free >= needed
    except OSError:
        return False


def is_already_materialized(path: Path, kind: str) -> bool:
    for candidate in local_cache_candidates(kind):
        try:
            if path.resolve().is_relative_to(candidate.resolve()):
                return True
        except (OSError, ValueError):
            continue
    return False


def _choose_local_cache_dir(needed_bytes: int, kind: str) -> Path | None:
    for candidate in local_cache_candidates(kind):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if _path_has_free_bytes(candidate, needed_bytes):
            return candidate
    return None


def _skip_local_copy() -> bool:
    return os.environ.get("WORLDATLAS_SKIP_LOCAL_CHECKPOINT_COPY") == "1"


def materialize_local_file(src: Path, *, kind: str, required: bool = True) -> Path:
    """Flock + copy one file to node-local scratch. Reuse if already present."""
    src = Path(src)
    if not src.is_file():
        if required:
            raise MetricLoadError(f"checkpoint missing: {src}")
        raise FileNotFoundError(src)
    needed = src.stat().st_size
    if _skip_local_copy():
        return src
    if is_already_materialized(src, kind) and src.is_file():
        from worldatlas.benchmark.sample_identity import file_sha256
        expected = src.parent.name
        if len(expected) == 64 and all(c in '0123456789abcdef' for c in expected) and file_sha256(src) != expected:
            raise MetricLoadError(f'Corrupted content-addressed checkpoint: {src}')
        return src
    from worldatlas.benchmark.sample_identity import file_sha256
    source_digest = file_sha256(src)
    for candidate in local_cache_candidates(kind):
        existing = candidate / source_digest / src.name
        try:
            if existing.is_file() and file_sha256(existing) == source_digest:
                return existing
        except OSError:
            continue
    cache_dir = _choose_local_cache_dir(needed + 64 * 1024**2, kind)
    if cache_dir is None:
        log_progress(
            "metric_load",
            metric=kind,
            status="local_copy_skip",
            checkpoint=str(src),
            note="no local scratch with enough free space; falling back to source path",
        )
        return src

    dest = cache_dir / source_digest / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dest.parent / f"{src.name}.lock"
    with exclusive_file_lock(lock_path):
        if dest.is_file() and file_sha256(dest) == source_digest:
            return dest
        temporary = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        if temporary.exists():
            temporary.unlink()
        log_progress(
            "metric_load",
            metric=kind,
            status="local_copy_start",
            src=str(src),
            dest=str(dest),
            bytes=needed,
        )
        try:
            shutil.copyfile(src, temporary)
            copied = temporary.stat().st_size
            if copied != needed:
                raise OSError(
                    f"local copy size mismatch: {temporary} has {copied}, expected {needed}"
                )
            if file_sha256(temporary) != source_digest or file_sha256(src) != source_digest:
                raise OSError('Checkpoint contents changed or were corrupted during local copy')
            os.replace(temporary, dest)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        log_progress(
            "metric_load",
            metric=kind,
            status="local_copy_ok",
            dest=str(dest),
            bytes=needed,
        )
        return dest


def _tree_nbytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _tree_content_digest(src: Path) -> str:
    import hashlib
    import json
    from worldatlas.benchmark.sample_identity import file_sha256
    files = {str(path.relative_to(src)): {'sha256': file_sha256(path),
             'link': os.readlink(path) if path.is_symlink() else None}
             for path in sorted(src.rglob('*')) if path.is_file()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def materialize_local_tree(src: Path, *, kind: str, dest_name: str | None = None) -> Path:
    """Flock + copy a directory tree (HF hub snapshots with relative blob links)."""
    src = Path(src)
    if not src.is_dir():
        raise MetricLoadError(f"checkpoint directory missing: {src}")
    if _skip_local_copy():
        return src
    if is_already_materialized(src, kind):
        expected = src.parent.name
        if len(expected) == 64 and all(c in '0123456789abcdef' for c in expected) and _tree_content_digest(src) != expected:
            raise MetricLoadError(f'Corrupted content-addressed checkpoint tree: {src}')
        return src
    needed = _tree_nbytes(src)
    name = dest_name or src.name
    source_digest = _tree_content_digest(src)
    for candidate in local_cache_candidates(kind):
        existing = candidate / source_digest / name
        try:
            if existing.is_dir() and _tree_content_digest(existing) == source_digest:
                return existing
        except OSError:
            continue
    cache_dir = _choose_local_cache_dir(needed + 256 * 1024**2, kind)
    if cache_dir is None:
        log_progress(
            "metric_load",
            metric=kind,
            status="local_copy_skip",
            checkpoint=str(src),
            note="no local scratch for tree copy; falling back to source path",
        )
        return src

    dest = cache_dir / source_digest / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dest.parent / f"{Path(name).name}.lock"
    with exclusive_file_lock(lock_path):
        if dest.is_dir() and _tree_content_digest(dest) == source_digest:
            return dest
        temporary = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        log_progress(
            "metric_load",
            metric=kind,
            status="local_copy_start",
            src=str(src),
            dest=str(dest),
            bytes=needed,
        )
        try:
            shutil.copytree(src, temporary, symlinks=True)
            if _tree_content_digest(temporary) != source_digest or _tree_content_digest(src) != source_digest:
                raise OSError('Checkpoint tree changed or was corrupted during local copy')
            os.replace(temporary, dest)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise
        log_progress(
            "metric_load",
            metric=kind,
            status="local_copy_ok",
            dest=str(dest),
            bytes=needed,
        )
        return dest


def sea_raft_checkpoint_path() -> Path:
    override = os.environ.get("WORLDATLAS_SEA_RAFT_CHECKPOINT", "").strip()
    if override:
        return Path(override).expanduser()
    return checkpoint_path("official_metrics", "Tartan-C-T-TSKH-spring540x960-M.pth", kind="file")


def megasam_root() -> Path:
    override = os.environ.get("MEGASAM_ROOT", "").strip()
    if override:
        return Path(override)
    return project_root() / "thirdparty" / "mega-sam"


def _usable_file(path: Path) -> bool:
    """True when ``path`` exists as a non-empty regular file (symlink targets count)."""
    try:
        resolved = path.resolve()
        return resolved.is_file() and resolved.stat().st_size > 0
    except OSError:
        return False


def source_hf_home() -> Path:
    """Persistent UniDepth source, independent of the node-local runtime cache."""
    override = os.environ.get("WORLDATLAS_SOURCE_HF_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return checkpoint_root(create=False) / "huggingface"


def default_hf_home() -> Path:
    localized = os.environ.get("WORLDATLAS_MEGASAM_HF_HOME", "").strip()
    if localized:
        candidate = Path(localized)
        if _usable_file(unidepth_weight_path(candidate)):
            return candidate
    override = os.environ.get("HF_HOME", "").strip()
    if override:
        candidate = Path(override)
        if not is_already_materialized(candidate, MEGASAM_KIND):
            return candidate
    return source_hf_home()


def unidepth_hub_dir(hf_home: Path | None = None) -> Path:
    return (hf_home or default_hf_home()) / "hub" / UNIDEPTH_HF_REPO


def unidepth_weight_path(hf_home: Path | None = None) -> Path:
    return unidepth_hub_dir(hf_home) / "snapshots" / UNIDEPTH_REVISION / "model.safetensors"


def megasam_checkpoint_path(mega_root: Path | None = None) -> Path:
    override = os.environ.get("WORLDATLAS_MEGASAM_CHECKPOINT", "").strip()
    if override:
        path = Path(override)
        if _usable_file(path):
            return path
    root = mega_root or megasam_root()
    return root / "checkpoints" / "megasam_final.pth"


def depth_anything_checkpoint_path(mega_root: Path | None = None) -> Path:
    override = os.environ.get("WORLDATLAS_DEPTH_ANYTHING_CHECKPOINT", "").strip()
    if override:
        path = Path(override)
        if _usable_file(path):
            return path
    root = mega_root or megasam_root()
    return root / "Depth-Anything" / "checkpoints" / "depth_anything_vitl14.pth"


def required_megasam_paths() -> list[tuple[str, Path]]:
    root = megasam_root()
    hf_home = source_hf_home()
    localized_hf = os.environ.get("WORLDATLAS_MEGASAM_HF_HOME", "").strip()
    unidepth = unidepth_weight_path(Path(localized_hf)) if localized_hf else unidepth_weight_path(hf_home)
    if not _usable_file(unidepth):
        unidepth = unidepth_weight_path(hf_home)
    return [
        ("thirdparty/mega-sam", root),
        ("megasam_final.pth", megasam_checkpoint_path(root)),
        ("depth_anything_vitl14.pth", depth_anything_checkpoint_path(root)),
        ("unidepth-v2-vitl14 snapshot", unidepth),
    ]


def missing_megasam_paths() -> list[str]:
    missing: list[str] = []
    for label, path in required_megasam_paths():
        if path.is_dir():
            continue
        if _usable_file(path):
            continue
        missing.append(f"{label} (got: {path})")
    return missing


def alexnet_official_checkpoint_path() -> Path:
    return checkpoint_path("official_metrics", ALEXNET_WEIGHTS_NAME, kind="file", required=False)


def alexnet_torch_hub_path() -> Path:
    torch_home = os.environ.get("TORCH_HOME", "").strip()
    if not torch_home:
        torch_home = str(checkpoint_cache_env(create=False).torch_home)
    return Path(torch_home).expanduser() / "hub" / "checkpoints" / ALEXNET_WEIGHTS_NAME


def _is_usable_alexnet(path: Path) -> bool:
    try:
        return _usable_file(path) and path.stat().st_size >= ALEXNET_MIN_BYTES
    except OSError:
        return False


def ensure_local_alexnet_weights() -> Path:
    """Point torchvision LPIPS at a local AlexNet file. Never download."""
    official = alexnet_official_checkpoint_path()
    hub = alexnet_torch_hub_path()
    src = next((candidate for candidate in (official, hub) if _is_usable_alexnet(candidate)), None)
    if src is None:
        raise MetricLoadError(
            "LPIPS AlexNet weights missing (offline). "
            f"Expected {official} or {hub}. "
            "Will not download from download.pytorch.org."
        )
    hub.parent.mkdir(parents=True, exist_ok=True)
    if _is_usable_alexnet(hub):
        return hub
    if hub.exists() or hub.is_symlink():
        hub.unlink()
    try:
        hub.symlink_to(src.resolve())
    except OSError:
        shutil.copyfile(src, hub)
    if not _is_usable_alexnet(hub):
        raise MetricLoadError(f"LPIPS AlexNet weights could not be placed at {hub}")
    return hub


def prepare_lpips_alexnet_checkpoint() -> dict[str, str]:
    dest = ensure_local_alexnet_weights()
    return {"WORLDATLAS_LPIPS_ALEXNET": str(dest)}


def _clip_processor_complete(model_dir: Path) -> bool:
    if not all(_usable_file(model_dir / filename) for filename in CLIP_REQUIRED_CONFIG_FILES):
        return False
    return any(
        all(_usable_file(model_dir / filename) for filename in tokenizer_group)
        for tokenizer_group in CLIP_TOKENIZER_FILE_GROUPS
    )


def _clip_weight_path(model_dir: Path) -> Path | None:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        try:
            if _usable_file(candidate) and candidate.stat().st_size >= CLIP_MIN_WEIGHT_BYTES:
                return candidate
        except OSError:
            continue
    return None


def resolve_local_clip_dir(repo_id: str) -> Path:
    """Resolve an offline CLIP tree. Never downloads."""
    try:
        model_dir = hf_local_dir(repo_id, required=True)
    except FileNotFoundError as exc:
        raise MetricLoadError(
            f"CLIP checkpoint missing for {repo_id} (offline): {exc}. "
            "Will not download from huggingface.co."
        ) from exc
    if not _clip_processor_complete(model_dir):
        raise MetricLoadError(
            f"CLIP processor files are incomplete under {model_dir} for {repo_id} (offline). "
            f"Expected {CLIP_REQUIRED_CONFIG_FILES} and tokenizer files. "
            "Will not download from huggingface.co."
        )
    weight = _clip_weight_path(model_dir)
    if weight is None:
        raise MetricLoadError(
            f"CLIP weights missing under {model_dir} for {repo_id} (offline). "
            "Expected model.safetensors or pytorch_model.bin. "
            "Will not download from huggingface.co."
        )
    return model_dir


def prepare_clip_checkpoint(repo_id: str) -> dict[str, str]:
    """Fail-fast offline CLIP resolve. Do not copy the ~600MB weight tree."""
    model_dir = resolve_local_clip_dir(repo_id)
    weight = _clip_weight_path(model_dir)
    assert weight is not None
    suffix = "B16" if "patch16" in repo_id else "B32"
    return {
        f"WORLDATLAS_CLIP_{suffix}": str(model_dir),
        f"WORLDATLAS_CLIP_{suffix}_WEIGHTS": str(weight),
    }


def prepare_sea_raft_checkpoint() -> dict[str, str]:
    src = sea_raft_checkpoint_path()
    dest = materialize_local_file(src, kind=SEA_RAFT_KIND)
    return {"WORLDATLAS_SEA_RAFT_CHECKPOINT": str(dest)}


def prepare_megasam_checkpoints() -> dict[str, str]:
    missing = missing_megasam_paths()
    if missing:
        raise MetricLoadError(
            "camera_error MegaSAM runtime is incomplete:\n  - "
            + "\n  - ".join(missing)
            + "\nRefusing to start so this job does not silently write per-sample N/A."
        )
    root = megasam_root()
    megasam_ckpt = materialize_local_file(megasam_checkpoint_path(root), kind=MEGASAM_KIND)
    depth_ckpt = materialize_local_file(depth_anything_checkpoint_path(root), kind=MEGASAM_KIND)
    hub_src = unidepth_hub_dir(source_hf_home())
    hub_dest = materialize_local_tree(
        hub_src,
        kind=MEGASAM_KIND,
        dest_name=f"hub/{UNIDEPTH_HF_REPO}",
    )
    local_hf = hub_dest.parent.parent
    weight = unidepth_weight_path(local_hf)
    if not _usable_file(weight):
        raise MetricLoadError(
            "UniDepth snapshot copy is unreadable after local materialize: "
            f"{weight} (source={unidepth_weight_path(source_hf_home())})"
        )
    return {
        "WORLDATLAS_MEGASAM_CHECKPOINT": str(megasam_ckpt),
        "WORLDATLAS_DEPTH_ANYTHING_CHECKPOINT": str(depth_ckpt),
        "WORLDATLAS_MEGASAM_HF_HOME": str(local_hf),
    }


def prepare_trajan_checkpoints() -> dict[str, str]:
    tracker = materialize_local_file(
        checkpoint_path(
            "TRAJAN",
            "bootstapir_checkpoint_v2.npy",
            kind="file",
            required=False,
        ),
        kind=TRAJAN_KIND,
    )
    autoencoder = materialize_local_file(
        checkpoint_path(
            "TRAJAN",
            "track_autoencoder_ckpt.npz",
            kind="file",
            required=False,
        ),
        kind=TRAJAN_KIND,
    )
    return {
        "WORLDATLAS_TRAJAN_TRACKER_CHECKPOINT": str(tracker),
        "WORLDATLAS_TRAJAN_AUTOENCODER_CHECKPOINT": str(autoencoder),
    }


def prepare_metric_checkpoints(metrics: Iterable[str]) -> dict[str, str]:
    """Copy shared weights once per node for the selected metrics."""
    names = {str(name).strip() for name in metrics if str(name).strip()}
    env: dict[str, str] = {}
    if names & {"optical_flow_aepe", "motion_magnitude"}:
        env.update(prepare_sea_raft_checkpoint())
    if "motion_smoothness" in names:
        env.update(prepare_lpips_alexnet_checkpoint())
    if names & {"prompt_alignment", "scene_alignment", "dynamic_alignment"}:
        env.update(prepare_clip_checkpoint(CLIP_VIT_B16_REPO))
    if "background_consistency" in names:
        env.update(prepare_clip_checkpoint(CLIP_VIT_B32_REPO))
    if "camera_error" in names:
        env.update(prepare_megasam_checkpoints())
    if "trajan" in names:
        env.update(prepare_trajan_checkpoints())
    if any("met3r" in name.lower() for name in names):
        from worldatlas.benchmark.met3r_runtime import stage_featup_native_extensions

        try:
            env.update(stage_featup_native_extensions(apply_sys_path=False))
        except FileNotFoundError as exc:
            raise MetricLoadError(str(exc)) from exc
    return env


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize WorldAtlas metric checkpoints locally.")
    parser.add_argument("--metrics", required=True, help="Comma-separated metric names.")
    parser.add_argument(
        "--export-env",
        action="store_true",
        help="Print KEY=value lines for bash eval/export.",
    )
    parser.add_argument("--write-env", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metrics = [part.strip() for part in args.metrics.split(",") if part.strip()]
    try:
        env_updates = prepare_metric_checkpoints(metrics)
    except MetricLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    os.environ.update(env_updates)
    lines = [f"{key}={shlex.quote(value)}" for key, value in env_updates.items()]
    text = "\n".join(lines) + ("\n" if lines else "")
    if args.write_env is not None:
        args.write_env.parent.mkdir(parents=True, exist_ok=True)
        args.write_env.write_text(text, encoding="utf-8")
    if args.export_env:
        print(text, end="")
    else:
        for line in lines:
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
