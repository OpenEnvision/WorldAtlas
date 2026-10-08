"""Bridge WorldAtlas batch rows to SolarWM's official Stage2 CLI."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
from typing import Any

import numpy as np

from worldatlas.benchmark.annotations import load_camera_matrices
from worldatlas.common.checkpoints import project_root as worldatlas_project_root
from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    begin_sample,
    copy_output,
    load_batch_spec,
    load_json,
    log_pipeline,
    print_status,
)


CAMERA_CONVENTION = "authoritative_source_c2w_no_axis_flip"
DEFAULT_CONFIG = "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml"
PUBLIC_DATASET = "worldatlas"
PROJECT_ROOT = worldatlas_project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas SolarWM batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _resolve_child(root: Path, value: Any, default: str) -> Path:
    raw = str(value or default).strip()
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def resolve_runtime_paths(
    repo_root: Path,
    checkpoint_root: Path,
    generation: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Resolve and validate the official config, base model, and Stage2 release."""
    config_path = _resolve_child(repo_root, generation.get("config_path"), DEFAULT_CONFIG)
    base_model = _resolve_child(
        checkpoint_root,
        generation.get("base_model_dir"),
        "SolarWM-5B-base",
    )
    stage2_checkpoint = _resolve_child(
        checkpoint_root,
        generation.get("stage2_checkpoint_dir"),
        "SolarWM-5B-sgf-stage2-81f",
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"SolarWM inference config not found: {config_path}")
    if not base_model.is_dir():
        raise FileNotFoundError(f"SolarWM base model not found: {base_model}")
    if not stage2_checkpoint.is_dir():
        raise FileNotFoundError(f"SolarWM Stage2 checkpoint not found: {stage2_checkpoint}")
    return config_path, base_model, stage2_checkpoint


def _tar_bytes(handle: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    handle.addfile(info, BytesIO(payload))


def _camera_npz(poses: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.savez(buffer, c2w=np.ascontiguousarray(poses, dtype=np.float32))
    return buffer.getvalue()


def resolve_ffmpeg(ffmpeg_bin: str) -> Path:
    candidates = (
        Path(ffmpeg_bin).expanduser() if "/" in ffmpeg_bin else None,
        Path(found) if (found := shutil.which(ffmpeg_bin)) else None,
        PROJECT_ROOT.parent / "conda/envs/worldatlas/bin/ffmpeg",
    )
    executable = next(
        (path.absolute() for path in candidates if path and path.is_file() and os.access(path, os.X_OK)),
        None,
    )
    if executable is None:
        checked = ", ".join(str(path) for path in candidates if path)
        raise FileNotFoundError(
            f"SolarWM requires executable ffmpeg; configured={ffmpeg_bin!r}, checked={checked}"
        )
    return executable


def build_solarwm_environment(generation: dict[str, Any]) -> dict[str, str]:
    """Expose the bridge's ffmpeg to upstream readiness checks and MP4 encoding."""
    executable = resolve_ffmpeg(str(generation.get("ffmpeg_bin", "ffmpeg")))
    env = os.environ.copy()
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", os.defpath)
    found = shutil.which("ffmpeg", path=env["PATH"])
    if found is None or not Path(found).samefile(executable):
        raise RuntimeError(f"SolarWM requires ffmpeg on child PATH; configured executable: {executable}")
    result = subprocess.run(
        [found, "-version"], env=env, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError("SolarWM ffmpeg runtime check failed:\n" + result.stderr + result.stdout)
    return env


def _render_conditioning_video(
    image_path: Path,
    output_path: Path,
    *,
    frame_count: int,
    fps: float,
    width: int,
    height: int,
    ffmpeg_bin: str,
) -> None:
    executable = resolve_ffmpeg(ffmpeg_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-loop",
        "1",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(image_path),
        "-vf",
        (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},format=yuv420p"
        ),
        "-frames:v",
        str(frame_count),
        "-r",
        f"{fps:g}",
        "-an",
        "-c:v",
        "libx264",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode or not output_path.is_file() or output_path.stat().st_size <= 0:
        detail = (result.stderr or result.stdout or "unknown ffmpeg failure").strip()
        raise RuntimeError(f"failed to build SolarWM conditioning video: {detail}")


def build_input_bundle(
    rows: list[dict[str, Any]],
    input_root: Path,
    generation: dict[str, Any],
) -> tuple[Path, dict[str, Path]]:
    """Materialize one canonical uncompressed WDS shard and its test index."""
    fps = float(generation.get("fps", 16.0))
    width = int(generation.get("width", 864))
    height = int(generation.get("height", 480))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"SolarWM fps must be positive and finite, got {fps!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"SolarWM input dimensions must be positive, got {width}x{height}")

    shard_key = "raw-wds/worldatlas/shards/solarwm-input-000000.tar"
    shard_path = input_root / shard_key
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    video_root = input_root / "_source_videos"
    index_rows: list[dict[str, Any]] = []
    published_outputs: dict[str, Path] = {}
    seen_ids: set[str] = set()
    ffmpeg_bin = str(generation.get("ffmpeg_bin", "ffmpeg"))

    with tarfile.open(shard_path, "w") as archive:
        for index, row in enumerate(rows):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"SolarWM batch has missing or duplicate sample_id: {sample_id!r}")
            seen_ids.add(sample_id)
            key = f"sample-{index:06d}"
            clip_id = key
            frame_count = int(row.get("source_frame_count", 0))
            if frame_count < 81:
                raise ValueError(f"SolarWM sample {sample_id!r} needs at least 81 frames")

            image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"conditioning image not found: {image_path}")
            annotation_path = str(row.get("annotation_path", ""))
            poses = load_camera_matrices(annotation_path, target_frames=frame_count)
            if poses is None:
                raise FileNotFoundError(
                    f"SolarWM camera poses not found for {sample_id!r}: {annotation_path}"
                )
            poses = np.asarray(poses, dtype=np.float32)
            if poses.shape != (frame_count, 4, 4) or not np.isfinite(poses).all():
                raise ValueError(
                    f"SolarWM camera poses for {sample_id!r} must be finite "
                    f"[{frame_count},4,4], got {poses.shape}"
                )

            video_path = video_root / f"{key}.mp4"
            _render_conditioning_video(
                image_path,
                video_path,
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                ffmpeg_bin=ffmpeg_bin,
            )
            video_member = f"{key}.mp4"
            camera_member = f"{key}.camera.npz"
            archive.add(video_path, arcname=video_member, recursive=False)
            _tar_bytes(archive, camera_member, _camera_npz(poses))

            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"SolarWM sample {sample_id!r} has an empty prompt")
            manifest = {
                "video": {"fps": fps, "num_frames": frame_count},
                "camera": {
                    "array_key": "c2w",
                    "convention": CAMERA_CONVENTION,
                    "dtype": "float32",
                    "finite": True,
                    "magnitude_audit_seconds": 10.0,
                    "max_camera_abs": 20.0,
                    "max_rel_translation": 20.0,
                    "shape": [frame_count, 4, 4],
                },
                "prompt": {"text": prompt},
                "metadata": {"scene": str(row.get("suite") or PUBLIC_DATASET)},
            }
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "key": key,
                    "shard": shard_key,
                    "video_member": video_member,
                    "camera_member": camera_member,
                    "caption": prompt,
                    "fps": fps,
                    "num_frames": frame_count,
                    "dataset": PUBLIC_DATASET,
                    "physical_generation": PUBLIC_DATASET,
                    "clip_id": clip_id,
                    "start_frame": 0,
                    "manifest": manifest,
                }
            )
            published_outputs[sample_id] = Path("generate") / PUBLIC_DATASET / f"{clip_id}.mp4"

    index_path = input_root / "test-index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    return index_path, published_outputs


def _override(name: str, value: Any) -> str:
    return f"{name}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"


def build_solarwm_command(
    *,
    config_path: Path,
    base_model: Path,
    stage2_checkpoint: Path,
    input_root: Path,
    publication_root: Path,
    run_id: str,
    sample_count: int,
    generation: dict[str, Any],
) -> list[str]:
    """Build the official one-GPU Stage2 camera-length inference command."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        "1",
        "-m",
        "solarwm",
        "infer",
        "--config",
        str(config_path),
    ]
    overrides: list[tuple[str, Any]] = [
        ("model.base_path", str(base_model)),
        ("checkpoint.path", str(stage2_checkpoint)),
        ("data.index_root", str(input_root)),
        ("data.transport.kind", "local"),
        ("data.transport.root", str(input_root)),
        ("data.test_index", "test-index.jsonl"),
        # SolarWM requires a positive worker count even for inference. These
        # are CPU data workers, independent of the resident model/GPU workers.
        ("data.num_workers", 1),
        ("validation.sample_count", int(sample_count)),
        ("validation.selection_seed", int(generation.get("selection_seed", 42))),
        ("validation.noise_seed", int(generation.get("seed", 42))),
        ("inference.length", "camera"),
        ("inference.output_layout", "dataset_triplet_v1"),
        ("inference.run_id", run_id),
        ("runtime.output_dir", str(publication_root)),
    ]
    extra = generation.get("solar_overrides", {})
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("generation.solar_overrides must be a mapping")
    reserved = {"runtime.output_dir", "inference.run_id", "inference.output_layout"}
    if reserved.intersection(extra or {}):
        raise ValueError("SolarWM publication paths/layout are managed by the batch adapter")
    overrides.extend(sorted((str(key), value) for key, value in (extra or {}).items()))
    for name, value in overrides:
        command.extend(["--set", _override(name, value)])
    return command


def validate_solarwm_command(
    command: list[str], repo_root: Path, *, env: dict[str, str] | None = None,
) -> None:
    """Run the upstream schema/backend checks before rendering bridge videos."""
    infer_index = command.index("infer")
    result = subprocess.run(
        [sys.executable, "-m", "solarwm", "config", "resolve", *command[infer_index + 1:]],
        cwd=str(repo_root), env=env, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "SolarWM config validation failed before input bridge preparation:\n"
            + result.stderr + result.stdout
        )


def validate_publication_filesystem(
    root: Path, repo_root: Path, env: dict[str, str],
) -> None:
    """Exercise the actual upstream commit primitives before loading a model."""
    probe = """
import sys
import tempfile
from pathlib import Path
from solarwm.runtime.create_only import publish_directory_no_replace, write_file_create_only, link_file_create_only
with tempfile.TemporaryDirectory(prefix='.publication-probe-', dir=sys.argv[1]) as temporary:
    root = Path(temporary)
    source, target = root / 'staging', root / 'committed'
    source.mkdir()
    write_file_create_only(source / 'artifact', b'probe', error_type=RuntimeError, label='preflight file')
    publish_directory_no_replace(source, target, error_type=RuntimeError, label='preflight directory')
    link_file_create_only(target / 'artifact', root / 'linked', error_type=RuntimeError, label='preflight link')
    source.mkdir()
    try:
        publish_directory_no_replace(source, target, error_type=RuntimeError, label='preflight collision')
    except RuntimeError:
        assert source.is_dir() and (target / 'artifact').read_bytes() == b'probe'
    else:
        raise RuntimeError('publication overwrote an existing target')
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(root)],
        cwd=str(repo_root), env=env, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"SolarWM publication filesystem check failed at {root}; "
            "set WORLDATLAS_SOLARWM_SCRATCH_ROOT to a node-local Linux filesystem:\n"
            + result.stderr + result.stdout
        )


@contextmanager
def local_publication_workspace(
    publication_root: Path, repo_root: Path, env: dict[str, str],
):
    """Keep upstream's strict create-only transactions off shared/FUSE storage.

    Each GPU worker owns a private local directory. Copy the completed tree back
    to its unique batch directory before removing local artifacts. On any failure
    retain the local tree for diagnosis/recovery, never overwrite a previous tree.
    """
    scratch_root = Path(env.get("WORLDATLAS_SOLARWM_SCRATCH_ROOT", "/tmp")).expanduser().resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    if publication_root.exists():
        raise FileExistsError(f"SolarWM publication already exists: {publication_root}")
    local_root = Path(tempfile.mkdtemp(prefix="worldatlas-solarwm-", dir=scratch_root))
    try:
        validate_publication_filesystem(local_root, repo_root, env)
        log_pipeline("publication_ready", label="SolarWM", local_root=str(local_root))
        yield local_root
        publication_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_root, publication_root)
    except BaseException:
        log_pipeline("publication_retained", label="SolarWM", local_root=str(local_root))
        raise
    else:
        # Only this helper's mkdtemp directory, after successful shared export.
        shutil.rmtree(local_root)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_root = Path(str(args.checkpoint_dir or "")).expanduser().resolve()
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    config_path = _resolve_child(repo_root, generation.get("config_path"), DEFAULT_CONFIG)

    batch_root = Path(args.batch_spec_path).expanduser().resolve().with_suffix(".solarwm")
    input_root = batch_root / "input"
    publication_root = batch_root / "publication"
    run_id = "worldatlas-" + batch_root.stem.replace("_", "-").replace(".", "-")

    if os.environ.get("WORLDATLAS_SOLARWM_DRY_RUN") == "1":
        command = build_solarwm_command(
            config_path=config_path,
            base_model=checkpoint_root / "SolarWM-5B-base",
            stage2_checkpoint=checkpoint_root / "SolarWM-5B-sgf-stage2-81f",
            input_root=input_root,
            publication_root=publication_root,
            run_id=run_id,
            sample_count=len(rows),
            generation=generation,
        )
        print(json.dumps({"dry_run": True, "samples": len(rows), "command": command}))
        return 0

    config_path, base_model, stage2_checkpoint = resolve_runtime_paths(
        repo_root,
        checkpoint_root,
        generation,
    )
    runtime_env = build_solarwm_environment(generation)
    with local_publication_workspace(publication_root, repo_root, runtime_env) as local_root:
        command = build_solarwm_command(
            config_path=config_path,
            base_model=base_model,
            stage2_checkpoint=stage2_checkpoint,
            input_root=input_root,
            publication_root=local_root,
            run_id=run_id,
            sample_count=len(rows),
            generation=generation,
        )
        validate_solarwm_command(command, repo_root, env=runtime_env)
        index_path, published_outputs = build_input_bundle(rows, input_root, generation)
        log_pipeline(
            "pipeline_loading",
            label="SolarWM",
            samples=len(rows),
            checkpoint=str(stage2_checkpoint),
            index=str(index_path),
        )
        result = subprocess.run(command, cwd=str(repo_root), env=runtime_env, check=False)
        if result.returncode:
            raise RuntimeError(f"SolarWM inference failed with exit code {result.returncode}")
        local_completion = local_root / "runs" / run_id / "COMPLETE.json"
        if not local_completion.is_file():
            raise FileNotFoundError(f"SolarWM run completion marker not found: {local_completion}")
    completion = publication_root / "runs" / run_id / "COMPLETE.json"
    if not completion.is_file():
        raise FileNotFoundError(f"SolarWM run completion marker not found: {completion}")
    log_pipeline("pipeline_loaded", label="SolarWM", samples=len(rows))

    failed = 0
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", index))
        begin_sample(sample_id, index=index + 1, total=len(rows), label="SolarWM")
        try:
            source = publication_root / published_outputs[sample_id]
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            copy_output(source, output_path)
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                pose_source=row.get("pose_source"),
                source_frame_count=row.get("source_frame_count"),
                solarwm_run=str(completion.parent),
            )
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 1 if failed == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_input_bundle", "build_solarwm_command", "resolve_runtime_paths"]
