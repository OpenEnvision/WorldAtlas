"""Batch subprocess for the official Matrix-Game 3.5 inference entry point."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices
from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)
from worldarena.models.adapters.camera_i2v_common import normalize_camera_path


MIN_CAMERA_POSES = 86
POSES_PER_BLOCK = 84


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_batch_args(parser)
    return parser.parse_args()


def _slug(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "sample"


def _image_size(row: dict[str, Any]) -> tuple[int, int]:
    with Image.open(Path(str(row["conditioning_image"])).expanduser()) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid conditioning image dimensions: {(width, height)}")
    return int(width), int(height)


def _intrinsics(width: int, height: int, horizontal_fov_degrees: float) -> np.ndarray:
    if not 1.0 < horizontal_fov_degrees < 179.0:
        raise ValueError(
            f"horizontal_fov_degrees must be between 1 and 179, got {horizontal_fov_degrees}"
        )
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0))
    return np.asarray(
        [focal, focal, (width - 1.0) / 2.0, (height - 1.0) / 2.0],
        dtype=np.float32,
    )


def write_camera_npz(
    row: dict[str, Any],
    generation: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Translate symbolic WorldArena camera segments to Matrix-Game 3.5 c2w poses."""
    camera_path = normalize_camera_path(row.get("camera_path"))
    num_blocks = max(int(generation.get("num_blocks", 1)), 1)
    pose_count = max(MIN_CAMERA_POSES, 1 + POSES_PER_BLOCK * num_blocks)
    trajectory = synthetic_camera_matrices(
        camera_path,
        target_frames=pose_count,
        runtime=generation,
    )
    width, height = _image_size(row)
    intrinsic = _intrinsics(
        width,
        height,
        float(generation.get("horizontal_fov_degrees", 75.0)),
    )
    intrinsics = np.repeat(intrinsic[None, :], repeats=pose_count, axis=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        extrinsics_c2w=np.asarray(trajectory.matrices, dtype=np.float32),
        intrinsics=intrinsics,
    )
    return {
        "camera_path": list(trajectory.camera_path),
        "camera_pose_count": pose_count,
        "camera_npz": str(destination),
        "camera_source": str(row.get("camera_source") or trajectory.source),
        "horizontal_fov_degrees": float(generation.get("horizontal_fov_degrees", 75.0)),
        "source_width": width,
        "source_height": height,
    }


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing {description}: {path}")
    return path.resolve()


def resolve_checkpoint_layout(checkpoint_dir: Path, person: str) -> dict[str, Path]:
    """Resolve and validate the exact upstream checkpoint layout."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    model_filename = f"{person}-person.safetensors"
    model_candidates = [
        checkpoint_dir / model_filename,
        checkpoint_dir / "Matrix-Game-3.5-Base" / model_filename,
    ]
    model_checkpoint = next(
        (path for path in model_candidates if path.is_file() and path.stat().st_size > 0),
        None,
    )
    if model_checkpoint is None:
        raise FileNotFoundError(
            f"missing Matrix-Game 3.5 {person}-person checkpoint; tried {model_candidates}"
        )

    wan_dir = checkpoint_dir / "Wan2.2-TI2V-5B"
    _require_file(wan_dir / "Wan2.2_VAE.pth", "Wan2.2 VAE")
    _require_file(wan_dir / "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2 T5 encoder")
    diffusion_files = sorted(wan_dir.glob("diffusion_pytorch_model*.safetensors"))
    if not diffusion_files or any(path.stat().st_size <= 0 for path in diffusion_files):
        raise FileNotFoundError(f"missing Wan2.2 diffusion safetensors under {wan_dir}")

    tokenizer_dir = wan_dir / "google" / "umt5-xxl"
    _require_file(tokenizer_dir / "tokenizer_config.json", "UMT5 tokenizer config")
    da3_dir = checkpoint_dir / "DA3NESTED-GIANT-LARGE-1.1"
    _require_file(da3_dir / "config.json", "Depth-Anything-3 config")
    _require_file(da3_dir / "model.safetensors", "Depth-Anything-3 weights")
    return {
        "model_checkpoint": model_checkpoint.resolve(),
        "wan_dir": wan_dir.resolve(),
        "tokenizer_dir": tokenizer_dir.resolve(),
        "da3_dir": da3_dir.resolve(),
    }


def _append_generation_options(command: list[str], generation: dict[str, Any]) -> None:
    values = {
        "num_blocks": "--num-blocks",
        "steps": "--steps",
        "cfg_scale": "--cfg-scale",
        "seed": "--seed",
    }
    for key, option in values.items():
        value = generation.get(key)
        if value is not None:
            command.extend([option, str(value)])
    for value in generation.get("extra_args", []):
        command.append(str(value))


def _upstream_environment(repo_root: Path) -> dict[str, str]:
    """Expose vendored DA3 before upstream imports ``frustum_handler``.

    Matrix-Game 3.5 adds the DA3 source tree later during estimator creation,
    but ``frustum_handler`` probes for DA3 at module-import time. Prepending the
    vendored package here keeps that probe from incorrectly selecting the
    legacy VideoDepthAnything API.
    """
    env = os.environ.copy()
    python_paths = [
        repo_root / "third_party" / "depth-anything-3" / "src",
        repo_root,
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        python_paths.extend(Path(value) for value in existing.split(os.pathsep) if value)
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    return env


def _write_action_sidecar(
    output_path: Path,
    camera_metadata: dict[str, Any],
    generation: dict[str, Any],
) -> None:
    num_blocks = max(int(generation.get("num_blocks", 1)), 1)
    payload = {
        **camera_metadata,
        "actions": camera_metadata["camera_path"],
        "native_action_space": "matrix_game_3_5_camera_poses",
        "output_fps": 16,
        # Upstream describes each block as 80 newly generated frames, while
        # the user-facing history video retains the anchor plus 84 frames per
        # block (85 frames for the common single-block run).
        "generated_frames": 80 * num_blocks,
        "total_frames": 1 + POSES_PER_BLOCK * num_blocks,
    }
    sidecar = output_path.with_name(f"{output_path.stem}_actions.json")
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_row(
    row: dict[str, Any],
    *,
    repo_root: Path,
    checkpoint_layout: dict[str, Path],
    generation: dict[str, Any],
    person: str,
    batch_spec_path: Path,
    dry_run: bool,
) -> None:
    sample_id = str(row["sample_id"])
    output_path = Path(str(row["output_path"])).expanduser().resolve()
    camera_dir = batch_spec_path.parent / "matrix_game_35_cameras"
    camera_path = camera_dir / f"{_slug(sample_id)}.npz"
    camera_metadata = write_camera_npz(row, generation, camera_path)
    if dry_run:
        print_status(
            sample_id,
            "dry_run",
            person=person,
            checkpoint=str(checkpoint_layout["model_checkpoint"]),
            **camera_metadata,
        )
        return

    run_parent = batch_spec_path.parent / "matrix_game_35_runs"
    run_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{_slug(sample_id)}_", dir=run_parent) as temp_raw:
        temp_dir = Path(temp_raw)
        run_name = _slug(sample_id)
        command = [
            os.fspath(Path(sys.executable).resolve()),
            os.fspath(repo_root / "infer.py"),
            "--person",
            person,
            "--image",
            str(Path(str(row["conditioning_image"])).expanduser().resolve()),
            "--camera",
            str(camera_path),
            "--prompt",
            str(row.get("prompt") or ""),
            "--output",
            str(temp_dir),
            "--name",
            run_name,
            "--ckpt",
            str(checkpoint_layout["model_checkpoint"]),
            "--wan-dir",
            str(checkpoint_layout["wan_dir"]),
            "--tokenizer-dir",
            str(checkpoint_layout["tokenizer_dir"]),
            "--da3-dir",
            str(checkpoint_layout["da3_dir"]),
        ]
        _append_generation_options(command, generation)
        subprocess.run(
            command,
            cwd=repo_root,
            env=_upstream_environment(repo_root),
            check=True,
        )
        person_dir = "first_person" if person == "first" else "third_person"
        result_path = temp_dir / person_dir / run_name / "result.mp4"
        copy_output(result_path, output_path)

    _write_action_sidecar(output_path, camera_metadata, generation)
    print_status(
        sample_id,
        "generated",
        prediction_path=str(output_path),
        person=person,
        **camera_metadata,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    person = str(generation.get("person", "first")).strip().lower()
    if person not in {"first", "third"}:
        raise ValueError(f"person must be 'first' or 'third', got {person!r}")
    if not (repo_root / "infer.py").is_file():
        raise FileNotFoundError(f"Matrix-Game 3.5 infer.py not found under {repo_root}")

    checkpoint_layout = resolve_checkpoint_layout(checkpoint_dir, person)
    rows = load_batch_spec(batch_spec_path)
    dry_run = os.environ.get("WORLDARENA_MATRIX_GAME_35_DRY_RUN", "") == "1"
    failures: list[tuple[str, str]] = []
    for row in rows:
        try:
            _run_row(
                row,
                repo_root=repo_root,
                checkpoint_layout=checkpoint_layout,
                generation=generation,
                person=person,
                batch_spec_path=batch_spec_path,
                dry_run=dry_run,
            )
        except Exception as exc:
            sample_id = str(row.get("sample_id") or "unknown")
            failures.append((sample_id, str(exc)))
            print_status(sample_id, "failed", error=str(exc))
    if failures:
        summary = "; ".join(f"{sample_id}: {error}" for sample_id, error in failures)
        raise RuntimeError(f"Matrix-Game 3.5 failed for {len(failures)} sample(s): {summary}")


if __name__ == "__main__":
    main()
