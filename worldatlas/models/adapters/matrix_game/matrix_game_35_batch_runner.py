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

from worldatlas.benchmark.synthetic_camera import synthetic_camera_matrices
from worldatlas.models.adapters.base import plan_rollout
from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    begin_sample,
    print_status,
)
from worldatlas.models.adapters.camera_i2v_common import normalize_camera_path


MIN_CAMERA_POSES = 86
POSES_PER_BLOCK = 84
OUTPUT_FPS = 16


def resolve_num_blocks(generation: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``num_blocks`` in place when a target rollout duration is configured.

    Block count is read by camera synthesis, the CLI options, and the action sidecar,
    so resolving it once here keeps those three views consistent.
    """
    target_seconds = generation.get("target_duration_seconds")
    if target_seconds is None:
        return generation
    plan = plan_rollout(
        target_seconds=float(target_seconds),
        native_fps=float(generation.get("output_fps", OUTPUT_FPS)),
        unit_frames=POSES_PER_BLOCK,
        base_frames=1,
    )
    generation["num_blocks"] = plan.unit_count
    generation["rollout_plan"] = plan.as_details()
    return generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_batch_args(parser)
    return parser.parse_args()


def _slug(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "sample"


def _parse_size(value: object) -> tuple[int, int]:
    """Parse a ``width*height`` (or ``widthxheight``) spec into (width, height)."""
    raw = str(value).lower().replace("x", "*")
    width_text, height_text = raw.split("*", 1)
    return int(width_text), int(height_text)


def _target_size(generation: dict[str, Any]) -> tuple[int, int] | None:
    """Resolve the configured conditioning_resize target, or None if unset."""
    value = generation.get("conditioning_resize")
    if not value:
        return None
    width, height = _parse_size(value)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid conditioning_resize: {value!r}")
    return width, height


def _cover_center_crop(image: "Image.Image", target_width: int, target_height: int) -> "Image.Image":
    """Resize-to-cover then center-crop to exactly (target_width, target_height).

    Replicates Matrix-Game's own center_crop_resize verbatim
    (diffsynth/core/data/batch_encode_videos.py): torch bilinear interpolation
    with antialias=True, int(round()) for the covered size, and max((new-t)//2,0)
    center-crop offsets. The pipeline itself would run this same op on the anchor
    frames, so doing it here (before infer.py) keeps the transform identical while
    shrinking the input the pipeline generates at 1280x704 (configs/infer_*_person
    height:704/width:1280). Needed because dataset originals reach 20-40 MP, whose
    full-res DA3 depth map overflows torch.quantile's 2**24-element cap in
    frustum _edge_aware_smooth.
    """
    import torch
    import torch.nn.functional as F

    raw = (
        torch.from_numpy(np.asarray(image.convert("RGB")).copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    old_height, old_width = raw.shape[-2:]
    scale = max(target_height / old_height, target_width / old_width)
    new_height = int(round(old_height * scale))
    new_width = int(round(old_width * scale))
    if (new_height, new_width) != (old_height, old_width):
        raw = F.interpolate(
            raw,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    top = max((new_height - target_height) // 2, 0)
    left = max((new_width - target_width) // 2, 0)
    cropped = raw[:, :, top : top + target_height, left : left + target_width]
    arr = cropped.squeeze(0).permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8).numpy()
    return Image.fromarray(arr)


def _image_size(row: dict[str, Any], generation: dict[str, Any] | None = None) -> tuple[int, int]:
    """Effective anchor size: the conditioning_resize target if set, else original.

    Intrinsics are expressed in pixels of the anchor image (see README), so this
    must return the same size the anchor image is actually fed to infer.py at.
    """
    if generation is not None:
        target = _target_size(generation)
        if target is not None:
            return target
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
    """Translate symbolic WorldAtlas camera segments to Matrix-Game 3.5 c2w poses."""
    camera_path = normalize_camera_path(row.get("camera_path"))
    num_blocks = max(int(generation.get("num_blocks", 1)), 1)
    pose_count = max(MIN_CAMERA_POSES, 1 + POSES_PER_BLOCK * num_blocks)
    trajectory = synthetic_camera_matrices(
        camera_path,
        target_frames=pose_count,
        runtime=generation,
    )
    width, height = _image_size(row, generation)
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
        "output_fps": int(generation.get("output_fps", OUTPUT_FPS)),
        # Upstream describes each block as 80 newly generated frames, while
        # the user-facing history video retains the anchor plus 84 frames per
        # block (85 frames for the common single-block run).
        "generated_frames": 80 * num_blocks,
        "total_frames": 1 + POSES_PER_BLOCK * num_blocks,
        **(
            {"rollout_plan": generation["rollout_plan"]}
            if generation.get("rollout_plan")
            else {}
        ),
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
        source_image = Path(str(row["conditioning_image"])).expanduser().resolve()
        anchor_image = source_image
        target = _target_size(generation)
        if target is not None:
            target_width, target_height = target
            with Image.open(source_image) as handle:
                image = handle.convert("RGB")
                if image.size != (target_width, target_height):
                    image = _cover_center_crop(image, target_width, target_height)
                anchor_image = temp_dir / f"{run_name}_anchor.png"
                image.save(anchor_image)
        command = [
            os.fspath(Path(sys.executable).resolve()),
            os.fspath(repo_root / "infer.py"),
            "--person",
            person,
            "--image",
            str(anchor_image),
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
    generation = resolve_num_blocks(
        load_json(Path(args.generation_config_path).expanduser().resolve())
    )
    person = str(generation.get("person", "first")).strip().lower()
    if person not in {"first", "third"}:
        raise ValueError(f"person must be 'first' or 'third', got {person!r}")
    if not (repo_root / "infer.py").is_file():
        raise FileNotFoundError(f"Matrix-Game 3.5 infer.py not found under {repo_root}")

    checkpoint_layout = resolve_checkpoint_layout(checkpoint_dir, person)
    rows = load_batch_spec(batch_spec_path)
    dry_run = os.environ.get("WORLDATLAS_MATRIX_GAME_35_DRY_RUN", "") == "1"
    failures: list[tuple[str, str]] = []
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "unknown")
        begin_sample(sample_id, index=row_index + 1, total=len(rows), label="MatrixGame35")
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
            failures.append((sample_id, str(exc)))
            print_status(sample_id, "failed", error=str(exc))
    if failures:
        summary = "; ".join(f"{sample_id}: {error}" for sample_id, error in failures)
        raise RuntimeError(f"Matrix-Game 3.5 failed for {len(failures)} sample(s): {summary}")


if __name__ == "__main__":
    main()
