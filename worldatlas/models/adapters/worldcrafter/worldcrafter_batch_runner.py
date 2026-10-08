"""Keep one official WorldCrafter model resident across a WorldAtlas batch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

from worldatlas.benchmark.annotations import load_camera_matrices
from worldatlas.common.media import probe_video
from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args, begin_sample, load_batch_spec, load_json, log_pipeline,
    print_status,
)
from worldatlas.models.adapters.worldcrafter.geometry import FRAMES_PER_CHUNK, output_frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_batch_args(parser)
    return parser.parse_args()


def install_local_lora_name_hint() -> None:
    """Let upstream's local LoRA calls work with WorldAtlas's offline HF runtime."""
    from diffusers.loaders import lora_base

    original = lora_base._best_guess_weight_name

    def guess(path, file_extension=".safetensors", local_files_only=False):
        candidate = Path(str(path)) / "pytorch_lora_weights.safetensors"
        if file_extension == ".safetensors" and candidate.is_file():
            return candidate.name
        return original(path, file_extension=file_extension, local_files_only=local_files_only)

    lora_base._best_guess_weight_name = guess


def prepare_camera(row: dict, directory: Path) -> Path:
    """Read authoritative or synthetic c2w poses and satisfy the upstream 33-frame contract."""
    frames = int(row["num_frames"])
    chunks = int(row["num_chunks"])
    if chunks < 1 or frames != chunks * FRAMES_PER_CHUNK:
        raise ValueError("WorldCrafter requires whole 33-frame camera chunks")
    poses = load_camera_matrices(row["annotation_path"], target_frames=frames)
    if poses is None or poses.shape != (frames, 4, 4):
        raise ValueError(f"Missing c2w trajectory for {row['sample_id']}")
    if row.get("pose_source") == "synthetic":
        # WorldAtlas's pedestal-up token moves +Y; WorldCrafter's up action
        # moves -Y in its right/down/forward camera convention.
        poses = poses.copy()
        poses[:, 1, 3] *= -1
    if not np.isfinite(poses).all():
        raise ValueError("WorldCrafter camera contains nonfinite values")
    rotation = poses[:, :3, :3]
    if not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=5e-3):
        raise ValueError("WorldCrafter camera rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1, atol=5e-3):
        raise ValueError("WorldCrafter camera rotations must have determinant +1")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{row['prediction_stem']}.camera.npy"
    np.save(path, poses[:, :3, :4].astype(np.float32), allow_pickle=False)
    return path


def verify_video(path: Path, *, expected_frames: int, fps: int) -> dict:
    probe = probe_video(path)
    actual_frames = int(probe.get("frame_count") or round(float(probe["duration_seconds"]) * float(probe["fps"])))
    if actual_frames != expected_frames or not math.isclose(float(probe["fps"]), fps, abs_tol=0.05):
        raise RuntimeError(f"WorldCrafter output timing mismatch: {path}: {probe}")
    if (int(probe["width"]), int(probe["height"])) != (640, 384):
        raise RuntimeError(f"WorldCrafter output geometry mismatch: {path}: {probe}")
    return probe


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if not (repo / "worldcrafter" / "inference.py").is_file():
        raise FileNotFoundError(f"WorldCrafter source is incomplete: {repo}")
    rows = load_batch_spec(Path(args.batch_spec_path).resolve())
    generation = load_json(Path(args.generation_config_path).resolve())
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"WorldCrafter checkpoint not found: {checkpoint}")

    from worldcrafter.inference import WorldCrafter
    install_local_lora_name_hint()

    model_type = str(generation.get("model_type", "fast"))
    fps = int(generation.get("fps", 16))
    seed = int(generation.get("seed", 42))
    log_pipeline("worldcrafter_load_start", model_type=model_type, checkpoint=str(checkpoint))
    model = WorldCrafter.from_pretrained(
        checkpoint,
        model_type=model_type,
        device=str(generation.get("device", "cuda:0")),
        height=384,
        width=640,
        seed=seed,
        attention_backend=str(generation.get("attention_backend", "native")),
        enable_compile=bool(generation.get("enable_compile", False)),
    )
    log_pipeline("worldcrafter_load_done", model_type=model_type, samples=len(rows))
    work = Path(args.batch_spec_path).resolve().with_suffix(".worldcrafter")
    negative = generation.get("negative_prompt")
    if negative is None:
        negative = (repo / "test" / "negative_prompt.txt").read_text(encoding="utf-8").strip()
    negative = str(negative)
    failures = 0
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        begin_sample(sample_id, index=index, total=len(rows), label="WorldCrafter")
        try:
            camera = prepare_camera(row, work / "camera")
            output = Path(row["output_path"]).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            pending = output.with_name(output.stem + ".pending.mp4")
            result = model.generate(
                mode="i2v",
                image_path=Path(row["conditioning_image"]),
                camera_path=camera,
                output_path=pending,
                prompt=str(row["prompt"]),
                negative_prompt=negative,
                num_chunks=int(row["num_chunks"]),
                num_inference_steps=int(generation.get("num_inference_steps", 6 if model_type == "fast" else 50)),
                guidance_scale=float(generation.get("guidance_scale", 1 if model_type == "fast" else 5)),
                seed=seed,
                fps=fps,
            )
            expected_output_frames = int(row.get("output_frame_count") or output_frame_count(model_type, int(row["num_frames"])))
            probe = verify_video(pending, expected_frames=expected_output_frames, fps=fps)
            pending.replace(output)
            summary = dict(result.summary)
            summary["output_path"] = str(output)
            summary["output_frame_count"] = expected_output_frames
            summary["achieved_duration_seconds"] = float(probe["duration_seconds"])
            if row.get("camera_control_recipe"):
                summary["camera_control_recipe"] = row["camera_control_recipe"]
            summary_path = output.with_suffix(".json")
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            Path(result.summary_path).unlink(missing_ok=True)
            print_status(sample_id, "generated", output_path=str(output),
                         pose_source=row.get("pose_source"), num_chunks=row["num_chunks"],
                         duration_seconds=probe["duration_seconds"],
                         summary_path=str(summary_path))
        except Exception as exc:
            failures += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
