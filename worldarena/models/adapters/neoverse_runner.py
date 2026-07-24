"""Subprocess runner for NeoVerse inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from worldarena.benchmark.annotations import load_camera_matrices
from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Arena NeoVerse subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--annotation_path", default=None, type=str)
    parser.add_argument("--reference_path", required=True, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--suite", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument(
        "--input_mode",
        choices=("auto", "conditioning_image", "reference_video"),
        default="auto",
        type=str,
    )
    parser.add_argument(
        "--trajectory_source",
        choices=("auto", "annotation", "static", "predefined"),
        default="auto",
        type=str,
    )
    parser.add_argument("--trajectory", default="static", type=str)
    parser.add_argument("--traj_mode", choices=("relative", "global"), default="relative", type=str)
    parser.add_argument("--angle", default=None, type=float)
    parser.add_argument("--distance", default=None, type=float)
    parser.add_argument("--orbit_radius", default=None, type=float)
    parser.add_argument("--zoom_ratio", default=1.0, type=float)
    parser.add_argument("--num_frames", default=81, type=int)
    parser.add_argument("--height", default=336, type=int)
    parser.add_argument("--width", default=560, type=int)
    parser.add_argument("--resize_mode", choices=("center_crop", "resize"), default="center_crop")
    parser.add_argument("--alpha_threshold", default=1.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_subdir", default="NeoVerse", type=str)
    parser.add_argument("--reconstructor_filename", default="reconstructor.ckpt", type=str)
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--vis_rendering", action="store_true")
    return parser.parse_args()


def _resolved_input_mode(input_mode: str, suite: str) -> str:
    if input_mode != "auto":
        return input_mode
    return "conditioning_image" if suite == "video_static" else "reference_video"


def _resolved_trajectory_source(
    trajectory_source: str,
    *,
    input_mode: str,
    annotation_path: str | None,
) -> str:
    if trajectory_source != "auto":
        return trajectory_source
    if input_mode == "conditioning_image" and annotation_path:
        return "annotation"
    return "static"


def _annotation_trajectory_payload(
    *,
    annotation_path: str,
    num_frames: int,
    zoom_ratio: float,
) -> dict[str, object]:
    camera_c2w = load_camera_matrices(annotation_path, target_frames=num_frames)
    if camera_c2w is None:
        raise FileNotFoundError(f"poses.npy not found under annotation path: {annotation_path}")

    first_inverse = np.linalg.inv(camera_c2w[0].astype(np.float64)).astype(np.float32)
    relative_c2w = np.matmul(first_inverse[None, ...], camera_c2w.astype(np.float32))
    relative_c2w[0] = np.eye(4, dtype=np.float32)
    return {
        "name": "worldarena_annotation",
        "mode": "global",
        "num_frames": len(relative_c2w),
        "zoom_ratio": float(zoom_ratio),
        "use_first_frame": True,
        "trajectory": {
            "frame_indices": list(range(len(relative_c2w))),
            "frame_matrices": relative_c2w.tolist(),
        },
    }


def _prepare_model_root(
    checkpoint_dir: Path | None,
    workspace_root: Path,
    model_subdir: str,
) -> tuple[Path, Path]:
    if checkpoint_dir is None:
        raise ValueError("NeoVerse runner requires checkpoint_dir")
    checkpoint_root = checkpoint_dir.expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"NeoVerse checkpoint directory not found: {checkpoint_root}")
    if (checkpoint_root / model_subdir).is_dir():
        return checkpoint_root, (checkpoint_root / model_subdir).resolve()

    model_root = workspace_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    model_dir = model_root / model_subdir
    model_dir.symlink_to(checkpoint_root, target_is_directory=True)
    return model_root, model_dir


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("NeoVerse runner requires checkpoint_dir")
    reference_path = Path(args.reference_path).expanduser().resolve()
    conditioning_image = Path(args.conditioning_image).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_mode = _resolved_input_mode(args.input_mode, args.suite)
    input_path = conditioning_image if input_mode == "conditioning_image" else reference_path
    static_scene = input_mode == "conditioning_image"
    trajectory_source = _resolved_trajectory_source(
        args.trajectory_source,
        input_mode=input_mode,
        annotation_path=args.annotation_path,
    )

    if trajectory_source == "annotation" and not static_scene:
        raise ValueError(
            "NeoVerse annotation trajectories require conditioning_image input in WorldAtlas Arena; "
            "use input_mode=conditioning_image or trajectory_source=static."
        )

    inference_script = (repo_root / "inference.py").resolve()
    if not inference_script.exists():
        raise FileNotFoundError(f"NeoVerse inference script not found: {inference_script}")

    with tempfile.TemporaryDirectory(prefix="worldarena_neoverse_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        model_root, model_dir = _prepare_model_root(
            checkpoint_dir=checkpoint_dir,
            workspace_root=temp_dir,
            model_subdir=args.model_subdir,
        )
        reconstructor_path = (model_dir / args.reconstructor_filename).resolve()
        if not reconstructor_path.exists():
            raise FileNotFoundError(f"NeoVerse reconstructor checkpoint not found: {reconstructor_path}")

        generated_output = temp_dir / "outputs" / f"{args.sample_name}.mp4"
        generated_output.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(inference_script),
            "--input_path",
            str(input_path),
            "--output_path",
            str(generated_output),
            "--prompt",
            args.prompt,
            "--negative_prompt",
            args.negative_prompt,
            "--model_path",
            str(model_root),
            "--reconstructor_path",
            str(reconstructor_path),
            "--num_frames",
            str(args.num_frames),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--resize_mode",
            args.resize_mode,
            "--alpha_threshold",
            str(args.alpha_threshold),
            "--seed",
            str(args.seed),
        ]
        if static_scene:
            command.append("--static_scene")
        if args.disable_lora:
            command.append("--disable_lora")
        if args.low_vram:
            command.append("--low_vram")
        if args.vis_rendering:
            command.append("--vis_rendering")

        if trajectory_source == "annotation":
            if not args.annotation_path:
                raise FileNotFoundError("NeoVerse annotation trajectory requires --annotation_path")
            trajectory_path = temp_dir / "trajectory.json"
            trajectory_path.write_text(
                json.dumps(
                    _annotation_trajectory_payload(
                        annotation_path=args.annotation_path,
                        num_frames=args.num_frames,
                        zoom_ratio=args.zoom_ratio,
                    ),
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            command.extend(["--trajectory_file", str(trajectory_path)])
        else:
            trajectory_name = args.trajectory if trajectory_source == "predefined" else "static"
            command.extend(
                [
                    "--trajectory",
                    trajectory_name,
                    "--traj_mode",
                    args.traj_mode,
                    "--zoom_ratio",
                    str(args.zoom_ratio),
                ]
            )
            if args.angle is not None:
                command.extend(["--angle", str(args.angle)])
            if args.distance is not None:
                command.extend(["--distance", str(args.distance)])
            if args.orbit_radius is not None:
                command.extend(["--orbit_radius", str(args.orbit_radius)])

        env = apply_checkpoint_env()
        pythonpath = [str(repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        subprocess.run(
            command,
            check=True,
            cwd=str(repo_root),
            env=env,
        )

        if not generated_output.exists():
            raise FileNotFoundError(f"NeoVerse output video was not written: {generated_output}")
        shutil.copy2(generated_output, output_path)


if __name__ == "__main__":
    main()
