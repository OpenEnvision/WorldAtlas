from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from worldarena.benchmark.annotations import load_camera_matrices, load_intrinsics_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Lyra-2 subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--annotation_path", required=True, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=832, type=int)
    parser.add_argument("--num_frames", default=81, type=int)
    parser.add_argument("--guidance", default=5.0, type=float)
    parser.add_argument("--shift", default=5.0, type=float)
    parser.add_argument("--num_sampling_step", default=35, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--fps", default=16, type=int)
    parser.add_argument("--pose_scale", default=1.0, type=float)
    parser.add_argument("--context_parallel_size", default=1, type=int)
    parser.add_argument("--prompt_suffix", default="", type=str)
    parser.add_argument("--use_moge_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--da3_model_name", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1", type=str)
    parser.add_argument("--da3_model_path_custom", default=None, type=str)
    parser.add_argument("--da3_frame_interval", default=8, type=int)
    parser.add_argument("--da3_max_history_frames", default=10, type=int)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--offload_when_prompt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--da3_include_ar_chunk_last_frames", action="store_true")
    parser.add_argument("--da3_use_predicted_pose", action="store_true")
    parser.add_argument("--da3_predicted_pose_continuation", action="store_true")
    parser.add_argument("--ablate_same_t5", action="store_true")
    parser.add_argument("--use_dmd_scheduler", action="store_true")
    parser.add_argument("--disable_cache_update", action="store_true")
    parser.add_argument("--offload_da3_diffusion", action="store_true")
    return parser.parse_args()


def _intrinsics_matrix(values: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(values, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        return intrinsics.astype(np.float32)
    if intrinsics.shape != (4,):
        raise ValueError(f"unsupported Lyra-2 intrinsics shape: {intrinsics.shape}")
    fx, fy, cx, cy = [float(value) for value in intrinsics]
    return np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _trajectory_payload(
    *,
    annotation_path: str,
    num_frames: int,
    height: int,
    width: int,
) -> dict[str, np.ndarray | int]:
    camera_c2w = load_camera_matrices(annotation_path, target_frames=num_frames)
    if camera_c2w is None:
        raise FileNotFoundError(f"poses.npy not found under annotation path: {annotation_path}")

    intrinsics = load_intrinsics_sequence(annotation_path, target_frames=len(camera_c2w))
    if intrinsics is None:
        raise FileNotFoundError(f"intrinsics.npy not found under annotation path: {annotation_path}")

    camera_w2c = np.linalg.inv(camera_c2w.astype(np.float64)).astype(np.float32)
    intrinsics_33 = np.stack(
        [_intrinsics_matrix(entry) for entry in np.asarray(intrinsics, dtype=np.float32)],
        axis=0,
    ).astype(np.float32)
    return {
        "w2c": camera_w2c,
        "intrinsics": intrinsics_33,
        "image_height": int(height),
        "image_width": int(width),
    }


def _checkpoint_tree(checkpoint_dir: Path) -> Path:
    resolved = checkpoint_dir.expanduser().resolve()
    if (resolved / "checkpoints").is_dir():
        return (resolved / "checkpoints").resolve()
    required_dirs = ("image_encoder", "lora", "model", "recon", "text_encoder", "vae")
    if all((resolved / name).exists() for name in required_dirs):
        return resolved
    raise FileNotFoundError(
        "Lyra-2 checkpoint tree not found. Expected either a base directory with "
        f"`checkpoints/` under it or a direct checkpoint tree at: {resolved}"
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_tree = _checkpoint_tree(checkpoint_dir)

    with tempfile.TemporaryDirectory(prefix="worldarena_lyra2_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        workspace_root = temp_dir / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)

        (workspace_root / "lyra_2").symlink_to(repo_root / "lyra_2", target_is_directory=True)
        (workspace_root / "checkpoints").symlink_to(checkpoint_tree, target_is_directory=True)

        input_root = temp_dir / "input"
        input_root.mkdir(parents=True, exist_ok=True)
        conditioning_source = Path(args.conditioning_image).expanduser().resolve()
        conditioning_copy = input_root / f"{args.sample_name}{conditioning_source.suffix or '.png'}"
        shutil.copy2(conditioning_source, conditioning_copy)

        trajectory_path = temp_dir / "trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            **_trajectory_payload(
                annotation_path=args.annotation_path,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
            ),
        )

        output_root = temp_dir / "outputs"
        command = [
            sys.executable,
            "-m",
            "lyra_2._src.inference.lyra2_custom_traj_inference",
            "--input_image_path",
            str(conditioning_copy),
            "--trajectory_path",
            str(trajectory_path),
            "--checkpoint_dir",
            "checkpoints/model",
            "--output_path",
            str(output_root),
            "--prompt",
            args.prompt,
            "--guidance",
            str(args.guidance),
            "--shift",
            str(args.shift),
            "--num_sampling_step",
            str(args.num_sampling_step),
            "--seed",
            str(args.seed),
            "--fps",
            str(args.fps),
            "--num_frames",
            str(args.num_frames),
            "--pose_scale",
            str(args.pose_scale),
            "--resolution",
            f"{args.height},{args.width}",
            "--context_parallel_size",
            str(args.context_parallel_size),
            "--prompt_suffix",
            args.prompt_suffix,
            "--da3_model_name",
            args.da3_model_name,
            "--da3_frame_interval",
            str(args.da3_frame_interval),
            "--da3_max_history_frames",
            str(args.da3_max_history_frames),
        ]
        if args.use_moge_scale:
            command.append("--use_moge_scale")
        else:
            command.append("--no-use_moge_scale")
        if args.da3_model_path_custom:
            command.extend(["--da3_model_path_custom", args.da3_model_path_custom])
        for flag_name in (
            "offload",
            "offload_when_prompt",
            "debug",
            "da3_include_ar_chunk_last_frames",
            "da3_use_predicted_pose",
            "da3_predicted_pose_continuation",
            "ablate_same_t5",
            "use_dmd_scheduler",
            "disable_cache_update",
            "offload_da3_diffusion",
        ):
            if getattr(args, flag_name):
                command.append(f"--{flag_name}")

        env = os.environ.copy()
        pythonpath = [str(workspace_root), str(repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        subprocess.run(
            command,
            check=True,
            cwd=str(workspace_root),
            env=env,
        )

        generated_video = output_root / f"{args.sample_name}.mp4"
        if not generated_video.exists():
            raise FileNotFoundError(f"Lyra-2 output video was not written: {generated_video}")
        shutil.copy2(generated_video, output_path)


if __name__ == "__main__":
    main()
