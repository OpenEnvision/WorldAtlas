"""Batch subprocess runner for Lyra-2 inference."""

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

from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path
from worldarena.models.adapters.lyra2_runner import (
    _configure_lyra2_runtime_env,
    _checkpoint_tree,
    _link_hf_repo_aliases,
    _prepare_transformer_engine_shim,
    _resolve_local_da3_model_name,
    _trajectory_payload,
    _use_transformer_engine_shim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Arena Lyra-2 batch subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--batch_spec_path", required=True, type=str)
    parser.add_argument("--experiment", default="lyra2", type=str)
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


def _load_batch_spec(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty Lyra-2 batch spec: {path}")
    return rows


def _stage_input_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("Lyra-2 batch runner requires checkpoint_dir")
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()
    rows = _load_batch_spec(batch_spec_path)
    output_dirs = {Path(row["output_path"]).expanduser().resolve().parent for row in rows}
    if len(output_dirs) != 1:
        raise ValueError("Lyra-2 batch runner requires all outputs to share one directory")
    output_dir = next(iter(output_dirs))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_tree = _checkpoint_tree(checkpoint_dir)
    da3_model_name = _resolve_local_da3_model_name(args.da3_model_name)
    da3_model_path_custom = None
    if args.da3_model_path_custom:
        da3_model_path_custom = resolve_checkpoint_path(
            args.da3_model_path_custom,
            kind="file",
            required=True,
        )
        if da3_model_path_custom is None:
            raise FileNotFoundError(
                f"Lyra-2 DA3 checkpoint path is missing: {args.da3_model_path_custom}"
            )

    with tempfile.TemporaryDirectory(prefix="worldarena_lyra2_batch_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        workspace_root = temp_dir / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)

        if _use_transformer_engine_shim():
            _prepare_transformer_engine_shim(workspace_root)
        _link_hf_repo_aliases(
            workspace_root,
            da3_model_name,
            use_moge_scale=args.use_moge_scale,
        )

        (workspace_root / "lyra_2").symlink_to(repo_root / "lyra_2", target_is_directory=True)
        (workspace_root / "checkpoints").symlink_to(checkpoint_tree, target_is_directory=True)

        input_root = temp_dir / "input"
        trajectory_root = temp_dir / "trajectories"
        prompt_root = temp_dir / "prompts"
        input_root.mkdir(parents=True, exist_ok=True)
        trajectory_root.mkdir(parents=True, exist_ok=True)
        prompt_root.mkdir(parents=True, exist_ok=True)

        for row in rows:
            stem = str(row["prediction_stem"])
            conditioning_source = Path(row["conditioning_image"]).expanduser().resolve()
            suffix = conditioning_source.suffix or ".png"
            _stage_input_image(conditioning_source, input_root / f"{stem}{suffix}")
            np.savez_compressed(
                trajectory_root / f"{stem}.npz",
                **_trajectory_payload(
                    annotation_path=str(Path(row["annotation_path"]).expanduser().resolve()),
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                ),
            )
            (prompt_root / f"{stem}.txt").write_text(str(row["prompt"]), encoding="utf-8")

        command = [
            sys.executable,
            "-m",
            "lyra_2._src.inference.lyra2_custom_traj_inference",
            "--input_image_path",
            str(input_root),
            "--trajectory_path",
            str(trajectory_root),
            "--checkpoint_dir",
            "checkpoints/model",
            "--experiment",
            args.experiment,
            "--output_path",
            str(output_dir),
            "--prompt_dir",
            str(prompt_root),
            "--num_samples",
            str(len(rows)),
            "--sample_start_idx",
            "0",
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
            da3_model_name,
            "--da3_frame_interval",
            str(args.da3_frame_interval),
            "--da3_max_history_frames",
            str(args.da3_max_history_frames),
        ]
        if args.use_moge_scale:
            command.append("--use_moge_scale")
        else:
            command.append("--no-use_moge_scale")
        if da3_model_path_custom is not None:
            command.extend(["--da3_model_path_custom", str(da3_model_path_custom)])
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

        env = apply_checkpoint_env()
        env["TRANSFORMERS_CACHE"] = env["HF_HUB_CACHE"]
        _configure_lyra2_runtime_env(env, sys.executable)
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


if __name__ == "__main__":
    main()
