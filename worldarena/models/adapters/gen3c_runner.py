from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from worldarena.common.checkpoints import (
    apply_checkpoint_env,
    hf_local_dir,
    resolve_checkpoint_path,
)

_T5_WEIGHT_FILES = (
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-00001-of-00002.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena GEN3C subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--num_video_frames", default=121, type=int)
    parser.add_argument("--height", default=704, type=int)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--fps", default=24, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--guidance", default=1.0, type=float)
    parser.add_argument("--num_steps", default=35, type=int)
    parser.add_argument("--trajectory", default="left", type=str)
    parser.add_argument("--camera_rotation", default="center_facing", type=str)
    parser.add_argument("--movement_distance", default=0.3, type=float)
    parser.add_argument("--filter_points_threshold", default=0.05, type=float)
    parser.add_argument("--negative_prompt", default=None, type=str)
    parser.add_argument("--prompt_upsampler_dir", default=None, type=str)
    parser.add_argument("--save_buffer", action="store_true")
    parser.add_argument("--foreground_masking", action="store_true")
    parser.add_argument("--offload_diffusion_transformer", action="store_true")
    parser.add_argument("--offload_tokenizer", action="store_true")
    parser.add_argument("--offload_text_encoder_model", action="store_true")
    parser.add_argument("--offload_prompt_upsampler", action="store_true")
    parser.add_argument("--offload_guardrail_models", action="store_true")
    parser.add_argument("--disable_prompt_encoder", action="store_true")
    return parser.parse_args()


def _checkpoint_layout(checkpoint_dir: Path) -> dict[str, Path]:
    resolved = checkpoint_dir.expanduser().resolve()
    gen3c_dir = resolved
    tokenizer_dir = hf_local_dir("nvidia/Cosmos-Tokenize1-CV8x8x8-720p", required=True)
    t5_dir = hf_local_dir("google-t5/t5-11b", required=True)

    missing: list[str] = []
    if not (gen3c_dir / "model.pt").is_file():
        missing.append("Gen3C-Cosmos-7B/model.pt")
    if not all((tokenizer_dir / name).exists() for name in ("encoder.jit", "decoder.jit", "mean_std.pt", "image_mean_std.pt")):
        missing.append(
            "Cosmos-Tokenize1-CV8x8x8-720p/{encoder.jit,decoder.jit,mean_std.pt,image_mean_std.pt}"
        )
    if not all((t5_dir / name).exists() for name in ("config.json", "spiece.model")) or not any(
        (t5_dir / name).exists() for name in _T5_WEIGHT_FILES
    ):
        missing.append("google-t5/t5-11b/{config.json,spiece.model,<model-weights>}")
    if missing:
        raise FileNotFoundError(
            "GEN3C checkpoint dependencies are incomplete. Missing: " + ", ".join(missing)
        )

    return {
        "gen3c_dir": gen3c_dir,
        "tokenizer_dir": tokenizer_dir,
        "t5_dir": t5_dir,
    }


def _prepare_checkpoint_root(target_root: Path, layout: dict[str, Path]) -> Path:
    checkpoints_root = target_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    (checkpoints_root / "Gen3C-Cosmos-7B").symlink_to(layout["gen3c_dir"], target_is_directory=True)
    (checkpoints_root / "Cosmos-Tokenize1-CV8x8x8-720p").symlink_to(
        layout["tokenizer_dir"],
        target_is_directory=True,
    )
    google_t5_root = checkpoints_root / "google-t5"
    google_t5_root.mkdir(parents=True, exist_ok=True)
    (google_t5_root / "t5-11b").symlink_to(layout["t5_dir"], target_is_directory=True)
    return checkpoints_root


def _link_repo_alias(workspace_root: Path, repo_id: str, target_path: Path) -> None:
    alias_path = workspace_root / repo_id
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def _build_env(repo_root: Path, workspace_root: Path) -> dict[str, str]:
    env = apply_checkpoint_env()
    pythonpaths = [str(workspace_root), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    return env


def _require_repo_layout(repo_root: Path) -> Path:
    script_path = repo_root / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_single_image.py"
    if not script_path.exists():
        raise FileNotFoundError(f"GEN3C entrypoint not found: {script_path}")
    return script_path.resolve()


def _inference_command(
    *,
    script_path: Path,
    checkpoints_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [sys.executable]
    if args.num_gpus > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(args.num_gpus),
            ]
        )
    command.extend(
        [
            str(script_path),
            "--checkpoint_dir",
            str(checkpoints_root),
            "--input_image_path",
            str(Path(args.conditioning_image).expanduser().resolve()),
            "--video_save_folder",
            str(output_dir),
            "--video_save_name",
            args.sample_name,
            "--prompt",
            args.prompt,
            "--num_gpus",
            str(args.num_gpus),
            "--num_video_frames",
            str(args.num_video_frames),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--fps",
            str(args.fps),
            "--seed",
            str(args.seed),
            "--guidance",
            str(args.guidance),
            "--num_steps",
            str(args.num_steps),
            "--trajectory",
            args.trajectory,
            "--camera_rotation",
            args.camera_rotation,
            "--movement_distance",
            str(args.movement_distance),
            "--filter_points_threshold",
            str(args.filter_points_threshold),
        ]
    )
    if args.negative_prompt:
        command.extend(["--negative_prompt", args.negative_prompt])
    if args.prompt_upsampler_dir:
        command.extend(["--prompt_upsampler_dir", args.prompt_upsampler_dir])
    for flag_name in (
        "save_buffer",
        "foreground_masking",
        "offload_diffusion_transformer",
        "offload_tokenizer",
        "offload_text_encoder_model",
        "offload_prompt_upsampler",
        "offload_guardrail_models",
        "disable_prompt_encoder",
    ):
        if getattr(args, flag_name):
            command.append(f"--{flag_name}")
    return command


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("GEN3C runner requires checkpoint_dir")
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = _require_repo_layout(repo_root)

    with tempfile.TemporaryDirectory(prefix="worldarena_gen3c_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        workspace_root = temp_dir / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)
        moge_dir = hf_local_dir("Ruicheng/moge-vitl", required=True)
        moge_checkpoint = moge_dir / "model.pt"
        if not moge_checkpoint.is_file():
            raise FileNotFoundError(f"GEN3C MoGe checkpoint not found: {moge_checkpoint}")
        checkpoints_root = _prepare_checkpoint_root(
            temp_dir,
            _checkpoint_layout(checkpoint_dir),
        )
        _link_repo_alias(
            workspace_root,
            "Ruicheng/moge-vitl",
            moge_checkpoint,
        )
        output_dir = temp_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = _inference_command(
            script_path=script_path,
            checkpoints_root=checkpoints_root,
            output_dir=output_dir,
            args=args,
        )
        subprocess.run(
            command,
            check=True,
            cwd=str(workspace_root),
            env=_build_env(repo_root, workspace_root),
        )

        generated_path = output_dir / f"{args.sample_name}.mp4"
        if not generated_path.exists():
            raise FileNotFoundError(f"GEN3C output was not written: {generated_path}")
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(generated_path), str(output_path))


if __name__ == "__main__":
    main()
