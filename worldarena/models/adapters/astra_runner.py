from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from worldarena.common.checkpoints import apply_checkpoint_env, hf_local_dir, resolve_checkpoint_path


WAN_REQUIRED_FILES = (
    "diffusion_pytorch_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Astra subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--cam_type", default=4, type=int)
    parser.add_argument("--wan_model_path", default=None, type=str)
    parser.add_argument("--dit_path", default=None, type=str)
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--initial_condition_frames", default=1, type=int)
    parser.add_argument("--frames_per_generation", default=8, type=int)
    parser.add_argument("--total_frames_to_generate", default=24, type=int)
    parser.add_argument("--max_history_frames", default=100, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--modality_type", default="sekai", choices=("sekai", "nuscenes", "openx"), type=str)
    parser.add_argument("--scene_info_path", default=None, type=str)
    parser.add_argument("--use_camera_cfg", default=None, type=str)
    parser.add_argument("--use_real_poses", default=None, type=str)
    parser.add_argument("--camera_guidance_scale", default=2.0, type=float)
    parser.add_argument("--text_guidance_scale", default=1.0, type=float)
    parser.add_argument("--moe_num_experts", default=3, type=int)
    parser.add_argument("--moe_top_k", default=1, type=int)
    parser.add_argument("--moe_hidden_dim", default=None, type=int)
    parser.add_argument("--use_gt_prompt", action="store_true")
    parser.add_argument("--add_icons", action="store_true")
    parser.add_argument("--gpu_index", default=0, type=int)
    parser.add_argument("--cuda_visible_devices", default=None, type=str)
    return parser.parse_args()


def _resolve_dit_path(checkpoint_dir: Path, explicit: str | None) -> Path:
    if explicit:
        resolved = resolve_checkpoint_path(explicit, kind="file", required=True)
        if resolved is None:
            raise FileNotFoundError(f"Astra DiT checkpoint is missing: {explicit}")
        return resolved
    candidate = checkpoint_dir / "models" / "Astra" / "checkpoints" / "diffusion_pytorch_model.ckpt"
    if not candidate.is_file():
        raise FileNotFoundError(f"Astra DiT checkpoint not found: {candidate}")
    return candidate.resolve()


def _resolve_wan_model_path(explicit: str | None) -> Path:
    if explicit:
        resolved = resolve_checkpoint_path(explicit, kind="dir", required=True)
        if resolved is None:
            raise FileNotFoundError(f"Astra Wan2.1 model path is missing: {explicit}")
        candidate = resolved
    else:
        candidate = hf_local_dir("Wan-AI/Wan2.1-T2V-1.3B", required=True)
    missing = [name for name in WAN_REQUIRED_FILES if not (candidate / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Astra Wan2.1 model files are incomplete under {candidate}. Missing: {missing}"
        )
    return candidate.resolve()


def _build_env(repo_root: Path, *, gpu_index: int, cuda_visible_devices: str | None) -> dict[str, str]:
    env = apply_checkpoint_env()
    pythonpaths = [str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    elif gpu_index >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise FileNotFoundError(f"Astra checkpoint directory is missing: {args.checkpoint_dir}")
    entrypoint = (repo_root / "scripts" / "infer_demo.py").resolve()
    if not entrypoint.exists():
        raise FileNotFoundError(f"Astra entrypoint not found: {entrypoint}")

    conditioning_image = Path(args.conditioning_image).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if not conditioning_image.exists():
        raise FileNotFoundError(f"Astra conditioning image not found: {conditioning_image}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dit_path = _resolve_dit_path(checkpoint_dir, args.dit_path)
    wan_model_path = _resolve_wan_model_path(args.wan_model_path)
    env = _build_env(
        repo_root,
        gpu_index=int(args.gpu_index),
        cuda_visible_devices=args.cuda_visible_devices,
    )

    command = [
        sys.executable,
        str(entrypoint),
        "--dit_path",
        str(dit_path),
        "--wan_model_path",
        str(wan_model_path),
        "--condition_image",
        str(conditioning_image),
        "--output_path",
        str(output_path),
        "--prompt",
        str(args.prompt),
        "--cam_type",
        str(int(args.cam_type)),
        "--start_frame",
        str(int(args.start_frame)),
        "--initial_condition_frames",
        str(int(args.initial_condition_frames)),
        "--frames_per_generation",
        str(int(args.frames_per_generation)),
        "--total_frames_to_generate",
        str(int(args.total_frames_to_generate)),
        "--max_history_frames",
        str(int(args.max_history_frames)),
        "--device",
        str(args.device),
        "--modality_type",
        str(args.modality_type),
        "--camera_guidance_scale",
        str(float(args.camera_guidance_scale)),
        "--text_guidance_scale",
        str(float(args.text_guidance_scale)),
        "--moe_num_experts",
        str(int(args.moe_num_experts)),
        "--moe_top_k",
        str(int(args.moe_top_k)),
    ]
    if args.scene_info_path:
        command.extend(["--scene_info_path", str(Path(args.scene_info_path).expanduser().resolve())])
    if args.use_camera_cfg is not None:
        command.extend(["--use_camera_cfg", str(args.use_camera_cfg)])
    if args.use_real_poses is not None:
        command.extend(["--use_real_poses", str(args.use_real_poses)])
    if args.moe_hidden_dim is not None:
        command.extend(["--moe_hidden_dim", str(int(args.moe_hidden_dim))])
    if args.use_gt_prompt:
        command.append("--use_gt_prompt")
    if args.add_icons:
        command.append("--add_icons")

    subprocess.run(
        command,
        check=True,
        cwd=str(repo_root),
        env=env,
    )
    if not output_path.exists():
        raise FileNotFoundError(f"Astra output video was not written: {output_path}")


if __name__ == "__main__":
    main()
