"""Subprocess runner for Matrix-Game 2 inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from worldatlas.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path
from worldatlas.models.adapters.windowed_rollout import align_matrix_game_2_latent_frames


_DEFAULT_CONFIG_BY_MODE = {
    "universal": "configs/inference_yaml/inference_universal.yaml",
    "gta_drive": "configs/inference_yaml/inference_gta_drive.yaml",
    "templerun": "configs/inference_yaml/inference_templerun.yaml",
}
_DEFAULT_CHECKPOINT_BY_MODE = {
    "universal": "base_distilled_model/base_distill.safetensors",
    "gta_drive": "gta_distilled_model/gta_keyboard2dim.safetensors",
    "templerun": "templerun_distilled_model/templerun_7dim_onlykey.safetensors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Matrix-Game-2 runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--actions_json", required=True, type=str)
    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--mode", default="universal", choices=sorted(_DEFAULT_CONFIG_BY_MODE), type=str)
    parser.add_argument("--config_path", default=None, type=str)
    parser.add_argument("--checkpoint_filename", default=None, type=str)
    parser.add_argument("--num_output_frames", default=15, type=int)
    parser.add_argument("--save_num_frames", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_fps", default=12, type=int)
    parser.add_argument("--compile_vae", action="store_true")
    return parser.parse_args()


def _resolve_under(base_dir: Path, value: str | None, default_relative: str) -> Path:
    raw = value or default_relative
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def resolve_num_output_frames(action_spec: dict, cli_value: int) -> int:
    """Use the planned latent count, snapped to the official 3-latent block."""
    planned = action_spec.get("num_output_frames")
    raw = int(cli_value) if planned is None else int(planned)
    return align_matrix_game_2_latent_frames(raw)


def pad_or_trim_action_condition(condition: np.ndarray, expected_frames: int) -> np.ndarray:
    """Match a keyboard/mouse sequence to the snapped latent → pixel length."""
    if expected_frames < 1:
        raise ValueError(f"expected_frames must be >= 1, got {expected_frames}")
    if condition.shape[0] == expected_frames:
        return condition
    if condition.shape[0] > expected_frames:
        return condition[:expected_frames]
    pad = np.repeat(condition[-1:], expected_frames - condition.shape[0], axis=0)
    return np.concatenate([condition, pad], axis=0)


def as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def settings_from_generation(generation: dict) -> argparse.Namespace:
    """CLI-equivalent settings from the WorldAtlas generation YAML."""
    return argparse.Namespace(
        mode=str(generation.get("mode") or "universal"),
        config_path=generation.get("config_path"),
        checkpoint_filename=generation.get("checkpoint_filename"),
        num_output_frames=int(generation.get("num_output_frames", 15)),
        save_num_frames=int(generation.get("save_num_frames") or 0),
        seed=int(generation.get("seed", 42)),
        output_fps=int(generation.get("output_fps", 12)),
        compile_vae=as_bool(generation.get("compile_vae"), False),
    )


def _resizecrop(image, target_height: int, target_width: int):
    width, height = image.size
    if height / width > target_height / target_width:
        new_width = int(width)
        new_height = int(new_width * target_height / target_width)
    else:
        new_height = int(height)
        new_width = int(new_height * target_width / target_height)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom))


def prepare_matrix_game_2_repo(repo_root: Path) -> Path:
    os.environ.update(apply_checkpoint_env())
    repo_root = repo_root.expanduser().resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def load_matrix_game_2(
    repo_root: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
):
    """Load DiT / VAE / CLIP once. Autotune compile is opt-in and usually slower."""
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from torchvision.transforms import v2

    from demo_utils.vae_block3 import VAEDecoderWrapper
    from pipeline import CausalInferencePipeline
    from utils.misc import set_seed
    from utils.wan_wrapper import WanDiffusionWrapper
    from wan.vae.wanx_vae import get_wanx_vae_wrapper

    if not torch.cuda.is_available():
        raise RuntimeError("Matrix-Game-2 requires CUDA for inference.")

    config_path = _resolve_under(repo_root, args.config_path, _DEFAULT_CONFIG_BY_MODE[args.mode])
    checkpoint_path = _resolve_under(
        checkpoint_dir,
        args.checkpoint_filename,
        _DEFAULT_CHECKPOINT_BY_MODE[args.mode],
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Matrix-Game-2 checkpoint not found: {checkpoint_path}")

    set_seed(args.seed)
    device = torch.device("cuda")
    weight_dtype = torch.bfloat16
    config = OmegaConf.load(config_path)
    generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)

    current_vae_decoder = VAEDecoderWrapper()
    vae_state_dict = torch.load(checkpoint_dir / "Wan2.1_VAE.pth", map_location="cpu")
    decoder_state_dict = {
        key: value
        for key, value in vae_state_dict.items()
        if "decoder." in key or "conv2" in key
    }
    current_vae_decoder.load_state_dict(decoder_state_dict)
    current_vae_decoder.to(device, torch.float16)
    current_vae_decoder.requires_grad_(False)
    current_vae_decoder.eval()
    if args.compile_vae:
        current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

    pipeline = CausalInferencePipeline(config, generator=generator, vae_decoder=current_vae_decoder)
    pipeline.generator.load_state_dict(load_file(checkpoint_path))
    pipeline = pipeline.to(device=device, dtype=weight_dtype)
    pipeline.vae_decoder.to(torch.float16)

    vae = get_wanx_vae_wrapper(str(checkpoint_dir), torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device, weight_dtype)
    frame_process = v2.Compose(
        [
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return {
        "pipeline": pipeline,
        "vae": vae,
        "frame_process": frame_process,
        "args": args,
        "device": device,
        "weight_dtype": weight_dtype,
    }


def generate_matrix_game_2(
    runtime: dict,
    *,
    action_spec: dict,
    image_path: Path,
    output_path: Path,
) -> None:
    """Run one causal rollout on a resident Matrix-Game-2 pipeline."""
    import imageio
    import torch
    from diffusers.utils import load_image
    from einops import rearrange

    args = runtime["args"]
    pipeline = runtime["pipeline"]
    vae = runtime["vae"]
    frame_process = runtime["frame_process"]
    device = runtime["device"]
    weight_dtype = runtime["weight_dtype"]

    num_output_frames = resolve_num_output_frames(action_spec, args.num_output_frames)
    expected_frames = (num_output_frames - 1) * 4 + 1
    keyboard_condition = pad_or_trim_action_condition(
        np.asarray(action_spec["keyboard_condition"], dtype=np.float32),
        expected_frames,
    )
    mouse_condition: np.ndarray | None = None
    if args.mode != "templerun":
        if "mouse_condition" not in action_spec:
            raise ValueError(f"Matrix-Game-2 mode={args.mode} requires mouse_condition")
        mouse_condition = pad_or_trim_action_condition(
            np.asarray(action_spec["mouse_condition"], dtype=np.float32),
            expected_frames,
        )

    image = load_image(str(image_path))
    image = _resizecrop(image, 352, 640)
    image = frame_process(image)[None, :, None, :, :].to(dtype=weight_dtype, device=device)
    padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (num_output_frames - 1), 1, 1)
    img_cond = torch.concat([image, padding_video], dim=2)
    tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
    img_cond = vae.encode(img_cond, device=device, **tiler_kwargs).to(device)
    mask_cond = torch.ones_like(img_cond)
    mask_cond[:, :, 1:] = 0
    cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
    visual_context = vae.clip.encode_video(image)
    sampled_noise = torch.randn(
        [1, 16, num_output_frames, 44, 80],
        device=device,
        dtype=weight_dtype,
    )
    conditional_dict = {
        "cond_concat": cond_concat.to(device=device, dtype=weight_dtype),
        "visual_context": visual_context.to(device=device, dtype=weight_dtype),
        "keyboard_cond": torch.tensor(keyboard_condition, device=device, dtype=weight_dtype).unsqueeze(0),
    }
    if mouse_condition is not None:
        conditional_dict["mouse_cond"] = torch.tensor(
            mouse_condition,
            device=device,
            dtype=weight_dtype,
        ).unsqueeze(0)

    with torch.no_grad():
        videos = pipeline.inference(
            noise=sampled_noise,
            conditional_dict=conditional_dict,
            return_latents=False,
            mode=args.mode,
            profile=False,
        )

    videos_tensor = torch.cat(videos, dim=1)
    frames = rearrange(videos_tensor, "b t c h w -> b t h w c")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
    if args.save_num_frames:
        if args.save_num_frames > frames.shape[0]:
            raise ValueError(
                f"save_num_frames={args.save_num_frames} exceeds generated frame count {frames.shape[0]}"
            )
        frames = frames[: args.save_num_frames]
    frames = np.ascontiguousarray(frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=args.output_fps)
    # Save the actual inference inputs, including block rounding. Do not change
    # the already-executed program by rebuilding it with a new token duration.
    executed_actions = {
        **action_spec, "variant": "matrix_game_2", "mode": args.mode,
        "num_output_frames": num_output_frames, "total_frames": expected_frames,
        "keyboard_condition": keyboard_condition.tolist(),
        "saved_video_frames": int(frames.shape[0]),
    }
    if mouse_condition is not None:
        executed_actions["mouse_condition"] = mouse_condition.tolist()
    sidecar = output_path.with_name(f"{output_path.stem}_actions.json")
    sidecar.write_text(json.dumps(executed_actions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    launch_cwd = Path.cwd()
    repo_root = prepare_matrix_game_2_repo((launch_cwd / Path(args.repo_root).expanduser()).resolve())
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("Matrix-Game-2 runner requires checkpoint_dir")
    actions_path = (launch_cwd / Path(args.actions_json).expanduser()).resolve()
    image_path = (launch_cwd / Path(args.image_path).expanduser()).resolve()
    output_path = (launch_cwd / Path(args.output_path).expanduser()).resolve()
    with actions_path.open("r", encoding="utf-8") as file:
        action_spec = json.load(file)
    runtime = load_matrix_game_2(repo_root, checkpoint_dir, args)
    generate_matrix_game_2(
        runtime,
        action_spec=action_spec,
        image_path=image_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
