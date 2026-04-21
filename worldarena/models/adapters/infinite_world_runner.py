from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infinite-World single-sample runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--actions_json", required=True, type=str)
    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--config_path", default="configs/infworld_config.yaml", type=str)
    parser.add_argument("--bucket_config_name", default="ASPECT_RATIO_627_F64", type=str)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_chunks", default=1, type=int)
    parser.add_argument("--num_sampling_steps", default=30, type=int)
    parser.add_argument("--sample_shift", default=None, type=float)
    parser.add_argument("--sample_guide_scale", default=None, type=float)
    parser.add_argument("--quality", default=8, type=int)
    parser.add_argument("--negative_prompt", default=None, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    return parser.parse_args()


def _load_torch_state_dict(path: str, *, map_location: str):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_path(path_value: str | None, *, base_dir: Path) -> str | None:
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _setup_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _set_single_process_context_parallel(cp_util_module) -> None:
    cp_util_module.dp_rank = 0
    cp_util_module.dp_size = 1
    cp_util_module.cp_rank = 0
    cp_util_module.cp_size = 1
    cp_util_module.dp_group = None
    cp_util_module.cp_group = None


def _resize_and_center_crop(frame: np.ndarray, target_size: tuple[int, int]):
    import torch
    from torchvision.transforms import functional as tf

    target_h, target_w = target_size
    orig_h, orig_w = frame.shape[:2]
    scale = max(target_h / orig_h, target_w / orig_w)
    resized_h = int(math.ceil(scale * orig_h))
    resized_w = int(math.ceil(scale * orig_w))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1)
    cropped = tf.center_crop(tensor, [target_h, target_w])
    return cropped.permute(1, 2, 0).contiguous().numpy()


def _load_condition_video(image_path: Path, bucket_config) -> Any:
    import torch

    if image_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        cap = cv2.VideoCapture(str(image_path))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(f"Failed to decode condition video: {image_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Failed to read condition image: {image_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    aspect_ratio = frame.shape[0] / frame.shape[1]
    closest_bucket = min(bucket_config, key=lambda key: abs(float(key) - aspect_ratio))
    target_h, target_w = bucket_config[closest_bucket][0]
    cropped = _resize_and_center_crop(frame, (int(target_h), int(target_w)))
    tensor = torch.from_numpy(cropped).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0
    return tensor.unsqueeze(0).unsqueeze(2)


def _load_action_sequence(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame_actions = payload.get("frame_actions", payload) if isinstance(payload, dict) else payload
    move_action_map = {
        "no-op": 0,
        "go forward": 1,
        "go back": 2,
        "go left": 3,
        "go right": 4,
        "go forward and go left": 5,
        "go forward and go right": 6,
        "go back and go left": 7,
        "go back and go right": 8,
        "uncertain": 9,
    }
    view_action_map = {
        "no-op": 0,
        "turn up": 1,
        "turn down": 2,
        "turn left": 3,
        "turn right": 4,
        "turn up and turn left": 5,
        "turn up and turn right": 6,
        "turn down and turn left": 7,
        "turn down and turn right": 8,
        "uncertain": 9,
    }
    move_ids = [move_action_map[str(item["move"])] for item in frame_actions]
    view_ids = [view_action_map[str(item["view"])] for item in frame_actions]
    return move_ids, view_ids


def _write_video(output_path: Path, video_tensor, *, fps: int, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = video_tensor.detach().cpu().float()[0]
    frames = tensor.permute(1, 2, 3, 0)
    frames = ((frames + 1.0) / 2.0).clamp(0.0, 1.0)
    frames = (frames.numpy() * 255.0).round().clip(0.0, 255.0).astype(np.uint8)
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=quality)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = Path(args.ckpt_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Infinite-World requires a CUDA-enabled runtime.")

    device = torch.device(args.device)
    torch.cuda.set_device(device.index or 0)
    _setup_seed(args.seed)

    import infworld.context_parallel.context_parallel_util as cp_util
    from infworld.configs import bucket_config as bucket_config_module
    from infworld.utils.prepare_dataloader import get_obj_from_str

    _set_single_process_context_parallel(cp_util)

    config_path = Path(_resolve_path(args.config_path, base_dir=repo_root) or "")
    cfg = OmegaConf.load(config_path)
    cfg.checkpoint_path = _resolve_path(cfg.get("checkpoint_path"), base_dir=checkpoint_dir)
    if "vae_cfg" in cfg and "vae_pth" in cfg.vae_cfg:
        cfg.vae_cfg.vae_pth = _resolve_path(cfg.vae_cfg.vae_pth, base_dir=checkpoint_dir)
    if "text_encoder_cfg" in cfg:
        if "checkpoint_path" in cfg.text_encoder_cfg:
            cfg.text_encoder_cfg.checkpoint_path = _resolve_path(
                cfg.text_encoder_cfg.checkpoint_path,
                base_dir=checkpoint_dir,
            )
        if "tokenizer_path" in cfg.text_encoder_cfg:
            cfg.text_encoder_cfg.tokenizer_path = _resolve_path(
                cfg.text_encoder_cfg.tokenizer_path,
                base_dir=checkpoint_dir,
            )

    vae = get_obj_from_str(cfg.vae_target)(**cfg.vae_cfg).to(device)
    text_encoder = get_obj_from_str(cfg.text_encoder_target)(device=device.index or 0, **cfg.text_encoder_cfg)
    text_encoder.t5.model.to(device)

    scheduler = get_obj_from_str(cfg.scheduler_target)(**cfg.val_scheduler_cfg)
    scheduler.num_sampling_steps = int(args.num_sampling_steps)
    scheduler.shift = float(
        args.sample_shift if args.sample_shift is not None else cfg.val_scheduler_cfg.get("shift", 7.0)
    )

    model_dtype = getattr(torch, cfg.get("amp_dtype", "bfloat16"))
    dit = get_obj_from_str(cfg.model_target)(
        out_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_context_parallel=False,
        **cfg.model_cfg,
    )
    dit = dit.to(dtype=model_dtype)
    dit.eval()

    state_dict = _load_torch_state_dict(str(cfg.checkpoint_path), map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    dit.load_state_dict(state_dict, strict=False)
    dit = dit.to(device)

    bucket_config = getattr(bucket_config_module, args.bucket_config_name)
    validation_num_frames = int(cfg.validation_data.get("num_frames", 81))
    guidance_scale = float(
        args.sample_guide_scale
        if args.sample_guide_scale is not None
        else cfg.val_scheduler_cfg.get("text_cfg_scale", 5.0)
    )
    negative_prompt = (
        args.negative_prompt
        or "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, "
        "image, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, "
        "extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, "
        "motionless image, cluttered background, three legs, crowded background, walking backwards."
    )

    condition_video = _load_condition_video(Path(args.image_path), bucket_config).to(device)
    move_ids, view_ids = _load_action_sequence(Path(args.actions_json))
    move_ids = torch.tensor(move_ids, dtype=torch.long)
    view_ids = torch.tensor(view_ids, dtype=torch.long)
    video_buffer = condition_video.detach().cpu().float()

    for _ in range(max(int(args.num_chunks), 1)):
        current_cond = video_buffer.to(device, dtype=torch.float32)
        current_latent = vae.encode(current_cond)

        latent_size = list(current_latent.shape)
        latent_size[2] = 21
        curr_start = int(video_buffer.shape[2] - 1)
        curr_end = curr_start + validation_num_frames
        move = move_ids[curr_start:curr_end]
        view = view_ids[curr_start:curr_end]
        if move.shape[0] < validation_num_frames:
            pad_len = validation_num_frames - move.shape[0]
            move = torch.cat([move, torch.zeros(pad_len, dtype=torch.long)], dim=0)
            view = torch.cat([view, torch.zeros(pad_len, dtype=torch.long)], dim=0)

        additional_args = {
            "image_cond": current_latent,
            "move": move.unsqueeze(0).to(device),
            "view": view.unsqueeze(0).to(device),
        }
        with torch.no_grad():
            samples = scheduler.sample(
                model=dit,
                text_encoder=text_encoder,
                null_embedder=dit.y_embedder,
                z_size=torch.Size(latent_size),
                prompts=[args.prompt],
                guidance_scale=guidance_scale,
                negative_prompts=[negative_prompt],
                device=device,
                additional_args=additional_args,
            )
            decoded_chunk = vae.decode(samples).detach().cpu()
            video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
        torch.cuda.empty_cache()

    _write_video(output_path, video_buffer, fps=args.fps, quality=args.quality)


if __name__ == "__main__":
    main()
