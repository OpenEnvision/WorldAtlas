from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Matrix-Game-1 runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--actions_json", required=True, type=str)
    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--video_length", default=65, type=int)
    parser.add_argument("--guidance_scale", default=6.0, type=float)
    parser.add_argument("--inference_steps", default=50, type=int)
    parser.add_argument("--shift", default=15.0, type=float)
    parser.add_argument("--num_pre_frames", default=5, type=int)
    parser.add_argument("--num_steps", default=50, type=int)
    parser.add_argument("--rel_l1_thresh", default=0.075, type=float)
    parser.add_argument("--resolution", nargs=2, default=[1280, 720], type=int)
    parser.add_argument("--fps", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bfloat16", action="store_true")
    return parser.parse_args()


def _resize_and_crop_image(image, target_size: tuple[int, int]):
    width, height = image.size
    target_width, target_height = target_size

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


def main() -> None:
    args = parse_args()
    launch_cwd = Path.cwd()
    repo_root = (launch_cwd / Path(args.repo_root).expanduser()).resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("Matrix-Game-1 runner requires checkpoint_dir")
    actions_path = (launch_cwd / Path(args.actions_json).expanduser()).resolve()
    image_path = (launch_cwd / Path(args.image_path).expanduser()).resolve()
    output_path = (launch_cwd / Path(args.output_path).expanduser()).resolve()

    os.environ.update(apply_checkpoint_env())
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import imageio
    import torch
    from diffusers.video_processor import VideoProcessor
    from einops import rearrange
    from PIL import Image

    from matrixgame.encoder_variants import get_text_enc
    from matrixgame.model_variants.matrixgame_dit_src import MGVideoDiffusionTransformerI2V
    from matrixgame.sample.flow_matching_scheduler_matrixgame import FlowMatchDiscreteScheduler
    from matrixgame.sample.pipeline_matrixgame import MatrixGameVideoPipeline
    from matrixgame.vae_variants import get_vae
    from teacache_forward import teacache_forward

    if not torch.cuda.is_available():
        raise RuntimeError("Matrix-Game-1 requires CUDA for inference.")

    with actions_path.open("r", encoding="utf-8") as file:
        action_spec = json.load(file)

    keyboard_condition = np.asarray(action_spec["keyboard_condition"], dtype=np.float32)
    mouse_condition = np.asarray(action_spec["mouse_condition"], dtype=np.float32)
    if keyboard_condition.shape[0] != args.video_length or mouse_condition.shape[0] != args.video_length:
        raise ValueError(
            "Matrix-Game-1 action sequence length mismatch: "
            f"video_length={args.video_length}, keyboard={keyboard_condition.shape[0]}, "
            f"mouse={mouse_condition.shape[0]}"
        )

    device = torch.device("cuda")
    weight_dtype = torch.bfloat16 if args.bfloat16 else torch.float32
    condition_dtype = torch.bfloat16 if args.bfloat16 else torch.float16

    scheduler = FlowMatchDiscreteScheduler(
        shift=args.shift,
        reverse=True,
        solver="euler",
    )

    vae = get_vae("matrixgame", str(checkpoint_dir / "vae"), torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    vae.enable_tiling()

    dit = MGVideoDiffusionTransformerI2V.from_pretrained(str(checkpoint_dir / "dit"))
    dit.requires_grad_(False)
    dit.eval()

    text_enc = get_text_enc(
        "matrixgame",
        str(checkpoint_dir),
        weight_dtype=weight_dtype,
        i2v_type="refiner",
    )

    pipeline = MatrixGameVideoPipeline(
        vae=vae.vae,
        text_encoder=text_enc,
        transformer=dit,
        scheduler=scheduler,
    ).to(weight_dtype).to(device)

    pipeline.transformer.__class__.enable_teacache = True
    pipeline.transformer.__class__.cnt = 0
    pipeline.transformer.__class__.num_steps = args.num_steps
    pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    pipeline.transformer.__class__.rel_l1_thresh = args.rel_l1_thresh
    pipeline.transformer.__class__.previous_modulated_input = None
    pipeline.transformer.__class__.previous_residual = None
    pipeline.transformer.__class__.forward = teacache_forward

    image = Image.open(image_path).convert("RGB")
    new_width, new_height = int(args.resolution[0]), int(args.resolution[1])
    initial_image = _resize_and_crop_image(image, (new_width, new_height))
    semantic_image = initial_image

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)
    initial_image_tensor = video_processor.preprocess(initial_image, height=new_height, width=new_width)
    if args.num_pre_frames > 0:
        past_frames = initial_image_tensor.repeat(args.num_pre_frames, 1, 1, 1)
        initial_image_tensor = torch.cat([initial_image_tensor, past_frames], dim=0)

    keyboard_tensor = torch.tensor(keyboard_condition, dtype=condition_dtype, device=device).unsqueeze(0)
    mouse_tensor = torch.tensor(mouse_condition, dtype=condition_dtype, device=device).unsqueeze(0)

    effective_prompt = args.prompt.strip() or "N.o.n.e"
    with torch.no_grad():
        video = pipeline(
            prompt=effective_prompt,
            height=new_height,
            width=new_width,
            video_length=args.video_length,
            mouse_condition=mouse_tensor,
            keyboard_condition=keyboard_tensor,
            initial_image=initial_image_tensor,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            embedded_guidance_scale=None,
            data_type="video",
            vae_ver="884-16c-hy",
            enable_tiling=True,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            i2v_type="refiner",
            semantic_images=semantic_image,
        ).videos[0]

    frames = rearrange(video.permute(1, 0, 2, 3) * 255, "t c h w -> t h w c").contiguous()
    frames = frames.cpu().numpy().astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=args.fps)


if __name__ == "__main__":
    main()
