from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WorldArena Echo-Infinity batch subprocess runner (torchrun multi-GPU)."
    )
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--batch_spec_path", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--num_output_frames", default=21, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fps", default=16, type=int)
    return parser.parse_args()


def _load_batch_spec(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty Echo-Infinity batch spec: {path}")
    return rows


def _load_conditioning_latent(
    pipeline: Any,
    conditioning_image: str,
    device: torch.device,
) -> torch.Tensor:
    from PIL import Image  # noqa: PLC0415
    from torchvision import transforms  # noqa: PLC0415

    image = Image.open(conditioning_image).convert("RGB").resize((832, 480))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    vae_dtype = next(pipeline.vae.model.parameters()).dtype
    # encode_to_latent expects [B, C, T, H, W]
    pixel = transform(image).unsqueeze(0).unsqueeze(2).to(device=device, dtype=vae_dtype)
    autocast_enabled = device.type == "cuda" and vae_dtype in {
        torch.float16,
        torch.bfloat16,
    }
    with torch.autocast(
        device_type=device.type,
        dtype=vae_dtype,
        enabled=autocast_enabled,
    ):
        latent = pipeline.vae.encode_to_latent(pixel)
    return latent[:, :1].to(device=device, dtype=vae_dtype)


def _wan_video_frame_count(latent_frames: int) -> int:
    if latent_frames <= 0:
        return 0
    return 4 * latent_frames - 3


def _decode_i2v_generated_video(
    pipeline: Any,
    latent_output: torch.Tensor,
    *,
    num_input_frames: int,
    num_generated_frames: int,
) -> torch.Tensor:
    # Decode with the conditioning latent present so the temporal VAE has the
    # same left context used by the diffusion model, then crop away the visible
    # conditioning prefix. Decoding generated latents alone makes the first
    # decoded segment noticeably soft because the VAE starts without context.
    decode_start = max(0, num_input_frames - 1)
    context_output = latent_output[
        :, decode_start : num_input_frames + num_generated_frames
    ]
    generated_output = latent_output[
        :, num_input_frames : num_input_frames + num_generated_frames
    ]
    if generated_output.shape[1] != num_generated_frames:
        raise RuntimeError(
            "Echo-Infinity I2V generated latent slice has unexpected length: "
            f"expected={num_generated_frames} actual={generated_output.shape[1]}"
        )

    video = pipeline.vae.decode_to_pixel(context_output, use_cache=False)
    # Wan VAE maps T latent frames to 4*T - 3 pixel frames.
    expected_video_frames = _wan_video_frame_count(num_generated_frames)
    prefix_video_frames = _wan_video_frame_count(num_input_frames - decode_start)
    if video.shape[1] < prefix_video_frames + expected_video_frames:
        raise RuntimeError(
            "Echo-Infinity I2V decoded fewer video frames than expected: "
            f"expected>={prefix_video_frames + expected_video_frames} actual={video.shape[1]}"
        )
    return video[:, prefix_video_frames : prefix_video_frames + expected_video_frames]


def _run_i2v_inference(
    pipeline: Any,
    *,
    prompt: str,
    conditioning_image: str,
    noise: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    image_latent = _load_conditioning_latent(pipeline, conditioning_image, device)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])

    batch_size, num_generated_frames, num_channels, height, width = noise.shape
    if num_generated_frames % pipeline.num_frame_per_block != 0:
        raise ValueError(
            "Echo-Infinity I2V generated latent frames must be divisible by "
            f"num_frame_per_block={pipeline.num_frame_per_block}; got {num_generated_frames}"
        )

    num_input_frames = image_latent.shape[1]
    num_output_frames = num_input_frames + num_generated_frames
    output = torch.zeros(
        [batch_size, num_output_frames, num_channels, height, width],
        device=device,
        dtype=noise.dtype,
    )
    output[:, :num_input_frames] = image_latent

    kv_cache_size = pipeline.local_attn_size * pipeline.frame_seq_length
    pipeline._initialize_kv_cache(
        batch_size=batch_size,
        dtype=noise.dtype,
        device=device,
        kv_cache_size_override=kv_cache_size,
    )
    pipeline._initialize_crossattn_cache(
        batch_size=batch_size,
        dtype=noise.dtype,
        device=device,
    )
    pipeline.generator.model.local_attn_size = pipeline.local_attn_size
    pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)

    cond_ts = torch.zeros([batch_size, 1], device=device, dtype=torch.int64)
    pipeline.generator(
        noisy_image_or_video=image_latent,
        conditional_dict=conditional_dict,
        timestep=cond_ts,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=0,
    )

    current_start_frame = num_input_frames
    all_num_frames = [
        pipeline.num_frame_per_block
        for _ in range(num_generated_frames // pipeline.num_frame_per_block)
    ]
    for block_size in all_num_frames:
        noise_start = current_start_frame - num_input_frames
        noisy_input = noise[:, noise_start : noise_start + block_size]
        for idx, current_timestep in enumerate(pipeline.denoising_step_list):
            ts = torch.ones([batch_size, block_size], device=device, dtype=torch.int64) * current_timestep
            if idx < len(pipeline.denoising_step_list) - 1:
                _, denoised_pred = pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=ts,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )
                next_ts = pipeline.denoising_step_list[idx + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_ts * torch.ones([batch_size * block_size], device=device, dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=ts,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )
        output[:, current_start_frame : current_start_frame + block_size] = denoised_pred
        ctx_ts = torch.ones_like(ts) * pipeline.args.context_noise
        # Match official inference: cache the clean final block. Adding noise here
        # creates periodic block-boundary artifacts in generated I2V clips.
        pipeline.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=ctx_ts,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )
        current_start_frame += block_size

    video = _decode_i2v_generated_video(
        pipeline,
        output,
        num_input_frames=num_input_frames,
        num_generated_frames=num_generated_frames,
    )
    return (video * 0.5 + 0.5).clamp(0, 1)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    # Set CUDA device before any CUDA calls or Echo-Infinity imports
    # (utils/memory.py reads current_device() at import time)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=3600),
        )

    repo_root = Path(args.repo_root).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()

    # Insert repo_root so 'pipeline', 'utils', 'wan', etc. are importable
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Must chdir so wan_wrapper.py can open relative paths like 'wan_models/...'
    os.chdir(str(repo_root))

    if rank == 0:
        logging.info(
            "Echo-Infinity batch runner starting: rank=%s world_size=%s ckpt=%s",
            rank,
            world_size,
            ckpt_path,
        )

    batch_spec = _load_batch_spec(batch_spec_path)
    my_rows = batch_spec[rank::world_size]

    if rank == 0:
        logging.info(
            "Total samples=%s; this rank (%s) handles %s",
            len(batch_spec),
            rank,
            len(my_rows),
        )

    # --- Load config from official yaml ---
    from omegaconf import OmegaConf  # noqa: PLC0415

    base_config_path = repo_root / "configs" / "echo_infinity_inference_std.yaml"
    config = OmegaConf.load(str(base_config_path))

    # Override with runner parameters
    config.generator_ckpt = str(ckpt_path)
    config.seed = args.seed + rank  # distinct seed per rank
    config.num_output_frames = args.num_output_frames
    config.num_samples = 1
    # Each rank runs independently; no collective ops needed inside the pipeline
    config.distributed = False

    torch.set_grad_enabled(False)

    # --- Import Echo-Infinity modules (after set_device so gpu constant is correct) ---
    from utils.memory import get_cuda_free_memory_gb  # noqa: PLC0415
    from utils.misc import set_seed  # noqa: PLC0415
    from pipeline import CausalInferencePipeline  # noqa: PLC0415

    set_seed(args.seed + rank)

    if rank == 0:
        logging.info("Loading Echo-Infinity pipeline on device=%s", device)

    pipeline = CausalInferencePipeline(config, device=device)

    # --- Load checkpoint exactly once per rank ---
    if rank == 0:
        logging.info("Loading checkpoint from %s", ckpt_path)
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict.get(
            "generator", state_dict.get("generator_ema")
        )
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in checkpoint: {ckpt_path}")

    cleaned = {
        k.replace("_fsdp_wrapped_module.", ""): v
        for k, v in raw_gen_state_dict.items()
    }
    missing, unexpected = pipeline.generator.load_state_dict(cleaned, strict=False)
    if rank == 0:
        if missing:
            logging.warning(
                "Checkpoint missing %s keys: %s ...", len(missing), missing[:8]
            )
        if unexpected:
            logging.warning(
                "Checkpoint unexpected %s keys: %s ...", len(unexpected), unexpected[:8]
            )

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    if rank == 0:
        logging.info(
            "Pipeline loaded; processing %s samples on rank=%s", len(my_rows), rank
        )

    # --- Inference loop: model loaded once, all samples processed sequentially ---
    failed = 0
    for row in my_rows:
        sample_id = str(row.get("sample_id", ""))
        output_path = Path(str(row["output_path"])).expanduser().resolve()

        if output_path.is_file() and output_path.stat().st_size > 0:
            logging.info("rank=%s skipping existing output: %s", rank, output_path)
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "status": "skipped",
                        "output_path": str(output_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        try:
            prompt = str(row["prompt"])
            low_memory = get_cuda_free_memory_gb(device) < 40
            conditioning_image = row.get("conditioning_image")

            noise = torch.randn(
                [1, config.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )

            if conditioning_image:
                video = _run_i2v_inference(
                    pipeline,
                    prompt=prompt,
                    conditioning_image=str(conditioning_image),
                    noise=noise,
                    device=device,
                )
            else:
                video, _ = pipeline.inference(
                    noise=noise,
                    text_prompts=[prompt],
                    return_latents=True,
                    low_memory=low_memory,
                    profile=False,
                )

            # video: [B, T, C, H, W] in [0, 1] range (float)
            from einops import rearrange  # noqa: PLC0415

            current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
            video_uint8 = (255.0 * current_video).clamp(0, 255).to(torch.uint8)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_name(
                f".tmp_{os.getpid()}_{output_path.name}"
            )
            from torchvision.io import write_video  # noqa: PLC0415

            write_video(str(tmp_path), video_uint8[0], fps=args.fps)
            tmp_path.rename(output_path)

            # Clear VAE streaming cache between samples
            pipeline.vae.model.clear_cache()

            logging.info(
                "rank=%s generated sample_id=%s -> %s", rank, sample_id, output_path
            )
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "status": "generated",
                        "output_path": str(output_path),
                        "num_output_frames": config.num_output_frames,
                        "fps": args.fps,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            failed += 1
            logging.error(
                "rank=%s failed sample_id=%s error=%s", rank, sample_id, exc
            )
            logging.error(
                "Echo-Infinity row traceback:\n%s", traceback.format_exc()
            )
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "status": "failed",
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    logging.info(
        "rank=%s completed rows=%s failed=%s", rank, len(my_rows), failed
    )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
