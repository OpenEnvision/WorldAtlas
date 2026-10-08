"""Persistent GEN3C subprocess runner that keeps weights loaded across samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from worldatlas.models.adapters.batch_runner_common import begin_sample, log_pipeline, print_status


DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent WorldAtlas GEN3C batch runtime.")
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--batch_input_path", required=True, type=str)
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
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT, type=str)
    parser.add_argument("--prompt_upsampler_dir", default="Pixtral-12B", type=str)
    parser.add_argument("--save_buffer", action="store_true")
    parser.add_argument("--foreground_masking", action="store_true")
    parser.add_argument("--offload_diffusion_transformer", action="store_true")
    parser.add_argument("--offload_tokenizer", action="store_true")
    parser.add_argument("--offload_text_encoder_model", action="store_true")
    parser.add_argument("--offload_prompt_upsampler", action="store_true")
    parser.add_argument("--offload_guardrail_models", action="store_true")
    parser.add_argument("--disable_prompt_encoder", action="store_true")
    return parser.parse_args()


def _load_items(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def _item_value(item: dict[str, Any], args: argparse.Namespace, key: str) -> Any:
    value = item.get(key)
    return getattr(args, key) if value is None else value


def _staged_output_path(output_path: Path) -> Path:
    """Keep partial encodes hidden until a complete video can replace the target."""
    return output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}"
    )


def _run_one_item(
    *,
    item: dict[str, Any],
    args: argparse.Namespace,
    pipeline: Any,
    moge_model: Any,
    device: torch.device,
    generator: torch.Generator,
    frame_buffer_max: int,
    sample_n_frames: int,
) -> None:
    from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
    from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
    from cosmos_predict1.diffusion.inference.gen3c_single_image import (
        _predict_moge_depth,
        _predict_moge_depth_from_tensor,
    )
    from cosmos_predict1.diffusion.inference.inference_utils import check_input_frames
    from cosmos_predict1.utils import log
    from cosmos_predict1.utils.io import save_video

    current_prompt = str(item.get("prompt") or "")
    current_image_path = str(item.get("visual_input") or "")
    output_path = Path(str(item["output_path"])).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_output_path = _staged_output_path(output_path)
    staged_output_path.unlink(missing_ok=True)

    if not current_prompt and not args.disable_prompt_encoder:
        raise ValueError("GEN3C persistent runtime requires non-empty prompt")
    if not current_image_path:
        raise ValueError("GEN3C persistent runtime requires visual_input")
    if not check_input_frames(current_image_path, 1):
        raise RuntimeError(f"input image is not valid: {current_image_path}")

    height = int(_item_value(item, args, "height"))
    width = int(_item_value(item, args, "width"))
    num_video_frames = int(_item_value(item, args, "num_video_frames"))
    filter_points_threshold = float(_item_value(item, args, "filter_points_threshold"))
    foreground_masking = bool(_item_value(item, args, "foreground_masking"))

    (
        moge_image_b1chw_float,
        moge_depth_b11hw,
        _moge_mask_b11hw,
        moge_initial_w2c_b144,
        moge_intrinsics_b133,
    ) = _predict_moge_depth(
        current_image_path,
        height,
        width,
        device,
        moge_model,
    )

    cache = Cache3D_Buffer(
        frame_buffer_max=frame_buffer_max,
        generator=generator,
        noise_aug_strength=float(item.get("noise_aug_strength", 0.0)),
        input_image=moge_image_b1chw_float[:, 0].clone(),
        input_depth=moge_depth_b11hw[:, 0],
        input_w2c=moge_initial_w2c_b144[:, 0],
        input_intrinsics=moge_intrinsics_b133[:, 0],
        filter_points_threshold=filter_points_threshold,
        foreground_masking=foreground_masking,
    )

    camera_path = item.get("camera_path")
    if isinstance(camera_path, list) and camera_path:
        from worldatlas.models.adapters.gen3c.gen3c_runner import generate_gen3c_camera_path

        generated_w2cs, generated_intrinsics = generate_gen3c_camera_path(
            camera_path=[str(token) for token in camera_path],
            generate_camera_trajectory=generate_camera_trajectory,
            initial_w2c=moge_initial_w2c_b144[0, 0],
            initial_intrinsics=moge_intrinsics_b133[0, 0],
            num_frames=num_video_frames,
            movement_distance=float(_item_value(item, args, "movement_distance")),
            center_depth=1.0,
            device=device.type,
        )
    else:
        generated_w2cs, generated_intrinsics = generate_camera_trajectory(
            trajectory_type=str(_item_value(item, args, "trajectory")),
            initial_w2c=moge_initial_w2c_b144[0, 0],
            initial_intrinsics=moge_intrinsics_b133[0, 0],
            num_frames=num_video_frames,
            movement_distance=float(_item_value(item, args, "movement_distance")),
            camera_rotation=str(_item_value(item, args, "camera_rotation")),
            center_depth=1.0,
            device=device.type,
        )

    rendered_warp_images, rendered_warp_masks = cache.render_cache(
        generated_w2cs[:, 0:sample_n_frames],
        generated_intrinsics[:, 0:sample_n_frames],
    )

    all_rendered_warps = []
    if args.save_buffer:
        all_rendered_warps.append(rendered_warp_images.clone().cpu())

    generated_output = pipeline.generate(
        prompt=current_prompt,
        image_path=current_image_path,
        negative_prompt=str(item.get("negative_prompt") or args.negative_prompt or ""),
        rendered_warp_images=rendered_warp_images,
        rendered_warp_masks=rendered_warp_masks,
    )
    if generated_output is None:
        raise RuntimeError("guardrail blocked GEN3C generation")
    video, _prompt = generated_output

    num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
    for num_iter in range(1, num_ar_iterations):
        start_frame_idx = num_iter * (sample_n_frames - 1)
        end_frame_idx = start_frame_idx + sample_n_frames
        log.info(f"Generating {start_frame_idx} - {end_frame_idx} frames")

        last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)
        pred_image_for_depth_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1) / 255.0
        pred_depth, _pred_mask = _predict_moge_depth_from_tensor(
            pred_image_for_depth_chw_0_1,
            moge_model,
        )
        cache.update_cache(
            new_image=pred_image_for_depth_chw_0_1.unsqueeze(0) * 2 - 1,
            new_depth=pred_depth,
            new_w2c=generated_w2cs[:, start_frame_idx],
            new_intrinsics=generated_intrinsics[:, start_frame_idx],
        )

        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs[:, start_frame_idx:end_frame_idx],
            generated_intrinsics[:, start_frame_idx:end_frame_idx],
        )
        if args.save_buffer:
            all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())

        pred_image_for_depth_bcthw_minus1_1 = pred_image_for_depth_chw_0_1.unsqueeze(0).unsqueeze(2) * 2 - 1
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_path=pred_image_for_depth_bcthw_minus1_1,
            negative_prompt=str(item.get("negative_prompt") or args.negative_prompt or ""),
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
        )
        video_new, _prompt = generated_output
        video = np.concatenate([video, video_new[1:]], axis=0)

    final_video_to_save = video
    final_width = width
    if args.save_buffer and all_rendered_warps:
        squeezed_warps = [tensor.squeeze(0) for tensor in all_rendered_warps]
        n_max = max(tensor.shape[1] for tensor in squeezed_warps)
        padded_tensors = []
        for tensor in squeezed_warps:
            pad_spec = (0, 0, 0, 0, 0, 0, 0, n_max - tensor.shape[1], 0, 0)
            padded_tensors.append(F.pad(tensor, pad_spec, mode="constant", value=-1.0))
        full_rendered_warp_tensor = torch.cat(padded_tensors, dim=0)
        t_total, _n, c_dim, h_dim, w_dim = full_rendered_warp_tensor.shape
        buffer_video = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
        buffer_video = buffer_video.contiguous().view(t_total, c_dim, h_dim, n_max * w_dim)
        buffer_video = (buffer_video * 0.5 + 0.5) * 255.0
        buffer_numpy = np.transpose(buffer_video.cpu().numpy().astype(np.uint8), (0, 2, 3, 1))
        final_video_to_save = np.concatenate([buffer_numpy, final_video_to_save], axis=2)
        final_width = width * (1 + n_max)

    save_video(
        video=final_video_to_save,
        fps=int(_item_value(item, args, "fps")),
        H=height,
        W=final_width,
        video_save_quality=5,
        video_save_path=str(staged_output_path),
    )
    if not staged_output_path.is_file() or staged_output_path.stat().st_size <= 0:
        raise RuntimeError(f"GEN3C staged output was not written: {staged_output_path}")
    staged_output_path.replace(output_path)


def main() -> None:
    args = parse_args()
    torch.enable_grad(False)

    from moge.model.v1 import MoGeModel

    from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
    from cosmos_predict1.utils import log, misc

    misc.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = _load_items(Path(args.batch_input_path).expanduser().resolve())

    pipeline = Gen3cPipeline(
        inference_type="video2world",
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=False,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=True,
        disable_prompt_encoder=args.disable_prompt_encoder,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
    )
    frame_buffer_max = pipeline.model.frame_buffer_max
    sample_n_frames = pipeline.model.chunk_size
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    log_pipeline("pipeline_loaded", label="GEN3C", samples=len(items))

    for index, item in enumerate(items):
        sample_id = str(item.get("sample_id", index))
        begin_sample(sample_id, index=index + 1, total=len(items), label="GEN3C")
        try:
            misc.set_random_seed(args.seed)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            _run_one_item(
                item=item,
                args=args,
                pipeline=pipeline,
                moge_model=moge_model,
                device=device,
                generator=generator,
                frame_buffer_max=frame_buffer_max,
                sample_n_frames=sample_n_frames,
            )
            print_status(sample_id, "generated", output_path=str(item.get("output_path", "")))
        except Exception as exc:
            raw_output_path = item.get("output_path")
            if raw_output_path:
                output_path = Path(str(raw_output_path)).expanduser().resolve()
                _staged_output_path(output_path).unlink(missing_ok=True)
            error_path = item.get("error_path")
            if error_path:
                Path(str(error_path)).write_text(traceback.format_exc(), encoding="utf-8")
            log.critical(traceback.format_exc())
            print_status(sample_id, "failed", error=str(exc))


if __name__ == "__main__":
    main()
