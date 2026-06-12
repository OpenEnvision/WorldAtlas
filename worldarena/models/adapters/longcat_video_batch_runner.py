from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena LongCat-Video batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _init_single_rank_dist(torch, dist) -> None:
    if dist.is_available() and dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    handle = tempfile.NamedTemporaryFile(prefix="worldarena_longcat_dist_", delete=True)
    dist.init_process_group(
        backend=backend,
        rank=0,
        world_size=1,
        init_method=f"file://{handle.name}",
        timeout=datetime.timedelta(seconds=3600 * 24),
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())

    import PIL.Image
    import torch
    import torch.distributed as dist
    from diffusers.utils import load_image
    from torchvision.io import write_video
    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.context_parallel import context_parallel_util
    from longcat_video.context_parallel.context_parallel_util import init_context_parallel
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

    checkpoint_dir = Path(args.checkpoint_dir or generation.get("checkpoint_dir", "")).expanduser().resolve()
    _init_single_rank_dist(torch, dist)
    local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    init_context_parallel(context_parallel_size=int(generation.get("context_parallel_size", 1)), global_rank=0, world_size=1)
    cp_split_hw = context_parallel_util.get_optimal_split(context_parallel_util.get_cp_size())

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=dtype)
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir,
        subfolder="dit",
        cp_split_hw=cp_split_hw,
        torch_dtype=dtype,
    )
    if generation.get("enable_compile", False):
        dit = torch.compile(dit)
    pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler, dit=dit)
    pipe.to(local_rank)

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            generator = torch.Generator(device=local_rank)
            generator.manual_seed(int(generation.get("seed", 42)) + row_index)
            image = load_image(str(row["conditioning_image"]))
            target_size = image.size
            output = pipe.generate_i2v(
                image=image,
                prompt=str(row["prompt"]),
                negative_prompt=str(generation.get("negative_prompt", "")),
                resolution=str(generation.get("resolution", "480p")),
                num_frames=int(generation.get("num_frames", 93)),
                num_inference_steps=int(generation.get("num_inference_steps", 50)),
                guidance_scale=float(generation.get("guidance_scale", 4.0)),
                generator=generator,
            )[0]
            frames = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
            frames = [PIL.Image.fromarray(frame).resize(target_size, PIL.Image.BICUBIC) for frame in frames]
            output_tensor = torch.from_numpy(np.array(frames))
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_video(
                str(output_path),
                output_tensor,
                fps=int(generation.get("fps", 15)),
                video_codec="libx264",
                options={"crf": str(int(generation.get("crf", 18)))},
            )
            print_status(sample_id, "generated", output_path=str(output_path), fps=int(generation.get("fps", 15)))
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
