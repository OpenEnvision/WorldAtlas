from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena HunyuanVideo-I2V batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def distributed_barrier() -> None:
    try:
        import torch.distributed as dist
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_for_row(generation: dict, row_index: int, num_gpus: int) -> int | None:
    seed = generation.get("seed")
    if seed is not None:
        return int(seed) + row_index
    if num_gpus > 1:
        return row_index
    return None


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    checkpoint_dir = Path(args.checkpoint_dir or generation.get("model_base", "ckpts")).expanduser().resolve()
    os.environ["MODEL_BASE"] = str(checkpoint_dir)

    from hyvideo.config import parse_args as parse_hunyuan_args
    from hyvideo.inference import HunyuanVideoSampler
    from hyvideo.utils.file_utils import save_videos_grid

    i2v_weight = str(
        generation.get(
            "i2v_dit_weight",
            checkpoint_dir / "hunyuan-video-i2v-720p" / "transformers" / "mp_rank_00_model_states.pt",
        )
    )
    argv = [
        "hunyuan_video_i2v_batch",
        "--model",
        str(generation.get("model", "HYVideo-T/2")),
        "--model-base",
        str(checkpoint_dir),
        "--i2v-dit-weight",
        i2v_weight,
        "--i2v-mode",
        "--i2v-resolution",
        str(generation.get("i2v_resolution", "720p")),
        "--i2v-condition-type",
        str(generation.get("i2v_condition_type", "token_replace")),
        "--infer-steps",
        str(int(generation.get("infer_steps", 50))),
        "--video-length",
        str(int(generation.get("video_length", 129))),
        "--cfg-scale",
        str(float(generation.get("cfg_scale", 1.0))),
        "--embedded-cfg-scale",
        str(float(generation.get("embedded_cfg_scale", 6.0))),
        "--flow-shift",
        str(float(generation.get("flow_shift", 7.0))),
    ]
    video_size = generation.get("video_size", [720, 1280])
    argv.extend(["--video-size", *[str(int(value)) for value in video_size]])
    if generation.get("flow_reverse", True):
        argv.append("--flow-reverse")
    if generation.get("i2v_stability", True):
        argv.append("--i2v-stability")
    if generation.get("use_cpu_offload", False):
        argv.append("--use-cpu-offload")
    num_gpus = int(generation.get("num_gpus", 1))
    ulysses_degree = int(generation.get("ulysses_degree", num_gpus))
    ring_degree = int(generation.get("ring_degree", 1))
    if num_gpus > 1:
        argv.extend(["--ulysses-degree", str(ulysses_degree), "--ring-degree", str(ring_degree)])

    old_argv = sys.argv
    sys.argv = argv
    try:
        hv_args = parse_hunyuan_args()
    finally:
        sys.argv = old_argv

    sampler = HunyuanVideoSampler.from_pretrained(checkpoint_dir, args=hv_args)
    hv_args = sampler.args
    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            distributed_barrier()
            conditioning_image = Path(str(row["conditioning_image"])).expanduser().resolve()
            if not conditioning_image.is_file():
                raise FileNotFoundError(f"conditioning image not found: {conditioning_image}")
            seed = seed_for_row(generation, row_index, num_gpus)
            outputs = sampler.predict(
                prompt=str(row["prompt"]),
                height=hv_args.video_size[0],
                width=hv_args.video_size[1],
                video_length=hv_args.video_length,
                seed=seed,
                negative_prompt=generation.get("neg_prompt"),
                infer_steps=hv_args.infer_steps,
                guidance_scale=hv_args.cfg_scale,
                num_videos_per_prompt=1,
                flow_shift=hv_args.flow_shift,
                batch_size=1,
                embedded_guidance_scale=hv_args.embedded_cfg_scale,
                i2v_mode=True,
                i2v_resolution=hv_args.i2v_resolution,
                i2v_image_path=str(conditioning_image),
                i2v_condition_type=hv_args.i2v_condition_type,
                i2v_stability=hv_args.i2v_stability,
                ulysses_degree=hv_args.ulysses_degree,
                ring_degree=hv_args.ring_degree,
            )
            if is_main_process:
                sample = outputs["samples"][0].unsqueeze(0)
                output_path = Path(str(row["output_path"])).expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_videos_grid(sample, str(output_path), fps=int(generation.get("fps", 24)))
                print_status(
                    sample_id,
                    "generated",
                    output_path=str(output_path),
                    conditioning_image=str(conditioning_image),
                    fps=int(generation.get("fps", 24)),
                    seed=outputs["seeds"][0],
                )
            distributed_barrier()
        except Exception as exc:
            failed += 1
            if is_main_process:
                print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
            if num_gpus > 1:
                raise
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
