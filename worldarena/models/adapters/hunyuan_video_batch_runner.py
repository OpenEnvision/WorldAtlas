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
    parser = argparse.ArgumentParser(description="WorldArena HunyuanVideo batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


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

    dit_weight = str(
        generation.get(
            "dit_weight",
            checkpoint_dir / "hunyuan-video-t2v-720p" / "transformers" / "mp_rank_00_model_states.pt",
        )
    )
    argv = [
        "hunyuan_video_batch",
        "--model-base",
        str(checkpoint_dir),
        "--dit-weight",
        dit_weight,
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
    if generation.get("use_cpu_offload", False):
        argv.append("--use-cpu-offload")
    if generation.get("use_fp8", False):
        argv.append("--use-fp8")
    old_argv = sys.argv
    sys.argv = argv
    try:
        hv_args = parse_hunyuan_args()
    finally:
        sys.argv = old_argv
    sampler = HunyuanVideoSampler.from_pretrained(checkpoint_dir, args=hv_args)
    hv_args = sampler.args

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            seed = generation.get("seed")
            if seed is not None:
                seed = int(seed) + row_index
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
            )
            sample = outputs["samples"][0].unsqueeze(0)
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_videos_grid(sample, str(output_path), fps=int(generation.get("fps", 24)))
            print_status(sample_id, "generated", output_path=str(output_path), fps=int(generation.get("fps", 24)))
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
