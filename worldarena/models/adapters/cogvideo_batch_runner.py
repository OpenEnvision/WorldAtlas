from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="WorldArena CogVideo batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _torch_dtype(torch, value: str):
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    return torch.bfloat16


_DEFAULT_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}


def _default_height_width(model_path: str) -> tuple[int, int] | None:
    return _DEFAULT_RESOLUTIONS.get(Path(model_path).name.lower())


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())

    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline, CogVideoXPipeline
    from diffusers.utils import export_to_video, load_image

    model_path = str(generation.get("model_path") or args.checkpoint_dir)
    generate_type = str(generation.get("generate_type", "i2v"))
    dtype = _torch_dtype(torch, str(generation.get("dtype", "bfloat16")))
    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        raise ValueError(f"WorldArena CogVideo runner supports generate_type=i2v/t2v, got {generate_type}")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    if generation.get("cpu_offload", "sequential") == "model":
        pipe.enable_model_cpu_offload()
    elif generation.get("cpu_offload", "sequential"):
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            default_height_width = _default_height_width(model_path)
            height = generation.get("height")
            width = generation.get("width")
            if default_height_width is not None:
                if height is None:
                    height = default_height_width[0]
                if width is None:
                    width = default_height_width[1]
            kwargs = {
                "height": int(height) if height is not None else None,
                "width": int(width) if width is not None else None,
                "prompt": str(row["prompt"]),
                "num_videos_per_prompt": 1,
                "num_inference_steps": int(generation.get("num_inference_steps", 50)),
                "num_frames": int(generation.get("num_frames", 81)),
                "use_dynamic_cfg": bool(generation.get("use_dynamic_cfg", True)),
                "guidance_scale": float(generation.get("guidance_scale", 6.0)),
                "generator": torch.Generator().manual_seed(int(generation.get("seed", 42)) + row_index),
            }
            if kwargs["height"] is None:
                kwargs.pop("height")
            if kwargs["width"] is None:
                kwargs.pop("width")
            if generate_type == "i2v":
                kwargs["image"] = load_image(str(row["conditioning_image"]))
            frames = pipe(**kwargs).frames[0]
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output_path), fps=int(generation.get("fps", 16)))
            print_status(sample_id, "generated", output_path=str(output_path), fps=int(generation.get("fps", 16)))
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
