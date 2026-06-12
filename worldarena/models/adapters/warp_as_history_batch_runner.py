from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import traceback

import numpy as np

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    print_status,
)


def _load_script(repo_root: Path):
    script_path = repo_root / "scripts" / "infer_warp_as_history.py"
    spec = importlib.util.spec_from_file_location("worldarena_warp_as_history_infer", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Warp-as-History batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    infer = _load_script(repo_root)
    if not generation.get("enable_optional_attention", False):
        infer.disable_diffusers_optional_attention()

    import torch
    from PIL import Image
    from warp_as_history import WarpAsHistoryPipeline

    device = str(generation.get("device", "cuda"))
    dtype = infer.torch_dtype_from_arg(str(generation.get("dtype", "auto")), device)
    model_path = infer.resolve_model_path(str(generation.get("model_path", "checkpoints/helios-distilled")))
    no_lora = bool(generation.get("no_lora", False))
    lora_path = None
    if not no_lora:
        lora_path = infer.resolve_lora_path(
            str(generation.get("lora_path", "checkpoints/warp-as-history/visible_lora_state_step1000.safetensors"))
        )

    pipe = WarpAsHistoryPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    height = int(generation.get("height", 384))
    width = int(generation.get("width", 640))
    fps = int(generation.get("fps", 16))
    requested_frames = int(generation.get("num_frames", 81))
    base_seed = int(generation.get("seed", 42))

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            annotation_path = Path(str(row["annotation_path"])).expanduser().resolve()
            poses = np.load(annotation_path / "poses.npy").astype(np.float32)
            num_frames = min(requested_frames, int(poses.shape[0]))
            generator = (
                torch.Generator(device=device).manual_seed(base_seed + row_index)
                if device.startswith("cuda")
                else None
            )
            result = pipe(
                prompt=str(row["prompt"]),
                image=Image.open(str(row["conditioning_image"])).convert("RGB"),
                camera_poses=poses[:num_frames],
                lora_path=lora_path,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
                output_type="np",
            )
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            infer.write_video(output_path, infer.unwrap_video_frames(result), fps=fps)
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                num_frames=num_frames,
                fps=fps,
                pose_source=row.get("pose_source"),
            )
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
