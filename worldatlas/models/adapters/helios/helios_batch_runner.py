"""One official Helios pipeline per GPU process, reused across a JSONL shard."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args, begin_sample, load_batch_spec, log_pipeline, print_status,
)
from .helios import UPSTREAM_REVISION, runtime_options, sample_seed


def validate_paths(repo_root: Path, checkpoint: Path, variant: str) -> str:
    revision = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Helios checkout must be pinned to {UPSTREAM_REVISION}, got {revision}")
    if subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Helios upstream checkout has tracked modifications")
    for name in ("pipeline_helios_diffusers.py", "transformer_helios_diffusers.py", "scheduling_helios_diffusers.py"):
        if not (repo_root / "helios/diffusers_version" / name).is_file():
            raise FileNotFoundError(name)
    index = checkpoint / "model_index.json"
    if not index.is_file():
        raise FileNotFoundError(f"Download the complete official Helios snapshot first: {index}")
    metadata = json.loads(index.read_text())
    if metadata.get("is_distilled") is not (variant == "distilled"):
        raise ValueError(f"Checkpoint is_distilled does not match configured variant={variant}")
    for name in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
        if not (checkpoint / name).is_dir():
            raise FileNotFoundError(f"Incomplete Helios checkpoint: {checkpoint / name}")
    return revision


def load_pipeline(repo_root: Path, checkpoint: Path, options: dict[str, Any]):
    # Imports and CUDA initialization live only in this model's dedicated env.
    sys.path.insert(0, str(repo_root))
    import torch
    from diffusers import AutoencoderKLWan
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
    from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel

    if not torch.cuda.is_available():
        raise RuntimeError("Helios production inference requires a CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use independent GPU shards, not torchrun, for this adapter")
    torch.cuda.set_device(0)
    local = dict(local_files_only=True)
    transformer = HeliosTransformer3DModel.from_pretrained(
        str(checkpoint), subfolder="transformer", torch_dtype=torch.bfloat16, **local)
    if options["use_official_kernels"]:
        from helios.modules.helios_kernels import (
            replace_rmsnorm_with_fp32, replace_all_norms_with_flash_norms, replace_rope_with_flash_rope,
        )
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    backend = options["attention_backend"]
    if backend == "auto":
        backend = "_flash_3_hub" if torch.cuda.get_device_capability()[0] >= 9 else "flash_hub"
    transformer.set_attention_backend(backend)
    vae = AutoencoderKLWan.from_pretrained(str(checkpoint), subfolder="vae", torch_dtype=torch.float32, **local)
    # Use the repository's unified scheduler exactly as infer_helios.py does,
    # including for the released Distilled model (not a substitute Wan scheduler).
    scheduler = HeliosScheduler.from_pretrained(str(checkpoint), subfolder="scheduler", **local)
    pipe = HeliosPipeline.from_pretrained(
        str(checkpoint), transformer=transformer, vae=vae, scheduler=scheduler,
        torch_dtype=torch.bfloat16, **local)
    if options["enable_low_vram_mode"]:
        pipe.enable_group_offload(onload_device=torch.device("cuda:0"), offload_device=torch.device("cpu"),
                                 offload_type="leaf_level", use_stream=True, record_stream=True)
    else:
        pipe = pipe.to("cuda:0")
    return pipe, backend


def generate_frames(pipe, row: dict[str, Any], options: dict[str, Any], seed: int):
    import torch
    from PIL import Image

    kwargs = copy.deepcopy(options["pipeline_kwargs"])
    with Image.open(row["conditioning_image"]) as source:
        image = source.convert("RGB").resize((kwargs["width"], kwargs["height"]))
    with torch.inference_mode():
        frames = pipe(prompt=row["prompt"], image=image,
                      generator=torch.Generator(device="cuda:0").manual_seed(seed), **kwargs).frames[0]
    if len(frames) < options["requested_frames"]:
        raise RuntimeError(f"Short Helios output: {len(frames)} < {options['requested_frames']}; refusing frame padding")
    return frames[:options["requested_frames"]], len(frames)


def write_video(frames, destination: Path, options: dict[str, Any]) -> None:
    import cv2
    from diffusers.utils import export_to_video

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.mp4")
    try:
        export_to_video(frames, str(temporary), fps=options["fps"])
        cap = cv2.VideoCapture(str(temporary))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            expected_size = (options['pipeline_kwargs']['width'], options['pipeline_kwargs']['height'])
            if (not cap.isOpened() or count != options["requested_frames"] or not math.isfinite(fps)
                    or abs(fps - options["fps"]) > 0.01 or size != expected_size):
                raise RuntimeError(f"Invalid encoded Helios video: frames={count}, fps={fps}")
        finally:
            cap.release()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_batch(repo_root: Path, checkpoint: Path, rows: list[dict[str, Any]], generation: dict[str, Any]) -> int:
    options = runtime_options(generation)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Helios batch requires unique non-empty sample IDs")
    if len({str(Path(r['output_path']).resolve()) for r in rows}) != len(rows):
        raise ValueError("Helios batch output paths must be unique")
    for row in rows:
        if not Path(row["conditioning_image"]).is_file() or not isinstance(row["prompt"], str):
            raise ValueError(f"Invalid Helios I2V input for {row['sample_id']}")
    revision = validate_paths(repo_root, checkpoint, options["variant"])
    log_pipeline("checkpoint_load_start", label="Helios", variant=options["variant"])
    pipe, backend = load_pipeline(repo_root, checkpoint, options)
    log_pipeline("checkpoint_load_done", label="Helios", checkpoint_load_policy="load_once", attention=backend)
    failures = 0
    for i, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        begin_sample(sample_id, index=i + 1, total=len(rows), label="Helios")
        try:
            seed = sample_seed(options["seed"], sample_id)
            frames, native_frames = generate_frames(pipe, row, options, seed)
            output = Path(row["output_path"]).expanduser().resolve()
            write_video(frames, output, options)
            metadata = {"sample_id": sample_id, "upstream_revision": revision, "checkpoint_dir": str(checkpoint),
                        "variant": options["variant"], "seed": seed, "attention_backend": backend,
                        "requested_frames": options["requested_frames"], "native_frames": native_frames,
                        "fps": options["fps"], "conditioning": "image_and_text_only",
                        "native_camera_or_action_control": False, "generation": generation,
                        "pipeline_kwargs": options["pipeline_kwargs"]}
            sidecar = output.with_suffix(".helios.json")
            temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(sidecar)
            print_status(sample_id, "generated", output_path=str(output), num_frames=options["requested_frames"],
                         fps=options["fps"], seed=seed, provenance=str(sidecar))
        except Exception as exc:
            failures += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    # Never let the parent treat stale output files as successful new generation.
    return int(failures > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_batch_args(parser)
    args = parser.parse_args()
    if not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required")
    generation = json.loads(Path(args.generation_config_path).read_text())
    return run_batch(Path(args.repo_root).expanduser().resolve(), Path(args.checkpoint_dir).expanduser().resolve(),
                     load_batch_spec(Path(args.batch_spec_path)), generation)


if __name__ == "__main__":
    raise SystemExit(main())
