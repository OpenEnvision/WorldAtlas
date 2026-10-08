"""Reuse one official Zing pipeline for every sample in a WorldAtlas shard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from typing import Any
from uuid import uuid4

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args, begin_sample, load_batch_spec, load_json, log_pipeline, print_status,
)
from .zing_world import (
    ACTION_ORDER, UPSTREAM_REVISION, build_message, keyboard_actions, runtime_options, sample_seed,
)


def resolve_runtime_paths(repo_root: Path, checkpoint_root: Path) -> tuple[Path, Path, Path]:
    """Validate the public source and complete generator/pretrained layout."""
    config_path = repo_root / "config/zing.yaml"
    pretrained = checkpoint_root / "pretrained"
    checkpoint = checkpoint_root / "generator/model.pt"
    required = [
        config_path, repo_root / "src/zing_v0_5/pipeline.py",
        repo_root / "src/zing_v0_5/processor.py", checkpoint,
        pretrained / "text_encoder/config.json",
        pretrained / "text_encoder/model.safetensors.index.json",
        pretrained / "tokenizer/tokenizer_config.json",
        pretrained / "tokenizer/spiece.model",
        pretrained / "vae/config.json",
        pretrained / "vae/diffusion_pytorch_model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Incomplete Zing runtime; missing files: " + ", ".join(missing))
    index = json.loads((pretrained / "text_encoder/model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        raise ValueError("Zing text encoder index has no weight_map")
    for shard in shards:
        path = pretrained / "text_encoder" / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing Zing text encoder shard: {path}")
    return config_path.resolve(), pretrained.resolve(), checkpoint.resolve()


def validate_revision(repo_root: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Zing checkout must be pinned to {UPSTREAM_REVISION}, got {revision}")
    changed = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if changed:
        raise ValueError("Zing upstream checkout has tracked modifications")
    return revision


def load_runtime(repo_root: Path, paths: tuple[Path, Path, Path], options: dict[str, Any]):
    """Import CUDA dependencies only inside the dedicated model process."""
    sys.path.insert(0, str(repo_root / "src"))
    import torch
    from zing_v0_5.config import load_config, with_cache_window
    from zing_v0_5.pipeline import InferencePipeline
    from zing_v0_5.processor import MessageProcessor

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Zing expects one GPU per process, not torchrun")
    if not torch.cuda.is_available():
        raise RuntimeError("Zing inference requires a CUDA GPU")
    torch.cuda.set_device(0)
    config_path, pretrained, checkpoint = paths
    config = with_cache_window(load_config(config_path), options["local_attn_size"], options["sink_size"])
    if config.inference.output_fps != options["fps"]:
        raise ValueError("Zing upstream output FPS does not match the WorldAtlas frame plan")
    pipeline = InferencePipeline(config, pretrained, checkpoint)
    return pipeline, MessageProcessor(config, pipeline.encode_reference)


def generate_frames(runtime, message: dict[str, Any], options: dict[str, Any], seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pipeline, processor = runtime
    # Match upstream main: reference encoding and generation both run under no_grad.
    with torch.no_grad():
        request = processor.process(message)
        video = pipeline.generate(request)
        expected = (1, options["num_frames"], 3, options["height"], options["width"])
        if tuple(video.shape) != expected:
            raise RuntimeError(f"Zing returned {tuple(video.shape)}, expected {expected}")
        if not torch.isfinite(video).all():
            raise RuntimeError("Zing returned non-finite video pixels")
        frames = video[0].permute(0, 2, 3, 1).clamp(0, 1).mul(255).to(torch.uint8).cpu()
    return frames.numpy()


def write_video(frames, destination: Path, options: dict[str, Any]) -> None:
    """Encode with upstream's PyAV fallback and publish only a complete MP4."""
    import av
    import numpy as np

    expected = (options["num_frames"], options["height"], options["width"], 3)
    if frames.shape != expected or frames.dtype != np.uint8:
        raise ValueError(f"Zing RGB frames must be uint8 with shape {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.mp4")
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream("libx264", rate=options["fps"])
            stream.width = options["width"]
            stream.height = options["height"]
            stream.pix_fmt = "yuv420p"
            for pixels in frames:
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(temporary)) as container:
            stream = container.streams.video[0]
            if (stream.frames != options["num_frames"] or stream.average_rate != options["fps"]
                    or stream.width != options["width"] or stream.height != options["height"]):
                raise RuntimeError("Encoded Zing video has incorrect frame count, FPS, or dimensions")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_batch(repo_root: Path, checkpoint_root: Path, rows: list[dict[str, Any]],
              generation: dict[str, Any]) -> int:
    options = runtime_options(generation)
    ids = [str(row["sample_id"]).strip() for row in rows]
    outputs = [Path(row["output_path"]).expanduser().resolve() for row in rows]
    if not ids or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("Zing batch requires unique non-empty sample IDs")
    if len(set(outputs)) != len(outputs) or any(path.suffix.lower() != ".mp4" for path in outputs):
        raise ValueError("Zing batch requires unique .mp4 output paths")
    # Validate every input before allocating model memory.
    for row in rows:
        build_message(row, options)
    paths = resolve_runtime_paths(repo_root, checkpoint_root)
    revision = validate_revision(repo_root)
    log_pipeline("checkpoint_load_start", label="Zing-0.5")
    runtime = load_runtime(repo_root, paths, options)
    log_pipeline("checkpoint_load_done", label="Zing-0.5", checkpoint_load_policy="load_once")
    failures = 0
    consecutive_failures = 0
    for index, (row, output) in enumerate(zip(rows, outputs), start=1):
        sample_id = str(row["sample_id"])
        begin_sample(sample_id, index=index, total=len(rows), label="Zing-0.5")
        try:
            seed = sample_seed(options["seed"], sample_id)
            message = build_message(row, options)
            frames = generate_frames(runtime, message, options, seed)
            write_video(frames, output, options)
            del frames
            _, approximations = keyboard_actions(row.get("camera_path", []), options["num_frames"])
            metadata = {
                "sample_id": sample_id, "upstream_revision": revision,
                "checkpoint_dir": str(checkpoint_root), **options,
                "seed": seed, "base_seed": options["seed"],
                "camera_path": row.get("camera_path", []), "action_order": ACTION_ORDER,
                "action_approximations": approximations,
                "generated_frames": options["num_frames"] - 1,
                "duration_seconds": options["num_frames"] / options["fps"],
                "conditioning_image": message["messages"][1]["uri"],
                "prompt": message["messages"][0]["content"],
            }
            sidecar = output.with_suffix(".zing.json")
            temporary = sidecar.with_name(f".{sidecar.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
                temporary.replace(sidecar)
            finally:
                temporary.unlink(missing_ok=True)
            print_status(sample_id, "generated", output_path=str(output), seed=seed,
                         num_frames=options["num_frames"], fps=options["fps"], provenance=str(sidecar))
            consecutive_failures = 0
        except Exception as exc:
            failures += 1
            consecutive_failures += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
            if consecutive_failures >= 3:
                break
    # A stale output must never turn a failed generation into success in the parent.
    return int(failures > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_batch_args(parser)
    args = parser.parse_args()
    if not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required")
    return run_batch(
        Path(args.repo_root).expanduser().resolve(), Path(args.checkpoint_dir).expanduser().resolve(),
        load_batch_spec(Path(args.batch_spec_path)), load_json(Path(args.generation_config_path)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
