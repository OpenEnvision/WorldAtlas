"""One official Astronex pipeline per WorldAtlas GPU shard."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import traceback
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args, begin_sample, load_batch_spec, load_json, print_status,
)
from .astronex_world import UPSTREAM_REVISION, WEIGHTS_REVISION, camera_controls, runtime_options, sample_seed


def validate_source(repo_root: Path) -> str:
    required = ["astronex_env.py", "inference/causal.yaml", "inference/bidirectional.yaml",
                "inference/pipelines/causal_diffusion_inference.py",
                "utils/default_config.yaml", "utils/checkpoint_io.py"]
    missing = [name for name in required if not (repo_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Astronex source missing: {missing}")
    revision = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Astronex requires source revision {UPSTREAM_REVISION}, got {revision}")
    changed = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if changed:
        raise ValueError("Astronex upstream checkout has tracked modifications")
    return revision


def validate_weights(root: Path) -> None:
    required = [root / name for name in (
        "transformer/config.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json", "tokenizer/tokenizer_config.json", "tokenizer/spiece.model",
        "model.safetensors.index.json", "text_encoder/model.safetensors.index.json",
    )]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Astronex checkpoint incomplete: " + ", ".join(missing))
    for index in (root / "model.safetensors.index.json", root / "text_encoder/model.safetensors.index.json"):
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        if not shards:
            raise ValueError(f"empty Astronex checkpoint index: {index}")
        for name in shards:
            shard = index.parent / name
            if not shard.is_file() or shard.stat().st_size == 0:
                raise FileNotFoundError(f"Astronex checkpoint shard missing: {shard}")


def prepare_config(repo_root: Path, weight_root: Path, options: dict) -> dict:
    config = yaml.safe_load((repo_root / "utils/default_config.yaml").read_text())
    config.update(yaml.safe_load((repo_root / "inference" / f"{options['mode']}.yaml").read_text()))
    config.update(sampling_steps=options["sampling_steps"], guidance_scale=options["guidance_scale"])
    config["image_or_video_shape"] = [1, options["total_latents"], 48, options["height"] // 16, options["width"] // 16]
    config["model_kwargs"].update(model_root=str(weight_root), model_name="transformer", base_dtype="bfloat16")
    if options["mode"] == "causal":
        config.update(free_cache_before_decode=options["free_cache_before_decode"],
                      offload_denoiser_for_decode=options["offload_denoiser_for_decode"])
    return config


def load_generator_weights(generator: Any, state: dict) -> None:
    if "generator" not in state:
        raise ValueError("Astronex release must contain generator weights (EMA fallback is not implicit)")
    weights = {key.replace("model._fsdp_wrapped_module.", "model.", 1): value
               for key, value in state["generator"].items()}
    # The six zero-initialized inverse-dynamics head tensors are shipped but
    # intentionally inactive for video inference in the official configs.
    weights = {key: value for key, value in weights.items() if not key.startswith("model.action_head.")}
    generator.load_state_dict(weights, strict=True)


def load_runtime(repo_root: Path, weight_root: Path, config: dict, options: dict):
    os.environ["ASTRONEX_WEIGHTS"] = str(weight_root)
    if options["offline"]:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Astronex I2V requires one GPU per worker; use independent shards")
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Astronex inference requires CUDA")
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 9):
        raise RuntimeError("Use Astronex's dedicated PyTorch 2.9.1 environment")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    sys.path.insert(0, str(repo_root))
    from inference.pipelines import CausalDiffusionInferencePipeline, BidirectionalDiffusionInferencePipeline
    from inference.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
    from utils.checkpoint_io import load_checkpoint

    pipeline_type = CausalDiffusionInferencePipeline if options["mode"] == "causal" else BidirectionalDiffusionInferencePipeline
    low_memory = get_cuda_free_memory_gb(device) < 40
    pipeline = pipeline_type(OmegaConf.create(config), device=device)
    load_generator_weights(pipeline.generator, load_checkpoint(weight_root, map_location="cpu"))
    pipeline = pipeline.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.vae_device = None
    gc.collect()
    return pipeline


def validate_sample(row: dict, options: dict) -> None:
    from PIL import Image

    for key in ("sample_id", "conditioning_image", "output_path", "prompt"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Astronex sample requires nonempty {key}")
    seed = row.get("seed", sample_seed(row["sample_id"], options))
    if int(seed) != seed or not 0 <= seed < 2**32:
        raise ValueError("Astronex sample seed must be an integer in [0, 2**32)")
    event = row.get("event_prompt")
    start = row.get("event_start_frame", 0)
    if int(start) != start:
        raise ValueError("event_start_frame must be an integer latent index")
    if event is not None and (not isinstance(event, str) or not event.strip()):
        raise ValueError("event_prompt must be nonempty text or null")
    if event and (options["mode"] != "causal" or start <= 0 or start % 8 or start >= options["total_latents"]):
        raise ValueError("Astronex timed events require a causal block boundary inside the rollout")
    if not event and start:
        raise ValueError("event_start_frame requires event_prompt")
    with Image.open(row["conditioning_image"]) as image:
        image.convert("RGB").load()


def generate_frames(pipeline: Any, row: dict, controls: dict, options: dict, *, device: str = "cuda:0"):
    import torch
    from PIL import Image
    from torchvision import transforms

    validate_sample(row, options)
    seed = int(row.get("seed", sample_seed(str(row["sample_id"]), options)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transform = transforms.Compose([
        transforms.Resize((options["height"], options["width"])),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    with Image.open(row["conditioning_image"]) as image:
        pixels = transform(image.convert("RGB")).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        initial = pipeline.vae.encode_to_latent(pixels.to(device=device, dtype=torch.bfloat16)).to(
            device=device, dtype=torch.bfloat16)
        # Causal concatenates the input latent; bidirectional replaces noise[0].
        noise_frames = options["generated_latents"] if options["mode"] == "causal" else options["total_latents"]
        noise = torch.randn(1, noise_frames, 48, options["height"] // 16, options["width"] // 16,
                            device=device, dtype=torch.bfloat16)
        event = row.get("event_prompt")
        start = int(row.get("event_start_frame", 0))
        try:
            video = pipeline.inference(
                noise=noise, text_prompts=[str(row["prompt"])], initial_latent=initial,
                viewmats=torch.from_numpy(controls["viewmats"]).unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                Ks=torch.from_numpy(controls["Ks"]).unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                event_prompts=[str(event)] if event else None, event_start_frame=start,
                return_latents=False,
            )
        finally:
            # Upstream restores the denoiser only after a successful VAE decode.
            # Restore on decode errors too so a resident worker can continue.
            if options["offload_denoiser_for_decode"]:
                pipeline.generator.to(device=device)
        expected = (1, options["output_frames"], 3, options["height"], options["width"])
        if tuple(video.shape) != expected or not torch.isfinite(video).all():
            raise RuntimeError(f"Astronex invalid video shape/pixels: {tuple(video.shape)}, expected {expected}")
        return video[0].permute(0, 2, 3, 1).clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()


def release_sample_state(pipeline: Any) -> None:
    import torch

    pipeline.vae.model.clear_cache()
    for name in getattr(pipeline, "_CACHES", ()):
        setattr(pipeline, name, None)
    for name in ("_prope_mirror", "_prev_retrieved"):
        if hasattr(pipeline, name):
            setattr(pipeline, name, [])
    for name in ("_cache_keep", "_sink_promote"):
        if hasattr(pipeline, name):
            setattr(pipeline, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_video(frames: np.ndarray, destination: Path, options: dict) -> None:
    import av

    expected = (options["output_frames"], options["height"], options["width"], 3)
    if frames.shape != expected or frames.dtype != np.uint8:
        raise ValueError(f"Astronex output array must be uint8 {expected}, got {frames.shape}/{frames.dtype}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.mp4")
    try:
        with av.open(str(temporary), "w") as container:
            stream = container.add_stream("libx264", rate=options["fps"])
            stream.width, stream.height = options["width"], options["height"]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18"}
            for pixels in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(temporary)) as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (options["width"], options["height"]) or abs(float(stream.average_rate) - options["fps"]) > 0.01:
                raise RuntimeError("Astronex encoded video dimensions/FPS mismatch")
            if sum(1 for _ in container.decode(stream)) != options["output_frames"]:
                raise RuntimeError("Astronex encoded frame count mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_batch(rows: list[dict], repo_root: Path, weight_root: Path, generation: dict,
              work: Path, *, dry_run: bool = False) -> int:
    options = runtime_options(generation)
    revision = validate_source(repo_root)
    if not dry_run:
        validate_weights(weight_root)
    config = prepare_config(repo_root, weight_root, options)
    work.mkdir(parents=True, exist_ok=True)
    config_file = work / "inference.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pipeline = None
    failures = 0
    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        begin_sample(sample_id, index=index + 1, total=len(rows), label="Astronex-World")
        runtime_failed = False
        try:
            validate_sample(row, options)
            controls = camera_controls(list(row["camera_tokens"]), options)
            camera_path = work / f"camera_{index:06d}.npz"
            np.savez(camera_path, **controls)
            if dry_run:
                print_status(sample_id, "dry_run", dry_run=True, config_path=str(config_file),
                             camera_path=str(camera_path), total_latents=options["total_latents"],
                             expected_frames=options["output_frames"])
                continue
            if pipeline is None:
                runtime_failed = True
                pipeline = load_runtime(repo_root, weight_root, config, options)
                runtime_failed = False
                print("[Astronex-World] checkpoint_loads=1", flush=True)
            try:
                frames = generate_frames(pipeline, row, controls, options)
            finally:
                release_sample_state(pipeline)
            output = Path(row["output_path"]).expanduser().resolve()
            write_video(frames, output, options)
            del frames
            record = {
                "sample_id": sample_id, "upstream_revision": revision,
                "release_weights_revision": WEIGHTS_REVISION, "weight_root": str(weight_root),
                "mode": options["mode"], "num_frames": options["output_frames"], "fps": options["fps"],
                "total_latents": options["total_latents"], "seed": int(row.get("seed", sample_seed(sample_id, options))),
                "camera_tokens": row["camera_tokens"], "camera_npz": str(camera_path),
                "prompt": row["prompt"], "event_prompt": row.get("event_prompt"),
                "event_start_latent": int(row.get("event_start_frame", 0)),
                "runtime_config": str(config_file), "checkpoint_load_count": 1,
            }
            sidecar = output.with_suffix(".astronex.json")
            temporary = sidecar.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(sidecar)
            print_status(sample_id, "generated", output_path=str(output), num_frames=options["output_frames"])
        except Exception as exc:
            failures += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
            if not options["continue_on_error"] or runtime_failed:
                break
    return int(failures > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Astronex-World persistent batch inference")
    add_common_batch_args(parser)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required")
    spec = Path(args.batch_spec_path).resolve()
    work = Path(tempfile.mkdtemp(prefix="astronex_", dir=spec.parent))
    return run_batch(load_batch_spec(spec), Path(args.repo_root).resolve(), Path(args.checkpoint_dir).resolve(),
                     load_json(Path(args.generation_config_path)), work, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
