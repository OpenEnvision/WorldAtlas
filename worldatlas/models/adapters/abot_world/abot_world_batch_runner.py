"""Long-lived ABot-World subprocess runner used by the WorldAtlas adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
from uuid import uuid4

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    begin_sample,
    print_status,
)


UPSTREAM_REQUIRED_FILES = (
    "configs/default_config.yaml",
    "configs/long_forcing_dmd.yaml",
    "pipeline/causal_inference.py",
    "utils/wan_wrapper.py",
    "web_client/pipeline_loader.py",
)
CHECKPOINT_REQUIRED_FILES = (
    "Wan2.2_VAE.pth",
    "taew2_2.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "diffusion_pytorch_model.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABot-World WorldAtlas batch runner")
    add_common_batch_args(parser)
    return parser.parse_args()


def _bool_value(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_repo_path(repo_root: Path, value: str | None, default: str) -> Path:
    path = Path(str(value or default)).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _validate_layout(repo_root: Path, checkpoint_dir: Path) -> None:
    missing_repo = [name for name in UPSTREAM_REQUIRED_FILES if not (repo_root / name).is_file()]
    if missing_repo:
        raise FileNotFoundError(
            f"ABot-World repo is incomplete at {repo_root}; missing: {', '.join(missing_repo)}"
        )
    missing_checkpoint = [
        name for name in CHECKPOINT_REQUIRED_FILES if not (checkpoint_dir / name).is_file()
    ]
    tokenizer_dir = checkpoint_dir / "google" / "umt5-xxl"
    if not tokenizer_dir.is_dir():
        missing_checkpoint.append("google/umt5-xxl/")
    if missing_checkpoint:
        raise FileNotFoundError(
            f"ABot-World checkpoint is incomplete at {checkpoint_dir}; missing: "
            + ", ".join(missing_checkpoint)
        )


def _patch_checkpoint_paths(config: Any, checkpoint_dir: Path) -> None:
    """Redirect every path in the upstream merged config to WorldAtlas's ckpt root."""
    config.taew2_2_checkpoint = str(checkpoint_dir / "taew2_2.pth")
    config.lightvae_encoder_checkpoint = str(checkpoint_dir / "Wan2.2_VAE.pth")
    config.model_kwargs.model_name = str(checkpoint_dir)
    config.text_encoder_kwargs.tokenizer_path = str(checkpoint_dir / "google" / "umt5-xxl")
    config.text_encoder_kwargs.encoder_pth_path = str(
        checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    config.vae_kwargs.pretrained_path = str(checkpoint_dir / "Wan2.2_VAE.pth")


def _load_upstream_config(
    repo_root: Path,
    checkpoint_dir: Path,
    generation: dict[str, Any],
) -> Any:
    from omegaconf import OmegaConf

    default_path = _resolve_repo_path(
        repo_root,
        generation.get("default_config_path"),
        "configs/default_config.yaml",
    )
    model_path = _resolve_repo_path(
        repo_root,
        generation.get("model_config_path"),
        "configs/long_forcing_dmd.yaml",
    )
    config = OmegaConf.merge(OmegaConf.load(default_path), OmegaConf.load(model_path))
    _patch_checkpoint_paths(config, checkpoint_dir)
    if generation.get("vae_type") is not None:
        config.vae_type = str(generation["vae_type"])
    if generation.get("use_fp8_gemm") is not None:
        config.use_fp8_gemm = _bool_value(generation, "use_fp8_gemm", True)
    if generation.get("quant_type") is not None:
        config.quant_type = str(generation["quant_type"])
    return config


def _load_pipeline(
    repo_root: Path,
    checkpoint_dir: Path,
    generation: dict[str, Any],
):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ABot-World requires a CUDA GPU")

    from pipeline import CausalInferencePipeline
    from utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
    from utils.misc import set_seed
    from utils.wan_wrapper import create_vae_from_config
    from wan.modules.helios_kernels import (
        replace_all_norms_with_flash_norms,
        replace_rope_with_flash_rope,
    )

    config = _load_upstream_config(repo_root, checkpoint_dir, generation)
    device = torch.device(str(generation.get("device", "cuda:0")))
    set_seed(int(generation.get("seed", 42)))
    torch.set_grad_enabled(False)

    vae = create_vae_from_config(config)
    pipeline = CausalInferencePipeline(config, device=device, vae=vae)
    replace_all_norms_with_flash_norms(pipeline.generator.model)
    replace_rope_with_flash_rope()
    pipeline = pipeline.to(dtype=torch.bfloat16)

    threshold = float(generation.get("low_memory_threshold_gb", 40.0))
    low_memory = get_cuda_free_memory_gb(device) < threshold
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # The full Wan VAE is used only to encode the conditioning frame.  Upstream
    # leaves it on CPU to fit a 32 GB RTX 5090; an 80 GB A100 can keep it on
    # CUDA, which avoids a very slow bfloat16 CPU encode during batch rollout.
    encoder_device = str(generation.get("encoder_device", "cpu")).strip().lower()
    if pipeline.encoder is not None and encoder_device not in {"", "cpu"}:
        pipeline.encoder.to(device=device)

    use_fp8 = _bool_value(
        generation,
        "use_fp8_gemm",
        bool(getattr(config, "use_fp8_gemm", True)),
    )
    quant_type = str(generation.get("quant_type", getattr(config, "quant_type", "fp8-per-token")))
    if use_fp8 and quant_type.lower() != "none":
        # Keep torchao optional for the BF16/A100 runtime.  Importing it against
        # the upstream torch pin otherwise emits an extension compatibility
        # warning even though no FP8 operation is requested.
        from quantizer import apply_fp8_quantization

        apply_fp8_quantization(model=pipeline.generator.model, quant_type=quant_type)

    pipeline.torch_dtype = torch.bfloat16
    return pipeline, config, device, low_memory


def _reference_cache_dir(
    repo_root: Path,
    image_path: Path,
    generation: dict[str, Any],
) -> Path:
    root_value = generation.get("reference_cache_root")
    if root_value:
        root = _resolve_repo_path(repo_root, str(root_value), "outputs/ref_image_cache")
    else:
        root = repo_root / "outputs" / "ref_image_cache"
    digest = hashlib.md5(image_path.read_bytes()).hexdigest()[:16]
    return root / digest


def _clear_streaming_vae_cache(pipeline: Any) -> None:
    vae = getattr(pipeline, "vae", None)
    model = getattr(vae, "model", None)
    if model is not None and hasattr(model, "clear_cache"):
        model.clear_cache()
        return
    taehv = getattr(vae, "taehv", None)
    if taehv is not None and hasattr(taehv, "reset"):
        taehv.reset()


def _run_request(
    *,
    pipeline: Any,
    device: Any,
    repo_root: Path,
    row: dict[str, Any],
    generation: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import torch

    image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"conditioning image not found: {image_path}")
    output_path = Path(str(row["output_path"])).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    block_actions = list(row.get("block_actions") or [])
    if not block_actions:
        raise ValueError("ABot-World request has no block_actions")

    base_seed = int(row.get("seed", generation.get("seed", 42)))
    increment_seed = _bool_value(generation, "increment_seed", True)
    seed = base_seed + row_index if increment_seed else base_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    prompt = str(row.get("prompt") or "").strip()
    height = int(generation.get("height", 704))
    width = int(generation.get("width", 1280))
    fps = int(generation.get("fps", 12))
    codec = str(generation.get("codec", "libx264"))
    crf = int(generation.get("crf", 18))
    preset = str(generation.get("preset", "fast"))

    pipeline.set_prompts([prompt], device=device)
    ref_cache_dir = _reference_cache_dir(repo_root, image_path, generation)
    pipeline.set_ref_latent_mask_from_exists_paths(ref_dir=str(ref_cache_dir), device=device)
    pipeline.reset_stream(batch_size=1, dtype=torch.bfloat16, device=device, initial_latent=None)
    pipeline.set_first_frame_latent(
        str(image_path),
        height=height,
        width=width,
        device=device,
    )

    num_fpb = int(pipeline.num_frame_per_block)
    vae_for_shape = pipeline.encoder if pipeline.encoder is not None else pipeline.vae
    upsampling = int(getattr(vae_for_shape, "upsampling_factor", 16))
    latent_channels = int(vae_for_shape.z_dim)
    latent_shape = (
        1,
        num_fpb,
        latent_channels,
        height // upsampling,
        width // upsampling,
    )

    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.tmp{output_path.suffix}"
    )
    writer = imageio.get_writer(
        str(temporary_path),
        fps=fps,
        format="FFMPEG",
        codec=codec,
        ffmpeg_params=[
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
        ],
    )
    frame_count = 0
    started_at = time.perf_counter()
    succeeded = False
    try:
        for block_index, raw_action in enumerate(block_actions):
            action = {str(key).upper(): bool(value) for key, value in dict(raw_action).items()}
            pipeline.set_act(
                action,
                height=height,
                width=width,
                num_frames=num_fpb,
                device=device,
            )
            noise = torch.randn(latent_shape, device=device, dtype=torch.bfloat16)
            latents = pipeline.generate_next_block(noise)
            if latents is None:
                raise RuntimeError(f"ABot-World returned no latents for block {block_index}")
            frames = []

            class _Writer:
                def append_data(self, frame) -> None:
                    frames.append(frame)

            pipeline.decode_block_and_write(latents, _Writer())
            for frame in frames:
                writer.append_data(frame)
                frame_count += 1
        succeeded = True
    finally:
        try:
            writer.close()
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        # Do not leave corrupt hidden MP4s behind when inference or encoding
        # fails midway through a sample.
        if not succeeded:
            temporary_path.unlink(missing_ok=True)

    if frame_count <= 0 or not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"ABot-World did not render any frames for {row['sample_id']}")
    temporary_path.replace(output_path)
    elapsed = time.perf_counter() - started_at
    _clear_streaming_vae_cache(pipeline)
    return {
        "prediction_path": str(output_path),
        "frame_count": frame_count,
        "fps": fps,
        "block_count": len(block_actions),
        "seed": seed,
        "wall_time_seconds": round(elapsed, 6),
        "reference_cache_dir": str(ref_cache_dir),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not args.checkpoint_dir:
        raise ValueError("ABot-World runner requires --checkpoint_dir")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    spec_path = Path(args.batch_spec_path).expanduser().resolve()
    generation_path = Path(args.generation_config_path).expanduser().resolve()
    rows = load_batch_spec(spec_path)
    generation = load_json(generation_path)
    _validate_layout(repo_root, checkpoint_dir)

    if os.environ.get("WORLDATLAS_ABOT_WORLD_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "persistent_engine": True,
                    "checkpoint_load_count": 1,
                    "request_count": len(rows),
                    "repo_root": str(repo_root),
                    "checkpoint_dir": str(checkpoint_dir),
                    "blocks": [len(row.get("block_actions") or []) for row in rows],
                }
            )
        )
        return

    pipeline, _, device, low_memory = _load_pipeline(repo_root, checkpoint_dir, generation)
    print(
        "[WorldAtlas:ABotWorld] persistent pipeline ready; checkpoint_loads=1",
        flush=True,
    )
    continue_on_error = _bool_value(generation, "continue_on_error", True)
    for row_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        begin_sample(
            sample_id,
            index=row_index + 1,
            total=len(rows),
            label="ABot-World",
        )
        try:
            result = _run_request(
                pipeline=pipeline,
                device=device,
                repo_root=repo_root,
                row=row,
                generation=generation,
                row_index=row_index,
            )
            print_status(
                sample_id,
                "generated",
                low_memory=low_memory,
                persistent_engine=True,
                checkpoint_load_count=1,
                **result,
            )
        except Exception as exc:
            print_status(
                sample_id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            _clear_streaming_vae_cache(pipeline)
            if not continue_on_error:
                raise


if __name__ == "__main__":
    main()
