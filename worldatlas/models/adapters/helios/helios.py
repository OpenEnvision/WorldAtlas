"""Persistent official Helios I2V adapter and dependency-free runtime contract."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from worldatlas.models.adapters.external_batch import ExternalBatchAdapter

UPSTREAM_REVISION = "babed9811266e4b5b111c9c1e0977a07899066ab"
NATIVE_FPS = 24
CHUNK_FRAMES = 33
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, "
    "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
    "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
    "many people in the background, walking backwards"
)


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != float(value) or int(value) <= 0:
        raise ValueError(f"Helios {name} must be a positive integer")
    return int(value)


def frame_plan(generation: dict[str, Any]) -> dict[str, int]:
    fps = positive_int(generation.get("fps", NATIVE_FPS), "fps")
    if fps != NATIVE_FPS:
        raise ValueError("Helios runs at native 24 FPS; changing container FPS would change task duration")
    seconds = generation.get("target_duration_seconds")
    if seconds is not None:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Helios target_duration_seconds must be positive and finite")
        requested = math.ceil(seconds * fps)
    else:
        requested = positive_int(generation.get("num_frames", 120), "num_frames")
    # Official pipeline generates 33-frame chunks, then trims the whole video
    # to 4*k+1 frames. Account for BOTH operations before exact tail cropping.
    vae_minimum = ((requested - 1 + 3) // 4) * 4 + 1
    inference_frames = ((vae_minimum + CHUNK_FRAMES - 1) // CHUNK_FRAMES) * CHUNK_FRAMES
    return {"fps": fps, "requested_frames": requested, "inference_frames": inference_frames,
            "native_output_frames": ((inference_frames - 1) // 4) * 4 + 1}


def sample_seed(base_seed: int, sample_id: str) -> int:
    """Stable under sharding, skipped samples and resume; never use row index."""
    value = hashlib.sha256(f"helios:{base_seed}:{sample_id}".encode()).digest()
    return int.from_bytes(value[:8], "big") % (2**63 - 1)


def runtime_options(generation: dict[str, Any]) -> dict[str, Any]:
    allowed = {"variant", "fps", "num_frames", "target_duration_seconds", "width", "height",
               "seed", "guidance_scale", "num_inference_steps", "negative_prompt",
               "image_noise_sigma_min", "image_noise_sigma_max", "attention_backend",
               "use_official_kernels", "enable_low_vram_mode", "require_persistent_batch",
               "nproc_per_node"}
    unknown = set(generation) - allowed
    if unknown:
        raise ValueError(f"Unsupported Helios options (not silently ignored): {sorted(unknown)}")
    if positive_int(generation.get("nproc_per_node", 1), "nproc_per_node") != 1:
        raise ValueError("Helios adapter uses one GPU per worker; parallelize independent manifest shards")
    variant = str(generation.get("variant", "distilled"))
    if variant not in {"base", "distilled"}:
        raise ValueError("Helios variant must be base or distilled")
    distilled = variant == "distilled"
    if distilled and "num_inference_steps" in generation:
        raise ValueError("Distilled uses fixed official pyramid steps [2,2,2], not num_inference_steps")
    width = positive_int(generation.get("width", 640), "width")
    height = positive_int(generation.get("height", 384), "height")
    divisor = 64 if distilled else 16
    if width % divisor or height % divisor:
        raise ValueError(f"Helios {variant} width/height must be divisible by {divisor}")
    guidance = float(generation.get("guidance_scale", 1.0 if distilled else 5.0))
    if not math.isfinite(guidance) or guidance < 1 or (distilled and guidance != 1):
        raise ValueError("Helios requires finite CFG >=1; Distilled requires guidance_scale=1")
    low = float(generation.get("image_noise_sigma_min", 0.111))
    high = float(generation.get("image_noise_sigma_max", 0.135))
    if not 0 <= low <= high <= 1:
        raise ValueError("Helios image noise must satisfy 0 <= min <= max <= 1")
    for key in ("use_official_kernels", "enable_low_vram_mode", "require_persistent_batch"):
        if key in generation and not isinstance(generation[key], bool):
            raise ValueError(f"Helios {key} must be a YAML boolean")
    attention = str(generation.get("attention_backend", "native"))
    if attention not in {"native", "auto", "flash_hub", "_flash_3_hub"}:
        raise ValueError(f"Unsupported Helios attention backend: {attention}")
    plan = frame_plan(generation)
    seed = generation.get("seed", 42)
    if isinstance(seed, bool) or not math.isfinite(float(seed)) or int(seed) != float(seed) or not 0 <= int(seed) < 2**63:
        raise ValueError("Helios seed must be an integer in [0, 2**63)")
    return {
        **plan, "variant": variant, "seed": int(seed),
        "attention_backend": attention,
        "use_official_kernels": generation.get("use_official_kernels", False),
        "enable_low_vram_mode": generation.get("enable_low_vram_mode", False),
        "pipeline_kwargs": {
            "height": height, "width": width, "num_frames": plan["inference_frames"],
            "num_inference_steps": positive_int(generation.get("num_inference_steps", 50), "num_inference_steps"),
            "guidance_scale": guidance, "negative_prompt": str(generation.get("negative_prompt", NEGATIVE_PROMPT)),
            "history_sizes": [16, 2, 1], "num_latent_frames_per_chunk": 9,
            "keep_first_frame": True, "is_enable_stage2": distilled,
            "pyramid_num_inference_steps_list": [2, 2, 2] if distilled else [20, 20, 20],
            "is_amplify_first_chunk": distilled, "is_skip_first_chunk": False,
            "use_zero_init": False, "zero_steps": 1,
            "image_noise_sigma_min": low, "image_noise_sigma_max": high,
            "output_type": "np",
        },
    }


class HeliosAdapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.helios.helios_batch_runner"
    label = "Helios"

    def __init__(self, config):
        super().__init__(config)
        runtime_options(config.generation)

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        if self.config.checkpoint_dir is None:
            raise ValueError("Helios requires a local checkpoint_dir")
        runtime_options(self.config.generation)
        return super()._build_command(spec_path, generation_path)
