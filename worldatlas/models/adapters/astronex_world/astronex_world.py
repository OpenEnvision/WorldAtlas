"""Image/camera/event controls for the released Astronex-World 5B model."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from worldatlas.benchmark.synthetic_camera import camera_delta, synthetic_camera_matrices
from worldatlas.models.adapters.alayaworld.alayaworld import _camera_tokens_for_sample
from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter

UPSTREAM_REVISION = "27584bf1a89da01ef35a03b48d83bc818d2adf2b"
WEIGHTS_REVISION = "f06ea330f2ce5cabef5fbf175287fe323ec1218c"


def runtime_options(generation: dict[str, Any]) -> dict[str, Any]:
    mode = generation.get("mode", "causal")
    if mode not in {"causal", "bidirectional"}:
        raise ValueError("Astronex mode must be causal or bidirectional")
    options = dict(
        mode=mode, num_frames=125 if mode == "causal" else 65,
        width=832, height=480, fps=24, sampling_steps=8 if mode == "causal" else 50,
        guidance_scale=3.0 if mode == "causal" else 8.0, seed=1234,
        nproc_per_node=1, offline=True, continue_on_error=True,
        translation_per_latent=0.08, rotation_degrees_per_latent=3.0,
        camera_path_mode="native_steps", translation_per_token=0.8,
        rotation_degrees_per_token=30.0,
        intrinsic=[0.5050505, 0.89786756, 0.5, 0.5],
        free_cache_before_decode=True, offload_denoiser_for_decode=False,
    )
    options.update(generation)
    for key in ("width", "height", "num_frames", "fps", "sampling_steps", "nproc_per_node"):
        value = int(options[key])
        if float(options[key]) != value or value <= 0:
            raise ValueError(f"Astronex {key} must be a positive integer")
        options[key] = value
    if options["width"] % 32 or options["height"] % 32:
        raise ValueError("Astronex dimensions must be multiples of 32")
    if options["fps"] != 24 or options["nproc_per_node"] != 1:
        raise ValueError("Astronex image inference uses 24 FPS and one GPU per worker")
    seconds = options.get("target_duration_seconds")
    if seconds is not None:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Astronex target_duration_seconds must be finite and positive")
        options["num_frames"] = max(5, math.ceil(seconds * options["fps"]))
    if options["num_frames"] < 5:
        raise ValueError("Astronex requires at least 5 requested pixel frames")
    for key in ("offline", "continue_on_error", "free_cache_before_decode", "offload_denoiser_for_decode"):
        if not isinstance(options[key], bool):
            raise ValueError(f"{key} must be a YAML boolean")
    if options["camera_path_mode"] not in {"native_steps", "closed_loop"}:
        raise ValueError("Astronex camera_path_mode must be native_steps or closed_loop")
    for key in ("translation_per_latent", "rotation_degrees_per_latent", "guidance_scale",
                "translation_per_token", "rotation_degrees_per_token"):
        options[key] = float(options[key])
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    # Wan2.2 VAE: pixels = 1 + 4*(total_latents-1). Causal I2V counts
    # the reference latent as part of an 8-latent block, unlike CLI --frames.
    total = math.ceil((options["num_frames"] - 1) / 4) + 1
    if mode == "causal":
        total = math.ceil(total / 8) * 8
    options["total_latents"] = total
    options["generated_latents"] = total - 1
    options["output_frames"] = 1 + 4 * (total - 1)
    return options


def camera_controls(tokens: list[str], options: dict) -> dict[str, np.ndarray]:
    intervals = options["generated_latents"]
    if not tokens or len(tokens) > intervals:
        raise ValueError("camera path must contain between one and generated_latents actions")
    closed_loop = options["camera_path_mode"] == "closed_loop"
    step = options["translation_per_token"] if closed_loop else options["translation_per_latent"]
    rotation = options["rotation_degrees_per_token"] if closed_loop else options["rotation_degrees_per_latent"]
    trajectory_options = {
        "synthetic_camera_forward_step": step, "synthetic_camera_lateral_step": step,
        # WorldAtlas's synthetic Y-up translations must become OpenCV Y-down.
        "synthetic_camera_vertical_step": -step,
        "synthetic_camera_rotation_degrees": rotation,
        "synthetic_camera_tilt_degrees": rotation,
        "synthetic_camera_roll_degrees": rotation,
        "synthetic_camera_orbit_radius": float(options.get("orbit_radius", 2.0)),
    }
    if closed_loop:
        # Resample fixed token endpoints, rather than repeating a native step
        # for unequal segment lengths. This preserves inverse legs and 90/360
        # degree turns even when 367 latent intervals cannot divide the path.
        c2w = synthetic_camera_matrices(
            tokens, target_frames=intervals + 1, runtime=trajectory_options,
        ).matrices
        if not np.allclose(c2w[-1], c2w[0], atol=1e-4, rtol=0):
            raise ValueError("Astronex closed_loop camera path does not return to its initial pose")
    else:
        base, extra = divmod(intervals, len(tokens))
        actions = [token for i, token in enumerate(tokens) for _ in range(base + (i < extra))]
        current = np.eye(4, dtype=np.float32)
        c2w = [current.copy()]
        for token in actions:
            current = current @ camera_delta(token, trajectory_options)
            c2w.append(current.copy())
    viewmats = np.linalg.inv(np.asarray(c2w)).astype(np.float32)
    intrinsic = np.asarray(options["intrinsic"], dtype=np.float32)
    if intrinsic.shape != (4,) or not np.isfinite(intrinsic).all() or np.any(intrinsic[:2] <= 0):
        raise ValueError("Astronex intrinsic must be normalized [fx, fy, cx, cy] with positive focal lengths")
    fx, fy, cx, cy = intrinsic
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return {"viewmats": viewmats, "Ks": np.repeat(K[None], len(c2w), axis=0)}


def sample_seed(sample_id: str, options: dict) -> int:
    offset = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:4], "big")
    return (int(options["seed"]) + offset) % (2**31)


class AstronexWorldAdapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.astronex_world.astronex_world_batch_runner"
    label = "Astronex-World"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        options = runtime_options(dict(self.config.generation))
        row = super().batch_spec_payload(request)
        tokens, source = _camera_tokens_for_sample(request.sample)
        camera_controls(tokens, options)  # Fail before loading weights if controls are invalid.
        row.update(camera_tokens=tokens, control_source=source,
                   seed=sample_seed(request.sample.sample_id, options),
                   num_frames=options["output_frames"], total_latents=options["total_latents"])
        # Static-suite segments describe successive camera views, not timed
        # scene events. Keep the complete target caption and drive all segments
        # through the camera trajectory, including three-segment itineraries.
        if request.sample.suite == "image_static":
            row.update(event_prompt=None, event_start_frame=0)
            return row
        sequence = []
        for prompt in request.sample.prompt_sequence:
            prompt = str(prompt).strip()
            if prompt and (not sequence or sequence[-1] != prompt):
                sequence.append(prompt)
        if len(sequence) > 2:
            raise ValueError("Astronex supports one mid-rollout event (at most two prompt segments)")
        if len(sequence) == 2:
            if options["mode"] != "causal":
                raise ValueError("Astronex bidirectional does not support timed prompt switching; use causal")
            row["prompt"] = sequence[0]
            row["event_prompt"] = sequence[1]
            # Events apply on causal block boundaries (including the reference).
            total = options["total_latents"]
            if total < 16:
                raise ValueError("a timed Astronex event needs at least two causal blocks")
            row["event_start_frame"] = max(8, (total // 2 // 8) * 8)
        else:
            row.update(event_prompt=None, event_start_frame=0)
        return row
