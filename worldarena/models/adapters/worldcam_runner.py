"""Subprocess entry point that runs WorldCam inference inside the model repo.

Loaded by ``WorldCamAdapter`` via ``python -m worldarena.models.adapters.worldcam_runner``.
The runner bootstraps Wan2.1 base weights, loads fine-tuned DiT checkpoints, and
writes the predicted video to ``--output_path``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from worldarena.benchmark.annotations import load_camera_matrices, load_intrinsics_sequence
from worldarena.common.checkpoints import apply_checkpoint_env, hf_local_dir, resolve_checkpoint_path


# Latent-window constants from WorldCam's autoregressive rollout; used to derive
# the minimum number of camera frames required for a given num_ar_steps.
CAMERA_WINDOW_LATENTS = 8
GENERATED_WINDOW_LATENTS = 8
CAMERA_SAMPLES_PER_LATENT = 4
CAMERA_PADDING_FRAMES = 100
DTYPE_MAP = {
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldCam subprocess runner for WorldAtlas Arena.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--input_video", required=True, type=str)
    parser.add_argument("--annotation_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=832, type=int)
    parser.add_argument("--cond_frames", default=65, type=int)
    parser.add_argument("--camera_frames", default=233, type=int)
    parser.add_argument("--output_fps", default=30.0, type=float)
    parser.add_argument("--output_frames", default=None, type=int)
    parser.add_argument("--output_quality", default=4, type=int)
    parser.add_argument("--cfg_scale", default=4.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_ar_steps", default=50, type=int)
    parser.add_argument("--attention_sink_inference", action="store_true")
    parser.add_argument("--long_term_memory_start", default=30, type=int)
    parser.add_argument("--long_term_memory_num_clips", default=4, type=int)
    parser.add_argument("--long_term_memory_ref_indices", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--torch_dtype", default="bfloat16", type=str)
    parser.add_argument("--weights_filename", default="finetuned_dit.safetensors", type=str)
    return parser.parse_args()


def _minimum_camera_frames(
    *,
    num_ar_steps: int,
    padding_frames: int = CAMERA_PADDING_FRAMES,
) -> int:
    """Return the shortest pose sequence that can sustain ``num_ar_steps`` AR rollout."""
    required_after_drop = CAMERA_SAMPLES_PER_LATENT * (
        num_ar_steps + CAMERA_WINDOW_LATENTS + GENERATED_WINDOW_LATENTS - 1
    )
    return 1 + max(0, required_after_drop - padding_frames)


def _conditioning_frames(video_path: Path, *, cond_frames: int, height: int, width: int) -> list[Image.Image]:
    """Decode the first ``cond_frames`` from the conditioning video, padding by repeating the last frame."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open conditioning video: {video_path}")

    frames: list[Image.Image] = []
    while len(frames) < cond_frames:
        success, frame = capture.read()
        if not success:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if image.size != (width, height):
            image = image.resize((width, height))
        frames.append(image)
    capture.release()

    if not frames:
        raise RuntimeError(f"failed to decode any conditioning frames from: {video_path}")
    while len(frames) < cond_frames:
        frames.append(frames[-1].copy())
    return frames


def _camera_inputs(annotation_path: str, *, camera_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Load and align GT camera extrinsics/intrinsics to the requested frame count."""
    extrinsics = load_camera_matrices(annotation_path, target_frames=camera_frames)
    intrinsics = load_intrinsics_sequence(annotation_path, target_frames=camera_frames)
    if extrinsics is None:
        raise FileNotFoundError(f"poses.npy not found under annotation path: {annotation_path}")
    if intrinsics is None:
        raise FileNotFoundError(f"intrinsics.npy not found under annotation path: {annotation_path}")
    return extrinsics.astype(np.float32), intrinsics.astype(np.float32)


def _parse_ref_indices(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _torch_dtype_name(raw: str) -> str:
    normalized = DTYPE_MAP.get(raw.lower())
    if normalized is None:
        raise ValueError(f"unsupported torch dtype: {raw}")
    return normalized


def _base_model_root() -> Path:
    """Locate the local Wan2.1-T2V-1.3B base weights required by WorldCam."""
    required_files = (
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "diffusion_pytorch_model.safetensors",
    )
    candidate = hf_local_dir("Wan-AI/Wan2.1-T2V-1.3B", required=True)
    if not all((candidate / name).exists() for name in required_files):
        raise FileNotFoundError(
            "WorldCam base model files are incomplete under "
            f"{candidate}. Expected {required_files}."
        )
    return candidate


def _load_base_models(
    *,
    model_manager,
    pipe,
) -> Path:
    existing_root = _base_model_root()
    for file_name in (
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "diffusion_pytorch_model.safetensors",
    ):
        model_manager.load_model(
            str(existing_root / file_name),
            device="cpu",
            torch_dtype=pipe.torch_dtype,
        )
    tokenizer_candidates = sorted((existing_root / "google").glob("*"))
    if not tokenizer_candidates:
        raise FileNotFoundError(f"tokenizer files not found under: {existing_root / 'google'}")
    return tokenizer_candidates[0]


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("WorldCam runner requires checkpoint_dir")
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ.update(apply_checkpoint_env())
    os.chdir(repo_root)

    import torch

    from diffsynth import save_video
    from diffsynth.models import ModelManager, load_state_dict
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline

    torch_dtype_name = _torch_dtype_name(args.torch_dtype)
    torch_dtype = getattr(torch, torch_dtype_name)
    camera_frames = max(args.camera_frames, _minimum_camera_frames(num_ar_steps=args.num_ar_steps))
    conditioning_video = _conditioning_frames(
        Path(args.input_video).expanduser().resolve(),
        cond_frames=args.cond_frames,
        height=args.height,
        width=args.width,
    )
    extrinsics_np, intrinsics_np = _camera_inputs(
        args.annotation_path,
        camera_frames=camera_frames,
    )

    pipe = WanVideoPipeline(torch_dtype=torch_dtype, device=args.device)
    model_manager = ModelManager()
    tokenizer_path = _load_base_models(
        model_manager=model_manager,
        pipe=pipe,
    )

    pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
    pipe.vae = model_manager.fetch_model("wan_video_vae")
    pipe.dit = model_manager.fetch_model("wan_video_dit")

    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(str(tokenizer_path))

    weights_path = checkpoint_dir / args.weights_filename
    if not weights_path.exists():
        raise FileNotFoundError(f"WorldCam fine-tuned weights not found: {weights_path}")
    pipe.dit.load_state_dict(load_state_dict(str(weights_path), device="cpu"), strict=True)

    pipe.text_encoder.to(pipe.device, dtype=pipe.torch_dtype)
    pipe.vae.to(pipe.device, dtype=pipe.torch_dtype)
    pipe.dit.to(pipe.device, dtype=pipe.torch_dtype)

    intrinsics = torch.from_numpy(intrinsics_np).to(pipe.device, dtype=torch.float32)[None, ...]
    extrinsics = torch.from_numpy(extrinsics_np).to(pipe.device, dtype=torch.float32)[None, ...]

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_video=conditioning_video,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        height=args.height,
        width=args.width,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=True,
        long_term_memory_start_step=args.long_term_memory_start,
        long_term_memory_num_clips=args.long_term_memory_num_clips,
        long_term_memory_ref_indices=_parse_ref_indices(args.long_term_memory_ref_indices),
        num_ar_steps=args.num_ar_steps,
        attention_sink_inference=args.attention_sink_inference,
    )
    if args.output_frames is not None:
        if args.output_frames <= 0:
            raise ValueError(f"output_frames must be positive, got {args.output_frames}")
        if len(video) < args.output_frames:
            raise RuntimeError(
                f"WorldCam generated {len(video)} frames, fewer than requested {args.output_frames}"
            )
        video = video[: args.output_frames]
    save_video(video, str(output_path), fps=args.output_fps, quality=args.output_quality)


if __name__ == "__main__":
    main()
