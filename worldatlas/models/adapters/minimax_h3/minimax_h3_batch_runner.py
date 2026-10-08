"""Persistent batch runner for WorldFoundry's NativeMiniMaxH3Pipeline.

This module runs *inside* the WorldFoundry-capable interpreter (the unified
``worldfoundry-unified-cu121`` env, which can import both ``worldatlas`` and
``worldfoundry``).  It loads the bespoke MiniMax H3 audio-video pipeline once and
then walks the JSONL batch spec, writing one mp4 (32 kHz stereo audio muxed) per
sample.

Task routing (WorldAtlas suite/generation.task -> H3 task):
- ``t2va``  text-to-video-audio      -> pipeline(prompt, task="t2va", ...)  [public API]
- ``fl2va`` first-frame-to-video     -> generate(..., keyframe_cond_rows=<image>)
- ``ref2va`` reference-to-video      -> generate(..., ref_blocks=[image block], keyframe_cond_rows=<image>)

t2va is fully supported by WorldFoundry's public ``__call__``.  fl2va / ref2va
have no public image->conditioning helper upstream, so this runner assembles the
keyframe/reference condition rows from documented public primitives
(``video_vae.encode_images`` + the exported ``minimax_h3_patchify_video_latent``
+ the exported keyframe signatures).  Those image-conditioned paths cannot be
smoke-tested without real weights + a GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    begin_sample,
    print_status,
)

# WorldAtlas task name -> H3 native task.
_SUITE_DEFAULT_TASK = {
    "image_static": "fl2va",
    "image_dynamic": "fl2va",
    "video_static": "ref2va",
    "video_dynamic": "ref2va",
}
_VALID_TASKS = {"t2va", "fl2va", "ref2va"}
_TASK_CHECKPOINT_PARTITION = {
    "t2va": "FL2VA",
    "fl2va": "FL2VA",
    "ref2va": "Ref2VA",
}
_COMPONENT_NAMES = ("transformer", "video_vae", "audio_vae", "text_encoder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas persistent MiniMax H3 batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _resolve_task(row: dict[str, Any], generation: dict[str, Any]) -> str:
    """Pick the H3 task for a row: explicit override > suite default > t2va."""
    explicit = row.get("task") or generation.get("task")
    if explicit:
        task = str(explicit).strip().lower()
    else:
        task = _SUITE_DEFAULT_TASK.get(str(row.get("suite", "")), "t2va")
    if task not in _VALID_TASKS:
        raise ValueError(f"unsupported MiniMax H3 task {task!r}; expected one of {sorted(_VALID_TASKS)}")
    return task


def _generation_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    """Translate the WorldAtlas generation block into H3 __call__/generate kwargs."""
    kwargs: dict[str, Any] = {}
    if generation.get("short_edge") is not None:
        kwargs["short_edge"] = int(generation["short_edge"])
    if generation.get("aspect_ratio") is not None:
        kwargs["aspect_ratio"] = str(generation["aspect_ratio"])
    if generation.get("duration_seconds") is not None:
        kwargs["duration_seconds"] = float(generation["duration_seconds"])
    elif generation.get("duration") is not None:
        kwargs["duration_seconds"] = float(generation["duration"])
    if generation.get("num_inference_steps") is not None:
        kwargs["num_inference_steps"] = int(generation["num_inference_steps"])
    elif generation.get("steps") is not None:
        kwargs["num_inference_steps"] = int(generation["steps"])
    if generation.get("flow_shift") is not None:
        kwargs["flow_shift"] = float(generation["flow_shift"])
    if generation.get("audio_flow_shift") is not None:
        kwargs["audio_flow_shift"] = float(generation["audio_flow_shift"])
    return kwargs


def _select_checkpoint_dir(
    checkpoint_dir: Path,
    rows: list[dict[str, Any]],
    generation: dict[str, Any],
) -> Path:
    """Select the native H3 checkpoint partition required by a batch.

    The public MiniMax-H3 snapshot contains both diffusers components at its
    root and WorldFoundry-native checkpoints under ``FL2VA`` / ``Ref2VA``.
    WorldFoundry's bespoke loader requires the latter.  A persistent pipeline
    can only serve tasks backed by one transformer partition, so mixed batches
    must be split before they reach this runner.
    """
    tasks = {_resolve_task(row, generation) for row in rows}
    partitions = {_TASK_CHECKPOINT_PARTITION[task] for task in tasks}
    if len(partitions) > 1:
        raise ValueError(
            "MiniMax H3 batch mixes tasks that require different checkpoint partitions "
            f"({', '.join(sorted(tasks))}); split FL2VA/t2va and Ref2VA requests into "
            "separate batches."
        )

    # Empty specs are rejected by load_batch_spec, but keep the helper total for
    # lightweight callers and tests.
    if not partitions:
        return checkpoint_dir

    required_partition = next(iter(partitions))
    configured_partition = {
        "fl2va": "FL2VA",
        "ref2va": "Ref2VA",
    }.get(checkpoint_dir.name.casefold())
    if configured_partition is not None:
        if configured_partition != required_partition:
            raise ValueError(
                f"MiniMax H3 task(s) {sorted(tasks)} require {required_partition}, but "
                f"checkpoint_dir points directly at {configured_partition}: {checkpoint_dir}"
            )
        return checkpoint_dir
    return checkpoint_dir / required_partition


def _resolve_component_devices(
    generation: dict[str, Any],
    default_device: str,
) -> dict[str, str]:
    """Resolve an optional per-component device map for large H3 modules."""
    devices = {name: default_device for name in _COMPONENT_NAMES}
    configured = generation.get("component_devices")
    if configured is None:
        return devices
    if not isinstance(configured, dict):
        raise ValueError("MiniMax H3 generation.component_devices must be a mapping")
    unknown = sorted(set(configured) - set(_COMPONENT_NAMES))
    if unknown:
        raise ValueError(f"unknown MiniMax H3 component device key(s): {unknown}")
    for name, value in configured.items():
        device = str(value).strip()
        if not device:
            raise ValueError(f"MiniMax H3 component device for {name!r} cannot be empty")
        devices[name] = device
    return devices


def _module_device(module: Any, fallback: str) -> Any:
    """Return a module's parameter device without assuming it has parameters."""
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return fallback


def _route_decode_inputs_to_component_devices(pipeline: Any) -> None:
    """Move final latents to separately placed VAE devices before decoding."""
    video_decode = pipeline._decode_video
    audio_decode = pipeline._decode_audio
    video_device = _module_device(pipeline.video_vae, pipeline.device)
    audio_device = _module_device(pipeline.audio_vae, pipeline.device)

    def decode_video(latent: Any) -> Any:
        return video_decode(latent.to(video_device))

    def decode_audio(latent: Any) -> Any:
        return audio_decode(latent.to(audio_device))

    pipeline._decode_video = decode_video
    pipeline._decode_audio = decode_audio


def _load_pipeline(
    pipeline_class: Any,
    *,
    checkpoint_dir: Path,
    generation: dict[str, Any],
    default_device: str,
) -> Any:
    """Load H3 on one device or split its resident components across devices."""
    devices = _resolve_component_devices(generation, default_device)
    if len(set(devices.values())) == 1:
        return pipeline_class.from_pretrained(
            model_path=str(checkpoint_dir),
            device=devices["transformer"],
        )

    print(
        "[WorldAtlas:MiniMaxH3] loading component device map: "
        + ", ".join(f"{name}={device}" for name, device in devices.items()),
        flush=True,
    )
    transformer = pipeline_class._load_transformer(
        checkpoint_dir,
        device=devices["transformer"],
    )
    video_vae = pipeline_class._load_video_vae(
        checkpoint_dir,
        device=devices["video_vae"],
    )
    audio_vae = pipeline_class._load_audio_vae(
        checkpoint_dir,
        device=devices["audio_vae"],
    )
    text_encoder = pipeline_class._load_text_encoder(
        checkpoint_dir,
        device=devices["text_encoder"],
    )
    tokenizer = pipeline_class._load_tokenizer(checkpoint_dir)
    pipeline = pipeline_class(
        transformer=transformer,
        video_vae=video_vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=devices["transformer"],
    )
    _route_decode_inputs_to_component_devices(pipeline)
    return pipeline


def _encode_prompt(pipeline: Any, prompt: str) -> Any:
    """Tokenize one prompt and satisfy H3 encoder's one-dimensional ID contract."""
    tokens = pipeline.tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"]
    if input_ids.ndim == 2:
        if int(input_ids.shape[0]) != 1:
            raise ValueError(f"MiniMax H3 expects one prompt, got input_ids shape {list(input_ids.shape)}")
        input_ids = input_ids[0]
    return pipeline.text_encoder.encode_ids(input_ids)


def _build_keyframe_cond_rows(pipeline: Any, image: Any, plan: dict[str, int]) -> Any:
    """Encode a conditioning image into normalized, patchified keyframe rows.

    Uses only documented public MiniMax H3 primitives.  The keyframe/reference is
    treated as a single full-canvas latent frame; the DiT operates in normalized
    latent space, so we forward-normalize (the inverse of the pipeline's exposed
    reverse-normalize) before patchifying.
    """
    import torch
    from worldfoundry.pipelines.minimax._minimax_h3 import minimax_h3_patchify_video_latent

    video_vae = pipeline.video_vae
    if not isinstance(video_vae, torch.nn.Module):
        raise RuntimeError(
            "MiniMax H3 image-conditioned tasks require a real video VAE; "
            "none is loaded (supply real weights via checkpoint_dir)."
        )
    device = torch.device(pipeline.device)
    resized = image.resize((int(plan["width"]), int(plan["height"])))
    latents = video_vae.encode_images([resized])
    latent = latents[0].to(device=device, dtype=torch.float32)
    while latent.ndim < 5:  # -> [B, C, T, H, W]
        latent = latent.unsqueeze(0)
    arch = video_vae.sglang_config.arch_config
    mean = torch.as_tensor(tuple(arch.latents_mean), device=device, dtype=torch.float32)
    std = torch.as_tensor(tuple(arch.latents_std), device=device, dtype=torch.float32)
    view = [1] * latent.ndim
    view[1] = int(mean.shape[0])
    latent = (latent - mean.view(*view)) / std.view(*view)
    return minimax_h3_patchify_video_latent(latent, patch_size=(1, 2, 2))


def _run_batch(
    *,
    rows: list[dict[str, Any]],
    checkpoint_dir: Path,
    generation: dict[str, Any],
) -> None:
    import torch
    from PIL import Image
    from worldfoundry.pipelines.minimax.pipeline_minimax_h3 import NativeMiniMaxH3Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = _load_pipeline(
        NativeMiniMaxH3Pipeline,
        checkpoint_dir=checkpoint_dir,
        generation=generation,
        default_device=device,
    )
    print(
        "[WorldAtlas:MiniMaxH3] persistent pipeline ready; checkpoint_loads=1",
        flush=True,
    )
    call_kwargs = _generation_kwargs(generation)
    base_seed = int(generation.get("seed", 0))
    increment_seed = bool(generation.get("increment_seed", True))
    continue_on_error = bool(generation.get("continue_on_error", True))

    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        begin_sample(sample_id, index=index + 1, total=len(rows), label="MiniMaxH3")
        output_path = Path(str(row["output_path"])).expanduser().resolve()
        seed = base_seed + index if increment_seed else base_seed
        try:
            task = _resolve_task(row, generation)
            prompt = str(row.get("prompt") or "")
            prompt_embeds = _encode_prompt(pipeline, prompt)
            if task == "t2va":
                result = pipeline.generate(
                    prompt_embeds=prompt_embeds,
                    task="t2va",
                    seed=seed,
                    **call_kwargs,
                )
            else:
                image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"conditioning image not found: {image_path}")
                image = Image.open(image_path).convert("RGB")
                plan = pipeline._resolve_plan(
                    task=task,
                    short_edge=call_kwargs.get("short_edge", pipeline.DEFAULT_SHORT_EDGE),
                    aspect_ratio=call_kwargs.get("aspect_ratio", pipeline.DEFAULT_ASPECT_RATIO),
                    duration_seconds=call_kwargs.get(
                        "duration_seconds", pipeline.DEFAULT_DURATION_SECONDS
                    ),
                )
                keyframe_cond_rows = _build_keyframe_cond_rows(pipeline, image, plan)
                generate_kwargs: dict[str, Any] = dict(call_kwargs)
                generate_kwargs["keyframe_cond_rows"] = keyframe_cond_rows
                if task == "fl2va":
                    generate_kwargs["keyframe_frame_indices"] = (0,)
                else:  # ref2va: single reference image block
                    generate_kwargs["ref_blocks"] = [
                        {"kind": "image", "latent_h": plan["latent_h"], "latent_w": plan["latent_w"]}
                    ]
                result = pipeline.generate(
                    prompt_embeds=prompt_embeds,
                    task=task,
                    seed=seed,
                    **generate_kwargs,
                )
            if "video" in result:
                pipeline._write_mp4_with_audio(
                    video=result["video"],
                    audio=result.get("audio"),
                    output_path=output_path,
                    fps=pipeline.DEFAULT_FPS,
                )
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise RuntimeError(f"MiniMax H3 did not write output: {output_path}")
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                seed=seed,
                task=task,
                persistent_engine=True,
                checkpoint_load_count=1,
            )
        except Exception as exc:  # noqa: BLE001 - report per-sample, continue batch
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            print_status(sample_id, "failed", error=error)
            if not continue_on_error:
                raise
        finally:
            gc.collect()


def _validate_paths(checkpoint_dir: Path | None) -> Path:
    if checkpoint_dir is None:
        raise ValueError("MiniMax H3 requires checkpoint_dir")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"MiniMax H3 checkpoint directory not found: {checkpoint_dir}")
    return checkpoint_dir


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else None
    )
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    selected_checkpoint_dir = (
        _select_checkpoint_dir(checkpoint_dir, rows, generation) if checkpoint_dir else None
    )

    if os.environ.get("WORLDATLAS_MINIMAX_H3_DRY_RUN", "").lower() in {"1", "true", "yes", "on"}:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "persistent_engine": True,
                    "checkpoint_load_count": 1,
                    "repo_root": str(repo_root),
                    "configured_checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
                    "checkpoint_dir": (
                        str(selected_checkpoint_dir) if selected_checkpoint_dir else None
                    ),
                    "request_count": len(rows),
                    "tasks": [_resolve_task(row, generation) for row in rows],
                    "call_kwargs": _generation_kwargs(generation),
                    "component_devices": _resolve_component_devices(generation, "cuda"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    selected_checkpoint_dir = _validate_paths(selected_checkpoint_dir)
    _run_batch(rows=rows, checkpoint_dir=selected_checkpoint_dir, generation=generation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
