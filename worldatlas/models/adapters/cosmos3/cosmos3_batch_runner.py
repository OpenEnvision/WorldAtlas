"""Persistent batch runner for WorldFoundry's Cosmos3Pipeline.

This module runs inside the dedicated Cosmos3 interpreter so the native
transformer / Wan VAE / AVAE / tokenizer load once per shard.
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
    begin_sample,
    load_batch_spec,
    load_json,
    print_status,
)

_DEFAULT_VARIANT = "cosmos3-nano"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas persistent Cosmos3 batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _variant_id(checkpoint_dir: Path, generation: dict[str, Any]) -> str:
    explicit = generation.get("variant_id") or generation.get("model_id")
    if explicit:
        return str(explicit).strip()
    name = checkpoint_dir.name.lower().replace("_", "-")
    if "nano" in name:
        return "cosmos3-nano"
    return _DEFAULT_VARIANT


def _log_resident_placement(pipeline: Any) -> None:
    try:
        model = pipeline.native_pipeline.components.denoiser.model
        handle = getattr(model, "_worldfoundry_device_map_handle", None)
    except AttributeError:
        return
    if handle is None or not getattr(handle, "enabled", False):
        return
    print(
        f"[WorldAtlas:Cosmos3] resident device_map layers={handle.layer_count} "
        f"devices={','.join(handle.devices)} placement={','.join(handle.placement)}",
        flush=True,
    )


def _generation_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    mapping = {
        "num_frames": int,
        "height": int,
        "width": int,
        "fps": float,
        "num_inference_steps": int,
        "guidance_scale": float,
        "enable_sound": bool,
    }
    aliases = {"steps": "num_inference_steps", "guidance": "guidance_scale"}
    for src, dest in aliases.items():
        if dest not in generation and src in generation:
            generation = {**generation, dest: generation[src]}
    for key, cast in mapping.items():
        if generation.get(key) is not None:
            kwargs[key] = cast(generation[key])
    return kwargs


def _run_batch(
    *,
    rows: list[dict[str, Any]],
    checkpoint_dir: Path,
    generation: dict[str, Any],
) -> None:
    import torch
    from worldfoundry.pipelines.cosmos.pipeline_cosmos3 import Cosmos3Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant_id = _variant_id(checkpoint_dir, generation)
    offload_mode = generation.get("offload_mode")
    device_map = generation.get("device_map")
    visible = torch.cuda.device_count() if device == "cuda" else 0
    load_kwargs: dict[str, Any] = {
        "model_path": str(checkpoint_dir),
        "device": device,
        "model_id": variant_id,
        "variant_id": variant_id,
    }
    if offload_mode is not None:
        load_kwargs["offload_mode"] = str(offload_mode)
    if device_map is not None:
        load_kwargs["device_map"] = str(device_map)
    if generation.get("vae_tiling") is not None:
        load_kwargs["vae_tiling"] = bool(generation["vae_tiling"])
    pipeline = Cosmos3Pipeline.from_pretrained(**load_kwargs)
    _log_resident_placement(pipeline)
    print(
        f"[WorldAtlas:Cosmos3] persistent pipeline ready variant={variant_id}; "
        f"checkpoint_loads=1 device={device} visible_gpus={visible} "
        f"offload_mode={offload_mode or '<default>'} device_map={device_map or '<default>'}",
        flush=True,
    )
    if device == "cuda":
        for index in range(visible):
            allocated = torch.cuda.memory_allocated(index) / (1024**3)
            reserved = torch.cuda.memory_reserved(index) / (1024**3)
            total = torch.cuda.get_device_properties(index).total_memory / (1024**3)
            print(
                f"[WorldAtlas:Cosmos3] gpu{index} allocated={allocated:.1f}GiB "
                f"reserved={reserved:.1f}GiB total={total:.1f}GiB",
                flush=True,
            )
    call_kwargs = _generation_kwargs(generation)
    base_seed = int(generation.get("seed", 0))
    increment_seed = bool(generation.get("increment_seed", True))
    continue_on_error = bool(generation.get("continue_on_error", True))

    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        begin_sample(sample_id, index=index + 1, total=len(rows), label="Cosmos3")
        output_path = Path(str(row["output_path"])).expanduser().resolve()
        seed = base_seed + index if increment_seed else base_seed
        try:
            image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"conditioning image not found: {image_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = pipeline(
                prompt=str(row.get("prompt") or ""),
                image_path=str(image_path),
                output_path=str(output_path),
                seed=seed,
                return_dict=True,
                **call_kwargs,
            )
            artifact = result.get("artifact_path") if isinstance(result, dict) else None
            if artifact and Path(str(artifact)) != output_path:
                written = Path(str(artifact))
                if written.is_file() and written != output_path:
                    written.replace(output_path)
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise RuntimeError(f"Cosmos3 did not write output: {output_path}")
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                seed=seed,
                variant_id=variant_id,
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


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if args.checkpoint_dir is None:
        raise ValueError("Cosmos3 requires checkpoint_dir")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Cosmos3 checkpoint directory not found: {checkpoint_dir}")
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())

    if os.environ.get("WORLDATLAS_COSMOS3_DRY_RUN", "").lower() in {"1", "true", "yes", "on"}:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "persistent_engine": True,
                    "checkpoint_load_count": 1,
                    "repo_root": str(repo_root),
                    "checkpoint_dir": str(checkpoint_dir),
                    "variant_id": _variant_id(checkpoint_dir, generation),
                    "request_count": len(rows),
                    "offload_mode": generation.get("offload_mode"),
                    "device_map": generation.get("device_map"),
                    "call_kwargs": _generation_kwargs(generation),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    _run_batch(rows=rows, checkpoint_dir=checkpoint_dir, generation=generation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
