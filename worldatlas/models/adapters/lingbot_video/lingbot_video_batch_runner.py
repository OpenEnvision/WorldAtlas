"""Persistent batch runner for the upstream LingBot-Video TI2V pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    begin_sample,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WorldAtlas persistent LingBot-Video TI2V batch runner."
    )
    add_common_batch_args(parser)
    return parser.parse_args()


def _bool_value(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _split_camera_prompt(prompt: str) -> tuple[str, str]:
    """Split WorldAtlas's leading ``Camera ...`` sentence from scene content."""
    first, separator, remainder = prompt.strip().partition(".")
    if separator and first.strip().lower().startswith("camera "):
        return first.strip() + ".", remainder.strip()
    return "", prompt.strip()


def _camera_path_description(row: dict[str, Any]) -> str:
    values = row.get("camera_path") or []
    if isinstance(values, str):
        values = [values]
    tokens = [str(value).strip().replace("_", " ") for value in values if str(value).strip()]
    if not tokens:
        return "The camera follows the movement requested by the text prompt."
    return "The camera follows this ordered motion path: " + ", then ".join(tokens) + "."


def structured_prompt_for_row(row: dict[str, Any], generation: dict[str, Any]) -> str:
    """Convert a benchmark prompt into the JSON-caption string expected by the DiT."""
    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"empty prompt for sample {row.get('sample_id', '<unknown>')}")

    strategy = str(generation.get("prompt_strategy", "structured")).strip().lower()
    if strategy in {"raw", "plain", "text"}:
        return prompt
    if strategy in {"json", "passthrough_json"}:
        parsed = json.loads(prompt)
        if not isinstance(parsed, (dict, list)):
            raise ValueError("prompt_strategy=json requires a JSON object or list")
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if strategy != "structured":
        raise ValueError(f"unsupported LingBot-Video prompt_strategy: {strategy}")

    camera_description, scene_description = _split_camera_prompt(prompt)
    if not camera_description:
        camera_description = _camera_path_description(row)
    if not scene_description:
        scene_description = prompt

    caption = {
        "comprehensive_description": {
            "scene_content_description": scene_description,
            "camera_movement_description": camera_description,
        },
        "camera_info": {},
        "world_knowledge": [],
        "prominent_elements": [],
    }
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


def _snap_video_frame_count(value: int) -> int:
    if value < 1:
        raise ValueError(f"num_frames must be positive, got {value}")
    if value == 1:
        return 1
    return int(math.ceil((value - 1) / 4.0) * 4 + 1)


def frame_count_from_generation(generation: dict[str, Any]) -> int:
    explicit = generation.get("num_frames")
    if explicit is not None:
        value = int(explicit)
        snapped = _snap_video_frame_count(value)
        if snapped != value:
            raise ValueError(f"LingBot-Video num_frames must be 4n+1, got {value}")
        return value
    duration = float(generation.get("duration", 5.0))
    fps = int(generation.get("fps", 24))
    return _snap_video_frame_count(int(duration * fps))


def runtime_summary(generation: dict[str, Any]) -> dict[str, Any]:
    height = int(generation.get("height", 480))
    width = int(generation.get("width", 832))
    if height % 16 or width % 16:
        raise ValueError(
            f"LingBot-Video height and width must be multiples of 16, got {height}x{width}"
        )
    nproc = int(generation.get("nproc_per_node", 1) or 1)
    context_degree = int(generation.get("context_parallel_degree", 1) or 1)
    if context_degree > 1 and context_degree != nproc:
        raise ValueError(
            "context_parallel_degree must equal nproc_per_node for LingBot-Video TI2V: "
            f"{context_degree} != {nproc}"
        )
    return {
        "backend": str(generation.get("backend", "diffusers")),
        "mode": "ti2v",
        "prompt_strategy": str(generation.get("prompt_strategy", "structured")),
        "height": height,
        "width": width,
        "num_frames": frame_count_from_generation(generation),
        "fps": int(generation.get("fps", 24)),
        "steps": int(generation.get("steps", 40)),
        "guidance_scale": float(generation.get("guidance_scale", 3.0)),
        "shift": float(generation.get("shift", 3.0)),
        "nproc_per_node": nproc,
        "context_parallel_degree": context_degree,
        "enable_fsdp_inference": _bool_value(generation, "enable_fsdp_inference"),
        "batch_cfg": _bool_value(generation, "batch_cfg", True),
    }


def _validate_paths(repo_root: Path, checkpoint_dir: Path | None) -> Path:
    if not (repo_root / "scripts" / "inference.py").is_file():
        raise FileNotFoundError(
            f"LingBot-Video checkout is missing scripts/inference.py: {repo_root}"
        )
    if checkpoint_dir is None:
        raise ValueError("LingBot-Video requires checkpoint_dir")
    required = (
        checkpoint_dir / "model_index.json",
        checkpoint_dir / "transformer",
        checkpoint_dir / "text_encoder",
        checkpoint_dir / "vae",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "incomplete LingBot-Video checkpoint; missing: " + ", ".join(missing)
        )
    return checkpoint_dir


def _first_distributed_error(torch: Any, local_error: str | None) -> str | None:
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return local_error
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    return next((error for error in errors if error), None)


def _generate_batch(
    *,
    rows: list[dict[str, Any]],
    repo_root: Path,
    checkpoint_dir: Path,
    generation: dict[str, Any],
) -> None:
    if _bool_value(generation, "run_refiner"):
        raise ValueError(
            "The persistent WorldAtlas runner currently supports LingBot-Video base TI2V only; "
            "set generation.run_refiner=false."
        )

    summary = runtime_summary(generation)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ.setdefault(
        "DIFFUSERS_ATTN_BACKEND",
        str(generation.get("diffusers_attn_backend", "_native_flash")),
    )
    # The upstream default requires the optional flash_attention_3 package,
    # which is not installed by its base requirements. SDPA is the portable
    # default; environments with FA3 can override it through YAML or env.
    os.environ.setdefault(
        "LINGBOT_QWEN_ATTN_IMPLEMENTATION",
        str(generation.get("qwen_attn_implementation", "sdpa")),
    )
    os.environ.setdefault("LINGBOT_MOE_PAD_BACKEND", "vectorized")
    os.environ.setdefault("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm")
    if _bool_value(generation, "quiet_progress"):
        os.environ["LINGBOT_QUIET_PROGRESS"] = "1"

    import torch
    from PIL import Image
    from lingbot_video import runner as upstream
    from lingbot_video.inference_backend import resolve_backend_engine

    engine = resolve_backend_engine(
        engine=None,
        backend=str(summary["backend"]),
        stderr=sys.stderr,
    )
    upstream_args = SimpleNamespace(
        model_dir=str(checkpoint_dir),
        engine=engine,
        mode="ti2v",
        transformer_subfolder=str(generation.get("transformer_subfolder", "transformer")),
        default_dtype=str(generation.get("default_dtype", "bf16")),
        transformer_dtype=str(generation.get("transformer_dtype", "bf16")),
        text_encoder_dtype=str(generation.get("text_encoder_dtype", "bf16")),
        vae_dtype=str(generation.get("vae_dtype", "fp32")),
    )

    rank = 0
    pipe = None
    try:
        (
            rank,
            _local_rank,
            _world_size,
            _cfg_parallel_group,
            context_parallel_mesh,
            _cfg_branch_rank,
            context_parallel_rank,
        ) = upstream._init_parallel(
            1,
            int(summary["context_parallel_degree"]),
            bool(summary["enable_fsdp_inference"]),
        )
        if _bool_value(generation, "allow_tf32", True) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        fsdp_mesh = None
        if summary["enable_fsdp_inference"]:
            if upstream.init_fsdp_inference_mesh is None:
                raise RuntimeError("LingBot-Video FSDP helpers are unavailable")
            fsdp_mesh = upstream.init_fsdp_inference_mesh()

        dtype_map = upstream._make_dtype_map(upstream_args)
        pipe, engine_name = upstream._load_pipe(
            upstream_args,
            dtype_map,
            defer_transformer_to_device=fsdp_mesh is not None,
        )
        if rank == 0:
            print(
                "[WorldAtlas:LingBotVideo] persistent pipeline ready; checkpoint_loads=1",
                flush=True,
            )
        upstream._configure_pipeline_logs(pipe)
        device = upstream._default_device()
        if int(summary["context_parallel_degree"]) > 1:
            upstream._enable_context_parallel(
                pipe.transformer,
                int(summary["context_parallel_degree"]),
                _bool_value(generation, "context_parallel_ulysses_anything", True),
                context_parallel_rank,
                device,
                context_parallel_mesh,
                True,
            )
        upstream._apply_fsdp_inference_if_requested(
            pipe,
            bool(summary["enable_fsdp_inference"]),
            fsdp_mesh,
        )

        if "negative_prompt" in generation:
            negative_prompt = str(generation.get("negative_prompt") or "")
        else:
            negative_prompt = upstream.DEFAULT_NEGATIVE_PROMPT

        base_seed = int(generation.get("seed", 42))
        increment_seed = _bool_value(generation, "increment_seed", True)
        reuse_conditions = _bool_value(generation, "reuse_condition_features", True)
        null_cond_clone_zero = _bool_value(generation, "null_cond_clone_zero")
        continue_on_error = _bool_value(generation, "continue_on_error", True)

        for row_index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            seed = base_seed + row_index if increment_seed else base_seed
            local_error: str | None = None
            result = None
            frames = None
            condition_cache = None
            image_tensor_cpu = None
            if rank == 0:
                begin_sample(
                    sample_id,
                    index=row_index + 1,
                    total=len(rows),
                    label="LingBot-Video",
                )
            try:
                prompt = structured_prompt_for_row(row, generation)
                image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"conditioning image not found: {image_path}")
                input_image = Image.open(image_path).convert("RGB")
                generator = torch.Generator(device=device).manual_seed(seed)

                if reuse_conditions or bool(summary["batch_cfg"]) or null_cond_clone_zero:
                    condition_cache, image_tensor_cpu = upstream._cache_ti2v_prompt_conditions(
                        pipe,
                        prompt,
                        negative_prompt,
                        input_image,
                        height=int(summary["height"]),
                        width=int(summary["width"]),
                        device=device,
                        null_cond_clone_zero=null_cond_clone_zero,
                    )

                call_kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": input_image,
                    "height": int(summary["height"]),
                    "width": int(summary["width"]),
                    "num_frames": int(summary["num_frames"]),
                    "num_inference_steps": int(summary["steps"]),
                    "guidance_scale": float(summary["guidance_scale"]),
                    "shift": float(summary["shift"]),
                    "generator": generator,
                    "output_type": "np",
                    "batch_cfg": bool(summary["batch_cfg"]),
                    "null_cond_clone_zero": null_cond_clone_zero,
                    **upstream._condition_call_kwargs(condition_cache, device),
                }
                if image_tensor_cpu is not None:
                    call_kwargs["image_tensor"] = image_tensor_cpu.to(device=device)

                with torch.no_grad():
                    result = pipe(**call_kwargs)
                if rank == 0:
                    frames = upstream._extract_frames(result)
                    upstream._save_frames(
                        frames,
                        "ti2v",
                        output_path,
                        int(summary["fps"]),
                    )
                    if not output_path.is_file() or output_path.stat().st_size <= 0:
                        raise RuntimeError(f"LingBot-Video did not write output: {output_path}")
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

            error = _first_distributed_error(torch, local_error)
            if rank == 0:
                if error:
                    print_status(sample_id, "failed", error=error)
                else:
                    print_status(
                        sample_id,
                        "generated",
                        output_path=str(output_path),
                        seed=seed,
                        engine=engine_name,
                        persistent_engine=True,
                        checkpoint_load_count=1,
                        prompt_strategy=str(summary["prompt_strategy"]),
                        num_frames=int(summary["num_frames"]),
                        fps=int(summary["fps"]),
                    )

            del result, frames, condition_cache, image_tensor_cpu
            gc.collect()
            if error and not continue_on_error:
                raise RuntimeError(error)
    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        try:
            upstream._destroy_parallel_if_needed()
        except (NameError, RuntimeError):
            pass


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else None
    )
    checkpoint_dir = _validate_paths(repo_root, checkpoint_dir)
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    summary = runtime_summary(generation)

    if os.environ.get("WORLDATLAS_LINGBOT_VIDEO_DRY_RUN", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "persistent_engine": True,
                    "checkpoint_load_count": 1,
                    "repo_root": str(repo_root),
                    "checkpoint_dir": str(checkpoint_dir),
                    "request_count": len(rows),
                    "runtime": summary,
                    "prompt_preview": structured_prompt_for_row(rows[0], generation),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    _generate_batch(
        rows=rows,
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        generation=generation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
