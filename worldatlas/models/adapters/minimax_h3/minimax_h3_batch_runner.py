"""Persistent batch runner for MiniMax H3 via SGLang DiffGenerator.

This module runs inside the MiniMax-H3 SGLang environment, starts one local
``sglang.multimodal_gen.DiffGenerator`` instance per shard, and walks the JSONL
batch spec to write one 24 fps MP4 with 32 kHz stereo audio per sample.

Task routing (WorldAtlas suite or ``generation.task`` -> MiniMax H3 task):
- ``t2va``   text-to-video-audio        -> ``model_variant="fl2va"``, ``conditions=[]``
- ``fl2va``  first-frame-to-video-audio -> ``model_variant="fl2va"``, keyframe at frame 0
- ``ref2va`` reference-to-video-audio   -> ``model_variant="ref2va"``, image reference
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
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)

_SUITE_DEFAULT_TASK = {
    "image_static": "fl2va",
    "image_dynamic": "fl2va",
    "video_static": "ref2va",
    "video_dynamic": "ref2va",
}
_VALID_TASKS = {"t2va", "fl2va", "ref2va"}
_TASK_MODEL_VARIANT = {
    "t2va": "fl2va",
    "fl2va": "fl2va",
    "ref2va": "ref2va",
}
_DEFAULT_SHORT_EDGE = 768
_DEFAULT_ASPECT_RATIO = "16:9"
_DEFAULT_DURATION_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas persistent MiniMax H3 batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _prepend_repo_pythonpath(repo_root: Path) -> None:
    python_dir = repo_root / "python"
    candidates = [python_dir, repo_root] if python_dir.is_dir() else [repo_root]
    for candidate in reversed(candidates):
        resolved = str(candidate)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _resolve_task(row: dict[str, Any], generation: dict[str, Any]) -> str:
    """Pick the MiniMax H3 task for a row: explicit override > suite default > t2va."""
    explicit = row.get("task") or generation.get("task")
    if explicit:
        task = str(explicit).strip().lower()
    else:
        task = _SUITE_DEFAULT_TASK.get(str(row.get("suite", "")), "t2va")
    if task not in _VALID_TASKS:
        raise ValueError(
            f"unsupported MiniMax H3 task {task!r}; expected one of {sorted(_VALID_TASKS)}"
        )
    return task


def _resolve_model_target(
    checkpoint_dir: Path,
    rows: list[dict[str, Any]],
    generation: dict[str, Any],
) -> tuple[Path, str | None, str]:
    """Resolve SGLang ``(model_path, model_variant, required_variant)`` for a batch.

    MiniMax H3 ships ``FL2VA`` and ``Ref2VA`` partitions under the Hugging Face
    snapshot root. SGLang selects a partition via ``ServerArgs.model_variant``
    when given the root directory, or loads a standalone partition directory
    directly when ``model_variant`` is omitted.
    """
    tasks = {_resolve_task(row, generation) for row in rows}
    variants = {_TASK_MODEL_VARIANT[task] for task in tasks}
    if len(variants) > 1:
        raise ValueError(
            "MiniMax H3 batch mixes tasks that require different model variants "
            f"({', '.join(sorted(tasks))}); split FL2VA/t2va and Ref2VA requests into "
            "separate batches."
        )

    required_variant = next(iter(variants)) if variants else "fl2va"
    configured_variant = checkpoint_dir.name.casefold()
    if configured_variant in {"fl2va", "ref2va"}:
        if configured_variant != required_variant:
            raise ValueError(
                f"MiniMax H3 task(s) {sorted(tasks)} require variant {required_variant!r}, "
                f"but checkpoint_dir points at {checkpoint_dir}"
            )
        if (checkpoint_dir.parent / "model_index.json").is_file():
            return checkpoint_dir.parent, required_variant, required_variant
        return checkpoint_dir, None, required_variant

    return checkpoint_dir, required_variant, required_variant


def _build_server_args_kwargs(
    *,
    model_path: Path,
    model_variant: str | None,
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Translate WorldAtlas generation settings into SGLang ``ServerArgs`` kwargs."""
    kwargs: dict[str, Any] = {"model_path": str(model_path)}
    if model_variant is not None:
        kwargs["model_variant"] = model_variant

    num_gpus = generation.get("num_gpus")
    if num_gpus is None:
        num_gpus = generation.get("nproc_per_node")
    if num_gpus is not None:
        kwargs["num_gpus"] = int(num_gpus)

    for key in ("ulysses_degree", "ring_degree", "tp_size"):
        if generation.get(key) is not None:
            kwargs[key] = int(generation[key])

    for key in ("performance_mode", "attention_backend", "quantization"):
        if generation.get(key) is not None:
            kwargs[key] = str(generation[key])

    for key in (
        "dit_cpu_offload",
        "dit_layerwise_offload",
        "text_encoder_cpu_offload",
        "vae_cpu_offload",
        "pin_cpu_memory",
        "enable_torch_compile",
    ):
        if generation.get(key) is not None:
            kwargs[key] = bool(generation[key])

    return kwargs


def _build_sampling_params_kwargs(
    row: dict[str, Any],
    *,
    task: str,
    seed: int,
    output_path: Path,
    generation: dict[str, Any],
    check_files: bool = True,
) -> dict[str, Any]:
    """Build canonical ``MiniMaxH3SamplingParams`` kwargs for one benchmark row."""
    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("MiniMax H3 requires a non-empty prompt")

    conditions: list[dict[str, Any]]
    if task == "t2va":
        conditions = []
    else:
        image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
        if check_files and not image_path.is_file():
            raise FileNotFoundError(f"conditioning image not found: {image_path}")
        if task == "fl2va":
            conditions = [
                {
                    "type": "image",
                    "uri": image_path.as_uri(),
                    "role": "keyframe",
                    "frame_index": 0,
                }
            ]
        else:
            conditions = [
                {
                    "type": "image",
                    "uri": image_path.as_uri(),
                    "role": "reference",
                }
            ]

    duration = generation.get("duration_seconds")
    if duration is None:
        duration = generation.get("duration", _DEFAULT_DURATION_SECONDS)
    target = {
        "short_edge": int(generation.get("short_edge", _DEFAULT_SHORT_EDGE)),
        "aspect_ratio": str(generation.get("aspect_ratio", _DEFAULT_ASPECT_RATIO)),
        "duration_seconds": float(duration),
    }

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "task": task,
        "conditions": conditions,
        "target": target,
        "seed": int(seed),
        "save_output": True,
        "output_path": str(output_path.parent),
        "output_file_name": output_path.name,
    }

    steps = generation.get("num_inference_steps")
    if steps is None:
        steps = generation.get("steps")
    if steps is not None:
        kwargs["num_inference_steps"] = int(steps)

    if generation.get("flow_shift") is not None:
        kwargs["flow_shift"] = float(generation["flow_shift"])
    if generation.get("audio_flow_shift") is not None:
        kwargs["audio_flow_shift"] = float(generation["audio_flow_shift"])

    return kwargs


def _run_batch(
    *,
    rows: list[dict[str, Any]],
    model_path: Path,
    model_variant: str | None,
    required_variant: str,
    generation: dict[str, Any],
) -> None:
    from sglang.multimodal_gen import DiffGenerator
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    server_args_kwargs = _build_server_args_kwargs(
        model_path=model_path,
        model_variant=model_variant,
        generation=generation,
    )
    server_args = ServerArgs.from_kwargs(**server_args_kwargs)

    base_seed = int(generation.get("seed", 0))
    increment_seed = bool(generation.get("increment_seed", True))
    continue_on_error = bool(generation.get("continue_on_error", True))

    with DiffGenerator.from_pretrained(server_args=server_args, local_mode=True) as generator:
        print(
            f"[WorldAtlas:MiniMaxH3] SGLang DiffGenerator ready variant={required_variant}; "
            "checkpoint_loads=1",
            flush=True,
        )
        for index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            begin_sample(sample_id, index=index + 1, total=len(rows), label="MiniMaxH3")
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            seed = base_seed + index if increment_seed else base_seed
            try:
                task = _resolve_task(row, generation)
                sampling_params_kwargs = _build_sampling_params_kwargs(
                    row,
                    task=task,
                    seed=seed,
                    output_path=output_path,
                    generation=generation,
                    check_files=True,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result = generator.generate(sampling_params_kwargs=sampling_params_kwargs)
                if isinstance(result, list):
                    result = result[0] if result else None
                if result is None:
                    raise RuntimeError(f"SGLang DiffGenerator returned no output for {sample_id}")
                produced = getattr(result, "output_file_path", None)
                if produced:
                    produced_path = Path(str(produced)).expanduser().resolve()
                    if produced_path != output_path:
                        copy_output(produced_path, output_path)
                if not output_path.is_file() or output_path.stat().st_size <= 0:
                    raise RuntimeError(f"MiniMax H3 did not write output: {output_path}")
                print_status(
                    sample_id,
                    "generated",
                    output_path=str(output_path),
                    seed=seed,
                    task=task,
                    model_variant=required_variant,
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
    _prepend_repo_pythonpath(repo_root)
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else None
    )
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    model_path, model_variant, required_variant = (
        _resolve_model_target(checkpoint_dir, rows, generation)
        if checkpoint_dir is not None
        else (None, None, "fl2va")
    )

    if os.environ.get("WORLDATLAS_MINIMAX_H3_DRY_RUN", "").lower() in {"1", "true", "yes", "on"}:
        first_row = rows[0]
        first_task = _resolve_task(first_row, generation)
        base_seed = int(generation.get("seed", 0))
        sample_preview = _build_sampling_params_kwargs(
            first_row,
            task=first_task,
            seed=base_seed,
            output_path=Path(str(first_row["output_path"])).expanduser().resolve(),
            generation=generation,
            check_files=False,
        )
        server_args = (
            _build_server_args_kwargs(
                model_path=model_path,
                model_variant=model_variant,
                generation=generation,
            )
            if model_path is not None
            else {}
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "persistent_engine": True,
                    "checkpoint_load_count": 1,
                    "repo_root": str(repo_root),
                    "configured_checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
                    "model_path": str(model_path) if model_path else None,
                    "model_variant": model_variant,
                    "required_variant": required_variant,
                    "server_args": server_args,
                    "request_count": len(rows),
                    "tasks": [_resolve_task(row, generation) for row in rows],
                    "sample_preview": sample_preview,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    checkpoint_dir = _validate_paths(checkpoint_dir)
    model_path, model_variant, required_variant = _resolve_model_target(
        checkpoint_dir, rows, generation
    )
    _run_batch(
        rows=rows,
        model_path=model_path,
        model_variant=model_variant,
        required_variant=required_variant,
        generation=generation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
