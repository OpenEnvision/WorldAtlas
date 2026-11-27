"""Persistent batch runner for NVIDIA's official cosmos-framework CLI.

This module runs inside the dedicated Cosmos3 interpreter, prepares one JSONL
batch input for ``cosmos_framework.scripts.inference`` so the model loads once
per shard, and copies each generated ``vision.mp4`` to its benchmark path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    begin_sample,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)

_VALID_RESOLUTIONS = {"256", "480", "720", "768"}
_VALID_ASPECT_RATIOS = {"1,1", "4,3", "3,4", "16,9", "9,16"}
_TEXT_ONLY_MODES = {"text2image", "text2video"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas persistent Cosmos3 batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _normalize_aspect_ratio(value: Any) -> str:
    normalized = str(value).strip().replace(":", ",")
    if normalized not in _VALID_ASPECT_RATIOS:
        raise ValueError(
            f"unsupported Cosmos3 aspect_ratio {value!r}; expected one of "
            f"{sorted(_VALID_ASPECT_RATIOS)}"
        )
    return normalized


def _resolve_resolution_tier(generation: dict[str, Any]) -> str | None:
    raw = generation.get("resolution")
    if raw is None and generation.get("height") is not None:
        raw = str(int(generation["height"]))
    if raw is None:
        return None
    tier = str(raw).strip().lower().removesuffix("p")
    if tier not in _VALID_RESOLUTIONS:
        raise ValueError(
            f"unsupported Cosmos3 resolution tier {raw!r}; expected one of "
            f"{sorted(_VALID_RESOLUTIONS)}"
        )
    return tier


def _safe_sample_name(index: int, sample_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._") or "sample"
    return f"{index:06d}_{stem}"


def _build_sample_payload(
    row: dict[str, Any],
    *,
    index: int,
    generation: dict[str, Any],
    check_files: bool = True,
) -> tuple[str, dict[str, Any]]:
    sample_id = str(row["sample_id"])
    sample_name = _safe_sample_name(index, sample_id)
    model_mode = str(generation.get("model_mode", "image2video")).strip()
    base_seed = int(generation.get("seed", 0))
    increment_seed = bool(generation.get("increment_seed", True))
    seed = base_seed + index if increment_seed else base_seed

    payload: dict[str, Any] = {
        "name": sample_name,
        "model_mode": model_mode,
        "prompt": str(row.get("prompt") or ""),
        "seed": seed,
    }
    if model_mode not in _TEXT_ONLY_MODES:
        image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
        if check_files and not image_path.is_file():
            raise FileNotFoundError(f"conditioning image not found: {image_path}")
        payload["vision_path"] = str(image_path)

    resolution = _resolve_resolution_tier(generation)
    if resolution is not None:
        payload["resolution"] = resolution
    if generation.get("aspect_ratio") is not None:
        payload["aspect_ratio"] = _normalize_aspect_ratio(generation["aspect_ratio"])
    if generation.get("num_frames") is not None:
        payload["num_frames"] = int(generation["num_frames"])
    if generation.get("fps") is not None:
        payload["fps"] = int(round(float(generation["fps"])))

    num_steps = generation.get("num_steps")
    if num_steps is None:
        num_steps = generation.get("num_inference_steps")
    if num_steps is None:
        num_steps = generation.get("steps")
    if num_steps is not None:
        payload["num_steps"] = int(num_steps)

    guidance = generation.get("guidance")
    if guidance is None:
        guidance = generation.get("guidance_scale")
    if guidance is not None:
        payload["guidance"] = float(guidance)

    for key, cast in (
        ("shift", float),
        ("sigma_max", float),
        ("negative_prompt", str),
        ("enable_sound", bool),
        ("video_save_quality", int),
        ("prompt_upsampling", bool),
    ):
        if generation.get(key) is not None:
            payload[key] = cast(generation[key])

    return sample_name, payload


def _build_inference_command(
    *,
    python_bin: str,
    input_jsonl: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    generation: dict[str, Any],
) -> list[str]:
    nproc_per_node = int(generation.get("nproc_per_node", 1) or 1)
    if nproc_per_node > 1:
        command = [
            python_bin,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc-per-node",
            str(nproc_per_node),
        ]
        if generation.get("master_port") is not None:
            command.extend(["--master-port", str(int(generation["master_port"]))])
        command.extend(["-m", "cosmos_framework.scripts.inference"])
    else:
        command = [python_bin, "-m", "cosmos_framework.scripts.inference"]

    command.extend(
        [
            "-i",
            str(input_jsonl),
            "-o",
            str(output_dir),
            "--checkpoint-path",
            str(checkpoint_dir),
        ]
    )

    parallelism_preset = generation.get("parallelism_preset")
    if parallelism_preset is None:
        parallelism_preset = "throughput" if nproc_per_node > 1 else "latency"
    command.extend(["--parallelism-preset", str(parallelism_preset)])

    for key in (
        "dp_shard_size",
        "dp_replicate_size",
        "cp_size",
        "cfgp_size",
        "max_num_seqs",
        "max_model_len",
    ):
        if generation.get(key) is not None:
            command.extend([f"--{key.replace('_', '-')}", str(int(generation[key]))])

    for key in (
        "sampler",
        "quantization_method",
        "quantization_fp8_granularity",
        "attention_io_layout",
        "config_file",
    ):
        if generation.get(key) is not None:
            command.extend([f"--{key.replace('_', '-')}", str(generation[key])])

    guardrails = bool(generation.get("guardrails", False))
    command.append("--guardrails" if guardrails else "--no-guardrails")

    if bool(generation.get("continue_on_error", True)):
        command.append("--keep-going")

    for key in (
        "offload_guardrail_models",
        "enable_cpu_offload",
        "enable_layerwise_offload",
        "use_torch_compile",
        "use_cuda_graphs",
        "diffusion_cache",
        "use_batched_cfg",
    ):
        if generation.get(key) is not None:
            flag = key.replace("_", "-")
            command.append(f"--{flag}" if bool(generation[key]) else f"--no-{flag}")

    experiment_overrides = generation.get("experiment_overrides")
    if experiment_overrides:
        if isinstance(experiment_overrides, str):
            overrides = [experiment_overrides]
        else:
            overrides = [str(item) for item in experiment_overrides]
        command.append("--experiment-overrides")
        command.extend(overrides)

    return command


def _locate_sample_video(sample_dir: Path) -> Path | None:
    direct = sample_dir / "vision.mp4"
    if direct.is_file() and direct.stat().st_size > 0:
        return direct
    outputs_json = sample_dir / "sample_outputs.json"
    if not outputs_json.is_file():
        return None
    try:
        payload = json.loads(outputs_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for output in payload.get("outputs") or ():
        for candidate_str in output.get("files") or ():
            candidate = Path(str(candidate_str))
            if not candidate.is_absolute():
                candidate = sample_dir / candidate
            if candidate.suffix.lower() == ".mp4" and candidate.is_file():
                return candidate
    return None


def _sample_error_detail(sample_dir: Path, returncode: int) -> str:
    outputs_json = sample_dir / "sample_outputs.json"
    if outputs_json.is_file():
        try:
            payload = json.loads(outputs_json.read_text(encoding="utf-8"))
            status = payload.get("status")
            message = payload.get("message")
            stack_trace = payload.get("stack_trace")
            parts = [
                part
                for part in (f"status={status}" if status else None, message, stack_trace)
                if part
            ]
            if parts:
                return "\n".join(str(part) for part in parts)
        except (OSError, ValueError):
            pass
    return f"cosmos-framework did not write vision.mp4 under {sample_dir} (exit code {returncode})"


def _build_subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(repo_root)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _run_batch(
    *,
    rows: list[dict[str, Any]],
    repo_root: Path,
    checkpoint_dir: Path,
    generation: dict[str, Any],
) -> None:
    continue_on_error = bool(generation.get("continue_on_error", True))
    with tempfile.TemporaryDirectory(prefix="worldatlas_cosmos3_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        input_jsonl = tmp_dir / "samples.jsonl"
        run_output_dir = tmp_dir / "outputs"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        scheduled: list[tuple[int, dict[str, Any], str, int]] = []
        payload_lines: list[str] = []
        for index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            begin_sample(sample_id, index=index + 1, total=len(rows), label="Cosmos3")
            try:
                sample_name, payload = _build_sample_payload(
                    row,
                    index=index,
                    generation=generation,
                    check_files=True,
                )
            except Exception as exc:  # noqa: BLE001 - report per-sample, continue batch
                print_status(sample_id, "failed", error=f"{type(exc).__name__}: {exc}")
                if not continue_on_error:
                    raise
                continue
            scheduled.append((index, row, sample_name, int(payload["seed"])))
            payload_lines.append(json.dumps(payload, ensure_ascii=False))

        if not scheduled:
            return

        input_jsonl.write_text("\n".join(payload_lines) + "\n", encoding="utf-8")
        command = _build_inference_command(
            python_bin=sys.executable,
            input_jsonl=input_jsonl,
            output_dir=run_output_dir,
            checkpoint_dir=checkpoint_dir,
            generation=generation,
        )
        print(
            f"[WorldAtlas:Cosmos3] launching cosmos-framework for {len(scheduled)} sample(s); "
            "checkpoint_loads=1",
            flush=True,
        )
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            env=_build_subprocess_env(repo_root),
            check=False,
        )

        for _, row, sample_name, seed in scheduled:
            sample_id = str(row["sample_id"])
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            sample_dir = run_output_dir / sample_name
            video_file = _locate_sample_video(sample_dir)
            if video_file is not None and video_file.stat().st_size > 0:
                copy_output(video_file, output_path)
                print_status(
                    sample_id,
                    "generated",
                    output_path=str(output_path),
                    seed=seed,
                    persistent_engine=True,
                    checkpoint_load_count=1,
                )
            else:
                error = _sample_error_detail(sample_dir, completed.returncode)
                print_status(sample_id, "failed", error=error)
                if not continue_on_error:
                    raise RuntimeError(error)

        if completed.returncode != 0:
            raise RuntimeError(
                f"cosmos_framework.scripts.inference failed with exit code {completed.returncode}"
            )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.checkpoint_dir is None:
        raise ValueError("Cosmos3 requires checkpoint_dir")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())

    if os.environ.get("WORLDATLAS_COSMOS3_DRY_RUN", "").lower() in {"1", "true", "yes", "on"}:
        _, sample_preview = _build_sample_payload(
            rows[0],
            index=0,
            generation=generation,
            check_files=False,
        )
        command = _build_inference_command(
            python_bin=sys.executable,
            input_jsonl=Path("<staging>/samples.jsonl"),
            output_dir=Path("<staging>/outputs"),
            checkpoint_dir=checkpoint_dir,
            generation=generation,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "persistent_engine": True,
                    "checkpoint_load_count": 1,
                    "repo_root": str(repo_root),
                    "checkpoint_dir": str(checkpoint_dir),
                    "request_count": len(rows),
                    "command": command,
                    "sample_preview": sample_preview,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Cosmos3 checkpoint directory not found: {checkpoint_dir}")

    _run_batch(
        rows=rows,
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        generation=generation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
