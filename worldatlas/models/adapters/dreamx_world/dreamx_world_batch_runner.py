from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from worldatlas.common.checkpoints import project_root as worldatlas_project_root
from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    begin_sample,
    log_pipeline,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas DreamX World batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _resolve_path(value: Any, *, bases: list[Path] | None = None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        path = value.expanduser()
    else:
        text = str(value).strip()
        if not text:
            return None
        path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    search_bases = bases or [Path.cwd()]
    for base in search_bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (search_bases[0] / path).resolve()


def _checkpoint_search_bases(repo_root: Path, checkpoint_dir: Path | None) -> list[Path]:
    """Prefer WorldAtlas/arena ckpt trees over the DreamX repo root.

    Model yaml uses paths like ``../ckpts/GD-ML--DreamX-World-5B-Cam`` relative to
    WorldAtlas, not ``model/DreamX-World``. Resolving them only against repo_root
    produced ``model/ckpts/...``, which does not exist.
    """
    arena_root = worldatlas_project_root()
    bases = [arena_root, arena_root.parent, repo_root]
    if checkpoint_dir is not None:
        bases.extend([checkpoint_dir.parent, checkpoint_dir])
    unique: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        resolved = base.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _as_int_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [part for part in value.replace(",", " ").split() if part]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = []
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return default


def _stage_image(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _action_name(action_seq: list[str]) -> str:
    return "_".join(action_seq) if action_seq else "default"


AR_FORCING_ENTRYPOINT = "inference_ar_forcing.py"


def _uses_ar_forcing(generation: dict[str, Any]) -> bool:
    """Whether this run takes the chunk-wise causal entrypoint.

    ``inference_dreamx5b.py`` denoises the whole clip at once and tops out near five
    seconds. ``inference_ar_forcing.py`` is the chunk-wise causal path upstream
    recommends for long videos, which is what a minute-long memory rollout needs.
    """
    return Path(str(generation.get("entrypoint", "inference_dreamx5b.py"))).name == AR_FORCING_ENTRYPOINT


def _dreamx_ar_command(
    *,
    repo_root: Path,
    checkpoint_dir: Path | None,
    generation: dict[str, Any],
    input_json: Path,
    output_dir: Path,
) -> list[str]:
    """Build the AR-forcing command; its CLI shares almost nothing with the 5B one."""
    ckpt_bases = _checkpoint_search_bases(repo_root, checkpoint_dir)
    config_path = _resolve_path(
        generation.get("config_path", "configs/dreamx-ar/causal_camera_forcing_5b.yaml"),
        bases=[repo_root],
    )
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(
            f"DreamX AR config not found: {generation.get('config_path')!r} under {repo_root}"
        )
    model_name = _resolve_path(
        generation.get("model_name") or generation.get("base_model_path"),
        bases=ckpt_bases,
    )
    if model_name is None or not model_name.exists():
        raise FileNotFoundError("DreamX AR requires generation.model_name (Wan2.2 base weights)")
    transformer_path = _resolve_path(
        generation.get("transformer_path", "configs/dreamx-ar"),
        bases=[repo_root, *ckpt_bases],
    )
    if transformer_path is None or not (transformer_path / "config.json").is_file():
        raise FileNotFoundError(
            f"DreamX AR transformer config.json not found under {transformer_path}"
        )
    base_checkpoint = _resolve_path(generation.get("base_checkpoint_path"), bases=ckpt_bases)
    if base_checkpoint is None or not base_checkpoint.exists():
        raise FileNotFoundError(
            "DreamX AR requires generation.base_checkpoint_path "
            "(DreamX-World-5B model.safetensors or a generator_ema .pt); "
            f"got {generation.get('base_checkpoint_path')!r}"
        )

    num_output_frames = int(generation.get("num_output_frames", 21))
    if num_output_frames % 3 != 0:
        raise ValueError(
            f"DreamX AR num_output_frames must be divisible by 3, got {num_output_frames}"
        )

    command = [
        sys.executable,
        str(repo_root / str(generation.get("entrypoint", AR_FORCING_ENTRYPOINT))),
        "--config_path",
        str(config_path),
        "--model_name",
        str(model_name),
        "--transformer_path",
        str(transformer_path),
        "--base_checkpoint_path",
        str(base_checkpoint),
        "--data_path",
        str(input_json),
        "--output_folder",
        str(output_dir),
        "--num_output_frames",
        str(num_output_frames),
        "--fps",
        str(int(generation.get("fps", 16))),
        "--seed",
        str(int(generation.get("seed", 42))),
        "--color_correction_strength",
        str(float(generation.get("color_correction_strength", 1.0))),
    ]
    # Upstream recommends per-chunk relative poses for long rollouts.
    if bool(generation.get("chunk_relative", True)):
        command.append("--chunk_relative")
    for key in ("vae_path", "checkpoint_path", "lora_ckpt"):
        resolved = _resolve_path(generation.get(key), bases=ckpt_bases)
        if resolved is not None:
            command.extend([f"--{key}", str(resolved)])
    return command


def _dreamx_command(
    *,
    repo_root: Path,
    checkpoint_dir: Path | None,
    generation: dict[str, Any],
    input_json: Path,
    output_dir: Path,
) -> list[str]:
    ckpt_bases = _checkpoint_search_bases(repo_root, checkpoint_dir)
    model_name = _resolve_path(
        generation.get("model_name") or generation.get("base_model_path"),
        bases=ckpt_bases,
    )
    transformer_path = _resolve_path(generation.get("transformer_path"), bases=ckpt_bases) or checkpoint_dir
    config_path = _resolve_path(
        generation.get("config_path", "configs/wan2.2/wan_ti2v_5b.yaml"),
        bases=[repo_root],
    )
    if model_name is None:
        raise ValueError("DreamX World requires generation.model_name")
    if transformer_path is None:
        raise ValueError("DreamX World requires checkpoint_dir or generation.transformer_path")
    if config_path is None:
        raise ValueError("DreamX World requires config_path")
    transformer_config = transformer_path / "config.json"
    if not transformer_config.is_file():
        raise FileNotFoundError(f"DreamX transformer config not found: {transformer_config}")
    if not model_name.exists():
        raise FileNotFoundError(f"DreamX Wan backbone not found: {model_name}")
    if not config_path.is_file():
        raise FileNotFoundError(f"DreamX config not found: {config_path}")

    ulysses_degree = int(generation.get("ulysses_degree", 8))
    ring_degree = int(generation.get("ring_degree", 1))
    nproc_per_node = int(generation.get("nproc_per_node", max(ulysses_degree * ring_degree, 1)))
    entrypoint = repo_root / str(generation.get("entrypoint", "inference_dreamx5b.py"))
    if nproc_per_node > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(nproc_per_node),
            "--master_addr",
            str(generation.get("master_addr", "127.0.0.1")),
            "--master_port",
            str(int(generation.get("master_port", 29627))),
            str(entrypoint),
        ]
    else:
        command = [sys.executable, str(entrypoint)]

    sample_height, sample_width = _as_int_pair(
        generation.get("sample_size"),
        (
            int(generation.get("sample_height", 704)),
            int(generation.get("sample_width", 1280)),
        ),
    )
    command.extend(
        [
            "--config_path",
            str(config_path),
            "--model_name",
            str(model_name),
            "--transformer_path",
            str(transformer_path),
            "--input_dir",
            str(input_json),
            "--output_dir",
            str(output_dir),
            "--cam_method",
            str(generation.get("cam_method", "prope")),
            "--sample_size",
            str(sample_height),
            str(sample_width),
            "--video_length",
            str(int(generation.get("video_length", 121))),
            "--fps",
            str(int(generation.get("fps", 24))),
            "--guidance_scale",
            str(float(generation.get("guidance_scale", 3.0))),
            "--num_inference_steps",
            str(int(generation.get("num_inference_steps", 50))),
            "--seed",
            str(int(generation.get("seed", 42))),
            "--weight_dtype",
            str(generation.get("weight_dtype", "bfloat16")),
            "--ulysses_degree",
            str(ulysses_degree),
            "--ring_degree",
            str(ring_degree),
        ]
    )
    if bool(generation.get("add_control_adapter", True)):
        command.append("--add_control_adapter")
    optional_paths = (
        "transformer_high_path",
        "vae_path",
        "lora_path",
        "lora_high_path",
    )
    for key in optional_paths:
        resolved = _resolve_path(generation.get(key), bases=ckpt_bases)
        if resolved is not None:
            command.extend([f"--{key}", str(resolved)])
    optional_values = (
        "negative_prompt",
        "sampler_name",
        "shift",
        "GPU_memory_mode",
        "lora_weight",
        "lora_high_weight",
    )
    for key in optional_values:
        if generation.get(key) is not None:
            command.extend([f"--{key}", str(generation[key])])
    return command


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = _resolve_path(
        args.checkpoint_dir,
        bases=_checkpoint_search_bases(repo_root, None),
    )
    generation = load_json(Path(args.generation_config_path))
    rows = load_batch_spec(Path(args.batch_spec_path))

    first_output = Path(rows[0]["output_path"]).expanduser().resolve()
    work_root = first_output.parent / "_dreamx_work" / Path(args.batch_spec_path).stem
    input_dir = work_root / "inputs"
    output_dir = work_root / "outputs"
    input_json = work_root / "dreamx_inputs.json"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ar_forcing = _uses_ar_forcing(generation)
    items: list[dict[str, Any]] = []
    expected_outputs: dict[str, Path] = {}
    for row in rows:
        source = Path(row["conditioning_image"]).expanduser().resolve()
        suffix = source.suffix or ".png"
        staged_image = _stage_image(source, input_dir / f"{row['prediction_stem']}{suffix}")
        action_seq = [str(item).strip().lower() for item in row.get("dreamx_action_seq", ["w"]) if str(item).strip()]
        if not action_seq:
            action_seq = ["w"]
        speed_list = [int(item) for item in row.get("dreamx_action_speed_list", [])]
        if not speed_list:
            speed_list = [int(generation.get("action_speed", 4))] * len(action_seq)
        while len(speed_list) < len(action_seq):
            speed_list.append(speed_list[-1])
        item: dict[str, Any] = {
            "image_path": str(staged_image),
            "caption": str(row["prompt"]).replace("\n", " ").replace("\r", " ").strip(),
            "action_seq": action_seq,
            "action_speed_list": speed_list[: len(action_seq)],
        }
        if ar_forcing:
            # The AR entrypoint names outputs '{task_id}_{parent}_{stem}.mp4' and
            # falls back to the loop index for task_id, so pin it to keep the
            # mapping stable when a shard is retried.
            task_id = str(row["prediction_stem"])
            item["task_id"] = task_id
            expected_outputs[row["sample_id"]] = (
                output_dir / f"{task_id}_{input_dir.name}_{staged_image.stem}.mp4"
            )
        else:
            expected_outputs[row["sample_id"]] = (
                output_dir / f"{staged_image.stem}_{_action_name(action_seq)}.mp4"
            )
        items.append(item)

    input_json.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    build_command = _dreamx_ar_command if ar_forcing else _dreamx_command
    command = build_command(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        generation=generation,
        input_json=input_json,
        output_dir=output_dir,
    )
    env = os.environ.copy()
    if generation.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = str(generation["cuda_visible_devices"])
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # torchvision 0.20 still sets frame.pict_type = "NONE"; PyAV 14+ wants an int.
    compat_dir = worldatlas_project_root() / "worldatlas" / "_compat" / "av_write_video"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(compat_dir)
        if not existing_pythonpath
        else f"{compat_dir}{os.pathsep}{existing_pythonpath}"
    )
    log_pipeline("pipeline_loading", label="DreamX", samples=len(rows))
    if rows:
        begin_sample(str(rows[0]["sample_id"]), index=1, total=len(rows), label="DreamX")
    subprocess.run(command, cwd=str(repo_root), env=env, check=True)

    for row_index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        generated_path = expected_outputs[sample_id]
        copy_output(generated_path, Path(row["output_path"]))
        print_status(
            sample_id,
            "generated",
            index=row_index,
            total=len(rows),
            output_path=row["output_path"],
            dreamx_output=str(generated_path),
            label="DreamX",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
