from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena DreamX World batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _resolve_path(value: Any, *, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base is not None:
        return (base / path).resolve()
    return path.resolve()


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


def _dreamx_command(
    *,
    repo_root: Path,
    checkpoint_dir: Path | None,
    generation: dict[str, Any],
    input_json: Path,
    output_dir: Path,
) -> list[str]:
    model_name = _resolve_path(
        generation.get("model_name") or generation.get("base_model_path"),
        base=repo_root,
    )
    transformer_path = _resolve_path(
        generation.get("transformer_path") or checkpoint_dir,
        base=repo_root,
    )
    config_path = _resolve_path(
        generation.get("config_path", "configs/wan2.2/wan_ti2v_5b.yaml"),
        base=repo_root,
    )
    if model_name is None:
        raise ValueError("DreamX World requires generation.model_name")
    if transformer_path is None:
        raise ValueError("DreamX World requires checkpoint_dir or generation.transformer_path")
    if config_path is None:
        raise ValueError("DreamX World requires config_path")

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
        resolved = _resolve_path(generation.get(key), base=repo_root)
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
    checkpoint_dir = _resolve_path(args.checkpoint_dir)
    generation = load_json(Path(args.generation_config_path))
    rows = load_batch_spec(Path(args.batch_spec_path))

    first_output = Path(rows[0]["output_path"]).expanduser().resolve()
    work_root = first_output.parent / "_dreamx_work" / Path(args.batch_spec_path).stem
    input_dir = work_root / "inputs"
    output_dir = work_root / "outputs"
    input_json = work_root / "dreamx_inputs.json"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        items.append(
            {
                "image_path": str(staged_image),
                "caption": str(row["prompt"]).replace("\n", " ").replace("\r", " ").strip(),
                "action_seq": action_seq,
                "action_speed_list": speed_list[: len(action_seq)],
            }
        )
        expected_outputs[row["sample_id"]] = output_dir / f"{staged_image.stem}_{_action_name(action_seq)}.mp4"

    input_json.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    command = _dreamx_command(
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
    subprocess.run(command, cwd=str(repo_root), env=env, check=True)

    for row in rows:
        sample_id = row["sample_id"]
        generated_path = expected_outputs[sample_id]
        copy_output(generated_path, Path(row["output_path"]))
        print_status(sample_id, "generated", output_path=row["output_path"], dreamx_output=str(generated_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
