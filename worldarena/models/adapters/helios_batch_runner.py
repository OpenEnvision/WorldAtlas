from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    latest_mp4,
    load_batch_spec,
    load_json,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Helios batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _append_value_option(command: list[str], name: str, value: Any, cast: type = str) -> None:
    if value is None:
        return
    command.extend([f"--{name}", str(cast(value))])


def _append_multi_value_option(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    command.append(f"--{name}")
    if isinstance(value, (list, tuple)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def _append_flag(command: list[str], name: str, enabled: Any) -> None:
    if enabled:
        command.append(f"--{name}")


def _build_infer_command(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    generation: dict[str, Any],
    row: dict[str, Any],
    output_folder: Path,
) -> list[str]:
    infer_script = repo_root / "infer_helios.py"
    if not infer_script.is_file():
        raise FileNotFoundError(f"Helios inference script not found: {infer_script}")

    num_gpus = int(generation.get("num_gpus", 1))
    if num_gpus > 1:
        torchrun_bin = str(Path(sys.executable).with_name("torchrun"))
        command = [
            torchrun_bin,
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(num_gpus),
            str(infer_script),
            "--enable_parallelism",
        ]
        _append_value_option(command, "cp_backend", generation.get("cp_backend", "ulysses"), str)
    else:
        command = [sys.executable, str(infer_script)]

    base_model_path = generation.get("base_model_path") or str(checkpoint_dir)
    transformer_path = generation.get("transformer_path") or base_model_path
    command.extend(
        [
            "--base_model_path",
            str(base_model_path),
            "--transformer_path",
            str(transformer_path),
            "--sample_type",
            str(generation.get("sample_type", "i2v")),
            "--image_path",
            str(row["conditioning_image"]),
            "--prompt",
            str(row["prompt"]),
            "--output_folder",
            str(output_folder),
        ]
    )

    value_options: dict[str, type] = {
        "height": int,
        "width": int,
        "num_frames": int,
        "fps": int,
        "num_inference_steps": int,
        "guidance_scale": float,
        "seed": int,
        "weight_dtype": str,
        "negative_prompt": str,
        "num_latent_frames_per_chunk": int,
        "zero_steps": int,
        "image_noise_sigma_min": float,
        "image_noise_sigma_max": float,
        "video_noise_sigma_min": float,
        "video_noise_sigma_max": float,
        "interpolation_steps": int,
        "interpolate_time": int,
        "group_offloading_type": str,
        "num_blocks_per_group": str,
    }
    for name, cast in value_options.items():
        _append_value_option(command, name, generation.get(name), cast)

    _append_multi_value_option(
        command,
        "pyramid_num_inference_steps_list",
        generation.get("pyramid_num_inference_steps_list"),
    )

    for flag in (
        "is_enable_stage2",
        "is_skip_first_chunk",
        "is_amplify_first_chunk",
        "use_zero_init",
        "use_interpolate_prompt",
        "enable_low_vram_mode",
        "enable_compile",
        "disable_flash_attention",
    ):
        _append_flag(command, flag, generation.get(flag, False))

    return command


def _safe_temp_prefix(value: Any) -> str:
    text = str(value).strip() or "sample"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)[
        :80
    ]


def _run_one(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    generation: dict[str, Any],
    row: dict[str, Any],
    row_index: int,
) -> None:
    output_path = Path(str(row["output_path"])).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"helios_{_safe_temp_prefix(row.get('prediction_stem', row_index))}_",
        dir=str(output_path.parent),
    ) as tmp_dir:
        output_folder = Path(tmp_dir)
        command = _build_infer_command(
            repo_root=repo_root,
            checkpoint_dir=checkpoint_dir,
            generation=generation,
            row=row,
            output_folder=output_folder,
        )
        env = os.environ.copy()
        pythonpaths = [str(repo_root)]
        if env.get("PYTHONPATH"):
            pythonpaths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("HF_ENABLE_PARALLEL_LOADING", "yes")
        env.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")

        subprocess.run(
            command,
            check=True,
            cwd=str(repo_root),
            env=env,
            text=True,
        )
        generated = latest_mp4(output_folder)
        copy_output(generated, output_path)
        try:
            shutil.rmtree(output_folder)
        except FileNotFoundError:
            pass


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir or "").expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Helios checkpoint_dir not found: {checkpoint_dir}")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            _run_one(
                repo_root=repo_root,
                checkpoint_dir=checkpoint_dir,
                generation=generation,
                row=row,
                row_index=row_index,
            )
            print_status(
                sample_id,
                "generated",
                output_path=str(Path(str(row["output_path"])).expanduser().resolve()),
            )
        except Exception as exc:
            failed += 1
            print_status(
                sample_id,
                "failed",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
