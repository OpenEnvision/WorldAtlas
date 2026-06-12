from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path

_DEFAULT_CHECKPOINT_FILENAME = {
    "standard": "mp_rank_00_model_states.pt",
    "distill": "mp_rank_00_model_states_distill.pt",
}
_MODEL_BASE_MARKERS = (
    "vae_3d/hyvae/config.json",
    "openai_clip-vit-large-patch14",
    "llava-llama-3-8b-v1_1-transformers",
)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Hunyuan-GameCraft subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--action_list", nargs="+", required=True)
    parser.add_argument("--action_speed_list", nargs="+", required=True, type=float)
    parser.add_argument("--model_variant", default="standard", choices=tuple(_DEFAULT_CHECKPOINT_FILENAME))
    parser.add_argument("--checkpoint_filename", default=None, type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--master_port", default=29605, type=int)
    parser.add_argument("--height", default=704, type=int)
    parser.add_argument("--width", default=1216, type=int)
    parser.add_argument("--cfg_scale", default=2.0, type=float)
    parser.add_argument("--sample_n_frames", default=33, type=int)
    parser.add_argument("--infer_steps", default=50, type=int)
    parser.add_argument("--flow_shift_eval_video", default=5.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--add_pos_prompt", default="", type=str)
    parser.add_argument("--add_neg_prompt", default="", type=str)
    parser.add_argument("--use_deepcache", default=1, type=int)
    parser.add_argument("--image_start", default=True, type=_parse_bool)
    parser.add_argument("--cpu_offload", default=True, type=_parse_bool)
    parser.add_argument("--disable_sp", default=True, type=_parse_bool)
    parser.add_argument("--use_fp8", default=False, type=_parse_bool)
    parser.add_argument("--use_sage", default=False, type=_parse_bool)
    parser.add_argument("--keep_work_dir", default=False, type=_parse_bool)
    return parser.parse_args()


def _weight_root(checkpoint_dir: str | None) -> Path:
    resolved = resolve_checkpoint_path(checkpoint_dir, kind="dir", required=True)
    if resolved is None:
        raise ValueError("Hunyuan-GameCraft requires checkpoint_dir")
    return resolved


def _find_model_base(weight_root: Path) -> Path | None:
    for candidate in (weight_root / "stdmodels", weight_root):
        if candidate.is_dir() and all((candidate / marker).exists() for marker in _MODEL_BASE_MARKERS):
            return candidate.resolve()
    return None


def _find_checkpoint_file(
    weight_root: Path,
    *,
    model_variant: str,
    checkpoint_filename: str | None,
) -> Path | None:
    filename = checkpoint_filename or _DEFAULT_CHECKPOINT_FILENAME[model_variant]
    for candidate in (weight_root / "gamecraft_models" / filename, weight_root / filename):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _require_repo_layout(repo_root: Path) -> Path:
    script_path = repo_root / "hymm_sp" / "sample_batch.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Hunyuan-GameCraft entrypoint not found: {script_path}")
    return script_path.resolve()


def _build_env(
    *,
    repo_root: Path,
    model_base: Path,
    cpu_offload: bool,
    disable_sp: bool,
) -> dict[str, str]:
    env = apply_checkpoint_env()
    runtime_bin = Path(os.path.abspath(sys.executable)).parent
    path_entries = [str(runtime_bin)]
    if env.get("PATH"):
        path_entries.append(env["PATH"])
    env["PATH"] = os.pathsep.join(path_entries)

    pythonpath_entries = [str(repo_root)]
    if env.get("WORLDARENA_HUNYUAN_FLASH_ATTN_FALLBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        project_root = Path(__file__).resolve().parents[3]
        flash_attn_fallback = project_root / "worldarena" / "_compat" / "flash_attn_fallback"
        pythonpath_entries.insert(0, str(flash_attn_fallback))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["MODEL_BASE"] = str(model_base)
    env["CPU_OFFLOAD"] = "1" if cpu_offload else "0"
    env["DISABLE_SP"] = "1" if disable_sp else "0"
    return env


def _distributed_command(
    *,
    script_path: Path,
    conditioning_image: Path,
    checkpoint_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.num_gpus),
        "--master_port",
        str(args.master_port),
        str(script_path),
        "--image-path",
        str(conditioning_image),
        "--prompt",
        args.prompt,
        "--ckpt",
        str(checkpoint_path),
        "--video-size",
        str(args.height),
        str(args.width),
        "--cfg-scale",
        str(args.cfg_scale),
        "--action-list",
        *args.action_list,
        "--action-speed-list",
        *(str(value) for value in args.action_speed_list),
        "--seed",
        str(args.seed),
        "--sample-n-frames",
        str(args.sample_n_frames),
        "--infer-steps",
        str(args.infer_steps),
        "--flow-shift-eval-video",
        str(args.flow_shift_eval_video),
        "--use-deepcache",
        str(args.use_deepcache),
        "--save-path",
        str(output_dir),
    ]
    if args.image_start:
        command.append("--image-start")
    if args.cpu_offload:
        command.append("--cpu-offload")
    if args.use_fp8:
        command.append("--use-fp8")
    if args.use_sage:
        command.append("--use-sage")
    if args.add_pos_prompt:
        command.extend(["--add-pos-prompt", args.add_pos_prompt])
    if args.add_neg_prompt:
        command.extend(["--add-neg-prompt", args.add_neg_prompt])
    return command


@contextlib.contextmanager
def _work_dir(keep_work_dir: bool):
    if keep_work_dir:
        path = Path(tempfile.mkdtemp(prefix="worldarena_hunyuan_gamecraft_"))
        try:
            yield path
        finally:
            pass
        return

    with tempfile.TemporaryDirectory(prefix="worldarena_hunyuan_gamecraft_") as temp_dir_raw:
        yield Path(temp_dir_raw)


def main() -> None:
    args = parse_args()
    if args.num_gpus <= 0:
        raise ValueError(f"Hunyuan-GameCraft num_gpus must be positive, got {args.num_gpus}")
    if len(args.action_list) != len(args.action_speed_list):
        raise ValueError(
            "Hunyuan-GameCraft action_list/action_speed_list length mismatch: "
            f"{len(args.action_list)} vs {len(args.action_speed_list)}"
        )

    repo_root = Path(args.repo_root).expanduser().resolve()
    conditioning_image = Path(args.conditioning_image).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not conditioning_image.exists():
        raise FileNotFoundError(f"Hunyuan-GameCraft conditioning image not found: {conditioning_image}")

    script_path = _require_repo_layout(repo_root)
    weight_root = _weight_root(args.checkpoint_dir)
    model_base = _find_model_base(weight_root)
    checkpoint_path = _find_checkpoint_file(
        weight_root,
        model_variant=args.model_variant,
        checkpoint_filename=args.checkpoint_filename,
    )
    if model_base is None:
        raise FileNotFoundError(
            f"Hunyuan-GameCraft stdmodels root not found under: {weight_root}"
        )
    if checkpoint_path is None:
        filename = args.checkpoint_filename or _DEFAULT_CHECKPOINT_FILENAME[args.model_variant]
        raise FileNotFoundError(
            f"Hunyuan-GameCraft checkpoint file {filename} not found under: {weight_root}"
        )

    with _work_dir(args.keep_work_dir) as work_dir:
        input_dir = work_dir / "inputs"
        output_dir = work_dir / "results"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        local_image = input_dir / f"{args.sample_name}{conditioning_image.suffix or '.png'}"
        shutil.copy2(conditioning_image, local_image)

        command = _distributed_command(
            script_path=script_path,
            conditioning_image=local_image,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            args=args,
        )
        subprocess.run(
            command,
            check=True,
            cwd=str(repo_root),
            env=_build_env(
                repo_root=repo_root,
                model_base=model_base,
                cpu_offload=bool(args.cpu_offload),
                disable_sp=bool(args.disable_sp),
            ),
        )

        generated_path = output_dir / f"{args.sample_name}.mp4"
        if not generated_path.exists():
            raise FileNotFoundError(f"Hunyuan-GameCraft output was not written: {generated_path}")
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(generated_path), str(output_path))


if __name__ == "__main__":
    main()
