"""Subprocess runner that invokes HY-WorldPlay's ``hyvideo/generate.py``.

The adapter writes pose JSON and conditioning paths; this module translates them
into the upstream CLI, runs inference in a temp output dir, then copies the
final mp4 into the benchmark predictions tree.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from worldatlas.common.checkpoints import apply_checkpoint_env, resolve_project_path
from worldatlas.models.adapters.batch_runner_common import begin_sample, print_status


def _str_to_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _str_to_optional_bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return _str_to_bool(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas HY-WorldPlay subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--entrypoint", default="hyvideo/generate.py", type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--pose_json", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--resolution", default="480p", type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--action_ckpt", required=True, type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--num_inference_steps", default=4, type=int)
    parser.add_argument("--video_length", default=125, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--sr", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--save_pre_sr_video", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--rewrite", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--offloading", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--group_offloading", type=_str_to_optional_bool, nargs="?", const=True, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", type=str)
    parser.add_argument("--enable_torch_compile", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--few_step", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--model_type", choices=("bi", "ar"), default="ar", type=str)
    parser.add_argument("--height", default=None, type=int)
    parser.add_argument("--width", default=None, type=int)
    parser.add_argument("--with_ui", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_sageattn", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--sage_blocks_range", default="0-53", type=str)
    parser.add_argument("--use_vae_parallel", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_fp8_gemm", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--quant_type", default="fp8-per-block", type=str)
    parser.add_argument("--include_patterns", default="double_blocks", type=str)
    parser.add_argument(
        "--transformer_resident_ar_rollout",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--launcher", choices=("python", "torchrun"), default="python", type=str)
    parser.add_argument("--nproc_per_node", default=1, type=int)
    parser.add_argument("--master_port", default=None, type=int)
    parser.add_argument("--gpu_index", default=0, type=int)
    parser.add_argument("--cuda_visible_devices", default=None, type=str)
    return parser.parse_args()


def _build_env(
    repo_root: Path,
    *,
    gpu_index: int,
    cuda_visible_devices: str | None,
    nproc_per_node: int,
) -> dict[str, str]:
    env = apply_checkpoint_env()
    pythonpaths = [str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    elif nproc_per_node == 1 and gpu_index >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def _entrypoint_command(args: argparse.Namespace, entrypoint_path: Path) -> list[str]:
    """Choose between plain python and torchrun based on launcher config."""
    nproc_per_node = max(int(args.nproc_per_node), 1)
    if args.launcher == "torchrun" or nproc_per_node > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(nproc_per_node),
        ]
        if args.master_port is not None:
            command.extend(["--master_port", str(int(args.master_port))])
        command.append(str(entrypoint_path))
        return command
    return [sys.executable, str(entrypoint_path)]


def _resolve_generated_video(output_dir: Path) -> Path:
    """Prefer super-resolved output when ``--sr`` is enabled."""
    for candidate in (output_dir / "gen_sr.mp4", output_dir / "gen.mp4"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"HY-WorldPlay output video was not written under: {output_dir}")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    entrypoint_path = (repo_root / args.entrypoint).resolve()
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"HY-WorldPlay entrypoint not found: {entrypoint_path}")

    conditioning_image = Path(args.conditioning_image).expanduser().resolve()
    pose_json = Path(args.pose_json).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if not conditioning_image.exists():
        raise FileNotFoundError(f"conditioning image not found: {conditioning_image}")
    if not pose_json.exists():
        raise FileNotFoundError(f"pose json not found: {pose_json}")

    env = _build_env(
        repo_root,
        gpu_index=int(args.gpu_index),
        cuda_visible_devices=args.cuda_visible_devices,
        nproc_per_node=max(int(args.nproc_per_node), 1),
    )

    with tempfile.TemporaryDirectory(prefix="worldatlas_hy_worldplay_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        runtime_output_dir = temp_dir / "outputs"
        runtime_output_dir.mkdir(parents=True, exist_ok=True)
        model_path = resolve_project_path(args.model_path)
        if model_path is None or not model_path.is_dir():
            raise FileNotFoundError(f"HY-WorldPlay model path is missing: {args.model_path}")
        action_ckpt = resolve_project_path(args.action_ckpt)
        if action_ckpt is None or not action_ckpt.is_file():
            raise FileNotFoundError(f"HY-WorldPlay action checkpoint is missing: {args.action_ckpt}")

        command = _entrypoint_command(args, entrypoint_path)
        command.extend(
            [
                "--pose",
                str(pose_json),
                "--prompt",
                str(args.prompt),
                "--negative_prompt",
                str(args.negative_prompt),
                "--resolution",
                str(args.resolution),
                "--model_path",
                str(model_path),
                "--action_ckpt",
                str(action_ckpt),
                "--aspect_ratio",
                str(args.aspect_ratio),
                "--num_inference_steps",
                str(int(args.num_inference_steps)),
                "--video_length",
                str(int(args.video_length)),
                "--seed",
                str(int(args.seed)),
                "--image_path",
                str(conditioning_image),
                "--output_path",
                str(runtime_output_dir),
                "--dtype",
                str(args.dtype),
                "--model_type",
                str(args.model_type),
                "--sr",
                "true" if bool(args.sr) else "false",
                "--save_pre_sr_video",
                "true" if bool(args.save_pre_sr_video) else "false",
                "--rewrite",
                "true" if bool(args.rewrite) else "false",
                "--offloading",
                "true" if bool(args.offloading) else "false",
                "--enable_torch_compile",
                "true" if bool(args.enable_torch_compile) else "false",
                "--few_step",
                "true" if bool(args.few_step) else "false",
                "--with-ui",
                "true" if bool(args.with_ui) else "false",
                "--use_sageattn",
                "true" if bool(args.use_sageattn) else "false",
                "--sage_blocks_range",
                str(args.sage_blocks_range),
                "--use_vae_parallel",
                "true" if bool(args.use_vae_parallel) else "false",
                "--use_fp8_gemm",
                "true" if bool(args.use_fp8_gemm) else "false",
                "--quant_type",
                str(args.quant_type),
                "--include_patterns",
                str(args.include_patterns),
                "--transformer_resident_ar_rollout",
                "true" if bool(args.transformer_resident_ar_rollout) else "false",
            ]
        )
        if args.group_offloading is not None:
            command.extend(["--group_offloading", "true" if bool(args.group_offloading) else "false"])
        if args.height is not None:
            command.extend(["--height", str(int(args.height))])
        if args.width is not None:
            command.extend(["--width", str(int(args.width))])

        sample_id = output_path.stem
        begin_sample(sample_id, index=1, total=1, label="HY-WorldPlay")
        try:
            subprocess.run(
                command,
                check=True,
                cwd=str(repo_root),
                env=env,
            )

            generated_video = _resolve_generated_video(runtime_output_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.unlink(missing_ok=True)
            shutil.copy2(generated_video, output_path)

            pre_sr_video = runtime_output_dir / "gen.mp4"
            if generated_video.name != "gen.mp4" and bool(args.save_pre_sr_video) and pre_sr_video.exists():
                pre_sr_output = output_path.with_name(f"{output_path.stem}.pre_sr{output_path.suffix}")
                pre_sr_output.unlink(missing_ok=True)
                shutil.copy2(pre_sr_video, pre_sr_output)
            print_status(sample_id, "generated", output_path=str(output_path))
        except Exception as exc:
            print_status(sample_id, "failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
