"""Batch subprocess runner for Hunyuan GameCraft inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path
from worldarena.models.adapters.hunyuan_gamecraft_runner import (
    _DEFAULT_CHECKPOINT_FILENAME,
    _apply_nccl_defaults,
    _build_env,
    _find_checkpoint_file,
    _find_model_base,
    _parse_bool,
    _require_repo_layout,
    _weight_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WorldAtlas Arena Hunyuan-GameCraft batch subprocess runner."
    )
    parser.add_argument("--distributed_worker", "--distributed-worker", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", default=None)
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--model_base", default=None, type=str)
    parser.add_argument("--batch_spec", required=True, type=str)
    parser.add_argument("--work_output_dir", default=None, type=str)
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


def _load_batch_spec(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("sample_id", "conditioning_image", "output_path", "sample_name", "prompt"):
                if not str(row.get(key, "")).strip():
                    raise ValueError(f"batch row {line_number} missing required key: {key}")
            actions = row.get("actions")
            action_speeds = row.get("action_speeds")
            if not isinstance(actions, list) or not actions:
                raise ValueError(f"batch row {line_number} has empty actions")
            if not isinstance(action_speeds, list) or len(action_speeds) != len(actions):
                raise ValueError(f"batch row {line_number} action_speeds length mismatch")
            rows.append(row)
    if not rows:
        raise ValueError(f"Hunyuan-GameCraft batch spec is empty: {path}")
    return rows


def _distributed_command(
    *,
    script_path: Path,
    checkpoint_path: Path,
    model_base: Path,
    batch_spec: Path,
    work_output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(args.num_gpus),
        "--master_port",
        str(args.master_port),
        str(script_path),
        "--distributed_worker",
        "--repo_root",
        str(Path(args.repo_root).expanduser().resolve()),
        "--checkpoint_path",
        str(checkpoint_path),
        "--model_base",
        str(model_base),
        "--batch_spec",
        str(batch_spec),
        "--work_output_dir",
        str(work_output_dir),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--cfg_scale",
        str(args.cfg_scale),
        "--sample_n_frames",
        str(args.sample_n_frames),
        "--infer_steps",
        str(args.infer_steps),
        "--flow_shift_eval_video",
        str(args.flow_shift_eval_video),
        "--seed",
        str(args.seed),
        "--add_pos_prompt",
        args.add_pos_prompt,
        "--add_neg_prompt",
        args.add_neg_prompt,
        "--use_deepcache",
        str(args.use_deepcache),
        "--image_start",
        "true" if bool(args.image_start) else "false",
        "--cpu_offload",
        "true" if bool(args.cpu_offload) else "false",
        "--disable_sp",
        "true" if bool(args.disable_sp) else "false",
        "--use_fp8",
        "true" if bool(args.use_fp8) else "false",
        "--use_sage",
        "true" if bool(args.use_sage) else "false",
    ]
    if args.checkpoint_filename:
        command.extend(["--checkpoint_filename", args.checkpoint_filename])
    return command


@contextlib.contextmanager
def _work_dir(keep_work_dir: bool):
    if keep_work_dir:
        path = Path(tempfile.mkdtemp(prefix="worldarena_hunyuan_gamecraft_batch_"))
        try:
            yield path
        finally:
            pass
        return

    with tempfile.TemporaryDirectory(prefix="worldarena_hunyuan_gamecraft_batch_") as temp_dir_raw:
        yield Path(temp_dir_raw)


def _parent_main(args: argparse.Namespace) -> None:
    if args.num_gpus <= 0:
        raise ValueError(f"Hunyuan-GameCraft num_gpus must be positive, got {args.num_gpus}")

    repo_root = Path(args.repo_root).expanduser().resolve()
    script_path = Path(__file__).resolve()
    _require_repo_layout(repo_root)

    batch_spec = Path(args.batch_spec).expanduser().resolve()
    rows = _load_batch_spec(batch_spec)
    for row in rows:
        conditioning_image = Path(row["conditioning_image"]).expanduser().resolve()
        if not conditioning_image.exists():
            raise FileNotFoundError(
                f"Hunyuan-GameCraft conditioning image not found: {conditioning_image}"
            )

    weight_root = _weight_root(args.checkpoint_dir)
    model_base = _find_model_base(weight_root)
    checkpoint_path = _find_checkpoint_file(
        weight_root,
        model_variant=args.model_variant,
        checkpoint_filename=args.checkpoint_filename,
    )
    if model_base is None:
        raise FileNotFoundError(f"Hunyuan-GameCraft stdmodels root not found under: {weight_root}")
    if checkpoint_path is None:
        filename = args.checkpoint_filename or _DEFAULT_CHECKPOINT_FILENAME[args.model_variant]
        raise FileNotFoundError(
            f"Hunyuan-GameCraft checkpoint file {filename} not found under: {weight_root}"
        )

    with _work_dir(bool(args.keep_work_dir)) as work_dir:
        work_output_dir = work_dir / "results"
        work_output_dir.mkdir(parents=True, exist_ok=True)
        command = _distributed_command(
            script_path=script_path,
            checkpoint_path=checkpoint_path,
            model_base=model_base,
            batch_spec=batch_spec,
            work_output_dir=work_output_dir,
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

        for row in rows:
            generated_path = work_output_dir / f"{row['sample_name']}.mp4"
            output_path = Path(row["output_path"]).expanduser().resolve()
            if output_path.exists():
                continue
            if not generated_path.exists():
                raise FileNotFoundError(
                    f"Hunyuan-GameCraft output was not written: {generated_path}"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(generated_path), str(output_path))


def _hymm_args_for(args: argparse.Namespace, first_row: dict[str, Any]) -> argparse.Namespace:
    from hymm_sp.config import parse_args as parse_hymm_args

    argv = [
        "--ckpt",
        str(args.checkpoint_path),
        "--video-size",
        str(args.height),
        str(args.width),
        "--cfg-scale",
        str(args.cfg_scale),
        "--sample-n-frames",
        str(args.sample_n_frames),
        "--infer-steps",
        str(args.infer_steps),
        "--flow-shift-eval-video",
        str(args.flow_shift_eval_video),
        "--seed",
        str(args.seed),
        "--add-pos-prompt",
        args.add_pos_prompt,
        "--add-neg-prompt",
        args.add_neg_prompt,
        "--use-deepcache",
        str(args.use_deepcache),
        "--prompt",
        str(first_row["prompt"]),
        "--image-path",
        str(first_row["conditioning_image"]),
        "--save-path",
        str(args.work_output_dir),
        "--action-list",
        *(str(action) for action in first_row["actions"]),
        "--action-speed-list",
        *(str(speed) for speed in first_row["action_speeds"]),
    ]
    if args.image_start:
        argv.append("--image-start")
    if args.cpu_offload:
        argv.append("--cpu-offload")
    if args.use_fp8:
        argv.append("--use-fp8")
    if args.use_sage:
        argv.append("--use-sage")
    previous_argv = sys.argv
    try:
        sys.argv = ["hunyuan_gamecraft_batch_worker", *argv]
        return parse_hymm_args()
    finally:
        sys.argv = previous_argv


def _sync_distributed() -> None:
    """Sync distributed."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _crop_resize_transform(height: int, width: int):
    import torchvision.transforms as transforms

    class CropResize:
        def __init__(self, size: tuple[int, int]) -> None:
            self.target_h, self.target_w = size

        def __call__(self, img):
            img_w, img_h = img.size
            scale = max(self.target_w / img_w, self.target_h / img_h)
            new_size = (int(img_h * scale), int(img_w * scale))
            resized_img = transforms.Resize(
                new_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )(img)
            return transforms.CenterCrop((self.target_h, self.target_w))(resized_img)

    return transforms.Compose(
        [
            CropResize((height, width)),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def _encode_reference_image(
    *,
    sampler,
    runtime_args: argparse.Namespace,
    image_path: Path,
    device,
    height: int,
    width: int,
):
    import torch
    from PIL import Image

    ref_image_transform = _crop_resize_transform(height, width)
    raw_ref_images = [Image.open(image_path).convert("RGB")]
    ref_images_pixel_values = [ref_image_transform(ref_image) for ref_image in raw_ref_images]
    ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        if runtime_args.cpu_offload:
            sampler.vae.quant_conv.to("cuda")
            sampler.vae.encoder.to("cuda")

        sampler.pipeline.vae.enable_tiling()
        raw_last_latents = sampler.vae.encode(
            ref_images_pixel_values
        ).latent_dist.sample().to(dtype=torch.float16)
        raw_last_latents.mul_(sampler.vae.config.scaling_factor)
        raw_ref_latents = raw_last_latents.clone()
        sampler.pipeline.vae.disable_tiling()

        if runtime_args.cpu_offload:
            sampler.vae.quant_conv.to("cpu")
            sampler.vae.encoder.to("cpu")

    return raw_ref_images, raw_last_latents, raw_ref_latents


def _generate_one(
    *,
    sampler,
    runtime_args: argparse.Namespace,
    row: dict[str, Any],
    args: argparse.Namespace,
    device,
    rank: int,
) -> None:
    import torch
    from hymm_sp.data_kits.data_tools import save_videos_grid

    image_path = Path(row["conditioning_image"]).expanduser().resolve()
    prompt = str(row["prompt"])
    actions = [str(action) for action in row["actions"]]
    action_speeds = [float(speed) for speed in row["action_speeds"]]
    seed = int(row.get("seed", args.seed)) if args.seed else random.randint(0, 1_000_000)
    negative_prompt = str(args.add_neg_prompt)

    if not args.image_start:
        raise ValueError("WorldAtlas Arena Hunyuan-GameCraft batch runner currently expects image_start")

    ref_images, last_latents, ref_latents = _encode_reference_image(
        sampler=sampler,
        runtime_args=runtime_args,
        image_path=image_path,
        device=device,
        height=int(args.height),
        width=int(args.width),
    )

    out_cat = None
    output_path = Path(row["output_path"]).expanduser().resolve()
    partial_path = Path(args.work_output_dir) / f"{row['sample_name']}.partial.mp4"
    for index, action_id in enumerate(actions):
        outputs = sampler.predict(
            prompt=prompt,
            action_id=action_id,
            action_speed=action_speeds[index],
            is_image=(index == 0 and args.image_start),
            size=(int(args.height), int(args.width)),
            seed=seed,
            last_latents=last_latents,
            ref_latents=ref_latents,
            video_length=int(args.sample_n_frames),
            guidance_scale=float(args.cfg_scale),
            num_images_per_prompt=runtime_args.num_images,
            negative_prompt=negative_prompt,
            infer_steps=int(args.infer_steps),
            flow_shift=float(args.flow_shift_eval_video),
            use_linear_quadratic_schedule=runtime_args.use_linear_quadratic_schedule,
            linear_schedule_end=runtime_args.linear_schedule_end,
            use_deepcache=int(args.use_deepcache),
            cpu_offload=bool(args.cpu_offload),
            ref_images=ref_images,
            output_dir=str(args.work_output_dir),
            return_latents=True,
            use_sage=bool(args.use_sage),
        )
        ref_latents = outputs["ref_latents"]
        last_latents = outputs["last_latents"]

        if rank == 0:
            sub_samples = outputs["samples"][0]
            out_cat = sub_samples if out_cat is None else torch.cat([out_cat, sub_samples], dim=2)
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            save_videos_grid(out_cat, str(partial_path), n_rows=1, fps=24)

    if rank == 0:
        if not partial_path.exists():
            raise FileNotFoundError(f"Hunyuan-GameCraft partial output was not written: {partial_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(partial_path), str(output_path))

    _sync_distributed()


def _worker_main(args: argparse.Namespace) -> None:
    if args.checkpoint_path is None:
        raise ValueError("distributed Hunyuan-GameCraft worker requires --checkpoint_path")
    if args.model_base is None:
        raise ValueError("distributed Hunyuan-GameCraft worker requires --model_base")
    if args.work_output_dir is None:
        raise ValueError("distributed Hunyuan-GameCraft worker requires --work_output_dir")

    repo_root = Path(args.repo_root).expanduser().resolve()
    sys.path.insert(0, str(repo_root))
    os.environ["MODEL_BASE"] = str(Path(args.model_base).expanduser().resolve())
    os.environ["CPU_OFFLOAD"] = "1" if bool(args.cpu_offload) else "0"
    os.environ["DISABLE_SP"] = "1" if bool(args.disable_sp) else "0"
    os.environ.update(apply_checkpoint_env())
    # Must run before torch import / process-group init so NCCL picks up the
    # transport overrides. Covers direct torchrun launches that bypass the
    # parent _build_env path.
    _apply_nccl_defaults(os.environ)

    import torch
    from diffusers.hooks import apply_group_offloading
    from hymm_sp.modules.parallel_states import initialize_distributed, nccl_info
    from hymm_sp.sample_inference import HunyuanVideoSampler

    rows = _load_batch_spec(Path(args.batch_spec).expanduser().resolve())
    Path(args.work_output_dir).mkdir(parents=True, exist_ok=True)
    runtime_args = _hymm_args_for(args, rows[0])

    initialize_distributed(args.seed)
    rank = 0
    device = torch.device("cuda")
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = torch.distributed.get_rank()

    sampler = HunyuanVideoSampler.from_pretrained(
        str(args.checkpoint_path),
        args=runtime_args,
        device=device if not runtime_args.cpu_offload else torch.device("cpu"),
    )
    runtime_args = sampler.args

    if runtime_args.cpu_offload:
        apply_group_offloading(
            sampler.pipeline.transformer,
            onload_device=torch.device("cuda"),
            offload_type="block_level",
            num_blocks_per_group=1,
        )

    for row in rows:
        _generate_one(
            sampler=sampler,
            runtime_args=runtime_args,
            row=row,
            args=args,
            device=device,
            rank=rank,
        )


def main() -> None:
    args = parse_args()
    if args.distributed_worker:
        _worker_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
