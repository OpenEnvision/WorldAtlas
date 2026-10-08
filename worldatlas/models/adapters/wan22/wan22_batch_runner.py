from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
from pathlib import Path
import sys
import traceback
import types
from typing import Any

from worldatlas.models.adapters.batch_runner_common import begin_sample, print_status


def _str2bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Wan2.2 batch subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--batch_spec_path", required=True, type=str)
    parser.add_argument("--task", default="i2v-A14B", type=str)
    parser.add_argument("--size", default="1280*720", type=str)
    parser.add_argument("--frame_num", default=None, type=int)
    parser.add_argument("--min_seconds", default=5.0, type=float)
    parser.add_argument("--sample_solver", default="unipc", type=str)
    parser.add_argument("--sample_steps", default=None, type=int)
    parser.add_argument("--sample_shift", default=None, type=float)
    parser.add_argument("--sample_guide_scale", default=None, nargs="+", type=float)
    parser.add_argument("--base_seed", default=-1, type=int)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--offload_model", default=None, type=_str2bool)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--t5_fsdp", action="store_true")
    parser.add_argument("--dit_fsdp", action="store_true")
    parser.add_argument("--ulysses_size", default=1, type=int)
    parser.add_argument("--convert_model_dtype", action="store_true")
    parser.add_argument("--device_id", default=0, type=int)
    return parser.parse_args()


def _load_batch_spec(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty Wan2.2 batch spec: {path}")
    return rows


def _snap_wan_frame_num(value: float) -> int:
    minimum = max(1, int(math.ceil(value)))
    return ((minimum - 1 + 3) // 4) * 4 + 1


def _frame_num_for_duration(cfg: Any, requested: int | None, min_seconds: float) -> int:
    base = requested if requested is not None else int(cfg.frame_num)
    duration_floor = float(cfg.sample_fps) * float(min_seconds)
    return max(_snap_wan_frame_num(base), _snap_wan_frame_num(duration_floor))


def _guide_scale(args: argparse.Namespace, cfg: Any) -> float | tuple[float, ...]:
    if args.sample_guide_scale is None:
        return cfg.sample_guide_scale
    if len(args.sample_guide_scale) == 1:
        return float(args.sample_guide_scale[0])
    return tuple(float(value) for value in args.sample_guide_scale)


def _seed_for_row(base_seed: int, row_index: int) -> int:
    if base_seed < 0:
        return -1
    return base_seed + row_index


def _resolve_base_seed(args: argparse.Namespace, runtime: dict[str, Any], rank: int) -> int:
    dist = runtime["torch"].distributed
    if args.base_seed < 0:
        base_seed = random.randint(0, sys.maxsize) if rank == 0 else 0
    else:
        base_seed = int(args.base_seed)

    if dist.is_available() and dist.is_initialized():
        payload = [base_seed if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        base_seed = int(payload[0])

    args.base_seed = base_seed
    return base_seed


def _install_lightweight_wan_package(repo_root: Path) -> None:
    package_root = repo_root / "wan"
    package = sys.modules.get("wan")
    if package is not None and getattr(package, "__path__", None):
        return
    package = types.ModuleType("wan")
    package.__path__ = [str(package_root)]
    package.__package__ = "wan"
    package.__file__ = str(package_root / "__init__.py")
    sys.modules["wan"] = package


def _patch_flash_attention_fallback() -> None:
    from wan.modules import attention as attention_module

    if attention_module.FLASH_ATTN_2_AVAILABLE or attention_module.FLASH_ATTN_3_AVAILABLE:
        return

    def fallback_flash_attention(*args: Any, **kwargs: Any):
        kwargs.pop("version", None)
        return attention_module.attention(*args, **kwargs)

    attention_module.flash_attention = fallback_flash_attention
    for module_name in (
        "wan.modules.model",
        "wan.distributed.ulysses",
        "wan.modules.s2v.model_s2v",
        "wan.modules.s2v.motioner",
        "wan.modules.animate.model_animate",
        "wan.modules.animate.clip",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "flash_attention"):
            module.flash_attention = fallback_flash_attention


def _load_wan_runtime(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    _install_lightweight_wan_package(repo_root)
    import torch
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
    _patch_flash_attention_fallback()
    from wan.image2video import WanI2V
    from wan.text2video import WanT2V
    from wan.textimage2video import WanTI2V
    from wan.utils.utils import save_video
    _patch_flash_attention_fallback()

    wan = types.SimpleNamespace(WanI2V=WanI2V, WanT2V=WanT2V, WanTI2V=WanTI2V)
    return {
        "Image": Image,
        "MAX_AREA_CONFIGS": MAX_AREA_CONFIGS,
        "SIZE_CONFIGS": SIZE_CONFIGS,
        "SUPPORTED_SIZES": SUPPORTED_SIZES,
        "WAN_CONFIGS": WAN_CONFIGS,
        "save_video": save_video,
        "torch": torch,
        "wan": wan,
    }


def _init_distributed(args: argparse.Namespace, runtime: dict[str, Any]) -> tuple[int, int, int]:
    torch = runtime["torch"]
    dist = torch.distributed
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", str(args.device_id)))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Wan2.2 distributed generation requires CUDA")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
        if args.ulysses_size > 1:
            if args.ulysses_size != world_size:
                raise ValueError(
                    f"ulysses_size must equal world_size: {args.ulysses_size} != {world_size}"
                )
            from wan.distributed.util import init_distributed_group

            init_distributed_group()
        if args.offload_model is None:
            args.offload_model = False
        return rank, world_size, local_rank

    if args.t5_fsdp or args.dit_fsdp:
        raise ValueError("t5_fsdp and dit_fsdp require torchrun/distributed execution")
    if args.ulysses_size > 1:
        raise ValueError("ulysses_size > 1 requires torchrun/distributed execution")
    if args.offload_model is None:
        args.offload_model = True
    return rank, world_size, args.device_id


def _sync_rank0_save_error(runtime: dict[str, Any], error: str | None) -> None:
    dist = runtime["torch"].distributed
    if not (dist.is_available() and dist.is_initialized()):
        if error:
            raise RuntimeError(error)
        return

    payload = [error]
    dist.broadcast_object_list(payload, src=0)
    if payload[0]:
        raise RuntimeError(str(payload[0]))


def _load_pipeline(
    args: argparse.Namespace,
    cfg: Any,
    wan_module: Any,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
):
    common_kwargs = {
        "device_id": local_rank,
        "rank": rank,
        "t5_fsdp": args.t5_fsdp,
        "dit_fsdp": args.dit_fsdp,
        "use_sp": args.ulysses_size > 1 and world_size > 1,
        "t5_cpu": args.t5_cpu,
        "convert_model_dtype": args.convert_model_dtype,
    }
    if args.task.startswith("i2v"):
        return wan_module.WanI2V(cfg, args.checkpoint_dir, **common_kwargs)
    if args.task.startswith("t2v"):
        return wan_module.WanT2V(cfg, args.checkpoint_dir, **common_kwargs)
    if args.task.startswith("ti2v"):
        return wan_module.WanTI2V(cfg, args.checkpoint_dir, **common_kwargs)
    raise ValueError(f"Wan2.2 batch runner does not support task={args.task}")


def _generate_one(
    *,
    args: argparse.Namespace,
    cfg: Any,
    runtime: dict[str, Any],
    pipeline: Any,
    row: dict[str, Any],
    row_index: int,
    frame_num: int,
    rank: int,
) -> Path:
    output_path = Path(str(row["output_path"])).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = str(row["prompt"])
    solver = args.sample_solver
    sample_steps = args.sample_steps if args.sample_steps is not None else cfg.sample_steps
    sample_shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift
    guide_scale = _guide_scale(args, cfg)
    offload_model = True if args.offload_model is None else bool(args.offload_model)
    seed = _seed_for_row(args.base_seed, row_index)
    negative_prompt = args.negative_prompt or ""

    if args.task.startswith("i2v"):
        image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
        image = runtime["Image"].open(image_path).convert("RGB")
        video = pipeline.generate(
            prompt,
            image,
            max_area=runtime["MAX_AREA_CONFIGS"][args.size],
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=solver,
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=offload_model,
        )
    elif args.task.startswith("ti2v"):
        image = None
        image_path_raw = str(row.get("conditioning_image") or "")
        if image_path_raw:
            image = runtime["Image"].open(Path(image_path_raw).expanduser().resolve()).convert("RGB")
        video = pipeline.generate(
            prompt,
            img=image,
            size=runtime["SIZE_CONFIGS"][args.size],
            max_area=runtime["MAX_AREA_CONFIGS"][args.size],
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=solver,
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=offload_model,
        )
    else:
        video = pipeline.generate(
            prompt,
            size=runtime["SIZE_CONFIGS"][args.size],
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=solver,
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=offload_model,
        )

    save_error = None
    try:
        if rank == 0:
            runtime["save_video"](
                tensor=video[None],
                save_file=str(output_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise RuntimeError(f"Wan2.2 save_video did not create output: {output_path}")
    except Exception as exc:
        save_error = str(exc)
    del video
    gc.collect()
    if runtime["torch"].cuda.is_available():
        runtime["torch"].cuda.empty_cache()

    _sync_rank0_save_error(runtime, save_error if rank == 0 else None)
    return output_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()
    args.checkpoint_dir = str(checkpoint_dir)

    runtime = _load_wan_runtime(repo_root)
    wan_configs = runtime["WAN_CONFIGS"]
    supported_sizes = runtime["SUPPORTED_SIZES"]
    if args.task not in wan_configs:
        raise ValueError(f"unsupported Wan2.2 task: {args.task}")
    if args.size not in supported_sizes[args.task]:
        raise ValueError(
            f"unsupported Wan2.2 size {args.size} for task {args.task}; "
            f"supported sizes: {', '.join(supported_sizes[args.task])}"
        )

    rows = _load_batch_spec(batch_spec_path)
    cfg = wan_configs[args.task]
    rank, world_size, local_rank = _init_distributed(args, runtime)
    if rank != 0:
        logging.getLogger().setLevel(logging.ERROR)
    if args.ulysses_size > 1 and cfg.num_heads % args.ulysses_size != 0:
        raise ValueError(
            f"cfg.num_heads={cfg.num_heads} cannot be divided by ulysses_size={args.ulysses_size}"
        )
    base_seed = _resolve_base_seed(args, runtime, rank)
    frame_num = _frame_num_for_duration(cfg, args.frame_num, args.min_seconds)
    duration_seconds = frame_num / float(cfg.sample_fps)
    logging.info(
        (
            "loading Wan2.2 task=%s checkpoint=%s rows=%s size=%s frame_num=%s "
            "fps=%s duration=%.3fs rank=%s world_size=%s local_rank=%s"
        ),
        args.task,
        checkpoint_dir,
        len(rows),
        args.size,
        frame_num,
        cfg.sample_fps,
        duration_seconds,
        rank,
        world_size,
        local_rank,
    )
    if rank == 0:
        logging.info("Wan2.2 base_seed=%s", base_seed)
    pipeline = _load_pipeline(
        args,
        cfg,
        runtime["wan"],
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    logging.info("Wan2.2 pipeline loaded once per rank; starting %s rows", len(rows))
    if rank == 0:
        from worldatlas.models.adapters.batch_runner_common import log_pipeline

        log_pipeline("pipeline_loaded", label="Wan2.2", samples=len(rows))

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        if rank == 0:
            begin_sample(sample_id, index=row_index + 1, total=len(rows), label="Wan2.2")
        try:
            output_path = _generate_one(
                args=args,
                cfg=cfg,
                runtime=runtime,
                pipeline=pipeline,
                row=row,
                row_index=row_index,
                frame_num=frame_num,
                rank=rank,
            )
            if rank == 0:
                print_status(
                    sample_id,
                    "generated",
                    output_path=str(output_path),
                    frame_num=frame_num,
                    fps=cfg.sample_fps,
                    duration_seconds=duration_seconds,
                )
        except Exception as exc:
            if rank == 0:
                failed += 1
                logging.exception("Wan2.2 row failed sample_id=%s error=%s", sample_id, exc)
                print_status(sample_id, "failed", error=str(exc))
    logging.info("Wan2.2 batch completed rows=%s failed=%s", len(rows), failed)
    if runtime["torch"].distributed.is_available() and runtime["torch"].distributed.is_initialized():
        runtime["torch"].distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
