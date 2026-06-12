from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path
import sys
import traceback
from typing import Any


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
    parser = argparse.ArgumentParser(description="WorldArena Wan2.2 batch subprocess runner.")
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


def _load_wan_runtime(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import torch
    from PIL import Image
    import wan
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
    from wan.utils.utils import save_video

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


def _load_pipeline(args: argparse.Namespace, cfg: Any, wan_module: Any):
    common_kwargs = {
        "device_id": args.device_id,
        "rank": 0,
        "t5_fsdp": False,
        "dit_fsdp": False,
        "use_sp": False,
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

    runtime["save_video"](
        tensor=video[None],
        save_file=str(output_path),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    del video
    gc.collect()
    if runtime["torch"].cuda.is_available():
        runtime["torch"].cuda.empty_cache()
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Wan2.2 save_video did not create output: {output_path}")
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
    frame_num = _frame_num_for_duration(cfg, args.frame_num, args.min_seconds)
    duration_seconds = frame_num / float(cfg.sample_fps)
    logging.info(
        "loading Wan2.2 task=%s checkpoint=%s rows=%s size=%s frame_num=%s fps=%s duration=%.3fs",
        args.task,
        checkpoint_dir,
        len(rows),
        args.size,
        frame_num,
        cfg.sample_fps,
        duration_seconds,
    )
    pipeline = _load_pipeline(args, cfg, runtime["wan"])
    logging.info("Wan2.2 pipeline loaded once; starting %s rows", len(rows))

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            output_path = _generate_one(
                args=args,
                cfg=cfg,
                runtime=runtime,
                pipeline=pipeline,
                row=row,
                row_index=row_index,
                frame_num=frame_num,
            )
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "status": "generated",
                        "output_path": str(output_path),
                        "frame_num": frame_num,
                        "fps": cfg.sample_fps,
                        "duration_seconds": duration_seconds,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            failed += 1
            logging.error("Wan2.2 row failed sample_id=%s error=%s", sample_id, exc)
            logging.debug("Wan2.2 row traceback:\n%s", traceback.format_exc())
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "status": "failed",
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    logging.info("Wan2.2 batch completed rows=%s failed=%s", len(rows), failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
