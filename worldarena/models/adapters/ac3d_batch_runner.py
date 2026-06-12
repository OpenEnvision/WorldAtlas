from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys
import traceback

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    latest_mp4,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena AC3D batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _variant(value: Any) -> str:
    text = str(value or "2b").strip().lower()
    if text in {"2", "2b", "cogvideox-2b"}:
        return "2b"
    if text in {"5", "5b", "cogvideox-5b"}:
        return "5b"
    raise ValueError(f"unsupported AC3D variant: {value!r}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _torch_dtype(torch: Any, value: Any) -> Any:
    text = str(value or "bfloat16").strip().lower()
    if text in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if text in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    return torch.bfloat16


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


def _default_controlnet_path(checkpoint_dir: Path | None, variant: str) -> Path | None:
    if checkpoint_dir is None:
        return None
    if checkpoint_dir.is_file():
        return checkpoint_dir.resolve()
    shard = "5B" if variant == "5b" else "2B"
    candidates = (
        checkpoint_dir / shard / "checkpoint-10000.pt",
        checkpoint_dir / "checkpoint-10000.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _default_base_model_path(checkpoint_dir: Path | None, variant: str) -> Path | None:
    if checkpoint_dir is None:
        return None
    name = "CogVideoX-5b" if variant == "5b" else "CogVideoX-2b"
    candidates = [
        checkpoint_dir.parent / name,
        checkpoint_dir.parent.parent / name,
    ]
    if checkpoint_dir.name.lower() == "ac3d":
        candidates.insert(0, checkpoint_dir.parent / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _generation_defaults(generation: dict[str, Any], variant: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "annotation_json": "annotations/test.json",
        "controlnet_weights": 1.0,
        "controlnet_guidance_start": 0.0,
        "controlnet_guidance_end": 0.4,
        "use_dynamic_cfg": True,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "num_videos_per_prompt": 1,
        "dtype": "bfloat16",
        "seed": 42,
        "num_frames": 49,
        "height": 480,
        "width": 720,
        "stride_min": 2,
        "stride_max": 2,
        "downscale_coef": 8,
        "controlnet_input_channels": 6,
        "controlnet_transformer_num_attn_heads": 4,
        "controlnet_transformer_attention_head_dim": 64 if variant == "5b" else 32,
        "controlnet_transformer_out_proj_dim_factor": 64,
        "controlnet_transformer_out_proj_dim_zero_init": True,
    }
    defaults.update(generation)
    return defaults


def _load_ac3d_generate_video(repo_root: Path):
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    module = import_module("inference.cli_demo_camera")
    return module


def _copy_ac3d_output(raw_dir: Path, output_path: Path, camera_index: int) -> Path:
    expected = raw_dir / f"{camera_index:05d}_out.mp4"
    if not expected.is_file():
        expected = latest_mp4(raw_dir)
    copy_output(expected, output_path)
    return expected


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else None
    )
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    variant = _variant(generation.get("variant", generation.get("model_variant", "2b")))
    options = _generation_defaults(generation, variant)

    video_root_dir = _resolve_path(options.get("video_root_dir"))
    if video_root_dir is None:
        raise ValueError("AC3D requires generation.video_root_dir")
    base_model_path = _resolve_path(options.get("base_model_path")) or _default_base_model_path(
        checkpoint_dir,
        variant,
    )
    controlnet_model_path = _resolve_path(
        options.get("controlnet_model_path") or options.get("checkpoint_path"),
        base=checkpoint_dir if checkpoint_dir is not None and checkpoint_dir.is_dir() else None,
    ) or _default_controlnet_path(checkpoint_dir, variant)
    if base_model_path is None:
        raise ValueError("AC3D requires generation.base_model_path or checkpoint_dir near CogVideoX weights")
    if controlnet_model_path is None:
        raise ValueError("AC3D requires checkpoint_dir or generation.controlnet_model_path")

    import torch

    ac3d_module = _load_ac3d_generate_video(repo_root)
    ac3d_module.args = SimpleNamespace(
        controlnet_transformer_num_attn_heads=options.get("controlnet_transformer_num_attn_heads")
    )
    dtype = _torch_dtype(torch, options.get("dtype"))

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        output_path = Path(str(row["output_path"])).expanduser().resolve()
        camera_index = int(row.get("camera_index", row.get("start_camera_idx", 0)))
        end_camera_idx = int(row.get("end_camera_idx", camera_index + 1))
        raw_dir = output_path.parent / "_ac3d_raw" / output_path.stem
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            ac3d_module.generate_video(
                prompt=str(row.get("prompt") or options.get("prompt") or ""),
                video_root_dir=str(video_root_dir),
                annotation_json=str(options.get("annotation_json")),
                base_model_path=str(base_model_path),
                controlnet_model_path=str(controlnet_model_path),
                controlnet_weights=float(options.get("controlnet_weights", 1.0)),
                controlnet_guidance_start=float(options.get("controlnet_guidance_start", 0.0)),
                controlnet_guidance_end=float(options.get("controlnet_guidance_end", 0.4)),
                use_dynamic_cfg=_as_bool(options.get("use_dynamic_cfg", True)),
                lora_path=(
                    str(_resolve_path(options.get("lora_path")))
                    if _resolve_path(options.get("lora_path")) is not None
                    else None
                ),
                lora_rank=int(options.get("lora_rank", 128)),
                output_path=str(raw_dir),
                num_inference_steps=int(options.get("num_inference_steps", 50)),
                guidance_scale=float(options.get("guidance_scale", 6.0)),
                num_videos_per_prompt=int(options.get("num_videos_per_prompt", 1)),
                dtype=dtype,
                seed=int(options.get("seed", 42)) + row_index,
                num_frames=int(options.get("num_frames", 49)),
                height=int(options.get("height", 480)),
                width=int(options.get("width", 720)),
                stride_min=int(options.get("stride_min", 2)),
                stride_max=int(options.get("stride_max", 2)),
                start_camera_idx=camera_index,
                end_camera_idx=end_camera_idx,
                controlnet_transformer_num_attn_heads=options.get(
                    "controlnet_transformer_num_attn_heads"
                ),
                controlnet_transformer_attention_head_dim=options.get(
                    "controlnet_transformer_attention_head_dim"
                ),
                controlnet_transformer_out_proj_dim_factor=options.get(
                    "controlnet_transformer_out_proj_dim_factor"
                ),
                controlnet_transformer_out_proj_dim_zero_init=_as_bool(
                    options.get("controlnet_transformer_out_proj_dim_zero_init", True)
                ),
                controlnet_transformer_num_layers=int(
                    options.get("controlnet_transformer_num_layers", 8)
                ),
                downscale_coef=int(options.get("downscale_coef", 8)),
                controlnet_input_channels=int(options.get("controlnet_input_channels", 6)),
            )
            raw_output = _copy_ac3d_output(raw_dir, output_path, camera_index)
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                raw_output_path=str(raw_output),
                camera_index=camera_index,
            )
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 1 if failed == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
