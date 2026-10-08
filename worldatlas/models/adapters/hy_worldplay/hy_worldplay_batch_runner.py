"""Load HY-WorldPlay once, then generate every sample in a shard.

The adapter materializes pose JSON sidecars. This process creates
``HunyuanVideo_1_5_Pipeline`` a single time and reuses it for the JSONL spec.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace
from typing import Any

from worldatlas.common.checkpoints import resolve_project_path
from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    begin_sample,
    copy_output,
    load_batch_spec,
    load_json,
    log_pipeline,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas HY-WorldPlay batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "", "none", "null"}:
        return False
    return default


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return _as_bool(value, default=False)


def _transformer_dtype(name: str):
    import torch

    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}. Must be 'bf16' or 'fp32'")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _row_int(row: dict[str, Any], generation: dict[str, Any], key: str, default: int) -> int:
    if row.get(key) is not None:
        return int(row[key])
    if generation.get(key) is not None:
        return int(generation[key])
    return int(default)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())

    model_path = resolve_project_path(generation.get("model_path"))
    if model_path is None or not model_path.is_dir() or not (model_path / "transformer").is_dir():
        raise FileNotFoundError(
            "HY-WorldPlay backbone is missing transformer/ under "
            f"{generation.get('model_path')}. Set generation.model_path to HunyuanVideo-1.5."
        )
    action_ckpt = resolve_project_path(generation.get("action_ckpt"))
    if action_ckpt is None or not action_ckpt.is_file():
        raise FileNotFoundError(f"HY-WorldPlay action checkpoint is missing: {generation.get('action_ckpt')}")

    resolution = str(generation.get("resolution", "480p")).strip()
    if resolution != "480p":
        raise ValueError(f"HY-WorldPlay currently only supports resolution=480p, got {resolution!r}")

    import torch
    # generate.py initializes its CUDA process mesh on import. Select the
    # local rank first so NCCL never initializes both ranks on device zero.
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    from hyvideo.commons.infer_state import initialize_infer_state
    from hyvideo.generate import pose_to_input, save_video
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

    initialize_infer_state(
        SimpleNamespace(
            sage_blocks_range=str(generation.get("sage_blocks_range", "0-53")),
            use_sageattn=_as_bool(generation.get("use_sageattn"), default=False),
            include_patterns=str(generation.get("include_patterns", "double_blocks")),
            enable_torch_compile=_as_bool(generation.get("enable_torch_compile"), default=False),
            use_fp8_gemm=_as_bool(generation.get("use_fp8_gemm"), default=False),
            quant_type=str(generation.get("quant_type", "fp8-per-block")),
            use_vae_parallel=_as_bool(generation.get("use_vae_parallel"), default=False),
        )
    )

    enable_sr = _as_bool(generation.get("sr"), default=False)
    offloading = _as_bool(generation.get("offloading"), default=True)
    group_offloading = _optional_bool(generation.get("group_offloading"))
    few_step = _as_bool(generation.get("few_step"), default=True)
    rewrite = _as_bool(generation.get("rewrite"), default=False)
    save_pre_sr_video = _as_bool(generation.get("save_pre_sr_video"), default=False)
    transformer_resident = _as_bool(
        generation.get("transformer_resident_ar_rollout"),
        default=True,
    )
    model_type = str(generation.get("model_type", "ar"))
    aspect_ratio = str(generation.get("aspect_ratio", "16:9"))
    negative_prompt = str(generation.get("negative_prompt", ""))
    num_inference_steps = int(
        generation.get("num_inference_steps", 4 if few_step else 50)
    )
    default_video_length = int(generation.get("video_length", 125))
    default_width = int(generation.get("width", 832))
    default_height = int(generation.get("height", 480))
    base_seed = int(generation.get("seed", 1))
    dtype_name = str(generation.get("dtype", "bf16"))

    log_pipeline(
        "pipeline_loading",
        label="HY-WorldPlay",
        samples=len(rows),
        model_path=str(model_path),
        action_ckpt=str(action_ckpt),
    )
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=str(model_path),
        transformer_version=f"{resolution}_i2v",
        enable_offloading=offloading,
        enable_group_offloading=group_offloading,
        create_sr_pipeline=enable_sr,
        force_sparse_attn=False,
        transformer_dtype=_transformer_dtype(dtype_name),
        action_ckpt=str(action_ckpt),
    )
    log_pipeline("pipeline_loaded", label="HY-WorldPlay", samples=len(rows))

    failed = 0
    rank = int(os.environ.get("RANK", "0"))
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        begin_sample(sample_id, index=row_index + 1, total=len(rows), label="HY-WorldPlay")
        try:
            conditioning_image = _require_file(
                Path(str(row["conditioning_image"])).expanduser().resolve(),
                "conditioning image",
            )
            pose_json = _require_file(
                Path(str(row["pose_json_path"])).expanduser().resolve(),
                "pose json",
            )
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            video_length = _row_int(row, generation, "video_length", default_video_length)
            width = _row_int(row, generation, "width", default_width)
            height = _row_int(row, generation, "height", default_height)
            latent_count = (video_length - 1) // 4 + 1
            viewmats, ks, action = pose_to_input(str(pose_json), latent_count)
            extra_kwargs = {"reference_image": str(conditioning_image)}
            out = pipe(
                enable_sr=enable_sr,
                prompt=str(row["prompt"]),
                aspect_ratio=aspect_ratio,
                num_inference_steps=num_inference_steps,
                sr_num_inference_steps=None,
                video_length=video_length,
                negative_prompt=negative_prompt,
                seed=base_seed,
                output_type="pt",
                prompt_rewrite=rewrite,
                return_pre_sr_video=save_pre_sr_video,
                viewmats=viewmats.unsqueeze(0),
                Ks=ks.unsqueeze(0),
                action=action.unsqueeze(0),
                few_step=few_step,
                chunk_latent_frames=4 if model_type == "ar" else 16,
                model_type=model_type,
                user_height=height,
                user_width=width,
                transformer_resident_ar_rollout=transformer_resident,
                **extra_kwargs,
            )
            if rank == 0:
                with tempfile.TemporaryDirectory(prefix="worldatlas_hy_worldplay_") as temp_dir_raw:
                    temp_dir = Path(temp_dir_raw)
                    video_path = temp_dir / "gen.mp4"
                    if enable_sr and hasattr(out, "sr_videos"):
                        save_video(out.sr_videos, str(video_path))
                        if save_pre_sr_video:
                            pre_sr_tmp = temp_dir / "gen_pre_sr.mp4"
                            save_video(out.videos, str(pre_sr_tmp))
                            copy_output(
                                pre_sr_tmp,
                                output_path.with_name(f"{output_path.stem}.pre_sr{output_path.suffix}"),
                            )
                    else:
                        save_video(out.videos, str(video_path))
                    copy_output(video_path, output_path)
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                pose_source=row.get("pose_source"),
                video_length=video_length,
            )
        except Exception as exc:
            failed += 1
            print_status(
                sample_id,
                "failed",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        finally:
            # Reusing the pipeline must not retain the previous sample's large
            # positive/negative history caches through text encoding or decode.
            pipe._kv_cache = []
            pipe._kv_cache_neg = []
            out = None
    return 1 if failed == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
