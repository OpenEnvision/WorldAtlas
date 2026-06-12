from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import traceback
from typing import Any

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena LTX-2.3 batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _add_ltx_repo_to_sys_path(repo_root: Path) -> None:
    candidates = [
        repo_root / "packages" / "ltx-pipelines" / "src",
        repo_root / "packages" / "ltx-core" / "src",
        repo_root / "packages" / "ltx-trainer" / "src",
        repo_root,
    ]
    for path in reversed(candidates):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _resolve_path(value: str | None, *, base_dir: Path, default: str | None = None) -> Path:
    raw = value if value not in (None, "") else default
    if raw is None:
        raise ValueError("missing required LTX-2.3 path")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_existing_path(
    value: str | None,
    *,
    base_dir: Path,
    default: str | None = None,
    label: str,
) -> str:
    path = _resolve_path(value, base_dir=base_dir, default=default)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def _snap_frame_count(value: float) -> int:
    minimum = max(1, int(value + 0.999999))
    return ((minimum - 1 + 7) // 8) * 8 + 1


def _frame_count(generation: dict[str, Any]) -> int:
    fps = float(generation.get("frame_rate", generation.get("fps", 24.0)))
    requested = int(generation.get("num_frames", 121))
    min_seconds = float(generation.get("min_seconds", 5.0))
    return max(_snap_frame_count(requested), _snap_frame_count(fps * min_seconds))


def _seed_for_row(generation: dict[str, Any], row_index: int) -> int:
    seed = int(generation.get("seed", 10))
    if bool(generation.get("increment_seed", True)):
        seed += row_index
    return seed


def _ltx_lora(path: Path, strength: float):
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

    return LoraPathStrengthAndSDOps(str(path), float(strength), LTXV_LORA_COMFY_RENAMING_MAP)


def _parse_lora_item(item: Any, *, base_dir: Path) -> Any:
    if isinstance(item, str):
        return _ltx_lora(_resolve_path(item, base_dir=base_dir), 1.0)
    if isinstance(item, dict):
        return _ltx_lora(
            _resolve_path(str(item["path"]), base_dir=base_dir),
            float(item.get("strength", 1.0)),
        )
    if isinstance(item, (list, tuple)) and item:
        return _ltx_lora(_resolve_path(str(item[0]), base_dir=base_dir), float(item[1] if len(item) > 1 else 1.0))
    raise ValueError(f"invalid LTX-2.3 LoRA spec: {item!r}")


def _parse_loras(value: Any, *, base_dir: Path) -> list[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, (str, dict)):
        return [_parse_lora_item(value, base_dir=base_dir)]
    return [_parse_lora_item(item, base_dir=base_dir) for item in value]


def _parse_quantization(value: str | None, *, checkpoint_path: str) -> Any:
    if not value:
        return None
    from ltx_pipelines.utils.quantization_factory import QuantizationKind

    return QuantizationKind(str(value)).to_policy(checkpoint_path=checkpoint_path)


def _guider_params(prefix: str, generation: dict[str, Any], default_params: Any) -> Any:
    from ltx_core.components.guiders import MultiModalGuiderParams

    return MultiModalGuiderParams(
        cfg_scale=float(generation.get(f"{prefix}_cfg_guidance_scale", default_params.cfg_scale)),
        stg_scale=float(generation.get(f"{prefix}_stg_guidance_scale", default_params.stg_scale)),
        rescale_scale=float(generation.get(f"{prefix}_rescale_scale", default_params.rescale_scale)),
        modality_scale=float(generation.get(f"{prefix}_modality_guidance_scale", default_params.modality_scale)),
        skip_step=int(generation.get(f"{prefix}_skip_step", default_params.skip_step)),
        stg_blocks=[int(value) for value in generation.get(f"{prefix}_stg_blocks", default_params.stg_blocks)],
    )


def _common_runtime(generation: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_base = checkpoint_dir
    distilled_checkpoint_path = _resolve_existing_path(
        generation.get("distilled_checkpoint_path"),
        base_dir=checkpoint_base,
        default="ltx-2.3-22b-distilled-1.1.safetensors",
        label="LTX-2.3 distilled checkpoint",
    )
    checkpoint_path = _resolve_path(
        generation.get("checkpoint_path"),
        base_dir=checkpoint_base,
        default="ltx-2.3-22b-dev.safetensors",
    )
    spatial_upsampler_path = _resolve_existing_path(
        generation.get("spatial_upsampler_path"),
        base_dir=checkpoint_base,
        default="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        label="LTX-2.3 spatial upsampler",
    )
    gemma_root = _resolve_existing_path(
        generation.get("gemma_root"),
        base_dir=checkpoint_base,
        default="../gemma-3-12b-it-qat-q4_0-unquantized",
        label="LTX-2.3 Gemma root",
    )
    return {
        "checkpoint_base": checkpoint_base,
        "checkpoint_path": str(checkpoint_path),
        "distilled_checkpoint_path": distilled_checkpoint_path,
        "spatial_upsampler_path": spatial_upsampler_path,
        "gemma_root": gemma_root,
        "height": int(generation.get("height", 512)),
        "width": int(generation.get("width", 768)),
        "num_frames": _frame_count(generation),
        "frame_rate": float(generation.get("frame_rate", generation.get("fps", 24.0))),
        "image_strength": float(generation.get("image_conditioning_strength", 1.0)),
        "image_crf": int(generation.get("image_crf", 33)),
        "enhance_prompt": bool(generation.get("enhance_prompt", False)),
    }


def _load_pipeline(generation: dict[str, Any], runtime: dict[str, Any]) -> Any:
    from ltx_pipelines.utils.types import OffloadMode

    pipeline_name = str(generation.get("pipeline", "distilled")).lower()
    offload_mode = OffloadMode(str(generation.get("offload", "none")))
    quantization = _parse_quantization(generation.get("quantization"), checkpoint_path=runtime["distilled_checkpoint_path"])

    if pipeline_name == "distilled":
        from ltx_pipelines.distilled import DistilledPipeline

        return DistilledPipeline(
            distilled_checkpoint_path=runtime["distilled_checkpoint_path"],
            gemma_root=runtime["gemma_root"],
            spatial_upsampler_path=runtime["spatial_upsampler_path"],
            loras=_parse_loras(generation.get("lora"), base_dir=runtime["checkpoint_base"]),
            quantization=quantization,
            offload_mode=offload_mode,
        )

    checkpoint_path = runtime["checkpoint_path"]
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"LTX-2.3 checkpoint not found: {checkpoint_path}")
    quantization = _parse_quantization(generation.get("quantization"), checkpoint_path=checkpoint_path)
    distilled_lora = _parse_loras(
        generation.get(
            "distilled_lora",
            [["ltx-2.3-22b-distilled-lora-384-1.1.safetensors", 0.8]],
        ),
        base_dir=runtime["checkpoint_base"],
    )
    loras = _parse_loras(generation.get("lora"), base_dir=runtime["checkpoint_base"])

    if pipeline_name in {"two_stage", "ti2vid_two_stages"}:
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        return TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=runtime["spatial_upsampler_path"],
            gemma_root=runtime["gemma_root"],
            loras=loras,
            quantization=quantization,
            offload_mode=offload_mode,
        )

    if pipeline_name in {"two_stage_hq", "hq", "ti2vid_two_stages_hq"}:
        from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

        return TI2VidTwoStagesHQPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            distilled_lora_strength_stage_1=float(generation.get("distilled_lora_strength_stage_1", 0.25)),
            distilled_lora_strength_stage_2=float(generation.get("distilled_lora_strength_stage_2", 0.5)),
            spatial_upsampler_path=runtime["spatial_upsampler_path"],
            gemma_root=runtime["gemma_root"],
            loras=tuple(loras),
            quantization=quantization,
            offload_mode=offload_mode,
        )

    raise ValueError(f"unsupported LTX-2.3 pipeline: {pipeline_name}")


def _generate_one(
    *,
    pipeline: Any,
    row: dict[str, Any],
    row_index: int,
    generation: dict[str, Any],
    runtime: dict[str, Any],
) -> Path:
    import torch
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, LTX_2_3_PARAMS
    from ltx_pipelines.utils.media_io import encode_video

    output_path = Path(str(row["output_path"])).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
    images = [
        ImageConditioningInput(
            path=str(image_path),
            frame_idx=0,
            strength=runtime["image_strength"],
            crf=runtime["image_crf"],
        )
    ]
    tiling_config = TilingConfig.default()
    pipeline_name = str(generation.get("pipeline", "distilled")).lower()
    common_kwargs = {
        "prompt": str(row["prompt"]),
        "seed": _seed_for_row(generation, row_index),
        "height": runtime["height"],
        "width": runtime["width"],
        "num_frames": runtime["num_frames"],
        "frame_rate": runtime["frame_rate"],
        "images": images,
        "tiling_config": tiling_config,
        "enhance_prompt": runtime["enhance_prompt"],
    }
    if pipeline_name == "distilled":
        video, audio = pipeline(**common_kwargs)
    else:
        params = LTX_2_3_HQ_PARAMS if "hq" in pipeline_name else LTX_2_3_PARAMS
        video, audio = pipeline(
            **common_kwargs,
            negative_prompt=str(generation.get("negative_prompt", "")),
            num_inference_steps=int(generation.get("num_inference_steps", params.num_inference_steps)),
            video_guider_params=_guider_params("video", generation, params.video_guider_params),
            audio_guider_params=_guider_params("audio", generation, params.audio_guider_params),
            max_batch_size=int(generation.get("max_batch_size", 1)),
        )

    encode_video(
        video=video,
        fps=runtime["frame_rate"],
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=get_video_chunks_number(runtime["num_frames"], tiling_config),
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"LTX-2.3 encode_video did not create output: {output_path}")
    return output_path


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.checkpoint_dir is None:
        raise ValueError("LTX-2.3 batch runner requires --checkpoint_dir")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    _add_ltx_repo_to_sys_path(repo_root)

    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    runtime = _common_runtime(generation, checkpoint_dir)
    pipeline = _load_pipeline(generation, runtime)

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            output_path = _generate_one(
                pipeline=pipeline,
                row=row,
                row_index=row_index,
                generation=generation,
                runtime=runtime,
            )
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                fps=runtime["frame_rate"],
                num_frames=runtime["num_frames"],
            )
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 1 if failed == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
