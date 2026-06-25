from __future__ import annotations

import argparse
import math
import sys
import traceback
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

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
    if text in {"5", "5b", "cogvideox-5b", "cogvideox-5b-i2v"}:
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


def _default_base_model_path(checkpoint_dir: Path | None, variant: str, mode: str) -> Path | None:
    if checkpoint_dir is None:
        return None
    if mode == "image" and variant == "5b":
        names = ("CogVideoX-5b-I2V", "CogVideoX1.5-5B-I2V", "CogVideoX-5b")
    else:
        names = ("CogVideoX-5b",) if variant == "5b" else ("CogVideoX-2b",)
    roots = [checkpoint_dir.parent, checkpoint_dir.parent.parent]
    candidates = [root / name for root in roots for name in names]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _conditioning_mode(generation: dict[str, Any], base_model_path: Path | None) -> str:
    explicit = str(generation.get("conditioning_mode") or "").strip().lower()
    if explicit in {"image", "i2v", "image_to_video", "first_frame", "reference_image"}:
        return "image"
    if explicit in {"dataset", "demo_dataset", "re10k", "realestate10k"}:
        return "dataset"
    if base_model_path is not None and "i2v" in str(base_model_path).lower():
        return "image"
    return "dataset" if generation.get("video_root_dir") else "image"


def _frames_for_min_duration(num_frames: int, fps: int, min_duration_seconds: float) -> int:
    requested = max(int(num_frames), 1)
    min_frames = int(math.ceil(float(min_duration_seconds) * max(int(fps), 1)))
    if requested >= min_frames:
        return requested
    # CogVideoX defaults to 4n+1 frame counts such as 49.
    return ((min_frames - 1 + 3) // 4) * 4 + 1


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
        "fps": 8,
        "min_duration_seconds": 5.0,
        "height": 480,
        "width": 720,
        "stride_min": 2,
        "stride_max": 2,
        "focal_scale": 0.9,
        "downscale_coef": 8,
        "controlnet_input_channels": 6,
        "controlnet_transformer_num_attn_heads": 4,
        "controlnet_transformer_attention_head_dim": 64 if variant == "5b" else 32,
        "controlnet_transformer_out_proj_dim_factor": 64,
        "controlnet_transformer_out_proj_dim_zero_init": True,
        "controlnet_transformer_num_layers": 8,
        "offload_mode": "sequential_cpu",
    }
    defaults.update(generation)
    return defaults


def _add_repo_to_path(repo_root: Path) -> None:
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)


def _load_repo_symbols(repo_root: Path) -> dict[str, Any]:
    _add_repo_to_path(repo_root)
    return {
        "T5Tokenizer": import_module("transformers").T5Tokenizer,
        "T5EncoderModel": import_module("transformers").T5EncoderModel,
        "AutoencoderKLCogVideoX": import_module("diffusers").AutoencoderKLCogVideoX,
        "CogVideoXDDIMScheduler": import_module("diffusers").CogVideoXDDIMScheduler,
        "CogVideoXDPMScheduler": import_module("diffusers").CogVideoXDPMScheduler,
        "export_to_video": import_module("diffusers.utils").export_to_video,
        "CustomCogVideoXTransformer3DModel": import_module(
            "cogvideo_transformer"
        ).CustomCogVideoXTransformer3DModel,
        "CogVideoXControlnet": import_module("cogvideo_controlnet").CogVideoXControlnet,
        "ControlnetCogVideoXPipeline": import_module(
            "controlnet_pipeline"
        ).ControlnetCogVideoXPipeline,
        "CogVideoXImageToVideoControlnetPipeline": import_module(
            "controlnet_img2vid_pipeline"
        ).CogVideoXImageToVideoControlnetPipeline,
        "RealEstate10KPoseControlnetDataset": import_module(
            "training.controlnet_datasets_camera"
        ).RealEstate10KPoseControlnetDataset,
        "ray_condition": import_module("training.controlnet_datasets_camera").ray_condition,
    }


def _controlnet_kwargs(options: dict[str, Any], variant: str) -> dict[str, Any]:
    original_heads = 48 if variant == "5b" else 30
    kwargs: dict[str, Any] = {}
    kwargs["num_attention_heads"] = int(
        options.get("controlnet_transformer_num_attn_heads") or original_heads
    )
    if options.get("controlnet_transformer_attention_head_dim") is not None:
        kwargs["attention_head_dim"] = int(options["controlnet_transformer_attention_head_dim"])
    if options.get("controlnet_transformer_out_proj_dim_factor") is not None:
        kwargs["out_proj_dim"] = original_heads * int(
            options["controlnet_transformer_out_proj_dim_factor"]
        )
    kwargs["out_proj_dim_zero_init"] = _as_bool(
        options.get("controlnet_transformer_out_proj_dim_zero_init", False)
    )
    return kwargs


def _camera_path_from_row(row: dict[str, Any]) -> list[str]:
    raw = row.get("camera_path") or row.get("camera_tokens") or []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, (tuple, list)):
        tokens = [str(token).strip() for token in raw if str(token).strip()]
        return tokens or ["fixed"]
    return ["fixed"]


def _synthetic_controlnet_latents(
    *,
    symbols: dict[str, Any],
    camera_path: list[str],
    num_frames: int,
    height: int,
    width: int,
    focal_scale: float,
) -> Any:
    import numpy as np
    import torch

    from worldarena.models.adapters.pose_synthesis import _synthetic_image_poses

    c2w = _synthetic_image_poses(camera_path, frame_count=num_frames).astype(np.float32)
    focal = float(max(width, height)) * float(focal_scale)
    intrinsics = np.asarray(
        [[focal, focal, width / 2.0, height / 2.0] for _ in range(num_frames)],
        dtype=np.float32,
    )
    plucker = symbols["ray_condition"](
        torch.as_tensor(intrinsics)[None],
        torch.as_tensor(c2w)[None],
        height,
        width,
        device="cpu",
    )[0]
    return plucker.permute(0, 3, 1, 2).contiguous()[None]


def _maybe_duplicate_for_guidance(tensor: Any, guidance_scale: float) -> Any:
    import torch

    if float(guidance_scale) <= 1.0:
        return tensor
    return torch.cat([tensor] * 2)


def _pipeline_device(pipe: Any, torch: Any) -> Any:
    try:
        return pipe._execution_device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class AC3DBatchGenerator:
    repo_root: Path
    checkpoint_dir: Path | None
    options: dict[str, Any]
    variant: str
    mode: str
    base_model_path: Path
    controlnet_model_path: Path
    dtype: Any
    torch: Any
    symbols: dict[str, Any]
    pipe: Any
    dataset: Any = None

    @classmethod
    def build(
        cls,
        *,
        repo_root: Path,
        checkpoint_dir: Path | None,
        options: dict[str, Any],
        variant: str,
        mode: str,
        base_model_path: Path,
        controlnet_model_path: Path,
    ) -> "AC3DBatchGenerator":
        import torch

        symbols = _load_repo_symbols(repo_root)
        dtype = _torch_dtype(torch, options.get("dtype"))
        print_status(
            "ac3d_runner",
            "loading_checkpoint",
            mode=mode,
            variant=variant,
            base_model_path=str(base_model_path),
            controlnet_model_path=str(controlnet_model_path),
        )
        tokenizer = symbols["T5Tokenizer"].from_pretrained(
            str(base_model_path),
            subfolder="tokenizer",
        )
        text_encoder = symbols["T5EncoderModel"].from_pretrained(
            str(base_model_path),
            subfolder="text_encoder",
        )
        transformer = symbols["CustomCogVideoXTransformer3DModel"].from_pretrained(
            str(base_model_path),
            subfolder="transformer",
        )
        vae = symbols["AutoencoderKLCogVideoX"].from_pretrained(
            str(base_model_path),
            subfolder="vae",
        )
        scheduler = symbols["CogVideoXDDIMScheduler"].from_pretrained(
            str(base_model_path),
            subfolder="scheduler",
        )
        controlnet = symbols["CogVideoXControlnet"](
            num_layers=int(options.get("controlnet_transformer_num_layers", 8)),
            downscale_coef=int(options.get("downscale_coef", 8)),
            in_channels=int(options.get("controlnet_input_channels", 6)),
            **_controlnet_kwargs(options, variant),
        )
        ckpt = torch.load(str(controlnet_model_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
        print_status(
            "ac3d_runner",
            "checkpoint_loaded",
            missing_keys=len(missing),
            unexpected_keys=len(unexpected),
        )
        pipe_cls = (
            symbols["CogVideoXImageToVideoControlnetPipeline"]
            if mode == "image"
            else symbols["ControlnetCogVideoXPipeline"]
        )
        pipe = pipe_cls(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        lora_path = _resolve_path(options.get("lora_path"))
        if lora_path is not None:
            pipe.load_lora_weights(
                str(lora_path),
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="worldarena",
            )
            pipe.fuse_lora(lora_scale=1 / int(options.get("lora_rank", 128)))
        pipe.scheduler = symbols["CogVideoXDPMScheduler"].from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        pipe = pipe.to(dtype=dtype)
        offload_mode = str(options.get("offload_mode", "sequential_cpu")).strip().lower()
        if offload_mode in {"none", "cuda", "gpu"}:
            pipe = pipe.to("cuda")
        elif offload_mode in {"model_cpu", "model"} and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print_status("ac3d_runner", "pipeline_ready", mode=mode, variant=variant)
        return cls(
            repo_root=repo_root,
            checkpoint_dir=checkpoint_dir,
            options=options,
            variant=variant,
            mode=mode,
            base_model_path=base_model_path,
            controlnet_model_path=controlnet_model_path,
            dtype=dtype,
            torch=torch,
            symbols=symbols,
            pipe=pipe,
        )

    def _dataset_controlnet_latents(self, camera_index: int, num_frames: int, height: int, width: int) -> Any:
        if self.dataset is None:
            video_root_dir = _resolve_path(self.options.get("video_root_dir"))
            if video_root_dir is None:
                raise ValueError("AC3D dataset mode requires generation.video_root_dir")
            self.dataset = self.symbols["RealEstate10KPoseControlnetDataset"](
                video_root_dir=str(video_root_dir),
                annotation_json=str(self.options.get("annotation_json")),
                image_size=(height, width),
                stride=(int(self.options.get("stride_min", 2)), int(self.options.get("stride_max", 2))),
                sample_n_frames=num_frames,
            )
        data = self.dataset[camera_index]
        return data["controlnet_video"][None]

    def _image_controlnet_latents(self, row: dict[str, Any], num_frames: int, height: int, width: int) -> Any:
        return _synthetic_controlnet_latents(
            symbols=self.symbols,
            camera_path=_camera_path_from_row(row),
            num_frames=num_frames,
            height=height,
            width=width,
            focal_scale=float(self.options.get("focal_scale", 0.9)),
        )

    def _prepared_controlnet_latents(
        self,
        row: dict[str, Any],
        *,
        camera_index: int,
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float,
    ) -> Any:
        if self.mode == "image":
            controlnet_latents = self._image_controlnet_latents(row, num_frames, height, width)
            controlnet_latents = controlnet_latents.to(
                device=_pipeline_device(self.pipe, self.torch),
                dtype=self.dtype,
            )
            return _maybe_duplicate_for_guidance(controlnet_latents, guidance_scale)
        return self._dataset_controlnet_latents(camera_index, num_frames, height, width)

    def generate_row(self, row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
        from PIL import Image

        prompt = str(row.get("prompt") or self.options.get("prompt") or "")
        output_path = Path(str(row["output_path"])).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        camera_index = int(row.get("camera_index", row.get("start_camera_idx", 0)))
        fps = int(self.options.get("fps", self.options.get("output_fps", 8)))
        height = int(self.options.get("height", 480))
        width = int(self.options.get("width", 720))
        num_frames = _frames_for_min_duration(
            int(self.options.get("num_frames", 49)),
            fps,
            float(self.options.get("min_duration_seconds", 5.0)),
        )
        guidance_scale = float(self.options.get("guidance_scale", 6.0))
        controlnet_latents = self._prepared_controlnet_latents(
            row,
            camera_index=camera_index,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
        generator = self.torch.Generator(device="cpu").manual_seed(
            int(self.options.get("seed", 42)) + row_index
        )
        call_kwargs = {
            "prompt": prompt,
            "controlnet_latents": controlnet_latents,
            "num_videos_per_prompt": int(self.options.get("num_videos_per_prompt", 1)),
            "num_inference_steps": int(self.options.get("num_inference_steps", 50)),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "use_dynamic_cfg": _as_bool(self.options.get("use_dynamic_cfg", True)),
            "guidance_scale": guidance_scale,
            "generator": generator,
            "controlnet_weights": float(self.options.get("controlnet_weights", 1.0)),
            "controlnet_guidance_start": float(self.options.get("controlnet_guidance_start", 0.0)),
            "controlnet_guidance_end": float(self.options.get("controlnet_guidance_end", 0.4)),
        }
        if self.mode == "image":
            conditioning_image = Path(str(row["conditioning_image"])).expanduser().resolve()
            image = Image.open(conditioning_image).convert("RGB")
            frames = self.pipe(image=image, **call_kwargs).frames[0]
            raw_output = output_path.parent / "_ac3d_raw" / f"{output_path.stem}_raw.mp4"
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            self.symbols["export_to_video"](frames, str(raw_output), fps=fps)
        else:
            raw_dir = output_path.parent / "_ac3d_raw" / output_path.stem
            raw_dir.mkdir(parents=True, exist_ok=True)
            frames = self.pipe(**call_kwargs).frames[0]
            raw_output = raw_dir / f"{camera_index:05d}_out.mp4"
            self.symbols["export_to_video"](frames, str(raw_output), fps=fps)
            if not raw_output.is_file():
                raw_output = latest_mp4(raw_dir)
        copy_output(raw_output, output_path)
        return {
            "output_path": str(output_path),
            "raw_output_path": str(raw_output),
            "camera_path": _camera_path_from_row(row),
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": round(num_frames / max(fps, 1), 6),
        }


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
    mode = _conditioning_mode(options, _resolve_path(options.get("base_model_path")))
    base_model_path = _resolve_path(options.get("base_model_path")) or _default_base_model_path(
        checkpoint_dir,
        variant,
        mode,
    )
    controlnet_model_path = _resolve_path(
        options.get("controlnet_model_path") or options.get("checkpoint_path"),
        base=checkpoint_dir if checkpoint_dir is not None and checkpoint_dir.is_dir() else None,
    ) or _default_controlnet_path(checkpoint_dir, variant)
    if base_model_path is None:
        raise ValueError("AC3D requires generation.base_model_path or checkpoint_dir near CogVideoX weights")
    if controlnet_model_path is None:
        raise ValueError("AC3D requires checkpoint_dir or generation.controlnet_model_path")

    generator = AC3DBatchGenerator.build(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        options=options,
        variant=variant,
        mode=mode,
        base_model_path=base_model_path,
        controlnet_model_path=controlnet_model_path,
    )
    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            payload = generator.generate_row(row, row_index=row_index)
            print_status(sample_id, "generated", **payload)
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 1 if failed == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
