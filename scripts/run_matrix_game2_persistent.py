from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


_DEFAULT_CONFIG_BY_MODE = {
    "universal": "configs/inference_yaml/inference_universal.yaml",
    "gta_drive": "configs/inference_yaml/inference_gta_drive.yaml",
    "templerun": "configs/inference_yaml/inference_templerun.yaml",
}
_DEFAULT_CHECKPOINT_BY_MODE = {
    "universal": "base_distilled_model/base_distill.safetensors",
    "gta_drive": "gta_distilled_model/gta_keyboard2dim.safetensors",
    "templerun": "templerun_distilled_model/templerun_7dim_onlykey.safetensors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Matrix-Game-2 batch generation runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suites", default="image_static,image_dynamic", type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--visible-devices", default=None, type=str)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _split_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def _prepare_cuda_visibility(args: argparse.Namespace) -> None:
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_under(base_dir: Path, value: str | None, default_relative: str) -> Path:
    raw = value or default_relative
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resizecrop(image, target_height: int, target_width: int):
    width, height = image.size
    if height / width > target_height / target_width:
        new_width = int(width)
        new_height = int(new_width * target_height / target_width)
    else:
        new_height = int(height)
        new_width = int(new_height * target_width / target_height)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom))


def _write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


class MatrixGame2Runtime:
    def __init__(
        self,
        *,
        model_config,
        project_root: Path,
        device_id: int,
    ) -> None:
        import torch

        self.model_config = model_config
        self.project_root = project_root
        self.repo_root = model_config.repo_root.resolve()
        if model_config.checkpoint_dir is None:
            raise ValueError("Matrix-Game-2 runner requires checkpoint_dir")
        self.checkpoint_dir = model_config.checkpoint_dir.resolve()
        self.generation = dict(model_config.generation)
        self.mode = str(self.generation.get("mode", "universal")).strip().lower()
        if self.mode not in _DEFAULT_CONFIG_BY_MODE:
            supported = ", ".join(sorted(_DEFAULT_CONFIG_BY_MODE))
            raise ValueError(f"unsupported Matrix-Game-2 mode: {self.mode}; supported: {supported}")

        self.num_output_frames = int(self.generation.get("num_output_frames", 150))
        self.expected_action_frames = int(self.generation.get("num_frames", (self.num_output_frames - 1) * 4 + 1))
        self.output_fps = int(self.generation.get("output_fps", 12))
        self.save_num_frames = int(self.generation.get("save_num_frames") or 0)
        self.seed = int(self.generation.get("seed", 42))
        self.weight_dtype = torch.bfloat16
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(self.device)

        os.chdir(self.repo_root)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        import torch as _torch
        from diffusers.utils import load_image
        import imageio
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from torchvision.transforms import v2

        from demo_utils.vae_block3 import VAEDecoderWrapper
        from pipeline import CausalInferencePipeline
        from utils.misc import set_seed
        from utils.wan_wrapper import WanDiffusionWrapper
        from wan.vae.wanx_vae import get_wanx_vae_wrapper

        self.torch = _torch
        self.load_image = load_image
        self.imageio = imageio
        self.set_seed = set_seed

        config_path = _resolve_under(
            self.repo_root,
            self.generation.get("config_path"),
            _DEFAULT_CONFIG_BY_MODE[self.mode],
        )
        checkpoint_path = _resolve_under(
            self.checkpoint_dir,
            self.generation.get("checkpoint_filename"),
            _DEFAULT_CHECKPOINT_BY_MODE[self.mode],
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Matrix-Game-2 config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Matrix-Game-2 checkpoint not found: {checkpoint_path}")
        vae_path = self.checkpoint_dir / "Wan2.1_VAE.pth"
        if not vae_path.exists():
            raise FileNotFoundError(f"Matrix-Game-2 VAE checkpoint not found: {vae_path}")

        self.set_seed(self.seed)
        config = OmegaConf.load(config_path)
        generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)

        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = _torch.load(vae_path, map_location="cpu")
        decoder_state_dict = {
            key: value
            for key, value in vae_state_dict.items()
            if "decoder." in key or "conv2" in key
        }
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, _torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        if bool(self.generation.get("compile_vae", True)):
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        pipeline = CausalInferencePipeline(config, generator=generator, vae_decoder=current_vae_decoder)
        pipeline.generator.load_state_dict(load_file(checkpoint_path))
        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(_torch.float16)
        self.pipeline.eval()

        vae = get_wanx_vae_wrapper(str(self.checkpoint_dir), _torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

        self.frame_process = v2.Compose(
            [
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_images(self, items: list[dict[str, Any]]):
        tensors = []
        for item in items:
            image = self.load_image(str(item["conditioning_image"]))
            image = _resizecrop(image, 352, 640)
            tensors.append(self.frame_process(image))
        image_batch = self.torch.stack(tensors, dim=0)[:, :, None, :, :]
        return image_batch.to(dtype=self.weight_dtype, device=self.device)

    def _build_conditions(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        keyboard_rows = []
        mouse_rows = []
        has_mouse = self.mode != "templerun"
        for item in items:
            action_payload = item["action_payload"]
            keyboard_condition = np.asarray(action_payload["keyboard_condition"], dtype=np.float32)
            if keyboard_condition.shape[0] != self.expected_action_frames:
                raise ValueError(
                    "Matrix-Game-2 keyboard sequence length mismatch: "
                    f"expected {self.expected_action_frames}, got {keyboard_condition.shape[0]}"
                )
            keyboard_rows.append(keyboard_condition)
            if has_mouse:
                mouse_condition = np.asarray(action_payload["mouse_condition"], dtype=np.float32)
                if mouse_condition.shape[0] != self.expected_action_frames:
                    raise ValueError(
                        "Matrix-Game-2 mouse sequence length mismatch: "
                        f"expected {self.expected_action_frames}, got {mouse_condition.shape[0]}"
                    )
                mouse_rows.append(mouse_condition)

        keyboard_tensor = self.torch.tensor(
            np.stack(keyboard_rows),
            device=self.device,
            dtype=self.weight_dtype,
        )
        conditional_dict = {"keyboard_cond": keyboard_tensor}
        if has_mouse:
            conditional_dict["mouse_cond"] = self.torch.tensor(
                np.stack(mouse_rows),
                device=self.device,
                dtype=self.weight_dtype,
            )
        return conditional_dict

    def _seeded_noise(self, batch_size: int):
        noises = []
        for _ in range(batch_size):
            generator = self.torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            noises.append(
                self.torch.randn(
                    [1, 16, self.num_output_frames, 44, 80],
                    generator=generator,
                    device=self.device,
                    dtype=self.weight_dtype,
                )
            )
        return self.torch.cat(noises, dim=0)

    def generate_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from einops import rearrange

        torch = self.torch
        image = self._load_images(items)
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        visual_context = self.vae.clip.encode_video(image)
        conditional_dict = self._build_conditions(items)
        conditional_dict["cond_concat"] = cond_concat.to(device=self.device, dtype=self.weight_dtype)
        conditional_dict["visual_context"] = visual_context.to(device=self.device, dtype=self.weight_dtype)
        sampled_noise = self._seeded_noise(len(items))

        with torch.no_grad():
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                mode=self.mode,
                profile=False,
            )

        videos_tensor = torch.cat(videos, dim=1)
        frames_tensor = rearrange(videos_tensor, "b t c h w -> b t h w c")
        results: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            frames = ((frames_tensor[index].float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            if self.save_num_frames:
                if self.save_num_frames > frames.shape[0]:
                    raise ValueError(
                        f"save_num_frames={self.save_num_frames} exceeds generated frame count {frames.shape[0]}"
                    )
                frames = frames[: self.save_num_frames]
            frames = np.ascontiguousarray(frames)
            output_path = item["prediction_path"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.imageio.mimsave(str(output_path), frames, fps=self.output_fps)
            results.append(
                {
                    "actions": item["action_payload"].get("actions", []),
                    "action_source": item["action_payload"].get("action_source"),
                    "camera_token": item["action_payload"].get("camera_token"),
                    "camera_path": item["action_payload"].get("camera_path"),
                    "control_source": item["action_payload"].get("control_source"),
                    "prompt_control_note": item["action_payload"].get("prompt_control_note"),
                    "mode": self.mode,
                    "num_output_frames": self.num_output_frames,
                    "output_fps": self.output_fps,
                    "prediction_path": str(output_path),
                    "saved_frames": int(frames.shape[0]),
                    "variant": "matrix_game_2",
                }
            )
        return results


def _generate_with_oom_split(
    runtime: MatrixGame2Runtime,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        return runtime.generate_batch(items)
    except RuntimeError as exc:
        if len(items) <= 1:
            raise
        runtime.torch.cuda.empty_cache()
        midpoint = len(items) // 2
        return _generate_with_oom_split(runtime, items[:midpoint]) + _generate_with_oom_split(
            runtime,
            items[midpoint:],
        )


def main() -> int:
    args = parse_args()
    _prepare_cuda_visibility(args)
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")

    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.benchmark.manifest import load_manifest
    from worldarena.common.checkpoints import apply_checkpoint_env
    from worldarena.generation.runner import _conditioning_asset_for, _select_samples
    from worldarena.models.adapters.matrix_game import MatrixGameAdapter, _build_action_payload
    from worldarena.models.config import load_model_config

    os.environ.update(apply_checkpoint_env())

    model_config = load_model_config(args.model_config.resolve())
    if model_config.family.lower() not in {"matrix_game_2", "matrix-game-2"}:
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")
    if model_config.repo_root is None:
        raise ValueError("Matrix-Game-2 runner requires repo_root")

    manifest = load_manifest(args.manifest.resolve())
    suites = _split_arg(args.suites)
    selected = _select_samples(
        manifest,
        suites=suites,
        limit=args.limit,
        supported_suites=set(model_config.supported_suites),
    )
    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"shard_index must be within [0, {args.num_shards - 1}]")
    selected = selected[args.shard_index :: args.num_shards]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning_root = (output_dir / f"_conditioning_shard{args.shard_index}").resolve()
    conditioning_root.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"generation_records.shard{args.shard_index}.jsonl"
    summary_path = output_dir / f"generation_manifest.shard{args.shard_index}.json"
    prompt_map_path = output_dir / f"prompt_map.shard{args.shard_index}.json"
    for path in (records_path, summary_path, prompt_map_path):
        path.unlink(missing_ok=True)

    adapter = MatrixGameAdapter(model_config)
    generation = dict(model_config.generation)
    pending: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    prompt_map: dict[str, str] = {}

    for offset, sample in enumerate(selected, start=1):
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        prompt = adapter.prompt_for(sample)
        action_spec_path = output_dir / f"{prediction_path.stem}_actions.json"
        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(prediction_path),
            "conditioning_image": None,
            "conditioning_input": None,
            "prompt": prompt,
            "status": "pending",
            "error": None,
            "shard_index": args.shard_index,
            "shard_position": offset,
            "shard_total": len(selected),
        }
        prompt_map[str(prediction_path)] = prompt
        try:
            conditioning_image = _conditioning_asset_for(
                sample,
                reference_mode=model_config.reference_mode,
                cache_root=conditioning_root,
                min_video_frames=int(generation.get("conditioning_video_frames", 8)),
            )
            record["conditioning_image"] = str(conditioning_image)
            record["conditioning_input"] = str(conditioning_image)
            if prediction_path.exists() and not args.overwrite:
                record["status"] = "skipped_existing"
                records.append(record)
                _write_record(records_path, record)
                continue
            action_payload = _build_action_payload(
                generation,
                sample.suite,
                family=model_config.family,
                sample=sample,
            )
            action_spec_path.write_text(
                json.dumps(action_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            pending.append(
                {
                    "sample": sample,
                    "record": record,
                    "conditioning_image": conditioning_image,
                    "prediction_path": prediction_path,
                    "action_payload": action_payload,
                    "action_spec_path": action_spec_path,
                }
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc) or repr(exc)
            records.append(record)
            _write_record(records_path, record)

    runtime = MatrixGame2Runtime(
        model_config=model_config,
        project_root=project_root,
        device_id=args.device_id,
    )

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        try:
            results = _generate_with_oom_split(runtime, batch)
            for item, payload in zip(batch, results, strict=True):
                record = item["record"]
                record.update(payload)
                record["action_spec_path"] = str(item["action_spec_path"])
                record["status"] = "generated"
                records.append(record)
                _write_record(records_path, record)
        except Exception as exc:
            error_text = str(exc) or repr(exc)
            for item in batch:
                record = item["record"]
                record["status"] = "failed"
                record["error"] = error_text
                record["action_spec_path"] = str(item["action_spec_path"])
                records.append(record)
                _write_record(records_path, record)

    prompt_map_path.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "model_name": model_config.name,
        "family": model_config.family,
        "output_dir": str(output_dir),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": len(records),
        "generated": sum(1 for item in records if item["status"] == "generated"),
        "skipped_existing": sum(1 for item in records if item["status"] == "skipped_existing"),
        "failed": sum(1 for item in records if item["status"] == "failed"),
        "batch_size": args.batch_size,
        "suites": sorted(suites) if suites else model_config.supported_suites,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
