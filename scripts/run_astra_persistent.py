from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Astra batch generation runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suites", default=None, type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--visible-devices", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _split_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def _prepare_environment(*, visible_devices: str | None, project_root: Path, repo_root: Path) -> None:
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for path in (project_root, repo_root):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)

    from worldarena.common.checkpoints import apply_checkpoint_env

    os.environ.update(apply_checkpoint_env())


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_astra_module(repo_root: Path):
    module_path = (repo_root / "scripts" / "infer_demo.py").resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Astra entrypoint not found: {module_path}")
    spec = importlib.util.spec_from_file_location("worldarena_astra_infer_demo", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load Astra module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class PersistentAstraGenerator:
    def __init__(
        self,
        *,
        astra_module,
        checkpoint_dir: Path,
        generation: dict[str, Any],
        device: str,
    ) -> None:
        import torch
        import torch.nn as nn

        from worldarena.models.adapters.astra_runner import _resolve_dit_path, _resolve_wan_model_path

        self.mod = astra_module
        self.torch = torch
        self.nn = nn
        self.device = device
        self.generation = dict(generation)

        dit_path = _resolve_dit_path(checkpoint_dir, self.generation.get("dit_path"))
        wan_model_path = _resolve_wan_model_path(self.generation.get("wan_model_path"))

        print("[astra-persistent] loading Astra/Wan models once", flush=True)
        self.mod.replace_dit_model_in_manager()
        model_manager = self.mod.ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(
            [
                str(wan_model_path / "diffusion_pytorch_model.safetensors"),
                str(wan_model_path / "models_t5_umt5-xxl-enc-bf16.pth"),
                str(wan_model_path / "Wan2.1_VAE.pth"),
            ]
        )
        pipe = self.mod.WanVideoAstraPipeline.from_model_manager(model_manager, device="cuda")

        dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in pipe.dit.blocks:
            block.cam_encoder = nn.Linear(13, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))

        self.mod.add_framepack_components(pipe.dit)
        self.mod.add_moe_components(
            pipe.dit,
            {
                "num_experts": int(self.generation.get("moe_num_experts", 3)),
                "top_k": int(self.generation.get("moe_top_k", 1)),
                "hidden_dim": self.generation.get("moe_hidden_dim") or dim * 2,
                "sekai_input_dim": 13,
                "nuscenes_input_dim": 8,
                "openx_input_dim": 13,
            },
        )

        dit_state_dict = torch.load(dit_path, map_location="cpu")
        pipe.dit.load_state_dict(dit_state_dict, strict=False)
        del dit_state_dict
        pipe = pipe.to(device)
        self.model_dtype = next(pipe.dit.parameters()).dtype
        if hasattr(pipe.dit, "clean_x_embedder"):
            pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=self.model_dtype)
        pipe.scheduler.set_timesteps(50)
        self.pipe = pipe
        print("[astra-persistent] model ready", flush=True)

    def _check_generation_compatible(self, generation: dict[str, Any]) -> None:
        for key in ("moe_num_experts", "moe_top_k", "moe_hidden_dim"):
            if generation.get(key) != self.generation.get(key):
                raise ValueError(
                    f"persistent Astra runner cannot change {key} after model initialization: "
                    f"{self.generation.get(key)} -> {generation.get(key)}"
                )

    def generate(
        self,
        *,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
        generation: dict[str, Any],
        cam_type: int,
    ) -> None:
        import imageio
        import numpy as np
        from PIL import Image

        torch = self.torch
        self._check_generation_compatible(generation)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start_frame = int(generation.get("start_frame", 0))
        initial_condition_frames = int(generation.get("initial_condition_frames", 1))
        frames_per_generation = int(generation.get("frames_per_generation", 8))
        total_frames_to_generate = int(generation.get("total_frames_to_generate", 24))
        max_history_frames = int(generation.get("max_history_frames", 100))
        modality_type = str(generation.get("modality_type", "sekai"))
        use_real_poses = bool(generation.get("use_real_poses", False))
        scene_info_path = generation.get("scene_info_path")
        use_camera_cfg = bool(generation.get("use_camera_cfg", False))
        camera_guidance_scale = float(generation.get("camera_guidance_scale", 2.0))
        text_guidance_scale = float(generation.get("text_guidance_scale", 1.0))
        use_gt_prompt = bool(generation.get("use_gt_prompt", False))
        add_icons = bool(generation.get("add_icons", False))

        print(f"[astra-persistent] generating {output_path.name}", flush=True)
        initial_latents, encoded_data = self.mod.load_or_encode_condition(
            None,
            None,
            str(conditioning_image),
            start_frame,
            initial_condition_frames,
            self.device,
            self.pipe,
        )

        target_height, target_width = 60, 104
        channels, _, height, width = initial_latents.shape
        if height > target_height or width > target_width:
            h_start = (height - target_height) // 2
            w_start = (width - target_width) // 2
            initial_latents = initial_latents[
                :, :, h_start : h_start + target_height, w_start : w_start + target_width
            ]
            height, width = target_height, target_width

        history_latents = initial_latents.to(self.device, dtype=self.model_dtype)

        if use_gt_prompt and "prompt_emb" in encoded_data:
            prompt_emb_pos = encoded_data["prompt_emb"]
            if "context" in prompt_emb_pos:
                prompt_emb_pos["context"] = prompt_emb_pos["context"].to(
                    self.device, dtype=self.model_dtype
                )
            if "context_mask" in prompt_emb_pos:
                prompt_emb_pos["context_mask"] = prompt_emb_pos["context_mask"].to(
                    self.device, dtype=self.model_dtype
                )
            prompt_emb_neg = self.pipe.encode_prompt("") if text_guidance_scale > 1.0 else None
        else:
            prompt_emb_pos = self.pipe.encode_prompt(prompt)
            prompt_emb_neg = self.pipe.encode_prompt("") if text_guidance_scale > 1.0 else None

        scene_info = None
        if modality_type == "nuscenes" and scene_info_path and Path(scene_info_path).exists():
            scene_info = json.loads(Path(scene_info_path).read_text(encoding="utf-8"))

        if modality_type == "sekai":
            camera_embedding_full = self.mod.generate_sekai_camera_embeddings_sliding(
                encoded_data.get("cam_emb", None),
                start_frame,
                initial_condition_frames,
                total_frames_to_generate,
                0,
                use_real_poses=use_real_poses,
                cam_type=int(cam_type),
            ).to(self.device, dtype=self.model_dtype)
        elif modality_type == "nuscenes":
            camera_embedding_full = self.mod.generate_nuscenes_camera_embeddings_sliding(
                scene_info,
                start_frame,
                initial_condition_frames,
                total_frames_to_generate,
            ).to(self.device, dtype=self.model_dtype)
        elif modality_type == "openx":
            camera_embedding_full = self.mod.generate_openx_camera_embeddings_sliding(
                encoded_data,
                start_frame,
                initial_condition_frames,
                total_frames_to_generate,
                use_real_poses=use_real_poses,
            ).to(self.device, dtype=self.model_dtype)
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")

        camera_embedding_uncond = torch.zeros_like(camera_embedding_full) if use_camera_cfg else None
        total_generated = 0
        all_generated_frames = []

        with torch.inference_mode():
            while total_generated < total_frames_to_generate:
                current_generation = min(
                    frames_per_generation,
                    total_frames_to_generate - total_generated,
                )
                framepack_data = self.mod.prepare_framepack_sliding_window_with_camera_moe(
                    history_latents,
                    current_generation,
                    camera_embedding_full,
                    modality_type,
                )

                clean_latents = framepack_data["clean_latents"].unsqueeze(0)
                clean_latents_2x = framepack_data["clean_latents_2x"].unsqueeze(0)
                clean_latents_4x = framepack_data["clean_latents_4x"].unsqueeze(0)
                camera_embedding = framepack_data["camera_embedding"].unsqueeze(0)
                modality_inputs = {modality_type: camera_embedding}
                if use_camera_cfg and camera_embedding_uncond is not None:
                    camera_embedding_uncond_batch = camera_embedding_uncond[
                        : camera_embedding.shape[1], :
                    ].unsqueeze(0)
                    modality_inputs_uncond = {modality_type: camera_embedding_uncond_batch}

                latent_indices = framepack_data["latent_indices"].unsqueeze(0).cpu()
                clean_latent_indices = framepack_data["clean_latent_indices"].unsqueeze(0).cpu()
                clean_latent_2x_indices = framepack_data["clean_latent_2x_indices"].unsqueeze(0).cpu()
                clean_latent_4x_indices = framepack_data["clean_latent_4x_indices"].unsqueeze(0).cpu()

                new_latents = torch.randn(
                    1,
                    channels,
                    current_generation,
                    height,
                    width,
                    device=self.device,
                    dtype=self.model_dtype,
                )
                extra_input = self.pipe.prepare_extra_input(new_latents)

                for timestep in self.pipe.scheduler.timesteps:
                    timestep_tensor = timestep.unsqueeze(0).to(self.device, dtype=self.model_dtype)

                    if use_camera_cfg and camera_guidance_scale > 1.0:
                        noise_pred_cond, _ = self.pipe.dit(
                            new_latents,
                            timestep=timestep_tensor,
                            cam_emb=camera_embedding,
                            modality_inputs=modality_inputs,
                            latent_indices=latent_indices,
                            clean_latents=clean_latents,
                            clean_latent_indices=clean_latent_indices,
                            clean_latents_2x=clean_latents_2x,
                            clean_latent_2x_indices=clean_latent_2x_indices,
                            clean_latents_4x=clean_latents_4x,
                            clean_latent_4x_indices=clean_latent_4x_indices,
                            **prompt_emb_pos,
                            **extra_input,
                        )
                        noise_pred_uncond, _ = self.pipe.dit(
                            new_latents,
                            timestep=timestep_tensor,
                            cam_emb=camera_embedding_uncond_batch,
                            modality_inputs=modality_inputs_uncond,
                            latent_indices=latent_indices,
                            clean_latents=clean_latents,
                            clean_latent_indices=clean_latent_indices,
                            clean_latents_2x=clean_latents_2x,
                            clean_latent_2x_indices=clean_latent_2x_indices,
                            clean_latents_4x=clean_latents_4x,
                            clean_latent_4x_indices=clean_latent_4x_indices,
                            **(prompt_emb_neg if prompt_emb_neg else prompt_emb_pos),
                            **extra_input,
                        )
                        noise_pred = noise_pred_uncond + camera_guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                        if text_guidance_scale > 1.0 and prompt_emb_neg:
                            noise_pred_text_uncond, _ = self.pipe.dit(
                                new_latents,
                                timestep=timestep_tensor,
                                cam_emb=camera_embedding,
                                modality_inputs=modality_inputs,
                                latent_indices=latent_indices,
                                clean_latents=clean_latents,
                                clean_latent_indices=clean_latent_indices,
                                clean_latents_2x=clean_latents_2x,
                                clean_latent_2x_indices=clean_latent_2x_indices,
                                clean_latents_4x=clean_latents_4x,
                                clean_latent_4x_indices=clean_latent_4x_indices,
                                **prompt_emb_neg,
                                **extra_input,
                            )
                            noise_pred = noise_pred_text_uncond + text_guidance_scale * (
                                noise_pred - noise_pred_text_uncond
                            )
                    elif text_guidance_scale > 1.0 and prompt_emb_neg:
                        noise_pred_cond, _ = self.pipe.dit(
                            new_latents,
                            timestep=timestep_tensor,
                            cam_emb=camera_embedding,
                            modality_inputs=modality_inputs,
                            latent_indices=latent_indices,
                            clean_latents=clean_latents,
                            clean_latent_indices=clean_latent_indices,
                            clean_latents_2x=clean_latents_2x,
                            clean_latent_2x_indices=clean_latent_2x_indices,
                            clean_latents_4x=clean_latents_4x,
                            clean_latent_4x_indices=clean_latent_4x_indices,
                            **prompt_emb_pos,
                            **extra_input,
                        )
                        noise_pred_uncond, _ = self.pipe.dit(
                            new_latents,
                            timestep=timestep_tensor,
                            cam_emb=camera_embedding,
                            modality_inputs=modality_inputs,
                            latent_indices=latent_indices,
                            clean_latents=clean_latents,
                            clean_latent_indices=clean_latent_indices,
                            clean_latents_2x=clean_latents_2x,
                            clean_latent_2x_indices=clean_latent_2x_indices,
                            clean_latents_4x=clean_latents_4x,
                            clean_latent_4x_indices=clean_latent_4x_indices,
                            **prompt_emb_neg,
                            **extra_input,
                        )
                        noise_pred = noise_pred_uncond + text_guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                    else:
                        noise_pred, _ = self.pipe.dit(
                            new_latents,
                            timestep=timestep_tensor,
                            cam_emb=camera_embedding,
                            modality_inputs=modality_inputs,
                            latent_indices=latent_indices,
                            clean_latents=clean_latents,
                            clean_latent_indices=clean_latent_indices,
                            clean_latents_2x=clean_latents_2x,
                            clean_latent_2x_indices=clean_latent_2x_indices,
                            clean_latents_4x=clean_latents_4x,
                            clean_latent_4x_indices=clean_latent_4x_indices,
                            **prompt_emb_pos,
                            **extra_input,
                        )

                    new_latents = self.pipe.scheduler.step(noise_pred, timestep, new_latents)

                new_latents_squeezed = new_latents.squeeze(0)
                history_latents = torch.cat([history_latents, new_latents_squeezed], dim=1)
                if history_latents.shape[1] > max_history_frames:
                    first_frame = history_latents[:, 0:1, :, :]
                    recent_frames = history_latents[:, -(max_history_frames - 1) :, :, :]
                    history_latents = torch.cat([first_frame, recent_frames], dim=1)
                all_generated_frames.append(new_latents_squeezed)
                total_generated += current_generation

            all_generated = torch.cat(all_generated_frames, dim=1)
            final_video = torch.cat(
                [initial_latents.to(all_generated.device), all_generated],
                dim=1,
            ).unsqueeze(0)
            decoded_video = self.pipe.decode_video(
                final_video,
                tiled=True,
                tile_size=(34, 34),
                tile_stride=(18, 16),
            )

        video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
        video_np = (video_np * 0.5 + 0.5).clip(0, 1)
        video_np = (video_np * 255).astype(np.uint8)

        icons = {}
        video_camera_poses = None
        if add_icons:
            icons_dir = Path(self.mod.ROOT_DIR) / "icons"
            icon_names = [
                "move_forward.png",
                "not_move_forward.png",
                "move_backward.png",
                "not_move_backward.png",
                "move_left.png",
                "not_move_left.png",
                "move_right.png",
                "not_move_right.png",
                "turn_up.png",
                "not_turn_up.png",
                "turn_down.png",
                "not_turn_down.png",
                "turn_left.png",
                "not_turn_left.png",
                "turn_right.png",
                "not_turn_right.png",
            ]
            for name in icon_names:
                path = icons_dir / name
                if path.exists():
                    icons[name] = Image.open(path).convert("RGBA").resize(
                        (50, 50),
                        Image.Resampling.LANCZOS,
                    )
            camera_poses = camera_embedding_full.detach().float().cpu().numpy()
            video_camera_poses = [pose for pose in camera_poses for _ in range(4)]

        with imageio.get_writer(str(output_path), fps=20) as writer:
            for index, frame in enumerate(video_np):
                image = Image.fromarray(frame)
                if add_icons and video_camera_poses is not None and icons:
                    pose_index = start_frame + index
                    if pose_index < len(video_camera_poses):
                        image = self.mod.overlay_controls(
                            image,
                            video_camera_poses[pose_index],
                            icons,
                        )
                writer.append_data(np.array(image))

        del (
            initial_latents,
            history_latents,
            all_generated_frames,
            all_generated,
            final_video,
            decoded_video,
            video_np,
            prompt_emb_pos,
            prompt_emb_neg,
            camera_embedding_full,
        )
        torch.cuda.empty_cache()
        gc.collect()


def main() -> int:
    args = parse_args()
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.benchmark.manifest import load_manifest
    from worldarena.generation.runner import _conditioning_asset_for, _select_samples
    from worldarena.models.adapters.astra import (
        AstraAdapter,
        _astra_cam_type_for_sample,
        _generation_for_suite,
    )
    from worldarena.models.config import load_model_config

    model_config = load_model_config(args.model_config.resolve())
    if model_config.family.lower() != "astra":
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")
    if model_config.repo_root is None:
        raise ValueError("Astra persistent runner requires repo_root")
    if model_config.checkpoint_dir is None:
        raise ValueError("Astra persistent runner requires checkpoint_dir")

    repo_root = model_config.repo_root.resolve()
    _prepare_environment(
        visible_devices=args.visible_devices,
        project_root=project_root,
        repo_root=repo_root,
    )

    astra_module = _load_astra_module(repo_root)
    adapter = AstraAdapter(model_config)
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

    records: list[dict[str, Any]] = []
    prompt_map: dict[str, str] = {}
    generator: PersistentAstraGenerator | None = None
    for offset, sample in enumerate(selected, start=1):
        generation = _generation_for_suite(model_config.generation, sample.suite)
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        prompt = adapter.prompt_for(sample)
        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()
        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(prediction_path),
            "conditioning_image": None,
            "conditioning_input": None,
            "prompt": prompt_text,
            "status": "pending",
            "error": None,
            "shard_index": args.shard_index,
            "shard_position": offset,
            "shard_total": len(selected),
        }
        prompt_map[str(prediction_path)] = prompt_text
        try:
            conditioning_image = _conditioning_asset_for(
                sample,
                reference_mode=model_config.reference_mode,
                cache_root=conditioning_root,
                min_video_frames=int(generation.get("conditioning_video_frames", 8)),
            )
            record["conditioning_image"] = str(conditioning_image)
            record["conditioning_input"] = str(conditioning_image)
            cam_type, cam_type_source, camera_path = _astra_cam_type_for_sample(sample, generation)
            record["cam_type"] = int(cam_type)
            record["cam_type_source"] = cam_type_source
            record["camera_path"] = list(camera_path)
            if prediction_path.exists() and not args.overwrite:
                record["status"] = "skipped_existing"
            else:
                if generator is None:
                    generator = PersistentAstraGenerator(
                        astra_module=astra_module,
                        checkpoint_dir=model_config.checkpoint_dir,
                        generation=dict(model_config.generation),
                        device=args.device,
                    )
                generator.generate(
                    conditioning_image=conditioning_image.expanduser().resolve(),
                    output_path=prediction_path,
                    prompt=prompt_text,
                    generation=generation,
                    cam_type=int(cam_type),
                )
                record["status"] = "generated"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
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
        "supported_suites": model_config.supported_suites,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
