from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Matrix-Game-3 batch generation runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suites", default="image_static,image_dynamic", type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--visible-devices", default=None, type=str)
    parser.add_argument("--device-id", default=0, type=int)
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


def _write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class MatrixGame3Runtime:
    def __init__(self, *, model_config, project_root: Path, output_dir: Path, device_id: int) -> None:
        import torch

        self.model_config = model_config
        self.project_root = project_root
        self.repo_root = model_config.repo_root.resolve()
        if model_config.checkpoint_dir is None:
            raise ValueError("Matrix-Game-3 runner requires checkpoint_dir")
        self.checkpoint_dir = model_config.checkpoint_dir.resolve()
        self.generation = dict(model_config.generation)
        self.output_dir = output_dir
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(self.device)

        os.chdir(self.repo_root)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from PIL import Image
        import imageio
        import torch as _torch

        from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
        import pipeline.inference_pipeline as mg_inference
        from utils.cam_utils import get_extrinsics
        from utils.transform import get_video_transform
        from utils.utils import compute_all_poses_from_actions
        from worldarena.models.adapters.matrix_game_runner import (
            normalize_to_neg_one_to_one,
            patch_torch_dynamo_config_aliases,
        )

        patch_torch_dynamo_config_aliases()

        self.Image = Image
        self.imageio = imageio
        self.torch = _torch
        self.mg_inference = mg_inference
        self.get_extrinsics = get_extrinsics
        self.get_video_transform = get_video_transform
        self.compute_all_poses_from_actions = compute_all_poses_from_actions
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.MAX_AREA_CONFIGS = MAX_AREA_CONFIGS

        self.size = str(self.generation.get("size", "704*1280"))
        self.height = int(self.size.split("*")[0])
        self.width = int(self.size.split("*")[1])
        self.fps = int(self.generation.get("fps", 17))
        self.save_num_frames = int(self.generation.get("save_num_frames") or 0)
        self.seed = int(self.generation.get("seed", 42))
        self.num_iterations = int(dict(self.generation.get("action_plan", {})).get("num_iterations", self.generation.get("num_iterations", 2)))

        self.runtime_args = SimpleNamespace(
            ckpt_dir=str(self.checkpoint_dir),
            size=self.size,
            save_name="generated_video",
            seed=self.seed,
            num_iterations=self.num_iterations,
            output_dir=str(output_dir),
            sample_shift=self.generation.get("sample_shift"),
            sample_guide_scale=float(self.generation.get("sample_guide_scale", 5.0)),
            num_inference_steps=int(self.generation.get("num_inference_steps", 3)),
            ulysses_size=int(self.generation.get("ulysses_size", 1)),
            t5_fsdp=bool(self.generation.get("t5_fsdp", False)),
            t5_cpu=bool(self.generation.get("t5_cpu", False)),
            dit_fsdp=bool(self.generation.get("dit_fsdp", False)),
            convert_model_dtype=bool(self.generation.get("convert_model_dtype", False)),
            use_base_model=bool(self.generation.get("use_base_model", False)),
            use_int8=bool(self.generation.get("use_int8", True)),
            verify_quant=bool(self.generation.get("verify_quant", False)),
            vae_type=str(self.generation.get("vae_type", "mg_lightvae_v2")),
            lightvae_pruning_rate=self.generation.get("lightvae_pruning_rate"),
            use_async_vae=bool(self.generation.get("use_async_vae", False)),
            async_vae_warmup_iters=int(self.generation.get("async_vae_warmup_iters", 0)),
            compile_vae=bool(self.generation.get("compile_vae", True)),
            fa_version=str(self.generation.get("fa_version", "0")),
        )

        cfg = WAN_CONFIGS["matrix_game3"]
        if self.runtime_args.sample_shift is None:
            self.runtime_args.sample_shift = cfg.sample_shift
        if self.runtime_args.sample_guide_scale is None:
            self.runtime_args.sample_guide_scale = cfg.sample_guide_scale

        def patched_process_video(
            input_video,
            output_video,
            config,
            mouse_icon_path,
            mouse_scale=1.0,
            mouse_rotation=0,
            default_frame_res=(704, 1280),
        ):
            del config, mouse_icon_path, mouse_scale, mouse_rotation, default_frame_res
            frames = input_video
            if self.save_num_frames:
                if self.save_num_frames > len(frames):
                    raise ValueError(
                        f"save_num_frames={self.save_num_frames} exceeds generated frame count {len(frames)}"
                    )
                frames = frames[: self.save_num_frames]
            output_path = Path(output_video)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.imageio.mimsave(str(output_path), np.ascontiguousarray(frames), fps=self.fps)

        mg_inference.process_video = patched_process_video

        self.pipeline = mg_inference.MatrixGame3Pipeline(
            config=cfg,
            checkpoint_dir=str(self.checkpoint_dir),
            device_id=device_id,
            rank=0,
            t5_fsdp=self.runtime_args.t5_fsdp,
            dit_fsdp=self.runtime_args.dit_fsdp,
            use_sp=False,
            t5_cpu=self.runtime_args.t5_cpu,
            convert_model_dtype=self.runtime_args.convert_model_dtype,
            args=self.runtime_args,
            fa_version=self.runtime_args.fa_version,
            use_base_model=self.runtime_args.use_base_model,
        )

    def _install_get_data(self, action_payload: dict[str, Any], pil_image) -> None:
        import torch

        keyboard_condition_np = np.asarray(action_payload["keyboard_condition"], dtype=np.float32)
        mouse_condition_np = np.asarray(action_payload["mouse_condition"], dtype=np.float32)
        expected_frames = int(action_payload["total_frames"])
        if keyboard_condition_np.shape[0] != expected_frames or mouse_condition_np.shape[0] != expected_frames:
            raise ValueError(
                "Matrix-Game-3 action sequence length mismatch: "
                f"expected {expected_frames}, got keyboard={keyboard_condition_np.shape[0]}, "
                f"mouse={mouse_condition_np.shape[0]}"
            )

        def custom_get_data(num_frames, height, width, _pil_image, device=None, dtype=None):
            del _pil_image
            if keyboard_condition_np.shape[0] != num_frames or mouse_condition_np.shape[0] != num_frames:
                raise ValueError(
                    "Custom action sequence length mismatch: "
                    f"expected {num_frames}, got keyboard={keyboard_condition_np.shape[0]}, "
                    f"mouse={mouse_condition_np.shape[0]}"
                )
            input_image = torch.from_numpy(np.asarray(pil_image)).unsqueeze(0).permute(0, 3, 1, 2)
            transform = self.get_video_transform(height, width, self.normalize_to_neg_one_to_one)
            input_image = transform(input_image).transpose(0, 1).unsqueeze(0)

            first_pose = np.zeros(5, dtype=np.float32)
            all_poses = self.compute_all_poses_from_actions(
                keyboard_condition_np,
                mouse_condition_np,
                first_pose=first_pose,
            )
            positions = all_poses[:, :3].tolist()
            rotations = np.concatenate(
                [np.zeros((all_poses.shape[0], 1), dtype=np.float32), all_poses[:, 3:5]],
                axis=1,
            ).tolist()
            extrinsics_all = self.get_extrinsics(rotations, positions)

            keyboard_tensor = torch.tensor(keyboard_condition_np, dtype=dtype, device=device).unsqueeze(0)
            mouse_tensor = torch.tensor(mouse_condition_np, dtype=dtype, device=device).unsqueeze(0)
            return input_image.to(device, dtype), extrinsics_all, keyboard_tensor, mouse_tensor

        self.mg_inference.get_data = custom_get_data

    def generate_one(self, item: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        pil_image = self.Image.open(item["conditioning_image"]).convert("RGB")
        self._install_get_data(item["action_payload"], pil_image)
        self.runtime_args.save_name = item["prediction_path"].stem
        self.runtime_args.output_dir = str(item["prediction_path"].parent)
        self.pipeline.output_dir = str(item["prediction_path"].parent)
        try:
            self.pipeline.generate(
                item["prompt"],
                pil_image,
                max_area=self.MAX_AREA_CONFIGS[self.size],
                shift=self.runtime_args.sample_shift,
                num_inference_steps=self.runtime_args.num_inference_steps,
                guide_scale=self.runtime_args.sample_guide_scale,
                seed=self.seed,
                use_base_model=self.runtime_args.use_base_model,
                args=self.runtime_args,
            )
        except SystemExit:
            pass
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if not item["prediction_path"].exists():
            raise FileNotFoundError(f"Matrix-Game-3 output was not written: {item['prediction_path']}")
        return {
            "actions": item["action_payload"].get("actions", []),
            "action_source": item["action_payload"].get("action_source"),
            "camera_token": item["action_payload"].get("camera_token"),
            "camera_path": item["action_payload"].get("camera_path"),
            "control_source": item["action_payload"].get("control_source"),
            "prompt_control_note": item["action_payload"].get("prompt_control_note"),
            "mode": "matrix_game_3",
            "num_iterations": self.num_iterations,
            "output_fps": self.fps,
            "saved_frames": self.save_num_frames or None,
            "prediction_path": str(item["prediction_path"]),
            "variant": "matrix_game_3",
        }


def main() -> int:
    args = parse_args()
    _prepare_cuda_visibility(args)
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
    os.environ.update(model_config.env)
    if model_config.family.lower().replace("-", "_") not in {"matrix_game", "matrix_game_3"}:
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")
    if model_config.repo_root is None:
        raise ValueError("Matrix-Game-3 runner requires repo_root")

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
                    "prompt": prompt,
                    "action_payload": action_payload,
                    "action_spec_path": action_spec_path,
                }
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc) or repr(exc)
            record["traceback"] = traceback.format_exc()
            records.append(record)
            _write_record(records_path, record)

    runtime = MatrixGame3Runtime(
        model_config=model_config,
        project_root=project_root,
        output_dir=output_dir,
        device_id=args.device_id,
    )

    for item in pending:
        record = item["record"]
        started_at = time.perf_counter()
        try:
            payload = runtime.generate_one(item)
            payload.setdefault(
                "generation_wall_time_seconds",
                round(time.perf_counter() - started_at, 6),
            )
            record.update(payload)
            record["action_spec_path"] = str(item["action_spec_path"])
            record["status"] = "generated"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc) or repr(exc)
            record["traceback"] = traceback.format_exc()
            record["action_spec_path"] = str(item["action_spec_path"])
            record["generation_wall_time_seconds"] = round(time.perf_counter() - started_at, 6)
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
        "suites": sorted(suites) if suites else model_config.supported_suites,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
