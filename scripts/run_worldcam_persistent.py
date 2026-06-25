from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent WorldCam video_static runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suites", default="video_static", type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--visible-devices", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _split_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _prepare_cuda_visibility(args: argparse.Namespace) -> None:
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _annotation_dir(sample) -> Path | None:
    if not sample.annotation_path:
        return None
    from worldarena.common.checkpoints import rehome_legacy_workspace_path

    remapped = rehome_legacy_workspace_path(sample.annotation_path)
    if remapped is None:
        return None
    path = Path(remapped).expanduser()
    return path.resolve() if path.is_dir() else None


def _reference_video(sample) -> Path:
    from worldarena.common.checkpoints import rehome_legacy_workspace_path

    remapped = rehome_legacy_workspace_path(sample.reference_path or sample.path)
    if remapped is None:
        raise FileNotFoundError(f"reference video not found: {sample.reference_path or sample.path}")
    path = Path(remapped).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"reference video not found: {path}")
    return path


def _target_frames_and_fps(sample, annotation_dir: Path | None, generation: dict[str, Any]) -> tuple[int, float, dict[str, Any]]:
    quality = _read_json_dict(annotation_dir / "quality_report.json") if annotation_dir is not None else {}
    frame_count = quality.get("source_video_frame_count") or quality.get("frame_count")
    if frame_count is None and annotation_dir is not None and (annotation_dir / "frame_indices.npy").is_file():
        frame_count = int(np.load(annotation_dir / "frame_indices.npy").shape[0])
    if frame_count is None and sample.duration_seconds and sample.fps:
        frame_count = int(round(float(sample.duration_seconds) * float(sample.fps)))
    if frame_count is None:
        configured_steps = generation.get("num_ar_steps")
        if _is_auto(configured_steps):
            frame_count = int(generation.get("fallback_frame_count", 233))
        else:
            frame_count = _decoded_frame_count(max(int(configured_steps), 1))

    fps = quality.get("fps") or sample.fps or generation.get("output_fps", 30)
    if _is_auto(fps) or str(fps).strip().lower() in {"match", "match_gt", "gt"}:
        fps = sample.fps or quality.get("fps") or 30
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    return max(int(frame_count), 1), fps, quality


def _is_auto(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "auto", "match", "match_gt", "gt"}
    return False


def _num_ar_steps_for_target_frames(target_frames: int, *, condition_latents: int) -> int:
    latent_steps_after_first = int(math.ceil(max(target_frames - 1, 1) / 4.0))
    return max(latent_steps_after_first - condition_latents, 1)


def _decoded_frame_count(num_ar_steps: int, *, condition_latents: int = 8) -> int:
    return 4 * (condition_latents + max(int(num_ar_steps), 1)) + 1


def _resolve_num_ar_steps(target_frames: int, generation: dict[str, Any], *, condition_latents: int) -> int:
    configured = generation.get("num_ar_steps")
    if bool(generation.get("match_gt_frame_count", True)) and _is_auto(configured):
        return _num_ar_steps_for_target_frames(target_frames, condition_latents=condition_latents)
    return max(int(configured), 1)


def _resolve_camera_frames(target_frames: int, num_ar_steps: int, generation: dict[str, Any], *, minimum_camera_frames) -> int:
    minimum = int(minimum_camera_frames(num_ar_steps=num_ar_steps))
    configured = generation.get("camera_frames")
    if _is_auto(configured):
        return max(target_frames, minimum)
    return max(int(configured), minimum)


def _trim_or_pad_video(frames: list[Any], target_frames: int) -> list[Any]:
    if target_frames <= 0:
        return frames
    if len(frames) >= target_frames:
        return frames[:target_frames]
    if not frames:
        return frames
    padded = list(frames)
    while len(padded) < target_frames:
        padded.append(padded[-1].copy())
    return padded


class WorldCamRuntime:
    def __init__(self, *, model_config, project_root: Path) -> None:
        self.model_config = model_config
        if model_config.repo_root is None:
            raise ValueError("WorldCam runner requires repo_root")
        if model_config.checkpoint_dir is None:
            raise ValueError("WorldCam runner requires checkpoint_dir")

        self.project_root = project_root
        self.repo_root = model_config.repo_root.resolve()
        self.checkpoint_dir = model_config.checkpoint_dir.resolve()
        self.generation = dict(model_config.generation)
        self.height = int(self.generation.get("height", 480))
        self.width = int(self.generation.get("width", 832))
        self.cond_frames = int(self.generation.get("cond_frames", 65))
        self.output_quality = int(self.generation.get("output_quality", 4))
        self.cfg_scale = float(self.generation.get("cfg_scale", 4.0))
        self.seed = int(self.generation.get("seed", 0))
        self.long_term_memory_start = int(self.generation.get("long_term_memory_start", 30))
        self.long_term_memory_num_clips = int(self.generation.get("long_term_memory_num_clips", 4))
        self.attention_sink_inference = bool(self.generation.get("attention_sink_inference", False))
        self.negative_prompt = str(self.generation.get("negative_prompt", ""))
        self.prompt_prefix = str(self.generation.get("prompt_prefix", ""))
        self.weights_filename = str(self.generation.get("weights_filename", "finetuned_dit.safetensors"))
        self.device = str(self.generation.get("device", "cuda"))

        os.chdir(self.repo_root)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from worldarena.common.checkpoints import apply_checkpoint_env

        os.environ.update(apply_checkpoint_env())

        import torch
        from diffsynth import save_video
        from diffsynth.models import ModelManager, load_state_dict
        from diffsynth.pipelines.wan_video_new import WanVideoPipeline
        from worldarena.models.adapters.worldcam import _ref_indices_arg
        from worldarena.models.adapters.worldcam_runner import (
            CAMERA_WINDOW_LATENTS,
            _camera_inputs,
            _conditioning_frames,
            _load_base_models,
            _minimum_camera_frames,
            _parse_ref_indices,
            _torch_dtype_name,
        )

        self.torch = torch
        self.save_video = save_video
        self._camera_inputs = _camera_inputs
        self._conditioning_frames = _conditioning_frames
        self._minimum_camera_frames = _minimum_camera_frames
        self._parse_ref_indices = _parse_ref_indices
        self.condition_latents = int(CAMERA_WINDOW_LATENTS)
        self.long_term_memory_ref_indices = _parse_ref_indices(
            _ref_indices_arg(self.generation.get("long_term_memory_ref_indices"))
        )

        torch_dtype_name = _torch_dtype_name(str(self.generation.get("torch_dtype", "bfloat16")))
        torch_dtype = getattr(torch, torch_dtype_name)

        self.pipe = WanVideoPipeline(torch_dtype=torch_dtype, device=self.device)
        model_manager = ModelManager()
        tokenizer_path = _load_base_models(model_manager=model_manager, pipe=self.pipe)

        self.pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        self.pipe.vae = model_manager.fetch_model("wan_video_vae")
        self.pipe.dit = model_manager.fetch_model("wan_video_dit")
        self.pipe.prompter.fetch_models(self.pipe.text_encoder)
        self.pipe.prompter.fetch_tokenizer(str(tokenizer_path))

        weights_path = self.checkpoint_dir / self.weights_filename
        if not weights_path.exists():
            raise FileNotFoundError(f"WorldCam fine-tuned weights not found: {weights_path}")
        self.pipe.dit.load_state_dict(load_state_dict(str(weights_path), device="cpu"), strict=True)

        self.pipe.text_encoder.to(self.pipe.device, dtype=self.pipe.torch_dtype)
        self.pipe.vae.to(self.pipe.device, dtype=self.pipe.torch_dtype)
        self.pipe.dit.to(self.pipe.device, dtype=self.pipe.torch_dtype)

    def generate_one(self, item: dict[str, Any]) -> dict[str, Any]:
        sample = item["sample"]
        annotation_dir = _annotation_dir(sample)
        if annotation_dir is None:
            raise FileNotFoundError(f"WorldCam sample has no annotation directory: {sample.sample_id}")
        reference_video = _reference_video(sample)
        target_frames, output_fps, quality = _target_frames_and_fps(sample, annotation_dir, self.generation)
        num_ar_steps = _resolve_num_ar_steps(
            target_frames,
            self.generation,
            condition_latents=self.condition_latents,
        )
        camera_frames = _resolve_camera_frames(
            target_frames,
            num_ar_steps,
            self.generation,
            minimum_camera_frames=self._minimum_camera_frames,
        )
        expected_decoded_frames = _decoded_frame_count(
            num_ar_steps,
            condition_latents=self.condition_latents,
        )
        prompt = self.prompt_prefix + str(item["prompt"])
        prompt = prompt.strip()

        conditioning_video = self._conditioning_frames(
            reference_video,
            cond_frames=self.cond_frames,
            height=self.height,
            width=self.width,
        )
        extrinsics_np, intrinsics_np = self._camera_inputs(str(annotation_dir), camera_frames=camera_frames)
        intrinsics = self.torch.from_numpy(intrinsics_np).to(self.pipe.device, dtype=self.torch.float32)[None, ...]
        extrinsics = self.torch.from_numpy(extrinsics_np).to(self.pipe.device, dtype=self.torch.float32)[None, ...]

        video = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            input_video=conditioning_video,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            height=self.height,
            width=self.width,
            cfg_scale=self.cfg_scale,
            seed=self.seed,
            tiled=True,
            long_term_memory_start_step=self.long_term_memory_start,
            long_term_memory_num_clips=self.long_term_memory_num_clips,
            long_term_memory_ref_indices=self.long_term_memory_ref_indices,
            num_ar_steps=num_ar_steps,
            attention_sink_inference=self.attention_sink_inference,
        )
        saved_video = _trim_or_pad_video(list(video), target_frames)
        output_path = item["prediction_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_video(saved_video, str(output_path), fps=output_fps, quality=self.output_quality)
        self.torch.cuda.empty_cache()
        if not output_path.exists():
            raise FileNotFoundError(f"WorldCam output was not written: {output_path}")
        return {
            "prediction_path": str(output_path),
            "prompt": prompt,
            "conditioning_input": str(reference_video),
            "conditioning_video_frame_count": self.cond_frames,
            "camera_annotation_path": str(annotation_dir),
            "camera_frames": camera_frames,
            "num_ar_steps": num_ar_steps,
            "expected_decoded_frames": expected_decoded_frames,
            "saved_frames": len(saved_video),
            "gt_video_frame_count": target_frames,
            "output_fps": output_fps,
            "gt_video_fps": output_fps,
            "gt_camera_pose_count": int(np.load(annotation_dir / "poses.npy").shape[0]) if (annotation_dir / "poses.npy").is_file() else None,
            "gt_camera_quality_status": quality.get("status"),
            "variant": "worldcam",
            "control_source": "vipe_intrinsics_extrinsics",
        }


def main() -> int:
    args = parse_args()
    _prepare_cuda_visibility(args)
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.benchmark.manifest import load_manifest
    from worldarena.common.checkpoints import apply_checkpoint_env
    from worldarena.generation.runner import _select_samples
    from worldarena.models.adapters.worldcam import WorldCamAdapter
    from worldarena.models.config import load_model_config

    model_config = load_model_config(args.model_config.resolve())
    os.environ.update(model_config.env)
    os.environ.update(apply_checkpoint_env())
    if model_config.family.lower() != "worldcam":
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")

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
    records_path = output_dir / f"generation_records.shard{args.shard_index}.jsonl"
    summary_path = output_dir / f"generation_manifest.shard{args.shard_index}.json"
    prompt_map_path = output_dir / f"prompt_map.shard{args.shard_index}.json"
    for path in (records_path, summary_path, prompt_map_path):
        path.unlink(missing_ok=True)

    adapter = WorldCamAdapter(model_config)
    runtime = WorldCamRuntime(model_config=model_config, project_root=project_root)
    records: list[dict[str, Any]] = []
    prompt_map: dict[str, str] = {}

    for offset, sample in enumerate(selected, start=1):
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(prediction_path),
            "conditioning_image": None,
            "conditioning_input": None,
            "prompt": None,
            "status": "pending",
            "error": None,
            "shard_index": args.shard_index,
            "shard_position": offset,
            "shard_total": len(selected),
        }
        started_at = time.perf_counter()
        try:
            prompt = adapter.prompt_for(sample)
            record["prompt"] = prompt
            prompt_map[str(prediction_path)] = prompt
            if prediction_path.exists() and not args.overwrite:
                record["status"] = "skipped_existing"
            else:
                payload = runtime.generate_one(
                    {
                        "sample": sample,
                        "prediction_path": prediction_path,
                        "prompt": prompt,
                    }
                )
                payload.setdefault("generation_wall_time_seconds", round(time.perf_counter() - started_at, 6))
                record.update(payload)
                record["status"] = "generated"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc) or repr(exc)
            record["traceback"] = traceback.format_exc()
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
