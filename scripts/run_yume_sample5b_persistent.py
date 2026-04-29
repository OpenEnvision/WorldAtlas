from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent YUME sample_5b.py batch runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suites", default=None, type=str)
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


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _prepare_environment(*, visible_devices: str | None, project_root: Path, repo_root: Path) -> None:
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    for path in (project_root, repo_root):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)

    from worldarena.common.checkpoints import apply_checkpoint_env

    os.environ.update(apply_checkpoint_env(create=False))


def _load_sample5b_module(repo_root: Path):
    module_path = (repo_root / "fastvideo" / "sample" / "sample_5b.py").resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"YUME sample_5b entrypoint not found: {module_path}")
    spec = importlib.util.spec_from_file_location("worldarena_yume_sample5b", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load YUME sample_5b module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stage_caption_file(source: Path, destination: Path, *, line_limit: int | None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = source.read_text(encoding="utf-8").splitlines()
    if line_limit is not None:
        lines = lines[: max(1, int(line_limit))]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _make_image_tuple(sample_mod, image_path: Path, *, stem: str, height: int, width: int, frame_count: int):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    video_tensor = torch.zeros(img_tensor.shape[0], frame_count, height, width)
    resized_frame = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    video_tensor[:, 0] = (resized_frame - 0.5) * 2
    return video_tensor.permute(1, 0, 2, 3), stem, str(image_path)


def _safe_sample5b_exporter(sample_mod, *, stem: str, target_frames: int | None, fps: int | None):
    original_export = sample_mod.export_to_video

    def export_to_video_short_name(video, filename, fps=16):
        path = Path(filename)
        if path.name.startswith(f"{stem}_"):
            suffix_match = re.search(r"_(\d+)_(\d+)_(\d+)\.mp4$", path.name)
            if suffix_match:
                filename = str(path.with_name(f"{stem}_{suffix_match.group(1)}_{suffix_match.group(2)}_{suffix_match.group(3)}.mp4"))
        if target_frames and isinstance(video, list):
            if len(video) > target_frames:
                video = video[:target_frames]
            elif video:
                video = video + [video[-1]] * (target_frames - len(video))
        return original_export(video, filename, fps=int(fps or 16))

    return original_export, export_to_video_short_name


class _DirectPromptCaptionModel:
    def __init__(self) -> None:
        self.current_prompt = ""

    def chat(self, tokenizer, pixel_values, question, generation_config):
        return self.current_prompt


class PersistentYumeSample5B:
    def __init__(
        self,
        *,
        sample_mod,
        repo_root: Path,
        checkpoint_dir: Path,
        caption_model_dir: Path,
        generation: dict[str, Any],
        device_id: int,
        runtime_output_dir: Path,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.sample_mod = sample_mod
        self.repo_root = repo_root
        self.generation = dict(generation)
        self.runtime_output_dir = runtime_output_dir
        self.runtime_output_dir.mkdir(parents=True, exist_ok=True)
        self.device_id = int(device_id)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.set_device(self.device_id)
        self.device = torch.cuda.current_device()

        print("[yume-sample5b] loading Yume-5B once", flush=True)
        cfg = sample_mod.WAN_CONFIGS["ti2v-5B"]
        self.wan_i2v = sample_mod.wan23.Yume(
            config=cfg,
            checkpoint_dir=str(checkpoint_dir),
            device_id=self.device,
        )
        self.transformer = self.wan_i2v.model.eval().requires_grad_(False)
        self.transformer = self.transformer.to(device=self.device, dtype=torch.bfloat16)
        self.vae = self.wan_i2v.vae
        self.wan_i2v.device = self.device
        self.denoiser = sample_mod.load_denoiser()

        if bool(self.generation.get("load_caption_model", False)):
            print("[yume-sample5b] loading InternVL caption model once", flush=True)
            self.caption_model = AutoModel.from_pretrained(
                str(caption_model_dir),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            ).eval().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(caption_model_dir),
                trust_remote_code=True,
                use_fast=False,
            )
        else:
            print("[yume-sample5b] direct prompt mode; InternVL caption model disabled", flush=True)
            self.caption_model = _DirectPromptCaptionModel()
            self.tokenizer = None
        print("[yume-sample5b] model ready", flush=True)

    def generate(
        self,
        *,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
        caption_path: Path,
    ) -> dict[str, Any]:
        import torch

        height = int(self.generation.get("height", 704))
        width = int(self.generation.get("width", 1280))
        num_euler_timesteps = int(self.generation.get("num_euler_timesteps", 4))
        rand_num_img = float(self.generation.get("rand_num_img", 0.6))
        fps = int(self.generation.get("fps", 16))
        frame_zero = int(self.generation.get("frame_zero", 32))
        latent_frame_zero = int(self.generation.get("latent_frame_zero", max(frame_zero // 4, 1)))
        output_frames = int(self.generation.get("output_frames", frame_zero))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        stem = output_path.stem
        before = set(self.runtime_output_dir.glob(f"{stem}_*.mp4"))
        dataset_ddp = [
            _make_image_tuple(
                self.sample_mod,
                conditioning_image,
                stem=stem,
                height=height,
                width=width,
                frame_count=frame_zero + 1,
            )
        ]

        print(f"[yume-sample5b] generating {output_path.name}", flush=True)
        original_export, export_wrapper = _safe_sample5b_exporter(
            self.sample_mod,
            stem=stem,
            target_frames=output_frames,
            fps=fps,
        )
        self.sample_mod.WORLDARENA_FRAME_ZERO = frame_zero
        self.sample_mod.WORLDARENA_LATENT_FRAME_ZERO = latent_frame_zero
        if isinstance(self.caption_model, _DirectPromptCaptionModel):
            self.caption_model.current_prompt = prompt
        self.sample_mod.export_to_video = export_wrapper
        try:
            self.sample_mod.sample_one(
                self.transformer,
                bool(self.generation.get("not_apply_cfg_solver", False)),
                self.device,
                dataset_ddp=dataset_ddp,
                video_output_dir=str(self.runtime_output_dir),
                rand_num_img=rand_num_img,
                num_euler_timesteps=num_euler_timesteps,
                vae=self.vae,
                step=1,
                wan_i2v=self.wan_i2v,
                denoiser=self.denoiser,
                rank=0,
                world_size=1,
                image_sample=True,
                caption_path=str(caption_path),
                camption_model=self.caption_model,
                tokenizer=self.tokenizer,
                t2v=False,
                prompt1=prompt,
            )
        finally:
            self.sample_mod.export_to_video = original_export
        generated = sorted(set(self.runtime_output_dir.glob(f"{stem}_*.mp4")) - before)
        if not generated:
            raise FileNotFoundError(f"YUME sample_5b did not write an output for {stem}")
        selected = generated[-1] if self.generation.get("output_index", "last") == "last" else generated[0]
        shutil.copy2(selected, output_path)

        torch.cuda.empty_cache()
        gc.collect()
        return {
            "prediction_path": str(output_path),
            "output_path": str(output_path),
            "prompt": prompt,
            "sample5b_source_path": str(selected),
            "num_euler_timesteps": num_euler_timesteps,
            "rand_num_img": rand_num_img,
            "fps": fps,
            "height": height,
            "width": width,
            "frame_zero": frame_zero,
            "latent_frame_zero": latent_frame_zero,
            "output_frames": output_frames,
            "duration_seconds": output_frames / max(float(fps), 1.0),
        }


def main() -> None:
    args = parse_args()
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.benchmark.manifest import load_manifest
    from worldarena.generation.runner import _conditioning_asset_for, _select_samples
    from worldarena.models.adapters.yume import YumeAdapter
    from worldarena.models.config import load_model_config
    from worldarena.common.checkpoints import resolve_project_path

    model_config = load_model_config(args.model_config)
    generation = dict(model_config.generation)
    repo_root = resolve_project_path(generation.get("sample5b_repo_root") or "/share/project/yesliu/hjd/model/tmp/YUME")
    if repo_root is None:
        raise ValueError("YUME sample_5B runner requires sample5b_repo_root")
    _prepare_environment(visible_devices=args.visible_devices, project_root=project_root, repo_root=repo_root)

    sample_mod = _load_sample5b_module(repo_root)
    checkpoint_dir = resolve_project_path(model_config.checkpoint_dir)
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"YUME checkpoint directory not found: {model_config.checkpoint_dir}")
    caption_model_value = generation.get("caption_model_dir", "ckpt/InternVL3-2B-Instruct")
    caption_model_dir = resolve_project_path(caption_model_value)
    if caption_model_dir is None or not caption_model_dir.is_dir():
        raise FileNotFoundError(f"YUME caption model directory not found: {caption_model_value}")
    caption_source = Path(generation.get("sample5b_caption_path") or (repo_root / "caption_re.txt")).expanduser()
    if not caption_source.is_absolute():
        caption_source = (repo_root / caption_source).resolve()
    if not caption_source.exists():
        raise FileNotFoundError(f"YUME sample_5B caption file not found: {caption_source}")

    manifest = load_manifest(args.manifest.resolve())
    selected = _select_samples(
        manifest,
        suites=_split_arg(args.suites),
        limit=args.limit,
        supported_suites=set(model_config.supported_suites),
    )
    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"shard_index must be within [0, {args.num_shards - 1}]")
    selected = selected[args.shard_index :: args.num_shards]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning_root = (output_dir / f"_conditioning_shard{args.shard_index}").resolve()
    conditioning_root.mkdir(parents=True, exist_ok=True)
    runtime_output_dir = (output_dir / f"_sample5b_runtime_shard{args.shard_index}").resolve()
    caption_path = _stage_caption_file(
        caption_source,
        output_dir / "_sample5b_caption" / f"caption_shard{args.shard_index}.txt",
        line_limit=generation.get("sample5b_caption_lines", 1),
    )

    adapter = YumeAdapter(model_config)
    runtime = PersistentYumeSample5B(
        sample_mod=sample_mod,
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        caption_model_dir=caption_model_dir,
        generation=generation,
        device_id=args.device_id,
        runtime_output_dir=runtime_output_dir,
    )

    records_path = output_dir / f"generation_records.shard{args.shard_index}.jsonl"
    summary_path = output_dir / f"generation_manifest.shard{args.shard_index}.json"
    records_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)
    records: list[dict[str, Any]] = []

    for offset, sample in enumerate(selected, start=1):
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        prompt = adapter.prompt_for(sample)
        record: dict[str, Any] = {
            "case_id": sample.sample_id,
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(prediction_path),
            "output_path": str(prediction_path),
            "prompt": prompt,
            "prompt_current": sample.prompt_current,
            "prompt_target": sample.prompt_target,
            "prompt_mode": model_config.prompt_mode,
            "prompt_source": model_config.prompt_mode,
            "caption_refinement": "disabled",
            "status": "pending",
            "error": None,
            "shard_index": args.shard_index,
            "shard_position": offset,
            "shard_total": len(selected),
        }
        try:
            conditioning_image = _conditioning_asset_for(
                sample,
                reference_mode=model_config.reference_mode,
                cache_root=conditioning_root,
                min_video_frames=int(generation.get("conditioning_video_frames", 8)),
            )
            record["conditioning_image"] = str(conditioning_image)
            if prediction_path.exists() and not args.overwrite:
                record["status"] = "skipped_existing"
            else:
                payload = runtime.generate(
                    conditioning_image=conditioning_image,
                    output_path=prediction_path,
                    prompt=prompt,
                    caption_path=caption_path,
                )
                record.update(payload)
                record["status"] = "generated"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
        records.append(record)
        _write_record(records_path, record)

    summary = {
        "model_name": model_config.name,
        "family": model_config.family,
        "runner": "sample_5b",
        "output_dir": str(output_dir),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": len(records),
        "generated": sum(1 for item in records if item["status"] == "generated"),
        "skipped_existing": sum(1 for item in records if item["status"] == "skipped_existing"),
        "failed": sum(1 for item in records if item["status"] == "failed"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
