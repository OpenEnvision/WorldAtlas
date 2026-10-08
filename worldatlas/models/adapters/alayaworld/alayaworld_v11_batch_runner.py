"""Persistent inference through AlayaWorld v1.1's official ViGeo validation path.

Only model/history/checkpoint initialization is extracted from trainer.setup():
optimizers, training data, the DMD critic and its EMA are unnecessary for inference.
Rollout, prompt switching, spatial memory and decoding remain upstream operations.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Any
from types import MethodType

import numpy as np
import yaml

from worldatlas.benchmark.synthetic_camera import synthetic_camera_matrices
from worldatlas.common.checkpoints import project_root
from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args, begin_sample, copy_output, load_batch_spec, load_json,
    print_status,
)
from .alayaworld_batch_runner import _trajectory_runtime
from .alayaworld_v11 import UPSTREAM_REVISION, runtime_options, sample_seed

PROJECT_ROOT = project_root()
TRAJECTORY_FRAMES = 8192  # Official CustomI2VDataset's fixed trajectory buffer.


def validate_source(repo_root: Path) -> str:
    required = ("configs/infer_i2v_camera.yaml", "configs/infer_i2v_camera_ar.yaml",
                "alaya/trainer/rollout_trainer.py", "alaya/data/custom_i2v.py",
                "alaya/model/loader.py", "alaya/checkpoint.py")
    missing = [p for p in required if not (repo_root / p).is_file()]
    if missing:
        raise FileNotFoundError(f"AlayaWorld v1.1 source missing at {repo_root}: {missing}")
    revision = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"AlayaWorld v1.1 requires revision {UPSTREAM_REVISION}, got {revision}")
    changes = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if changes:
        raise ValueError("AlayaWorld v1.1 source has tracked modifications")
    return revision


def runtime_paths(checkpoint_root: Path, options: dict[str, Any]) -> dict[str, Path]:
    defaults = {
        "base_model": checkpoint_root / "ltx-2.3-22b-dev.safetensors",
        "ar_checkpoint": checkpoint_root / "stage2b",
        "dmd_checkpoint": checkpoint_root / "stage3",
        "gemma_dir": checkpoint_root / "gemma-3-12b-it-qat-q4_0-unquantized",
        "vigeo_repo": PROJECT_ROOT / "thirdparty" / "ViGeo",
        "vigeo_checkpoint": checkpoint_root / "ViGeo1.1",
    }
    paths = {}
    for key, default in defaults.items():
        path = Path(options.get(key) or default).expanduser()
        paths[key] = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    return paths


def validate_weights(paths: dict[str, Path], variant: str) -> None:
    required = [paths["base_model"], paths["ar_checkpoint"] / "transformer.pt",
                paths["ar_checkpoint"] / "history_encoder.pt",
                paths["gemma_dir"] / "config.json",
                paths["gemma_dir"] / "model.safetensors.index.json",
                paths["gemma_dir"] / "tokenizer.json",
                paths["vigeo_repo"] / "vigeo" / "__init__.py",
                paths["vigeo_checkpoint"] / "vigeo.pt"]
    if variant == "dmd":
        required.append(paths["dmd_checkpoint"] / "lora.safetensors")
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("AlayaWorld v1.1 missing runtime files:\n" + "\n".join(missing))
    index = json.loads((paths["gemma_dir"] / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        raise ValueError("Gemma checkpoint index has no weight_map")
    for shard in shards:
        path = paths["gemma_dir"] / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Gemma checkpoint shard missing: {path}")


def camera_trajectory(row: dict[str, Any], options: dict[str, Any]) -> np.ndarray:
    actions = list(row.get("round_actions") or [])
    if not actions:
        raise ValueError("AlayaWorld v1.1 requires nonempty round_actions")
    needed = 25 + len(actions) * 32
    if needed > TRAJECTORY_FRAMES:
        raise ValueError(f"rollout needs {needed} camera frames; upstream limit is {TRAJECTORY_FRAMES}")
    motion_options = {
        "forward_distance_per_chunk": 0.1568, "lateral_distance_per_chunk": 0.1568,
        "vertical_distance_per_chunk": 0.1568, "rotation_degrees_per_chunk": 4.8,
        "tilt_degrees_per_chunk": 4.8, "roll_degrees_per_chunk": 4.8,
    }
    motion_options.update(options)
    motion = synthetic_camera_matrices(
        list(row.get("trajectory_actions") or actions), target_frames=len(actions) * 32 + 1,
        runtime=_trajectory_runtime(motion_options),
    ).matrices
    # Seed pixels are copies of one still image, so all 25 seed poses agree.
    # The output starts at pose 24; its first generated pixel uses pose 25.
    poses = np.concatenate([np.repeat(motion[:1], 24, axis=0), motion], axis=0)
    # CustomI2VDataset cycles short trajectories. Fill its entire camera buffer
    # with a static tail so it cannot jump to frame zero during a long rollout.
    return np.concatenate([
        poses, np.repeat(poses[-1:], TRAJECTORY_FRAMES - len(poses), axis=0),
    ]).astype(np.float32)


def materialize_inputs(row: dict[str, Any], directory: Path, options: dict[str, Any]) -> dict:
    from PIL import Image

    source = Path(row["conditioning_image"]).expanduser().resolve()
    images = directory / "images"
    images.mkdir(parents=True, exist_ok=True)
    # Fixed name, fresh directory per request: no filename sorting/pose pairing ambiguity.
    with Image.open(source) as image:
        image.convert("RGB").save(images / "reference.png")
    poses = camera_trajectory(row, options)
    camera_file = directory / "camera.npz"
    entry = {"assigned_image": "reference.png", "pose_path": str(camera_file)}
    arrays = {"cam_c2w": poses}
    if options.get("intrinsic") is not None:
        intrinsic = np.asarray(options["intrinsic"], dtype=np.float32)
        if intrinsic.shape != (4,) or not np.isfinite(intrinsic).all() or np.any(intrinsic[:2] <= 0):
            raise ValueError("intrinsic must be normalized [fx, fy, cx, cy] with positive focal lengths")
        entry["intrinsic"] = intrinsic.tolist()
    # Without calibrated intrinsics the official dataset selects ViGeo-fitted K.
    np.savez(camera_file, **arrays)
    (directory / "pose.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    (directory / "captions.json").write_text(
        json.dumps({"reference.png": str(row["prompt"])}, ensure_ascii=False), encoding="utf-8",
    )
    return {
        "source": "custom_i2v", "image_dir": str(images),
        "captions_json": str(directory / "captions.json"),
        "pose_jsonl": str(directory / "pose.jsonl"), "pose_offset": 0,
    }


def prepare_config(repo_root: Path, paths: dict[str, Path], options: dict, work: Path) -> dict:
    filename = "infer_i2v_camera_ar.yaml" if options["variant"] == "ar" else "infer_i2v_camera.yaml"
    cfg = yaml.safe_load((repo_root / "configs" / filename).read_text())
    cfg["paths"].update(
        base_transformer=str(paths["base_model"]), vae=str(paths["base_model"]),
        gemma=str(paths["gemma_dir"]), resume_checkpoint=str(paths["ar_checkpoint"]),
        history_encoder=str(paths["ar_checkpoint"] / "history_encoder.pt"),
        dmd_resume=str(paths["dmd_checkpoint"]) if options["variant"] == "dmd" else None,
        resume_reset_step=True,
    )
    cfg["run"].update(output_dir=str(work / "outputs"), log_dir=str(work / "logs"), seed=int(options["seed"]))
    cfg["sample"].update(height=options["height"], width=options["width"], fps=24.0)
    attention = options["attention_type"]
    cfg["runtime"].update(
        fsdp=False, gradient_checkpointing=False, dataloader_workers=0,
        # Upstream names this enum FA3, but dispatches to FA2 with ALAYA_USE_FA3=0.
        attention_type="flash_attention_3" if attention == "flash_attention_2" else attention,
        text_embed_cache_dir=str(work / "text_cache"),
        vae_latent_cache_dir=str(work / "vae_cache"),
    )
    cfg["spatial_memory"].update(
        vigeo_repo_path=str(paths["vigeo_repo"]), vigeo_checkpoint=str(paths["vigeo_checkpoint"]),
    )
    cfg["lora"].update(enabled=options["variant"] == "dmd", train=False)
    cfg["memory"]["train"] = False
    cfg["validation"].update(save_joystick=False, save_videos=True, max_samples=1, sample_offset=0)
    mode = cfg["validation"]["modes"]["custom_i2v"]
    mode["layout"].update(height=options["height"], width=options["width"])
    mode["rollout_rounds"] = options["rounds"]
    return cfg


def initialize_trainer(cfg: Any, upstream: Any) -> Any:
    """Same inference weights as upstream setup, without training-only allocations."""
    trainer = upstream.RolloutTrainer(cfg)
    dmd_enabled = cfg.dmd.enabled
    try:
        cfg.dmd.enabled = False  # Do not allocate the teacher/critic/EMA a second time.
        trainer.components = upstream.build_model_components(cfg, trainer.dist.device, trainer.dtype)
    finally:
        cfg.dmd.enabled = dmd_enabled
    projection = trainer.components.transformer.patchify_proj
    trainer.history_encoder = upstream.build_history_encoder(
        cfg.memory, in_channels=projection.in_features, out_channels=projection.out_features,
        device=trainer.dist.device, dtype=trainer.dtype, checkpoint_path=cfg.paths.history_encoder,
    )
    if cfg.memory.use_lr_branch:
        trainer.history_encoder.setup_lr_proj_from_patchify(projection)
    trainer.global_step = upstream.load_checkpoint_weights(
        cfg=cfg, dist_state=trainer.dist, transformer=trainer.components.transformer,
        history_encoder=trainer.history_encoder, lora_manager=trainer.components.lora_manager,
    )
    trainer.components.transformer.eval().requires_grad_(False)
    trainer.history_encoder.eval().requires_grad_(False)
    return trainer


def _bounded_mask_video(self, mask_latent, *, like_frames):
    """Render the diagnostic mask with bounded GPU memory, preserving nearest sampling."""
    import torch
    import torch.nn.functional as F

    count, height, width = (int(v) for v in like_frames.shape[:3])
    mask = mask_latent[:1, :1].to(device=self.dist.device, dtype=torch.float32)
    # Temporal nearest resampling is independent of spatial nearest resampling.
    mask = F.interpolate(mask, size=(count, *mask.shape[-2:]), mode="nearest")
    frames = []
    for start in range(0, count, 32):
        part = mask[:, :, start:start + 32]
        part = F.interpolate(part, size=(part.shape[2], height, width), mode="nearest")
        part = (part * 255.0).round().clamp(0, 255).to(torch.uint8)
        frames.append(part.squeeze(0).permute(1, 2, 3, 0).cpu().expand(-1, -1, -1, 3))
    return torch.cat(frames, dim=0)


def load_runtime(repo_root: Path, config_path: Path, options: dict):
    os.environ["ALAYA_USE_FA3"] = "0"
    if options["offline"]:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("AlayaWorld v1.1 uses one GPU per worker; launch independent sample shards")
    # The pinned attention module imports this extension even with FA3 disabled.
    # Loading it together with ViGeo/xformers can abort in C++, before Python can recover.
    if importlib.util.find_spec("flash_attn_interface") is not None:
        raise RuntimeError("Use a dedicated v1.1 environment without FlashAttention-3 (flash_attn_interface)")
    import torch

    version = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:3])
    if version < (2, 7, 1):
        raise RuntimeError("AlayaWorld v1.1 requires torch>=2.7.1; use the separate v1.1 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("AlayaWorld v1.1 inference requires CUDA")
    if options["attention_type"] == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError("flash-attn (FA2) is missing; install it or configure attention_type: pytorch")
    if importlib.util.find_spec("av") is None:
        raise RuntimeError("PyAV is required for AlayaWorld v1.1 video encoding and validation")
    sys.path.insert(0, str(repo_root))
    from worldatlas._compat.av_write_video.sitecustomize import install_torchvision_write_video_compat
    from ltx2.modules import attention
    from .alayaworld_upstream_runner import _install_attention_backend_override

    install_torchvision_write_video_compat()
    if options["attention_type"] == "xformers":
        from xformers.ops import memory_efficient_attention

        attention.memory_efficient_attention = memory_efficient_attention
    if options["attention_type"] != "flash_attention_2":
        _install_attention_backend_override(options["attention_type"], attention)
    from alaya.config.loader import load_config
    from alaya.trainer import rollout_trainer

    trainer = initialize_trainer(load_config(config_path), rollout_trainer)
    if options.get("bounded_mask_video", False):
        trainer._mask_latent_to_video_frames = MethodType(_bounded_mask_video, trainer)
    return trainer


def run_sample(trainer: Any, mode_payload: dict, mode_name: str, seed: int) -> Path:
    from alaya.config.schema import TrainConfig
    from alaya.utils.seed import seed_everything

    # Validate the mode in the full runtime config: the official converter also
    # validates model paths and training/layout fields, even for inference.
    payload = asdict(trainer.cfg)
    payload["validation"]["modes"] = {mode_name: mode_payload}
    mode = TrainConfig.from_mapping(payload).validation.modes[mode_name]
    trainer.cfg.validation.modes = {mode_name: mode}
    trainer.validation_datasets.clear()
    seed_everything(seed)
    trainer.cfg.run.seed = seed
    try:
        trainer.validate(trainer.global_step)
        step = f"step-{trainer.global_step:06d}{trainer.cfg.validation.step_dir_suffix}"
        return (Path(trainer.cfg.run.output_dir) / "validation" / step / mode_name
                / "rank-000_global-000000_sample-000000_pred_clean.mp4")
    finally:
        trainer.validation_datasets.clear()
        trainer._cleanup_after_validation()


def validate_video(path: Path, *, expected_frames: int, options: dict) -> dict:
    import av

    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 0)
        if abs(fps - options["fps"]) > 0.01:
            raise RuntimeError(f"AlayaWorld v1.1 FPS mismatch: {fps}")
        for frame in container.decode(stream):
            if (frame.width, frame.height) != (options["width"], options["height"]):
                raise RuntimeError("AlayaWorld v1.1 output resolution mismatch")
            count += 1
    if count != expected_frames:
        raise RuntimeError(f"AlayaWorld v1.1 frame count mismatch: {count}, expected {expected_frames}")
    return {"num_frames": count, "fps": fps, "width": options["width"], "height": options["height"]}


def run_batch(rows: list[dict], repo_root: Path, checkpoint_root: Path, generation: dict,
              work: Path, *, dry_run: bool = False) -> int:
    options = runtime_options(generation)
    revision = validate_source(repo_root)
    paths = runtime_paths(checkpoint_root, options)
    # Dry-run deliberately needs only the source and inputs, not large checkpoints.
    if not dry_run:
        validate_weights(paths, options["variant"])
    payload = prepare_config(repo_root, paths, options, work)
    template = deepcopy(payload["validation"]["modes"]["custom_i2v"])
    config_path = work / "inference.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    trainer = None
    failed = 0
    for index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        begin_sample(sample_id, index=index + 1, total=len(rows), label="AlayaWorld v1.1")
        try:
            mode_name = f"sample_{index:06d}"
            inputs = materialize_inputs(row, work / mode_name, options)
            rounds = len(row["round_actions"])
            schedule = list(row.get("prompt_schedule") or [])
            if schedule and len(schedule) != rounds:
                raise ValueError("prompt_schedule must contain exactly one prompt per round")
            mode = deepcopy(template)
            mode.update(dataset=inputs, rollout_rounds=rounds, prompt_schedule=schedule)
            seed = int(row.get("seed", sample_seed(sample_id, options)))
            payload["run"]["seed"] = seed
            payload["validation"]["modes"] = {mode_name: mode}
            sample_config = work / mode_name / "inference.yaml"
            sample_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            if dry_run:
                print_status(sample_id, "dry_run", dry_run=True, config_path=str(sample_config),
                             round_count=rounds, expected_frames=1 + rounds * 32,
                             seed=seed, upstream_revision=revision)
                continue
            if trainer is None:
                trainer = load_runtime(repo_root, sample_config, options)
                print("[AlayaWorld v1.1] checkpoint_loads=1", flush=True)
            source = run_sample(trainer, mode, mode_name, seed)
            video = validate_video(source, expected_frames=1 + rounds * 32, options=options)
            output = Path(row["output_path"]).expanduser().resolve()
            copy_output(source, output)
            sidecar = output.with_suffix(".alayaworld_v11.json")
            record = dict(video, sample_id=sample_id, variant=options["variant"], seed=seed,
                          upstream_revision=revision, checkpoint_load_count=1,
                          round_actions=row["round_actions"], prompt=str(row["prompt"]),
                          prompt_schedule=schedule, runtime_config=str(sample_config),
                          weights={k: str(v) for k, v in paths.items()},
                          camera_npz=str(work / mode_name / "camera.npz"))
            temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
            temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(sidecar)
            # Keep controls/config/diagnostics, but not duplicate large upstream videos.
            for video_path in source.parent.glob("*.mp4"):
                video_path.unlink()
            print_status(sample_id, "generated", output_path=str(output), **video)
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
            if not options["continue_on_error"] or (trainer is None and not dry_run):
                break
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AlayaWorld v1.1 ViGeo batch inference")
    add_common_batch_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="Prepare official inputs/configs without weights or CUDA")
    args = parser.parse_args()
    if not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required")
    spec_path = Path(args.batch_spec_path).resolve()
    work = Path(tempfile.mkdtemp(prefix="alayaworld_v11_", dir=spec_path.parent))
    return run_batch(load_batch_spec(spec_path), Path(args.repo_root).resolve(),
                     Path(args.checkpoint_dir).resolve(), load_json(Path(args.generation_config_path)),
                     work, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
