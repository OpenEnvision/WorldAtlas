"""WorldArena batch runner for the official AlayaWorld inference CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices
from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    latest_mp4,
    load_batch_spec,
    load_json,
    print_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_REQUIRED_FILES = (
    "configs/infer.yaml",
    "inference/run.py",
    "flash_alaya/utils/pipeline.py",
    "flash_alaya/alaya/memory/da3_depth.py",
)


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    model_file: Path
    gemma_dir: Path
    da3_repo: Path
    da3_cache: Path
    taehv_file: Path | None


@dataclass(frozen=True, slots=True)
class PersistentRuntime:
    cfg: Any
    engine: Any
    pipeline_type: Any
    load_input_sample: Any
    check_input_resolution: Any
    plan_rollout: Any
    apply_joystick_overlay: Any
    output_latent_frames: int
    history_latent_frames: int
    gap_steps: int
    condition_latent_frames: int
    flex_attn: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena AlayaWorld batch runner")
    add_common_batch_args(parser)
    return parser.parse_args()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: Any, *, bases: list[Path]) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()


def _runtime_paths(
    *,
    checkpoint_dir: Path,
    generation: dict[str, Any],
) -> RuntimePaths:
    checkpoint_root = checkpoint_dir.parent
    model_value = generation.get("model_file")
    model_file = (
        _resolve_path(model_value, bases=[PROJECT_ROOT, checkpoint_dir])
        if model_value
        else checkpoint_dir / "merged_infer.safetensors"
    )
    gemma_value = generation.get("gemma_dir")
    gemma_dir = (
        _resolve_path(gemma_value, bases=[PROJECT_ROOT, checkpoint_root])
        if gemma_value
        else checkpoint_root / "gemma-3-12b-it-qat-q4_0-unquantized"
    )
    da3_repo_value = generation.get("da3_repo")
    da3_repo = (
        _resolve_path(da3_repo_value, bases=[PROJECT_ROOT])
        if da3_repo_value
        else PROJECT_ROOT / "thirdparty" / "Depth-Anything-3"
    )
    da3_cache_value = generation.get("da3_cache")
    da3_cache = (
        _resolve_path(da3_cache_value, bases=[PROJECT_ROOT, checkpoint_root])
        if da3_cache_value
        else checkpoint_root / "huggingface"
    )
    taehv_value = generation.get("taehv_file")
    taehv_file = (
        _resolve_path(taehv_value, bases=[PROJECT_ROOT, checkpoint_root])
        if taehv_value
        else None
    )
    return RuntimePaths(
        model_file=model_file.resolve(),
        gemma_dir=gemma_dir.resolve(),
        da3_repo=da3_repo.resolve(),
        da3_cache=da3_cache.resolve(),
        taehv_file=taehv_file.resolve() if taehv_file is not None else None,
    )


def _validate_runtime(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    paths: RuntimePaths,
) -> None:
    missing_repo = [name for name in UPSTREAM_REQUIRED_FILES if not (repo_root / name).is_file()]
    if missing_repo:
        raise FileNotFoundError(
            f"AlayaWorld repo is incomplete at {repo_root}; missing: {', '.join(missing_repo)}"
        )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"AlayaWorld checkpoint directory not found: {checkpoint_dir}")
    if not paths.model_file.is_file() or paths.model_file.stat().st_size <= 0:
        raise FileNotFoundError(f"AlayaWorld merged checkpoint not found: {paths.model_file}")
    if not paths.gemma_dir.is_dir() or not (paths.gemma_dir / "config.json").is_file():
        raise FileNotFoundError(f"AlayaWorld Gemma directory is incomplete: {paths.gemma_dir}")
    if not (paths.da3_repo / "src" / "depth_anything_3").is_dir():
        raise FileNotFoundError(f"Depth-Anything-3 source checkout not found: {paths.da3_repo}")
    if not paths.da3_cache.is_dir():
        raise FileNotFoundError(f"Depth-Anything-3 cache directory not found: {paths.da3_cache}")
    if paths.taehv_file is not None and not paths.taehv_file.is_file():
        raise FileNotFoundError(f"optional TAEHV checkpoint not found: {paths.taehv_file}")


def _validate_python_runtime(generation: dict[str, Any]) -> None:
    """Fail before loading 60 GB of models when an inference extra is absent."""
    missing: list[str] = []
    if importlib.util.find_spec("av") is None:
        missing.append("av (PyAV, required by torchvision MP4 encoding)")
    attention_type = str(generation.get("attention_type", "xformers")).strip().lower()
    if attention_type == "xformers" and importlib.util.find_spec("xformers") is None:
        missing.append("xformers (configured AlayaWorld attention backend)")
    if missing:
        raise RuntimeError(
            "AlayaWorld Python runtime is incomplete; install: " + ", ".join(missing)
        )


def _patch_inference_config(
    *,
    repo_root: Path,
    paths: RuntimePaths,
    generation: dict[str, Any],
    output_path: Path,
) -> Path:
    source_value = generation.get("config_file")
    source_path = (
        _resolve_path(source_value, bases=[PROJECT_ROOT, repo_root])
        if source_value
        else repo_root / "configs" / "infer.yaml"
    )
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    payload.setdefault("paths", {}).update(
        {
            "model": str(paths.model_file),
            "gemma": str(paths.gemma_dir),
            "da3_repo": str(paths.da3_repo),
            "da3_model": str(
                generation.get("da3_model", "depth-anything/DA3NESTED-GIANT-LARGE-1.1")
            ),
            "da3_cache": str(paths.da3_cache),
            "taehv": str(paths.taehv_file) if paths.taehv_file is not None else "",
        }
    )
    payload.setdefault("run", {})["output_dir"] = str(output_path.parent)
    payload.setdefault("sample", {}).update(
        {
            "height": int(generation.get("height", 544)),
            "width": int(generation.get("width", 960)),
            "fps": float(generation.get("fps", 24.0)),
            "temporal_stride": int(generation.get("temporal_stride", 8)),
        }
    )
    payload.setdefault("layout", {}).update(
        {
            "sink_latent_frames": int(generation.get("sink_latent_frames", 1)),
            "history_latent_frames": int(generation.get("history_latent_frames", 16)),
        }
    )
    payload.setdefault("validation", {})["save_joystick"] = _as_bool(
        generation.get("joystick"), False
    )
    payload.setdefault("runtime", {})["attention_type"] = str(
        generation.get("attention_type", "xformers")
    )
    if generation.get("sampling_steps") is not None:
        payload["validation"]["sampling_steps"] = int(generation["sampling_steps"])
    if generation.get("da3_process_res") is not None:
        payload.setdefault("spatial_memory", {})["da3_process_res"] = int(
            generation["da3_process_res"]
        )
    if generation.get("action_scale") is not None:
        payload.setdefault("control", {})["action_scale"] = str(generation["action_scale"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def _trajectory_runtime(generation: dict[str, Any]) -> dict[str, float]:
    return {
        "synthetic_camera_forward_step": float(
            generation.get("forward_distance_per_chunk", 2.54)
        ),
        "synthetic_camera_lateral_step": float(
            generation.get("lateral_distance_per_chunk", 0.5348)
        ),
        "synthetic_camera_vertical_step": float(
            generation.get("vertical_distance_per_chunk", 1.1076)
        ),
        "synthetic_camera_rotation_degrees": float(
            generation.get("rotation_degrees_per_chunk", 12.5)
        ),
        "synthetic_camera_tilt_degrees": float(
            generation.get("tilt_degrees_per_chunk", 10.0)
        ),
        "synthetic_camera_roll_degrees": float(
            generation.get("roll_degrees_per_chunk", 8.0)
        ),
        "synthetic_camera_orbit_radius": float(generation.get("orbit_radius", 2.0)),
    }


def _build_camera_metadata(
    row: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    import torch

    round_actions = [str(value) for value in row.get("round_actions") or ["fixed"]]
    temporal_stride = max(int(generation.get("temporal_stride", 8)), 1)
    chunk_latent_frames = max(int(generation.get("chunk_latent_frames", 4)), 1)
    sink_latent_frames = max(int(generation.get("sink_latent_frames", 1)), 0)
    history_latent_frames = max(int(generation.get("history_latent_frames", 16)), 0)
    frames_per_chunk = temporal_stride * chunk_latent_frames
    trajectory_runtime = _trajectory_runtime(generation)
    motion = synthetic_camera_matrices(
        round_actions,
        target_frames=len(round_actions) * frames_per_chunk + 1,
        runtime=trajectory_runtime,
    )
    prefix_intervals = (sink_latent_frames + history_latent_frames) * temporal_stride
    prefix_mode = str(generation.get("camera_prefix_mode", "extend_first_action")).strip().lower()
    if prefix_mode == "static":
        prefix = np.repeat(
            np.eye(4, dtype=np.float32)[None],
            repeats=prefix_intervals + 1,
            axis=0,
        )
    elif prefix_mode == "extend_first_action":
        # The official playground trajectories move from pixel frame zero even
        # though image inputs replicate their first frame to seed sink/history.
        # Extend the first requested action through that visual prefix so target
        # action index 17 is non-zero and spatial memory sees the intended pose.
        warmup_chunks = max(math.ceil(prefix_intervals / frames_per_chunk), 1)
        warmup = synthetic_camera_matrices(
            [round_actions[0]] * warmup_chunks,
            target_frames=warmup_chunks * frames_per_chunk + 1,
            runtime=trajectory_runtime,
        ).matrices
        prefix = warmup[: prefix_intervals + 1]
    else:
        raise ValueError(
            f"unsupported AlayaWorld camera_prefix_mode {prefix_mode!r}; "
            "expected extend_first_action or static"
        )

    motion_global = prefix[-1][None] @ motion.matrices[1:]
    cam_c2w_np = np.concatenate([prefix, motion_global], axis=0).astype(np.float32)
    cam_c2w = torch.from_numpy(cam_c2w_np)

    intrinsic = torch.tensor(
        [
            [float(generation.get("intrinsic_fx", 0.4482019544)), 0.0, 0.5],
            [0.0, float(generation.get("intrinsic_fy", 0.7965529561)), 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    frame_count = int(cam_c2w.shape[0])
    return {
        "intrinsic": intrinsic,
        "cam_c2w": cam_c2w,
        "video_id": str(row.get("sample_id", "worldarena")),
        "has_camera": True,
        "source": "worldarena_camera_path",
        "caption_type": "overall",
        "pose_orig_w": float(generation.get("pose_orig_width", 1280.0)),
        "pose_orig_h": float(generation.get("pose_orig_height", 720.0)),
        "frame_start": 0,
        "frame_end": frame_count,
        "intrinsic_raw": intrinsic.clone(),
        "cam_c2w_raw": cam_c2w.clone(),
    }


def _safe_stem(value: Any, fallback: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return stem or fallback


def _skill_prompt(row: dict[str, Any], generation: dict[str, Any]) -> str | None:
    if not _as_bool(generation.get("enable_prompt_switching"), True):
        return None
    explicit = str(row.get("skill_prompt") or "").strip()
    if explicit:
        return explicit
    sequence = [str(value).strip() for value in row.get("prompt_sequence") or [] if str(value).strip()]
    if sequence and sequence[-1] != str(row.get("prompt") or "").strip():
        return sequence[-1]
    return None


def _materialize_case(
    *,
    row: dict[str, Any],
    row_index: int,
    case_root: Path,
    generation: dict[str, Any],
) -> tuple[Path, dict[str, Any], str | None]:
    import torch

    source = Path(str(row["conditioning_image"])).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"AlayaWorld conditioning image not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise ValueError(f"unsupported AlayaWorld conditioning image extension: {suffix}")

    stem = _safe_stem(row.get("prediction_stem"), f"sample_{row_index:06d}")
    row_root = case_root / f"{row_index:06d}_{stem}"
    row_root.mkdir(parents=True, exist_ok=True)
    prefix = row_root / stem
    shutil.copy2(source, Path(str(prefix) + f"_image{suffix}"))
    metadata = _build_camera_metadata(row, generation)
    torch.save(metadata, Path(str(prefix) + "_camera.pt"))
    Path(str(prefix) + "_prompt.txt").write_text(
        str(row.get("prompt") or "").strip() + "\n",
        encoding="utf-8",
    )
    skill_prompt = _skill_prompt(row, generation)
    if skill_prompt:
        Path(str(prefix) + "_skill.txt").write_text(skill_prompt + "\n", encoding="utf-8")
    return prefix, metadata, skill_prompt


def _official_command(
    *,
    prefix: Path,
    config_path: Path,
    output_dir: Path,
    rounds: int,
    seed: int,
    skill_prompt: str | None,
    generation: dict[str, Any],
) -> list[str]:
    args = [
        "-m",
        "worldarena.models.adapters.alayaworld_upstream_runner",
        "--input",
        str(prefix),
        "--cfg",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--rounds",
        str(rounds),
        "--seed",
        str(seed),
        "--compile",
        str(generation.get("compile", "reduce-overhead")),
        "--video-crf",
        str(int(generation.get("video_crf", 28))),
    ]
    args.append("--joystick" if _as_bool(generation.get("joystick"), False) else "--no-joystick")
    if not _as_bool(generation.get("flex_attn"), True):
        args.append("--no-flex-attn")
    if _as_bool(generation.get("ttc"), False):
        args.append("--ttc")

    if skill_prompt:
        skill_seconds = float(
            generation.get(
                "skill_seconds",
                int(generation.get("temporal_stride", 8))
                * int(generation.get("chunk_latent_frames", 4))
                / float(generation.get("fps", 24.0)),
            )
        )
        args.extend(["--skill-sec", str(skill_seconds), "--skill-prompt", skill_prompt])
        if _as_bool(generation.get("skill_keep_wrap"), False):
            args.append("--skill-keep-wrap")
    else:
        args.extend(["--skill-sec", "0"])

    nproc_per_node = max(int(generation.get("nproc_per_node", 1)), 1)
    if nproc_per_node == 1:
        return [sys.executable, *args]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc_per_node}",
    ]
    if generation.get("master_port") is not None:
        command.append(f"--master-port={int(generation['master_port'])}")
    command.extend(args)
    return command


def _runtime_env(generation: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(PROJECT_ROOT)
    )
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["WORLDARENA_ALAYAWORLD_ATTENTION_TYPE"] = str(
        generation.get("attention_type", "xformers")
    )
    if _as_bool(generation.get("offline"), True):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _empty_cuda_cache() -> None:
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_persistent_runtime(
    *,
    repo_root: Path,
    config_path: Path,
    generation: dict[str, Any],
) -> PersistentRuntime:
    """Load AlayaWorld exactly once for every WorldArena shard process."""
    if max(int(generation.get("nproc_per_node", 1)), 1) != 1:
        raise ValueError(
            "persistent AlayaWorld batching currently requires nproc_per_node=1; "
            "use one resident process on each GPU"
        )

    os.environ.update(_runtime_env(generation))
    repo_value = str(repo_root)
    if repo_value not in sys.path:
        sys.path.insert(0, repo_value)

    from worldarena.models.adapters.alayaworld_upstream_runner import (
        _install_attention_backend_override,
    )

    _install_attention_backend_override(str(generation.get("attention_type", "xformers")))

    from flash_alaya.alaya.config.loader import load_config
    from flash_alaya.utils.pipeline import FlashAlayaPipeline
    from flash_alaya.utils.rollout_utils import (
        apply_joystick_overlay,
        build_engine,
        check_input_resolution,
        load_input_sample,
        plan_rollout,
    )
    from inference.da3_patch import apply_da3_robust_scale

    cfg = load_config(str(config_path))
    cfg.validation.save_joystick = _as_bool(generation.get("joystick"), False)
    compile_mode = str(generation.get("compile", "none"))
    flex_attn = _as_bool(generation.get("flex_attn"), False) and compile_mode != "none"
    engine = build_engine(
        cfg,
        compile_mode=compile_mode,
        compile_aux=False,
        bank_taehv=False,
        verbose=True,
    )
    if apply_da3_robust_scale():
        print(
            "[AlayaWorld] DA3 robust scale-only fallback enabled "
            "(colinear-safe depth scaling)",
            flush=True,
        )

    mode_cfg = next(iter(cfg.validation.modes.values()))
    output_latent_frames = int(mode_cfg.layout.output_latent_frames)
    history_latent_frames = int(
        cfg.layout.history_latent_frames
        if mode_cfg.layout.history_latent_frames is None
        else mode_cfg.layout.history_latent_frames
    )
    gap_steps = int(
        float(mode_cfg.layout.max_gap_sec or 0.0)
        * cfg.sample.fps
        / cfg.sample.temporal_stride
    )
    condition_latent_frames = int(mode_cfg.layout.condition_latent_frames)
    print(
        "[WorldArena:AlayaWorld] persistent engine ready; checkpoint_loads=1",
        flush=True,
    )
    return PersistentRuntime(
        cfg=cfg,
        engine=engine,
        pipeline_type=FlashAlayaPipeline,
        load_input_sample=load_input_sample,
        check_input_resolution=check_input_resolution,
        plan_rollout=plan_rollout,
        apply_joystick_overlay=apply_joystick_overlay,
        output_latent_frames=output_latent_frames,
        history_latent_frames=history_latent_frames,
        gap_steps=gap_steps,
        condition_latent_frames=condition_latent_frames,
        flex_attn=flex_attn,
    )


def _release_rollout_only_state(cache: Any) -> None:
    """Free per-sample conditioning while retaining the resident engine."""
    import torch

    cache.spatial_bank = None
    cache.history = None
    cache.history_action_t_indices = None
    cache.explicit_nearby = None
    cache.context = None
    cache.negative_context = None
    cache.sink_latent = None
    cache.sink_indices = None
    cache.video_pixels = torch.empty(0)
    _empty_cuda_cache()


def _run_persistent_sample(
    *,
    runtime: PersistentRuntime,
    prefix: Path,
    requested_output: Path,
    requested_rounds: int,
    seed: int,
    skill_prompt: str | None,
    generation: dict[str, Any],
) -> tuple[int, int]:
    """Run one case without releasing or reconstructing model weights."""
    import torch

    cfg = runtime.cfg
    engine = runtime.engine
    video_pixels, caption, metadata = runtime.load_input_sample(
        str(prefix),
        image_target_hw=(int(cfg.sample.height), int(cfg.sample.width)),
    )
    runtime.check_input_resolution(video_pixels, cfg)
    video_pixels, metadata, rounds, max_rounds, needed_latents = runtime.plan_rollout(
        cfg,
        video_pixels,
        metadata,
        rounds_cap=requested_rounds,
        K=runtime.output_latent_frames,
        N=runtime.history_latent_frames,
        gap_steps=runtime.gap_steps,
        cond_end=runtime.condition_latent_frames,
    )
    if rounds != requested_rounds:
        raise RuntimeError(
            f"camera trajectory only supports {max_rounds} chunks, "
            f"but {requested_rounds} were requested"
        )

    pipe = runtime.pipeline_type(
        engine,
        control_modes=list(next(iter(cfg.validation.modes.values())).control),
        use_memory=bool(next(iter(cfg.validation.modes.values())).use_memory),
        action_cfg_scale=float(next(iter(cfg.validation.modes.values())).action_cfg_scale),
        flex_attn=runtime.flex_attn,
        seed=seed,
        ttc=_as_bool(generation.get("ttc"), False),
        ttc_levels=tuple(int(value) for value in cfg.validation.ttc.levels),
        ttc_strength=float(cfg.validation.ttc.strength),
        ttc_ref_action=bool(cfg.validation.ttc.ref_action),
    )
    cache = pipe.initialize_cache(
        video_pixels,
        caption,
        metadata,
        rounds=rounds,
        K=runtime.output_latent_frames,
        cond_end=runtime.condition_latent_frames,
        needed_latents=needed_latents,
    )

    skill_context = None
    skill_start = rounds
    skill_seconds = float(generation.get("skill_seconds", 1.3333333333))
    if skill_prompt and skill_seconds > 0:
        frames_per_chunk = runtime.output_latent_frames * int(cfg.sample.temporal_stride)
        skill_chunks = min(
            rounds,
            max(1, math.ceil(skill_seconds * float(cfg.sample.fps) / frames_per_chunk)),
        )
        skill_start = rounds - skill_chunks
        skill_context = engine.encode_caption(skill_prompt)

    for chunk_index in range(rounds):
        if skill_context is not None and chunk_index == skill_start:
            cache.context = skill_context
            if not _as_bool(generation.get("skill_keep_wrap"), False):
                cache.spatial_bank = None
        started = time.perf_counter()
        pred = pipe.generate(chunk_index, cache)
        generated_at = time.perf_counter()
        pipe.finalize(chunk_index, cache, pred)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"[AlayaWorld:persistent] chunk {chunk_index + 1}/{rounds}: "
            f"generate={generated_at - started:.2f}s "
            f"finalize={time.perf_counter() - generated_at:.2f}s",
            flush=True,
        )

    # The official single-case CLI destroys DiT/Gemma/DA3 before VAE decode.
    # On an 80 GB A100, releasing only the completed rollout's spatial/history
    # state leaves enough room for the tiled decoder while preserving all weights.
    _release_rollout_only_state(cache)
    decode_started = time.perf_counter()
    frames = pipe.decode(cache)
    if cfg.validation.save_joystick:
        frames = runtime.apply_joystick_overlay(cfg, cache, frames)

    requested_output.parent.mkdir(parents=True, exist_ok=True)
    staged_output = requested_output.with_name(
        f".{requested_output.stem}.{os.getpid()}.tmp{requested_output.suffix}"
    )
    staged_output.unlink(missing_ok=True)
    try:
        engine.write_video(
            staged_output,
            frames,
            crf=int(generation.get("video_crf", 28)),
        )
        if not staged_output.is_file() or staged_output.stat().st_size <= 0:
            raise RuntimeError(f"persistent AlayaWorld output was not written: {staged_output}")
        staged_output.replace(requested_output)
    except Exception:
        staged_output.unlink(missing_ok=True)
        raise
    frame_count = int(frames.shape[0])
    print(
        f"[AlayaWorld:persistent] decode {time.perf_counter() - decode_started:.1f}s "
        f"frames={tuple(frames.shape)} -> saved {requested_output}",
        flush=True,
    )
    del frames, cache, pipe, video_pixels, metadata
    _empty_cuda_cache()
    return rounds, frame_count


def _run_persistent_rows(
    *,
    rows: list[dict[str, Any]],
    repo_root: Path,
    config_path: Path,
    case_root: Path,
    generation: dict[str, Any],
) -> int:
    old_cwd = Path.cwd()
    failed = 0
    continue_on_error = _as_bool(generation.get("continue_on_error"), True)
    increment_seed = _as_bool(generation.get("increment_seed"), True)
    try:
        os.chdir(repo_root)
        runtime = _load_persistent_runtime(
            repo_root=repo_root,
            config_path=config_path,
            generation=generation,
        )
        print(
            f"[WorldArena:AlayaWorld] persistent batch requests={len(rows)}",
            flush=True,
        )
        for row_index, row in enumerate(rows):
            sample_id = str(row.get("sample_id", row_index))
            requested_output = Path(str(row["output_path"])).expanduser().resolve()
            if requested_output.is_file() and requested_output.stat().st_size > 0:
                print_status(
                    sample_id,
                    "skipped_existing",
                    prediction_path=str(requested_output),
                    reason="output appeared after batch planning",
                )
                continue
            prefix: Path | None = None
            try:
                prefix, metadata, skill_prompt = _materialize_case(
                    row=row,
                    row_index=row_index,
                    case_root=case_root,
                    generation=generation,
                )
                round_actions = list(row.get("round_actions") or ["fixed"])
                base_seed = int(row.get("seed", generation.get("seed", 1234)))
                seed = base_seed + row_index if increment_seed else base_seed
                print(
                    f"[WorldArena:AlayaWorld] persistent sample "
                    f"{row_index + 1}/{len(rows)} id={sample_id} seed={seed}",
                    flush=True,
                )
                rounds, frame_count = _run_persistent_sample(
                    runtime=runtime,
                    prefix=prefix,
                    requested_output=requested_output,
                    requested_rounds=len(round_actions),
                    seed=seed,
                    skill_prompt=skill_prompt,
                    generation=generation,
                )
                print_status(
                    sample_id,
                    "generated",
                    prediction_path=str(requested_output),
                    round_count=rounds,
                    round_actions=round_actions,
                    generated_frame_count=frame_count,
                    trajectory_frame_count=int(metadata["cam_c2w"].shape[0]),
                    seed=seed,
                    prompt_switching=skill_prompt is not None,
                    persistent_engine=True,
                    checkpoint_load_count=1,
                )
                if not _as_bool(generation.get("keep_work_files"), False):
                    shutil.rmtree(prefix.parent)
            except Exception as exc:
                failed += 1
                fatal_cuda_oom = (
                    type(exc).__name__ == "OutOfMemoryError"
                    or "CUDA out of memory" in str(exc)
                )
                print_status(
                    sample_id,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                    persistent_engine=True,
                )
                if fatal_cuda_oom or not continue_on_error:
                    raise
            finally:
                _empty_cuda_cache()
        return 0 if failed < len(rows) else 1
    finally:
        os.chdir(old_cwd)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not args.checkpoint_dir:
        raise ValueError("AlayaWorld runner requires --checkpoint_dir")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()
    generation_path = Path(args.generation_config_path).expanduser().resolve()
    rows = load_batch_spec(batch_spec_path)
    generation = load_json(generation_path)
    paths = _runtime_paths(checkpoint_dir=checkpoint_dir, generation=generation)
    _validate_runtime(repo_root=repo_root, checkpoint_dir=checkpoint_dir, paths=paths)

    output_parents = {
        Path(str(row["output_path"])).expanduser().resolve().parent for row in rows
    }
    if len(output_parents) != 1:
        raise ValueError("AlayaWorld batch runner requires one output directory")
    output_parent = next(iter(output_parents))
    work_root = output_parent / "_alayaworld_work" / batch_spec_path.stem
    case_root = work_root / "cases"
    official_root = work_root / "official_outputs"
    config_path = _patch_inference_config(
        repo_root=repo_root,
        paths=paths,
        generation=generation,
        output_path=work_root / "infer.worldarena.yaml",
    )

    if os.environ.get("WORLDARENA_ALAYAWORLD_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        row = rows[0]
        prefix, metadata, skill_prompt = _materialize_case(
            row=row,
            row_index=0,
            case_root=case_root,
            generation=generation,
        )
        rounds = len(row.get("round_actions") or ["fixed"])
        command = _official_command(
            prefix=prefix,
            config_path=config_path,
            output_dir=official_root / "000000",
            rounds=rounds,
            seed=int(row.get("seed", generation.get("seed", 1234))),
            skill_prompt=skill_prompt,
            generation=generation,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "request_count": len(rows),
                    "repo_root": str(repo_root),
                    "model_file": str(paths.model_file),
                    "gemma_dir": str(paths.gemma_dir),
                    "da3_repo": str(paths.da3_repo),
                    "da3_cache": str(paths.da3_cache),
                    "round_actions": list(row.get("round_actions") or ["fixed"]),
                    "trajectory_frame_count": int(metadata["cam_c2w"].shape[0]),
                    "persistent_engine": _as_bool(generation.get("persistent"), True),
                    "checkpoint_load_count": 1,
                    "command": command,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    _validate_python_runtime(generation)

    if _as_bool(generation.get("persistent"), True):
        return _run_persistent_rows(
            rows=rows,
            repo_root=repo_root,
            config_path=config_path,
            case_root=case_root,
            generation=generation,
        )

    failed = 0
    continue_on_error = _as_bool(generation.get("continue_on_error"), True)
    increment_seed = _as_bool(generation.get("increment_seed"), True)
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        requested_output = Path(str(row["output_path"])).expanduser().resolve()
        if requested_output.is_file() and requested_output.stat().st_size > 0:
            print_status(
                sample_id,
                "skipped_existing",
                prediction_path=str(requested_output),
                reason="output appeared after batch planning",
            )
            continue
        try:
            prefix, metadata, skill_prompt = _materialize_case(
                row=row,
                row_index=row_index,
                case_root=case_root,
                generation=generation,
            )
            round_actions = list(row.get("round_actions") or ["fixed"])
            base_seed = int(row.get("seed", generation.get("seed", 1234)))
            seed = base_seed + row_index if increment_seed else base_seed
            row_output_dir = official_root / f"{row_index:06d}"
            row_output_dir.mkdir(parents=True, exist_ok=True)
            command = _official_command(
                prefix=prefix,
                config_path=config_path,
                output_dir=row_output_dir,
                rounds=len(round_actions),
                seed=seed,
                skill_prompt=skill_prompt,
                generation=generation,
            )
            subprocess.run(
                command,
                check=True,
                cwd=str(repo_root),
                env=_runtime_env(generation),
            )
            generated_path = latest_mp4(row_output_dir)
            copy_output(generated_path, requested_output)
            print_status(
                sample_id,
                "generated",
                prediction_path=str(requested_output),
                official_output_path=str(generated_path),
                round_count=len(round_actions),
                round_actions=round_actions,
                trajectory_frame_count=int(metadata["cam_c2w"].shape[0]),
                seed=seed,
                prompt_switching=skill_prompt is not None,
            )
        except Exception as exc:
            failed += 1
            print_status(
                sample_id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            if not continue_on_error:
                raise
    return 0 if failed < len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
