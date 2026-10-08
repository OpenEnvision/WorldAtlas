"""WorldAtlas batch runner for the official Alaya-EVOKE inference launchers.

Materializes WorldAtlas samples as Evoke jsonl cases (image + vipe pose + prompt)
and drives ``scripts/inference/infer_batch.py`` with the post-distill recipe.
The official driver already loads the pipeline once per shard
(``IN_PROCESS_BATCH=1``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any

import numpy as np

from worldatlas.benchmark.synthetic_camera import synthetic_camera_matrices
from worldatlas.common.checkpoints import project_root as worldatlas_project_root
from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    begin_sample,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)
from worldatlas.models.adapters.evoke.evoke import (
    LATENT_FRAMES_PER_CHUNK,
    PIXEL_FRAMES_PER_CHUNK,
    resolve_num_chunks,
)


PROJECT_ROOT = worldatlas_project_root()
UPSTREAM_REQUIRED_FILES = (
    "scripts/inference/infer_batch.py",
    "scripts/inference/infer_single.py",
    "evoke/pipelines/pipeline_evoke.py",
)


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    base_ckpt: Path
    transformer_path: Path
    vigeo_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Alaya-EVOKE batch runner")
    add_common_batch_args(parser)
    return parser.parse_args()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


_JOYSTICK_HUD_CHOICES = ("auto", "on", "off", "both")
_JOYSTICK_HUD_ALIASES = {
    "1": "on",
    "true": "on",
    "yes": "on",
    "on": "on",
    "0": "off",
    "false": "off",
    "no": "off",
    "off": "off",
    "auto": "auto",
    "both": "both",
}


def _joystick_hud_choice(value: Any, default: str = "off") -> str:
    """Map config values onto infer_single.py --joystick_hud choices.

    YAML 1.1 treats ``off`` / ``on`` as booleans, so ``joystick_hud: off`` in
    evoke.yaml arrives here as ``False``. ``str(False)`` is ``'False'``, which
    the official CLI rejects.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if not text:
        return default
    choice = _JOYSTICK_HUD_ALIASES.get(text)
    if choice not in _JOYSTICK_HUD_CHOICES:
        raise ValueError(
            f"joystick_hud must be one of {_JOYSTICK_HUD_CHOICES}, got {value!r}"
        )
    return choice


def _resolve_path(value: Any, *, bases: list[Path]) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()


def _runtime_paths(*, checkpoint_dir: Path, generation: dict[str, Any]) -> RuntimePaths:
    base_value = generation.get("base_ckpt") or generation.get("base_ckpt_relpath", "evoke-base")
    transformer_value = generation.get("transformer_path") or generation.get(
        "transformer_relpath", "evoke/stage3_post_distillation"
    )
    vigeo_value = generation.get("vigeo_weights") or generation.get("vigeo_relpath", "ViGeo1.1")
    return RuntimePaths(
        base_ckpt=_resolve_path(base_value, bases=[checkpoint_dir, PROJECT_ROOT]),
        transformer_path=_resolve_path(transformer_value, bases=[checkpoint_dir, PROJECT_ROOT]),
        vigeo_dir=_resolve_path(vigeo_value, bases=[checkpoint_dir, PROJECT_ROOT]),
    )


def _validate_runtime(*, repo_root: Path, checkpoint_dir: Path, paths: RuntimePaths) -> None:
    missing_repo = [name for name in UPSTREAM_REQUIRED_FILES if not (repo_root / name).is_file()]
    if missing_repo:
        raise FileNotFoundError(
            f"EVOKE repo is incomplete at {repo_root}; missing: {', '.join(missing_repo)}"
        )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"EVOKE checkpoint directory not found: {checkpoint_dir}")
    if not paths.base_ckpt.is_dir():
        raise FileNotFoundError(f"EVOKE base components not found: {paths.base_ckpt}")
    if not (paths.transformer_path / "transformer").is_dir():
        raise FileNotFoundError(
            f"EVOKE transformer not found under {paths.transformer_path}/transformer"
        )
    vigeo_file = paths.vigeo_dir / "vigeo.pt"
    if not vigeo_file.is_file():
        raise FileNotFoundError(f"ViGeo weights not found: {vigeo_file}")


def _runtime_env(generation: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(PROJECT_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if _as_bool(generation.get("offline"), True):
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def _trajectory_runtime(generation: dict[str, Any]) -> dict[str, float]:
    return {
        "synthetic_camera_forward_step": float(generation.get("forward_distance_per_chunk", 1.2)),
        "synthetic_camera_lateral_step": float(generation.get("lateral_distance_per_chunk", 0.4)),
        "synthetic_camera_vertical_step": float(generation.get("vertical_distance_per_chunk", 0.35)),
        "synthetic_camera_rotation_degrees": float(
            generation.get("rotation_degrees_per_chunk", 8.0)
        ),
        "synthetic_camera_tilt_degrees": float(generation.get("tilt_degrees_per_chunk", 6.0)),
        "synthetic_camera_roll_degrees": float(generation.get("roll_degrees_per_chunk", 5.0)),
        "synthetic_camera_orbit_radius": float(generation.get("orbit_radius", 2.0)),
    }


def _pixel_intrinsics(generation: dict[str, Any]) -> np.ndarray:
    height = max(int(generation.get("height", 384)), 1)
    width = max(int(generation.get("width", 640)), 1)
    focal_scale = float(generation.get("intrinsic_focal_scale", 0.7))
    focal = float(max(width, height)) * focal_scale
    return np.asarray([focal, focal, width / 2.0, height / 2.0], dtype=np.float32)


def build_pose_npz(
    chunk_actions: list[str],
    generation: dict[str, Any],
    *,
    trajectory_actions: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build a vipe-style pose archive: ``cam_c2w [T,4,4]`` + ``intrinsics [4]``."""
    actions = [str(value) for value in chunk_actions] or ["fixed"]
    frame_count = max(len(actions) * PIXEL_FRAMES_PER_CHUNK, 1)
    motion = synthetic_camera_matrices(
        trajectory_actions or actions,
        target_frames=frame_count,
        runtime=_trajectory_runtime(generation),
    )
    cam_c2w = np.asarray(motion.matrices, dtype=np.float32)
    fx, fy, cx, cy = _pixel_intrinsics(generation)
    # Official load_pose_for_v2v reads `intrinsics` before `K` and treats the
    # value as a 3x3 matrix (or [N,3,3]). A 4-vector [fx,fy,cx,cy] is the
    # *output* of that loader, not the on-disk vipe layout.
    k_matrix = np.asarray(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return {
        "cam_c2w": cam_c2w,
        "intrinsics": k_matrix,
        "K": k_matrix,
    }


def _safe_stem(value: Any, fallback: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return stem or fallback


def _materialize_case(
    *,
    row: dict[str, Any],
    row_index: int,
    case_root: Path,
    generation: dict[str, Any],
) -> dict[str, Any]:
    source = Path(str(row["conditioning_image"])).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"EVOKE conditioning image not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise ValueError(f"unsupported EVOKE conditioning image extension: {suffix}")

    stem = _safe_stem(row.get("prediction_stem") or row.get("sample_id"), f"sample_{row_index:06d}")
    row_root = case_root / f"{row_index:06d}_{stem}"
    row_root.mkdir(parents=True, exist_ok=True)
    image_path = row_root / f"{stem}_image{suffix}"
    shutil.copy2(source, image_path)

    chunk_actions = [str(value) for value in row.get("chunk_actions") or ["fixed"]]
    pose_payload = build_pose_npz(
        chunk_actions, generation, trajectory_actions=row.get("trajectory_actions"),
    )
    pose_path = row_root / f"{stem}_pose.npz"
    np.savez(pose_path, **pose_payload)

    prompt = str(row.get("prompt") or "").strip()
    prompt_path = row_root / f"{stem}_prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    schedule = list(row.get("segment_prompts") or [])
    schedule_path = None
    if schedule:
        schedule_path = row_root / f"{stem}_schedule.json"
        schedule_path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8")

    height = int(generation.get("height", 384))
    width = int(generation.get("width", 640))
    record: dict[str, Any] = {
        "name": stem,
        "image_path": str(image_path),
        "pose_path": str(pose_path),
        "prompt": prompt,
        "prompt_path": str(prompt_path),
        "pose_fps": int(generation.get("fps", 24)),
        "pose_source_resolution": [height, width],
        "seed": int(row.get("seed", generation.get("seed", 44))),
    }
    if schedule:
        record["segment_prompts"] = schedule
        record["segment_prompts_path"] = str(schedule_path)
    if str(row.get("sample_type", "i2v")).lower() == "v2v":
        video = Path(str(row.get("reference_video") or "")).expanduser()
        if not video.is_file():
            raise FileNotFoundError(f"EVOKE v2v reference video not found: {video}")
        record["video_path"] = str(video.resolve())
        record["video_fps"] = float(generation.get("fps", 24))
        caption_path = row_root / f"{stem}_caption.json"
        caption_path.write_text(
            json.dumps({"overall": {"full_prompt": prompt}}, ensure_ascii=False),
            encoding="utf-8",
        )
        record["prompt_path"] = str(caption_path)
    return {
        "stem": stem,
        "row_root": row_root,
        "record": record,
        "pose": pose_payload,
        "chunk_actions": chunk_actions,
        "schedule": schedule,
    }


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    sample_type = str(row.get("sample_type", "i2v")).strip().lower()
    warp = "on" if _as_bool(row.get("warp_enabled"), True) else "off"
    return sample_type, warp


def _infer_env(
    *,
    generation: dict[str, Any],
    paths: RuntimePaths,
    jsonl_path: Path,
    out_root: Path,
    sample_type: str,
    warp: str,
    num_chunks: int,
) -> dict[str, str]:
    env = _runtime_env(generation)
    steps = generation.get("stage2_steps", [1, 1, 1])
    if isinstance(steps, (list, tuple)):
        steps_text = " ".join(str(int(value)) for value in steps)
    else:
        steps_text = str(steps)
    env.update(
        {
            "MODE": sample_type,
            "JSONL": str(jsonl_path),
            "OUT_ROOT": str(out_root),
            "TRANSFORMER_PATH": str(paths.transformer_path),
            "BASE_CKPT": str(paths.base_ckpt),
            "VIGEO_WEIGHTS": str(paths.vigeo_dir),
            "MAX_CASES": "0",
            "NUM_CHUNKS": str(int(num_chunks)),
            "NUM_FRAMES": str(int(num_chunks) * LATENT_FRAMES_PER_CHUNK),
            "HEIGHT": str(int(generation.get("height", 384))),
            "WIDTH": str(int(generation.get("width", 640))),
            "FPS": str(int(generation.get("fps", 24))),
            "GUIDANCE_SCALE": str(generation.get("guidance_scale", 1.0)),
            "SEED": str(int(generation.get("seed", 44))),
            "VAE_DECODE_TYPE": str(generation.get("vae_decode_type", "persistent")),
            "IS_STAGE2": "1" if _as_bool(generation.get("is_stage2"), True) else "0",
            "STAGE2_NUM_STAGES": str(int(generation.get("stage2_num_stages", 3))),
            "STAGE2_STEPS": steps_text,
            "NUM_INFERENCE_STEPS": str(int(generation.get("num_inference_steps", 3))),
            "RESTRICT": "1" if _as_bool(generation.get("restrict_self_attn"), False) else "0",
            "GEO_WARP_STAGE0_ONLY": (
                "1" if _as_bool(generation.get("geo_warp_stage0_only"), True) else "0"
            ),
            "WARP_SIGMA_MAX": str(generation.get("warp_sigma_max", 0.135)),
            "NOISE_CENTER": "1" if _as_bool(generation.get("noise_center"), False) else "0",
            "RENDER_MODE": str(generation.get("render_mode", "backward_zbuf")),
            "BW_FILL_ITERS": str(int(generation.get("bw_fill_iters", 12))),
            "DEPTH_BACKEND": str(generation.get("depth_backend", "vigeo")),
            "VIGEO_DEPTH_MEDIAN_TARGET": str(
                generation.get("vigeo_depth_median_target", 5)
            ),
            "ZBUF_DESPECKLE": "1" if _as_bool(generation.get("zbuf_despeckle"), False) else "0",
            "WARP_MODE": str(generation.get("warp_mode", "fixed_mem")),
            "WARP": warp,
            "IN_PROCESS_BATCH": "1" if _as_bool(generation.get("in_process_batch"), True) else "0",
            "DUMP_GEO": "1" if _as_bool(generation.get("dump_geo"), False) else "0",
            "SAVE_SEGMENTS": "1" if _as_bool(generation.get("save_segments"), False) else "0",
            "JOYSTICK_HUD": _joystick_hud_choice(generation.get("joystick_hud"), "off"),
            "REF_VIDEO_SEC": str(generation.get("ref_video_sec", 0.0 if sample_type != "v2v" else 2.0)),
            "START_SECONDS": str(generation.get("start_seconds", 0.0)),
            "SHARD": "0",
            "NSHARD": "1",
            "INFER_VERBOSE": "1",
            "QUIET": "0",
        }
    )
    return env


def _official_command(repo_root: Path) -> list[str]:
    return [sys.executable, str(repo_root / "scripts" / "inference" / "infer_batch.py")]


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not args.checkpoint_dir:
        raise ValueError("EVOKE runner requires --checkpoint_dir")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()
    generation_path = Path(args.generation_config_path).expanduser().resolve()
    rows = load_batch_spec(batch_spec_path)
    generation = resolve_num_chunks(load_json(generation_path))
    paths = _runtime_paths(checkpoint_dir=checkpoint_dir, generation=generation)
    _validate_runtime(repo_root=repo_root, checkpoint_dir=checkpoint_dir, paths=paths)

    output_parents = {
        Path(str(row["output_path"])).expanduser().resolve().parent for row in rows
    }
    if len(output_parents) != 1:
        raise ValueError("EVOKE batch runner requires one output directory")
    output_parent = next(iter(output_parents))
    work_root = output_parent / "_evoke_work" / batch_spec_path.stem
    case_root = work_root / "cases"
    official_root = work_root / "official_outputs"
    case_root.mkdir(parents=True, exist_ok=True)
    official_root.mkdir(parents=True, exist_ok=True)

    dry_run = os.environ.get("WORLDATLAS_EVOKE_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    increment_seed = _as_bool(generation.get("increment_seed"), True)
    materialized: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        seed = int(row.get("seed", generation.get("seed", 44)))
        if increment_seed:
            seed = seed + row_index
        row_with_seed = dict(row)
        row_with_seed["seed"] = seed
        materialized.append(
            {
                "row": row_with_seed,
                "case": _materialize_case(
                    row=row_with_seed,
                    row_index=row_index,
                    case_root=case_root,
                    generation=generation,
                ),
            }
        )

    if dry_run:
        first = materialized[0]
        sample_type, warp = _group_key(first["row"])
        jsonl_path = work_root / "cases.jsonl"
        jsonl_path.write_text(
            json.dumps(first["case"]["record"], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        env = _infer_env(
            generation=generation,
            paths=paths,
            jsonl_path=jsonl_path,
            out_root=official_root / f"{sample_type}_{warp}",
            sample_type=sample_type,
            warp=warp,
            num_chunks=int(first["row"].get("num_chunks", generation["num_chunks"])),
        )
        command = _official_command(repo_root)
        pose = first["case"]["pose"]
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "request_count": len(rows),
                    "repo_root": str(repo_root),
                    "base_ckpt": str(paths.base_ckpt),
                    "transformer_path": str(paths.transformer_path),
                    "vigeo_dir": str(paths.vigeo_dir),
                    "sample_type": sample_type,
                    "warp": warp,
                    "chunk_actions": first["case"]["chunk_actions"],
                    "trajectory_frame_count": int(pose["cam_c2w"].shape[0]),
                    "persistent_engine": _as_bool(generation.get("persistent"), True),
                    "checkpoint_load_count": 1,
                    "command": command,
                    "env": {
                        key: env[key]
                        for key in (
                            "MODE",
                            "TRANSFORMER_PATH",
                            "BASE_CKPT",
                            "NUM_CHUNKS",
                            "WARP",
                            "IN_PROCESS_BATCH",
                            "JOYSTICK_HUD",
                        )
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in materialized:
        groups.setdefault(_group_key(item["row"]), []).append(item)

    failed = 0
    continue_on_error = _as_bool(generation.get("continue_on_error"), True)
    for (sample_type, warp), items in groups.items():
        jsonl_path = work_root / f"cases_{sample_type}_{warp}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item["case"]["record"], ensure_ascii=False) + "\n")
        out_root = official_root / f"{sample_type}_{warp}"
        out_root.mkdir(parents=True, exist_ok=True)
        num_chunks = max(int(item["row"].get("num_chunks", generation["num_chunks"])) for item in items)
        env = _infer_env(
            generation=generation,
            paths=paths,
            jsonl_path=jsonl_path,
            out_root=out_root,
            sample_type=sample_type,
            warp=warp,
            num_chunks=num_chunks,
        )
        command = _official_command(repo_root)
        print(
            f"[WorldAtlas:EVOKE] infer_batch mode={sample_type} warp={warp} "
            f"cases={len(items)} chunks={num_chunks}",
            flush=True,
        )
        try:
            subprocess.run(
                command,
                check=True,
                cwd=str(repo_root),
                env=env,
            )
        except Exception:
            if not continue_on_error:
                raise
            for item in items:
                failed += 1
                print_status(
                    str(item["row"].get("sample_id")),
                    "failed",
                    error="infer_batch failed",
                    traceback=traceback.format_exc(),
                )
            continue

        for item in items:
            row = item["row"]
            sample_id = str(row.get("sample_id"))
            begin_sample(sample_id, index=1, total=len(rows), label="Alaya-EVOKE")
            requested_output = Path(str(row["output_path"])).expanduser().resolve()
            generated = out_root / item["case"]["stem"] / "geo_pred.mp4"
            try:
                copy_output(generated, requested_output)
                pose = item["case"]["pose"]
                print_status(
                    sample_id,
                    "generated",
                    prediction_path=str(requested_output),
                    official_output_path=str(generated),
                    sample_type=sample_type,
                    warp=warp,
                    chunk_actions=item["case"]["chunk_actions"],
                    trajectory_frame_count=int(pose["cam_c2w"].shape[0]),
                    persistent_engine=True,
                    checkpoint_load_count=1,
                )
                if not _as_bool(generation.get("keep_work_files"), False):
                    shutil.rmtree(item["case"]["row_root"], ignore_errors=True)
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
