from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    print_status,
)


SUPPORTED_CAMERA_ACTIONS = {
    "orbit_left",
    "orbit_right",
    "pan_left",
    "pan_right",
    "push_in",
    "pull_out",
    "move_left",
    "move_right",
    "fixed",
}
CONDITION_VERSION = "worldarena_voyager_condition_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena HunyuanWorld-Voyager batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return rank() % max(device_count, 1)


def device_for_rank() -> torch.device:
    if torch.cuda.is_available():
        device_index = local_rank()
        torch.cuda.set_device(device_index)
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def is_main_process() -> bool:
    return rank() == 0


def distributed_barrier() -> None:
    try:
        import torch.distributed as dist
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_for_row(generation: dict[str, Any], row_index: int, num_gpus: int) -> int | None:
    seed = generation.get("seed")
    if seed is not None:
        return int(seed) + row_index
    if num_gpus > 1:
        return row_index
    return None


def camera_actions_for_row(row: dict[str, Any]) -> list[str]:
    camera_path = row.get("camera_path") or []
    if isinstance(camera_path, str):
        camera_path = [camera_path]
    actions: list[str] = []
    for action in camera_path:
        action = str(action)
        if action in SUPPORTED_CAMERA_ACTIONS:
            actions.append(action)
    return actions or ["push_in"]


def camera_action_for_row(row: dict[str, Any]) -> str:
    return camera_actions_for_row(row)[0]


def _look_at_w2c(camera_center: np.ndarray, target_point: np.ndarray) -> np.ndarray:
    z_axis = target_point - camera_center
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y_axis = np.cross(z_axis, x_hint)
    if np.linalg.norm(y_axis) < 1e-6:
        x_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        y_axis = np.cross(z_axis, x_hint)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ camera_center
    return w2c


def _single_action_camera_list(
    *,
    num_frames: int,
    action: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
) -> tuple[np.ndarray, np.ndarray]:
    cx = width // 2
    cy = height // 2
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.stack([intrinsic] * num_frames)

    start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    end_pos = start_pos.copy()
    target_start = np.array([0.0, 0.0, 100.0], dtype=np.float32)
    target_end = target_start.copy()

    if action == "push_in":
        end_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    elif action == "pull_out":
        end_pos = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    elif action == "move_left":
        end_pos = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    elif action == "move_right":
        end_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif action == "pan_left":
        target_end = np.array([-100.0, 0.0, 0.0], dtype=np.float32)
    elif action == "pan_right":
        target_end = np.array([100.0, 0.0, 0.0], dtype=np.float32)

    if action in {"orbit_left", "orbit_right"}:
        sign = -1.0 if action == "orbit_left" else 1.0
        angles = np.linspace(0.0, sign * np.pi / 8.0, num_frames)
        radius = 2.5
        center = np.array([0.0, 0.0, radius], dtype=np.float32)
        camera_centers = np.stack(
            [
                radius * np.sin(angles),
                np.zeros_like(angles),
                center[2] - radius * np.cos(angles),
            ],
            axis=1,
        ).astype(np.float32)
        target_points = np.stack([center] * num_frames)
    else:
        camera_centers = np.linspace(start_pos, end_pos, num_frames).astype(np.float32)
        if action in {"pan_left", "pan_right"}:
            target_points = np.linspace(target_start, target_end, num_frames * 2)[:num_frames].astype(np.float32)
        else:
            target_points = np.linspace(target_start, target_end, num_frames).astype(np.float32)
        if action in {"move_left", "move_right"}:
            target_points = camera_centers + target_start

    extrinsics = np.stack(
        [_look_at_w2c(camera_center, target_point) for camera_center, target_point in zip(camera_centers, target_points)]
    )
    return intrinsics, extrinsics


def worldarena_camera_list(
    *,
    num_frames: int,
    action: str | list[str],
    width: int,
    height: int,
    fx: float,
    fy: float,
) -> tuple[np.ndarray, np.ndarray]:
    actions = [action] if isinstance(action, str) else [str(value) for value in action]
    actions = [value for value in actions if value in SUPPORTED_CAMERA_ACTIONS] or ["push_in"]
    if len(actions) == 1:
        return _single_action_camera_list(
            num_frames=num_frames,
            action=actions[0],
            width=width,
            height=height,
            fx=fx,
            fy=fy,
        )

    cx = width // 2
    cy = height // 2
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.stack([intrinsic] * num_frames)

    target_frames = max(int(num_frames), 1)
    if target_frames <= 1:
        return intrinsics, np.repeat(np.eye(4, dtype=np.float32)[None], repeats=target_frames, axis=0)

    remaining_intervals = target_frames - 1
    segment_count = len(actions)
    base_intervals = remaining_intervals // segment_count
    remainder = remaining_intervals % segment_count

    current_c2w = np.eye(4, dtype=np.float32)
    segments: list[np.ndarray] = []
    for index, segment_action in enumerate(actions):
        segment_intervals = base_intervals + (1 if index < remainder else 0)
        if segment_intervals <= 0:
            continue
        _, segment_w2c = _single_action_camera_list(
            num_frames=segment_intervals + 1,
            action=segment_action,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
        )
        relative_c2w = np.linalg.inv(segment_w2c).astype(np.float32)
        segment_c2w = np.matmul(current_c2w[None, ...], relative_c2w).astype(np.float32)
        segment_abs_w2c = np.linalg.inv(segment_c2w).astype(np.float32)
        if index:
            segment_abs_w2c = segment_abs_w2c[1:]
        segments.append(segment_abs_w2c)
        current_c2w = segment_c2w[-1]

    extrinsics = np.concatenate(segments, axis=0).astype(np.float32)
    if len(extrinsics) != target_frames:
        raise ValueError(f"expected {target_frames} camera poses, got {len(extrinsics)}")
    return intrinsics, extrinsics


def condition_dir_for(row: dict[str, Any], generation: dict[str, Any]) -> Path:
    output_path = Path(str(row["output_path"])).expanduser().resolve()
    root = generation.get("condition_root")
    if root:
        base = Path(str(root)).expanduser().resolve()
    else:
        base = output_path.parent / "_voyager_conditions"
    stem = str(row.get("prediction_stem") or row.get("sample_id") or output_path.stem)
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return base / safe_stem


def expected_condition_metadata(
    *,
    row: dict[str, Any],
    generation: dict[str, Any],
    image_path: Path,
    camera_actions: list[str],
    video_length: int,
) -> dict[str, Any]:
    return {
        "version": CONDITION_VERSION,
        "camera_path": list(camera_actions),
        "conditioning_image": str(image_path),
        "video_length": int(video_length),
        "moge_image_size": [int(value) for value in generation.get("moge_image_size", [720, 1280])],
        "moge_ref_fx": float(generation.get("moge_ref_fx", 256)),
        "moge_ref_fy": float(generation.get("moge_ref_fy", 256)),
        "moge_render_fx": float(generation.get("moge_render_fx", 128)),
        "moge_render_fy": float(generation.get("moge_render_fy", 128)),
        "sample_id": str(row.get("sample_id", "")),
    }


def condition_metadata_matches(condition_dir: Path, expected_metadata: dict[str, Any]) -> bool:
    metadata_path = condition_dir / "condition_meta.json"
    if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(actual.get(key) == value for key, value in expected_metadata.items())


def condition_files_ready(condition_dir: Path, video_length: int) -> bool:
    video_input = condition_dir / "video_input"
    required = [
        condition_dir / "ref_image.png",
        condition_dir / "ref_depth.exr",
    ]
    for frame_idx in range(video_length):
        required.extend(
            [
                video_input / f"render_{frame_idx:04d}.png",
                video_input / f"depth_{frame_idx:04d}.exr",
                video_input / f"mask_{frame_idx:04d}.png",
            ]
        )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def condition_is_ready(
    condition_dir: Path,
    video_length: int,
    *,
    expected_metadata: dict[str, Any],
    allow_legacy_single_action: bool,
) -> bool:
    if not condition_files_ready(condition_dir, video_length):
        return False
    metadata_path = condition_dir / "condition_meta.json"
    if metadata_path.exists():
        return condition_metadata_matches(condition_dir, expected_metadata)
    camera_path = expected_metadata.get("camera_path", [])
    legacy_safe_single = len(camera_path) == 1 and camera_path[0] not in {"pan_left", "pan_right"}
    return allow_legacy_single_action and legacy_safe_single


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def wait_for_json(path: Path, *, poll_seconds: float = 5.0) -> dict[str, Any]:
    while True:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(poll_seconds)


def load_moge_model(generation: dict[str, Any], device: torch.device):
    from moge.model.v1 import MoGeModel

    checkpoint = generation.get(
        "moge_checkpoint",
        "/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt/Ruicheng--moge-vitl/model.pt",
    )
    model = MoGeModel.from_pretrained(str(checkpoint), local_files_only=True).to(device)
    model.eval()
    return model


def prepare_condition(
    *,
    row: dict[str, Any],
    generation: dict[str, Any],
    moge_model,
    device: torch.device,
    repo_root: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root / "data_engine"))
    from create_input import (  # type: ignore
        create_video_input,
        depth_to_world_coords_points,
        render_from_cameras_videos,
    )

    video_length = int(generation.get("video_length", 129))
    condition_dir = condition_dir_for(row, generation)
    camera_actions = camera_actions_for_row(row)
    image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
    expected_metadata = expected_condition_metadata(
        row=row,
        generation=generation,
        image_path=image_path,
        camera_actions=camera_actions,
        video_length=video_length,
    )
    if condition_is_ready(
        condition_dir,
        video_length,
        expected_metadata=expected_metadata,
        allow_legacy_single_action=len(camera_actions) == 1,
    ) and not generation.get(
        "overwrite_conditions", False
    ):
        return {
            "status": "ready",
            "condition_dir": str(condition_dir),
            "camera_action": camera_actions[0],
            "camera_actions": camera_actions,
            "reused": True,
        }

    image_size = generation.get("moge_image_size", [720, 1280])
    height, width = int(image_size[0]), int(image_size[1])
    if not image_path.is_file():
        raise FileNotFoundError(f"conditioning image not found: {image_path}")

    condition_dir.mkdir(parents=True, exist_ok=True)
    image = np.array(Image.open(image_path).convert("RGB").resize((width, height)))
    image_tensor = torch.tensor(
        image / 255.0,
        dtype=torch.float32,
        device=device,
    ).permute(2, 0, 1)
    with torch.inference_mode():
        output = moge_model.infer(image_tensor)
    depth = np.array(output["depth"].detach().cpu())
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        raise ValueError(f"MoGE produced no finite depth values for {image_path}")
    depth[~finite_mask] = depth[finite_mask].max() + 1e4

    intrinsics, extrinsics = worldarena_camera_list(
        num_frames=1,
        action=camera_actions[0],
        width=width,
        height=height,
        fx=float(generation.get("moge_ref_fx", 256)),
        fy=float(generation.get("moge_ref_fy", 256)),
    )
    point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
    points = point_map.reshape(-1, 3)
    colors = image.reshape(-1, 3)

    intrinsics, extrinsics = worldarena_camera_list(
        num_frames=video_length,
        action=camera_actions,
        width=width // 2,
        height=height // 2,
        fx=float(generation.get("moge_render_fx", 128)),
        fy=float(generation.get("moge_render_fy", 128)),
    )
    render_list, mask_list, depth_list = render_from_cameras_videos(
        points,
        colors,
        extrinsics,
        intrinsics,
        height=height // 2,
        width=width // 2,
    )
    create_video_input(
        render_list,
        mask_list,
        depth_list,
        str(condition_dir),
        separate=True,
        ref_image=image,
        ref_depth=depth,
        Width=width,
        Height=height,
    )
    (condition_dir / "prompt.txt").write_text(str(row.get("prompt", "")), encoding="utf-8")
    write_json_atomic(condition_dir / "condition_meta.json", expected_metadata)
    del output, image_tensor
    torch.cuda.empty_cache()
    if not condition_is_ready(
        condition_dir,
        video_length,
        expected_metadata=expected_metadata,
        allow_legacy_single_action=False,
    ):
        raise FileNotFoundError(f"Voyager condition files were not fully written: {condition_dir}")
    return {
        "status": "ready",
        "condition_dir": str(condition_dir),
        "camera_action": camera_actions[0],
        "camera_actions": camera_actions,
        "reused": False,
    }


def prepare_all_conditions(
    *,
    rows: list[dict[str, Any]],
    generation: dict[str, Any],
    repo_root: Path,
    status_path: Path,
) -> dict[str, Any]:
    current_rank = rank()
    current_world_size = world_size()
    shard_path = status_path.with_name(f"{status_path.stem}.rank{current_rank:04d}{status_path.suffix}")

    device = device_for_rank()
    statuses: dict[str, dict[str, Any]] = {}
    moge_model = load_moge_model(generation, device)
    try:
        for row_index, row in enumerate(rows):
            if row_index % current_world_size != current_rank:
                continue
            sample_id = str(row.get("sample_id", row_index))
            try:
                status = prepare_condition(
                    row=row,
                    generation=generation,
                    moge_model=moge_model,
                    device=device,
                    repo_root=repo_root,
                )
                statuses[sample_id] = status
                log_status = {key: value for key, value in status.items() if key != "status"}
                print_status(
                    sample_id,
                    "condition_ready",
                    rank=current_rank,
                    world_size=current_world_size,
                    **log_status,
                )
            except Exception as exc:
                statuses[sample_id] = {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                print_status(sample_id, "condition_failed", rank=current_rank, error=str(exc))
    finally:
        del moge_model
        torch.cuda.empty_cache()

    write_json_atomic(shard_path, {"rank": current_rank, "world_size": current_world_size, "rows": statuses})
    if not is_main_process():
        return wait_for_json(status_path)

    merged: dict[str, dict[str, Any]] = {}
    for shard_rank in range(current_world_size):
        rank_path = status_path.with_name(f"{status_path.stem}.rank{shard_rank:04d}{status_path.suffix}")
        payload = wait_for_json(rank_path)
        merged.update(payload.get("rows", {}))
    payload = {"rows": merged}
    write_json_atomic(status_path, payload)
    return payload


def build_voyager_args(generation: dict[str, Any], checkpoint_dir: Path) -> Any:
    from voyager.config import parse_args as parse_voyager_args

    num_gpus = int(generation.get("num_gpus", 1))
    argv = [
        "hunyuanworld_voyager_batch",
        "--model",
        str(generation.get("model", "HYVideo-T/2")),
        "--model-base",
        str(checkpoint_dir),
        "--infer-steps",
        str(int(generation.get("infer_steps", 50))),
        "--video-length",
        str(int(generation.get("video_length", 129))),
        "--cfg-scale",
        str(float(generation.get("cfg_scale", 6.0))),
        "--flow-shift",
        str(float(generation.get("flow_shift", 7.0))),
        "--embedded-cfg-scale",
        str(float(generation.get("embedded_cfg_scale", 6.0))),
        "--i2v-resolution",
        str(generation.get("i2v_resolution", "720p")),
        "--i2v-condition-type",
        str(generation.get("i2v_condition_type", "latent_concat")),
    ]
    video_size = generation.get("video_size", [512, 768])
    argv.extend(["--video-size", *[str(int(value)) for value in video_size]])
    if generation.get("flow_reverse", True):
        argv.append("--flow-reverse")
    if generation.get("i2v_stability", True):
        argv.append("--i2v-stability")
    if generation.get("use_context_block", False):
        argv.append("--use-context-block")
    if generation.get("use_cpu_offload", False):
        argv.append("--use-cpu-offload")
    if generation.get("i2v_dit_weight"):
        argv.extend(["--i2v-dit-weight", str(generation["i2v_dit_weight"])])
    if generation.get("seed") is not None:
        argv.extend(["--seed", str(int(generation["seed"]))])
    if num_gpus > 1:
        ulysses_degree = int(generation.get("ulysses_degree", num_gpus))
        ring_degree = int(generation.get("ring_degree", 1))
        argv.extend(["--ulysses-degree", str(ulysses_degree), "--ring-degree", str(ring_degree)])

    old_argv = sys.argv
    sys.argv = argv
    try:
        return parse_voyager_args()
    finally:
        sys.argv = old_argv


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    checkpoint_dir = Path(args.checkpoint_dir or generation.get("model_base", "ckpts")).expanduser().resolve()
    os.environ["MODEL_BASE"] = str(checkpoint_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    spec_path = Path(args.batch_spec_path).expanduser().resolve()
    status_path = spec_path.with_suffix(".voyager_conditions.json")
    condition_payload = prepare_all_conditions(
        rows=rows,
        generation=generation,
        repo_root=repo_root,
        status_path=status_path,
    )
    condition_statuses = condition_payload.get("rows", {})
    valid_rows = [
        (row_index, row, condition_statuses.get(str(row.get("sample_id", row_index)), {}))
        for row_index, row in enumerate(rows)
        if condition_statuses.get(str(row.get("sample_id", row_index)), {}).get("status") == "ready"
    ]
    if not valid_rows:
        return 0

    from voyager.inference import HunyuanVideoSampler
    from voyager.utils.file_utils import save_videos_grid

    hv_args = build_voyager_args(generation, checkpoint_dir)
    sampler = HunyuanVideoSampler.from_pretrained(checkpoint_dir, args=hv_args)
    hv_args = sampler.args
    num_gpus = int(generation.get("num_gpus", 1))
    fps = int(generation.get("fps", 24))
    video_length = int(hv_args.video_length)
    failed = 0

    for row_index, row, condition_status in valid_rows:
        sample_id = str(row.get("sample_id", row_index))
        try:
            distributed_barrier()
            condition_dir = Path(str(condition_status["condition_dir"])).expanduser().resolve()
            seed = seed_for_row(generation, row_index, num_gpus)
            outputs = sampler.predict(
                prompt=str(row["prompt"]),
                height=hv_args.video_size[0],
                width=hv_args.video_size[1],
                video_length=video_length,
                seed=seed,
                negative_prompt=generation.get("neg_prompt"),
                infer_steps=hv_args.infer_steps,
                guidance_scale=hv_args.cfg_scale,
                num_videos_per_prompt=1,
                flow_shift=hv_args.flow_shift,
                batch_size=1,
                embedded_guidance_scale=hv_args.embedded_cfg_scale,
                i2v_mode=True,
                i2v_resolution=hv_args.i2v_resolution,
                i2v_image_path=str(condition_dir / "ref_image.png"),
                i2v_condition_type=hv_args.i2v_condition_type,
                i2v_stability=hv_args.i2v_stability,
                ulysses_degree=hv_args.ulysses_degree,
                ring_degree=hv_args.ring_degree,
                ref_images=[(str(condition_dir / "ref_image.png"), str(condition_dir / "ref_depth.exr"))],
                partial_cond=[
                    (
                        str(condition_dir / "video_input" / f"render_{frame_idx:04d}.png"),
                        str(condition_dir / "video_input" / f"depth_{frame_idx:04d}.exr"),
                    )
                    for frame_idx in range(video_length)
                ],
                partial_mask=[
                    (
                        str(condition_dir / "video_input" / f"mask_{frame_idx:04d}.png"),
                        str(condition_dir / "video_input" / f"mask_{frame_idx:04d}.png"),
                    )
                    for frame_idx in range(video_length)
                ],
            )
            if is_main_process():
                sample = outputs["samples"][0].unsqueeze(0)
                requested_height = int(hv_args.video_size[0])
                raw_height = int(sample.shape[-2])
                cropped_rgb = False
                if raw_height in (requested_height * 2, requested_height * 2 + 16):
                    sample = sample[..., :requested_height, :].contiguous()
                    cropped_rgb = True
                output_path = Path(str(row["output_path"])).expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_videos_grid(sample, str(output_path), fps=fps)
                print_status(
                    sample_id,
                    "generated",
                    output_path=str(output_path),
                    condition_dir=str(condition_dir),
                    camera_action=str(condition_status.get("camera_action", "")),
                    fps=fps,
                    video_length=video_length,
                    duration_seconds=round(video_length / max(fps, 1), 6),
                    output_height=int(sample.shape[-2]),
                    output_width=int(sample.shape[-1]),
                    cropped_rgb=cropped_rgb,
                    seed=outputs["seeds"][0],
                )
            distributed_barrier()
            torch.cuda.empty_cache()
        except Exception as exc:
            failed += 1
            if is_main_process():
                print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
            if num_gpus > 1:
                raise

    distributed_barrier()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
