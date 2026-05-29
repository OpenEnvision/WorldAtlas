from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

WORLDPLAY_DEFAULT_WIDTH = 832
WORLDPLAY_DEFAULT_HEIGHT = 480
WORLDPLAY_DEFAULT_VIDEO_LENGTH = 125
WORLDPLAY_DEFAULT_FOCAL_SCALE = 0.5050505
WORLDPLAY_FORWARD_STEP = 1.0
WORLDPLAY_LATERAL_STEP = 0.5
WORLDPLAY_VERTICAL_STEP = 0.35
WORLDPLAY_ROTATION_DEGREES = 30.0
WORLDPLAY_TILT_DEGREES = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent HY-WorldPlay benchmark runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suites", default="image_static,image_dynamic", type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--visible-devices", default=None, type=str)
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
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.8")


def _write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _rank() -> int:
    return _env_int("RANK", 0)


def _world_size() -> int:
    return _env_int("WORLD_SIZE", 1)


def _is_rank0() -> bool:
    return _rank() == 0


def _dist_barrier() -> None:
    if _world_size() <= 1:
        return
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y", "on"}


def _is_fatal_cuda_error(exc: BaseException) -> bool:
    text = (str(exc) or repr(exc)).lower()
    markers = (
        "cuda error: out of memory",
        "cuda out of memory",
        "out of memory",
        "internal assert failed",
        "cuda driver error",
        "cusolver_status_internal_error",
        "cublas_status",
        "illegal memory access",
    )
    return any(marker in text for marker in markers)


def _video_to_uint8_frames(video: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(video):
            if video.ndim == 5:
                if video.shape[0] != 1:
                    raise ValueError(f"expected batch size 1 video tensor, got {tuple(video.shape)}")
                video = video[0]
            frames = (video.detach() * 255).clamp(0, 255).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).contiguous().cpu().numpy()
            return np.ascontiguousarray(frames)
    except ImportError:
        pass

    frames = np.asarray(video)
    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise ValueError(f"expected batch size 1 video array, got {frames.shape}")
        frames = frames[0]
    if frames.ndim == 4 and frames.shape[0] in {1, 3, 4}:
        frames = np.transpose(frames, (1, 2, 3, 0))
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


def _save_video_frames(frames: np.ndarray, temp_path: Path, output_path: Path, fps: int) -> None:
    import imageio

    temp_path.unlink(missing_ok=True)
    imageio.mimwrite(str(temp_path), frames, fps=int(fps))
    temp_path.replace(output_path)


def _select_samples(
    manifest: list[Any],
    *,
    suites: set[str] | None,
    limit: int | None,
    supported_suites: set[str],
) -> list[Any]:
    selected: list[Any] = []
    for sample in manifest:
        if sample.suite not in supported_suites:
            continue
        if suites is not None and sample.suite not in suites:
            continue
        selected.append(sample)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _image_conditioning_asset_for(sample: Any) -> Path:
    from worldarena.common.checkpoints import rehome_legacy_workspace_path

    raw_path = sample.conditioning_path or sample.reference_path
    remapped = rehome_legacy_workspace_path(raw_path)
    if remapped is None:
        raise FileNotFoundError(f"conditioning asset not found: {raw_path}")
    path = Path(remapped).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"conditioning asset not found: {path}")
    return path


def _prompt_for_sample(model_config: Any, sample: Any) -> str:
    if model_config.prompt_mode == "none":
        return ""
    prompt = sample.prompt_current if model_config.prompt_mode == "prompt_current" else sample.prompt_target
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError(
            f"{model_config.name} requires non-empty {model_config.prompt_mode} for sample {sample.sample_id}"
        )
    return prompt


def _generation_for_suite(generation: dict[str, Any], suite: str) -> dict[str, Any]:
    resolved = dict(generation)
    for key, value in generation.items():
        if not key.endswith("_by_suite") or not isinstance(value, dict):
            continue
        base_key = key[: -len("_by_suite")]
        if suite in value:
            resolved[base_key] = value[suite]
    return resolved


def _latent_frame_count(video_length: int) -> int:
    return (int(video_length) - 1) // 4 + 1


def _validated_video_length(value: int) -> int:
    video_length = int(value)
    if video_length <= 0:
        raise ValueError(f"video_length must be positive, got {video_length}")
    if (video_length - 1) % 4 != 0:
        raise ValueError(f"HY-WorldPlay requires video_length where (video_length - 1) % 4 == 0; got {video_length}")
    latent_count = _latent_frame_count(video_length)
    if latent_count % 4 != 0:
        raise ValueError(
            "HY-WorldPlay requires the latent frame count to be divisible by 4; "
            f"got video_length={video_length}, latent_count={latent_count}"
        )
    return video_length


def _translation_matrix(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray([x, y, z], dtype=np.float32)
    return matrix


def _rotation_matrix_x(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float32,
    )


def _rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
        dtype=np.float32,
    )


def _rotation_matrix_z(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _rigid_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    if rotation is not None:
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float32)
    if translation is not None:
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def _camera_delta(token: str) -> np.ndarray:
    token = str(token or "").strip().lower()
    if not token or token == "fixed":
        return np.eye(4, dtype=np.float32)
    if token == "push_in":
        return _translation_matrix(z=-WORLDPLAY_FORWARD_STEP)
    if token == "pull_out":
        return _translation_matrix(z=WORLDPLAY_FORWARD_STEP)
    if token == "move_left":
        return _translation_matrix(x=-WORLDPLAY_LATERAL_STEP)
    if token == "move_right":
        return _translation_matrix(x=WORLDPLAY_LATERAL_STEP)
    if token == "pedestal_up":
        return _translation_matrix(y=WORLDPLAY_VERTICAL_STEP)
    if token == "pedestal_down":
        return _translation_matrix(y=-WORLDPLAY_VERTICAL_STEP)
    if token == "pan_left":
        return _rigid_transform(
            rotation=_rotation_matrix_y(WORLDPLAY_ROTATION_DEGREES),
            translation=np.asarray([-WORLDPLAY_LATERAL_STEP, 0.0, WORLDPLAY_FORWARD_STEP], dtype=np.float32),
        )
    if token == "pan_right":
        return _rigid_transform(
            rotation=_rotation_matrix_y(-WORLDPLAY_ROTATION_DEGREES),
            translation=np.asarray([WORLDPLAY_LATERAL_STEP, 0.0, WORLDPLAY_FORWARD_STEP], dtype=np.float32),
        )
    if token == "orbit_left":
        return _rigid_transform(
            rotation=_rotation_matrix_y(-WORLDPLAY_ROTATION_DEGREES),
            translation=np.asarray([-WORLDPLAY_LATERAL_STEP, 0.0, -WORLDPLAY_FORWARD_STEP], dtype=np.float32),
        )
    if token == "orbit_right":
        return _rigid_transform(
            rotation=_rotation_matrix_y(WORLDPLAY_ROTATION_DEGREES),
            translation=np.asarray([WORLDPLAY_LATERAL_STEP, 0.0, -WORLDPLAY_FORWARD_STEP], dtype=np.float32),
        )
    if token == "tilt_up":
        return _rigid_transform(rotation=_rotation_matrix_x(WORLDPLAY_TILT_DEGREES))
    if token == "tilt_down":
        return _rigid_transform(rotation=_rotation_matrix_x(-WORLDPLAY_TILT_DEGREES))
    if token == "roll_cw":
        return _rigid_transform(rotation=_rotation_matrix_z(-WORLDPLAY_ROTATION_DEGREES))
    if token == "roll_ccw":
        return _rigid_transform(rotation=_rotation_matrix_z(WORLDPLAY_ROTATION_DEGREES))
    return np.eye(4, dtype=np.float32)


def _camera_path_for_sample(sample: Any) -> list[str]:
    camera_path = [str(token).strip() for token in sample.camera_path if str(token).strip()]
    if camera_path:
        return camera_path
    return ["fixed"] if sample.modality == "image" else []


def _interpolate_pose_segment(start_pose: np.ndarray, end_pose: np.ndarray, *, frames: int) -> np.ndarray:
    from worldarena.benchmark.annotations import _matrix_to_pose_vector, _pose_vector_to_matrix, _resample_pose_vectors

    pose_vectors = np.stack(
        [
            _matrix_to_pose_vector(start_pose.astype(np.float32)),
            _matrix_to_pose_vector(end_pose.astype(np.float32)),
        ],
        axis=0,
    ).astype(np.float32)
    indices = np.asarray([0.0, 1.0], dtype=np.float32)
    interpolated = _resample_pose_vectors(pose_vectors, indices, frames)
    return np.stack([_pose_vector_to_matrix(pose_row) for pose_row in interpolated], axis=0).astype(np.float32)


def _synthetic_image_poses(camera_path: list[str], *, latent_count: int) -> np.ndarray:
    target_frames = max(int(latent_count), 1)
    if target_frames <= 1:
        return np.repeat(np.eye(4, dtype=np.float32)[None, ...], repeats=target_frames, axis=0)
    normalized_path = [token for token in camera_path if token] or ["fixed"]
    keyframes: list[np.ndarray] = [np.eye(4, dtype=np.float32)]
    current_pose = np.eye(4, dtype=np.float32)
    for token in normalized_path:
        current_pose = (current_pose @ _camera_delta(token)).astype(np.float32)
        keyframes.append(current_pose.copy())

    segment_count = max(1, len(normalized_path))
    remaining_intervals = target_frames - 1
    base_intervals = remaining_intervals // segment_count
    remainder = remaining_intervals % segment_count
    segments: list[np.ndarray] = []
    for index in range(segment_count):
        segment_intervals = base_intervals + (1 if index < remainder else 0)
        segment = _interpolate_pose_segment(
            keyframes[index],
            keyframes[index + 1],
            frames=segment_intervals + 1,
        )
        if index:
            segment = segment[1:]
        segments.append(segment.astype(np.float32))
    poses = np.concatenate(segments, axis=0).astype(np.float32)
    if len(poses) != target_frames:
        raise ValueError(f"expected {target_frames} synthetic poses, got {len(poses)}")
    return poses


def _intrinsics_sequence(latent_count: int, *, image_width: int, image_height: int, focal_scale: float) -> np.ndarray:
    focal_length = float(max(image_width, image_height)) * float(focal_scale)
    intrinsics = np.asarray([focal_length, focal_length, image_width / 2.0, image_height / 2.0], dtype=np.float32)
    return np.repeat(intrinsics[None, :], repeats=max(int(latent_count), 1), axis=0).astype(np.float32)


def _intrinsics_matrix(values: np.ndarray) -> list[list[float]]:
    fx, fy, cx, cy = np.asarray(values, dtype=np.float32).reshape(4).tolist()
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _resolved_pose_json_path(
    *,
    output_dir: Path,
    model_config: Any,
    sample: Any,
    latent_count: int,
    image_width: int,
    image_height: int,
) -> tuple[Path, str]:
    camera_path = _camera_path_for_sample(sample)
    focal_scale = float(model_config.generation.get("trajectory_focal_scale", WORLDPLAY_DEFAULT_FOCAL_SCALE))
    cache_key_payload = {
        "sample_id": sample.sample_id,
        "prediction_stem": sample.prediction_stem,
        "relative_path": sample.relative_path,
        "camera_path": camera_path,
        "latent_count": int(latent_count),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "focal_scale": focal_scale,
    }
    key = hashlib.sha1(json.dumps(cache_key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[
        :12
    ]
    pose_json_path = output_dir / "_pose_json" / f"{sample.prediction_stem}--{key}.json"
    if pose_json_path.exists():
        return pose_json_path, "synthetic"

    poses = _synthetic_image_poses(camera_path, latent_count=latent_count)
    intrinsics = _intrinsics_sequence(
        latent_count,
        image_width=image_width,
        image_height=image_height,
        focal_scale=focal_scale,
    )
    payload = {
        str(index): {
            "extrinsic": np.asarray(pose, dtype=np.float32).tolist(),
            "K": _intrinsics_matrix(intrinsics_row),
        }
        for index, (pose, intrinsics_row) in enumerate(zip(poses, intrinsics, strict=True))
    }
    pose_json_path.parent.mkdir(parents=True, exist_ok=True)
    pose_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return pose_json_path, "synthetic"


class HYWorldPlayRuntime:
    def __init__(self, *, model_config, project_root: Path) -> None:
        from worldarena.common.checkpoints import apply_checkpoint_env, resolve_project_path

        if model_config.repo_root is None:
            raise ValueError("HY-WorldPlay persistent runner requires repo_root")
        if model_config.checkpoint_dir is None:
            raise ValueError("HY-WorldPlay persistent runner requires checkpoint_dir")

        self.model_config = model_config
        self.project_root = project_root
        self.repo_root = model_config.repo_root.expanduser().resolve()
        self.generation = dict(model_config.generation)
        self._save_executor: ThreadPoolExecutor | None = None
        self._save_futures: list[Future] = []
        self._max_pending_saves = max(1, int(self.generation.get("async_save_queue", 2)))
        self.model_path = resolve_project_path(model_config.checkpoint_dir)
        if self.model_path is None or not self.model_path.is_dir():
            raise FileNotFoundError(f"HY-WorldPlay model path is missing: {model_config.checkpoint_dir}")
        action_ckpt_value = self.generation.get("action_ckpt")
        self.action_ckpt = resolve_project_path(action_ckpt_value)
        if self.action_ckpt is None or not self.action_ckpt.is_file():
            raise FileNotFoundError(f"HY-WorldPlay action checkpoint is missing: {action_ckpt_value}")

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        os.environ.update(apply_checkpoint_env())
        os.chdir(self.repo_root)

        import importlib
        import torch

        self.torch = torch
        self.mod = importlib.import_module("hyvideo.generate")
        self.mod.initialize_infer_state(
            SimpleNamespace(
                use_sageattn=_as_bool(self.generation.get("use_sageattn", False)),
                sage_blocks_range=str(self.generation.get("sage_blocks_range", "0-53")),
                enable_torch_compile=_as_bool(self.generation.get("enable_torch_compile", False)),
                use_fp8_gemm=_as_bool(self.generation.get("use_fp8_gemm", False)),
                quant_type=str(self.generation.get("quant_type", "fp8-per-block")),
                include_patterns=str(self.generation.get("include_patterns", "double_blocks")),
                use_vae_parallel=_as_bool(self.generation.get("use_vae_parallel", False)),
            )
        )

        resolution = str(self.generation.get("resolution", "480p"))
        transformer_version = f"{resolution}_i2v"
        if transformer_version != "480p_i2v":
            raise ValueError(f"HY-WorldPlay persistent runner only supports 480p_i2v, got {transformer_version}")
        dtype = str(self.generation.get("dtype", "bf16"))
        if dtype == "bf16":
            transformer_dtype = torch.bfloat16
        elif dtype == "fp32":
            transformer_dtype = torch.float32
        else:
            raise ValueError(f"unsupported HY-WorldPlay dtype: {dtype}")

        print("[hy-worldplay-persistent] loading pipeline once", flush=True)
        self.pipe = self.mod.HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=str(self.model_path),
            transformer_version=transformer_version,
            enable_offloading=_as_bool(self.generation.get("offloading", True)),
            enable_group_offloading=self.generation.get("group_offloading"),
            create_sr_pipeline=_as_bool(self.generation.get("sr", False)),
            force_sparse_attn=False,
            transformer_dtype=transformer_dtype,
            action_ckpt=str(self.action_ckpt),
        )
        if _is_rank0() and _as_bool(self.generation.get("async_save", True)):
            self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hy-worldplay-save")
        print("[hy-worldplay-persistent] pipeline ready", flush=True)

    def _drain_save_futures(self, *, wait: bool) -> None:
        if not _is_rank0():
            return
        remaining: list[Future] = []
        for future in self._save_futures:
            if wait or future.done():
                future.result()
            else:
                remaining.append(future)
        self._save_futures = remaining

    def _submit_video_save(self, *, video: Any, output_path: Path, fps: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f".{output_path.name}.tmp.mp4")
        frames = _video_to_uint8_frames(video)

        if self._save_executor is None:
            _save_video_frames(frames, temp_path, output_path, fps)
            return

        self._drain_save_futures(wait=False)
        while len(self._save_futures) >= self._max_pending_saves:
            self._save_futures.pop(0).result()
            self._drain_save_futures(wait=False)
        self._save_futures.append(
            self._save_executor.submit(_save_video_frames, frames, temp_path, output_path, fps)
        )

    def close(self) -> None:
        self._drain_save_futures(wait=True)
        if self._save_executor is not None:
            self._save_executor.shutdown(wait=True)
            self._save_executor = None

    def generate(self, item: dict[str, Any]) -> dict[str, Any]:
        generation = dict(item["generation"])
        latent_count = int(item["latent_count"])
        viewmats, ks, action = self.mod.pose_to_input(str(item["pose_json_path"]), latent_count)
        enable_sr = _as_bool(generation.get("sr", False))
        save_pre_sr = _as_bool(generation.get("save_pre_sr_video", False))
        out = self.pipe(
            enable_sr=enable_sr,
            prompt=str(item["prompt"]),
            aspect_ratio=str(generation.get("aspect_ratio", "16:9")),
            num_inference_steps=int(generation.get("num_inference_steps", 4)),
            sr_num_inference_steps=None,
            video_length=int(item["video_length"]),
            negative_prompt=str(generation.get("negative_prompt", "")),
            seed=int(generation.get("seed", 1)),
            output_type="pt",
            prompt_rewrite=_as_bool(generation.get("rewrite", False)),
            return_pre_sr_video=save_pre_sr,
            viewmats=viewmats.unsqueeze(0),
            Ks=ks.unsqueeze(0),
            action=action.unsqueeze(0),
            few_step=_as_bool(generation.get("few_step", True)),
            chunk_latent_frames=4 if str(generation.get("model_type", "ar")) == "ar" else 16,
            model_type=str(generation.get("model_type", "ar")),
            user_height=int(generation.get("height", item["height"])),
            user_width=int(generation.get("width", item["width"])),
            transformer_resident_ar_rollout=_as_bool(
                generation.get("transformer_resident_ar_rollout", False)
            ),
            reference_image=str(item["conditioning_image"]),
        )

        output_path = Path(item["prediction_path"])
        if _is_rank0():
            video = out.sr_videos if enable_sr and hasattr(out, "sr_videos") else out.videos
            self._submit_video_save(video=video, output_path=output_path, fps=int(generation.get("fps", 24)))
            if enable_sr and save_pre_sr and hasattr(out, "videos"):
                pre_sr_path = output_path.with_name(f"{output_path.stem}.pre_sr{output_path.suffix}")
                self._submit_video_save(video=out.videos, output_path=pre_sr_path, fps=int(generation.get("fps", 24)))
        if _as_bool(generation.get("empty_cache_each_sample", False)):
            self.torch.cuda.empty_cache()
        _dist_barrier()
        return {
            "prediction_path": str(output_path),
            "output_path": str(output_path),
            "prompt": item["prompt"],
            "pose_json_path": str(item["pose_json_path"]),
            "pose_path": str(item["pose_json_path"]),
            "camera_pose_path": str(item["pose_json_path"]),
            "pose_source": item["pose_source"],
            "control_source": item["pose_source"],
            "camera_path": item["camera_path"],
            "video_length": int(item["video_length"]),
            "width": int(item["width"]),
            "height": int(item["height"]),
        }


def main() -> int:
    args = parse_args()
    _prepare_cuda_visibility(args)
    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"shard_index must be within [0, {args.num_shards - 1}]")

    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.benchmark.manifest import load_manifest
    from worldarena.common.media import probe_image
    from worldarena.models.config import load_model_config

    model_config = load_model_config(args.model_config.resolve())
    if model_config.family.lower() not in {"hy_worldplay", "hy-worldplay", "worldplay"}:
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")

    manifest = load_manifest(args.manifest.resolve())
    suites = _split_arg(args.suites)
    selected = _select_samples(
        manifest,
        suites=suites,
        limit=args.limit,
        supported_suites=set(model_config.supported_suites),
    )
    selected = selected[args.shard_index :: args.num_shards]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"generation_records.shard{args.shard_index}.jsonl"
    summary_path = output_dir / f"generation_manifest.shard{args.shard_index}.json"
    prompt_map_path = output_dir / f"prompt_map.shard{args.shard_index}.json"
    if _is_rank0():
        for path in (records_path, summary_path, prompt_map_path):
            path.unlink(missing_ok=True)

    records: list[dict[str, Any]] = []
    prompt_map: dict[str, str] = {}
    pending: list[dict[str, Any]] = []

    for offset, sample in enumerate(selected, start=1):
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(prediction_path),
            "output_path": str(prediction_path),
            "conditioning_image": None,
            "conditioning_input": None,
            "prompt": None,
            "prompt_source": None,
            "pose_json_path": None,
            "pose_path": None,
            "camera_pose_path": None,
            "pose_source": None,
            "control_source": None,
            "camera_path": None,
            "checkpoint_dir": str(getattr(model_config, "checkpoint_dir", "")),
            "action_ckpt": None,
            "width": None,
            "height": None,
            "video_length": None,
            "fps": None,
            "num_inference_steps": None,
            "status": "pending",
            "error": None,
            "shard_index": args.shard_index,
            "shard_position": offset,
            "shard_total": len(selected),
        }
        try:
            prompt = _prompt_for_sample(model_config, sample)
            generation = _generation_for_suite(model_config.generation, sample.suite)
            video_length = _validated_video_length(
                int(generation.get("video_length", WORLDPLAY_DEFAULT_VIDEO_LENGTH))
            )
            latent_count = _latent_frame_count(video_length)
            record.update(
                {
                    "prompt": prompt,
                    "prompt_source": model_config.prompt_mode,
                    "action_ckpt": str(generation.get("action_ckpt", "")),
                    "video_length": video_length,
                    "fps": int(generation.get("fps", 24)),
                    "num_inference_steps": int(generation.get("num_inference_steps", 4)),
                }
            )
            prompt_map[str(prediction_path)] = prompt
            if prediction_path.exists() and not args.overwrite:
                record["status"] = "skipped_existing"
                records.append(record)
                if _is_rank0():
                    _write_record(records_path, record)
                continue
            conditioning_image = _image_conditioning_asset_for(sample)
            image_probe = probe_image(conditioning_image)
            width = int(generation.get("width", image_probe.get("width", WORLDPLAY_DEFAULT_WIDTH)))
            height = int(generation.get("height", image_probe.get("height", WORLDPLAY_DEFAULT_HEIGHT)))
            pose_json_path, pose_source = _resolved_pose_json_path(
                output_dir=output_dir,
                model_config=model_config,
                sample=sample,
                latent_count=latent_count,
                image_width=width,
                image_height=height,
            )
            camera_path = _camera_path_for_sample(sample)
            record.update(
                {
                    "conditioning_image": str(conditioning_image),
                    "conditioning_input": str(conditioning_image),
                    "prompt": prompt,
                    "prompt_source": model_config.prompt_mode,
                    "pose_json_path": str(pose_json_path),
                    "pose_path": str(pose_json_path),
                    "camera_pose_path": str(pose_json_path),
                    "pose_source": pose_source,
                    "control_source": pose_source,
                    "camera_path": camera_path,
                    "width": width,
                    "height": height,
                }
            )
            pending.append(
                {
                    "sample": sample,
                    "generation": generation,
                    "prediction_path": prediction_path,
                    "conditioning_image": conditioning_image,
                    "prompt": prompt,
                    "pose_json_path": pose_json_path,
                    "pose_source": pose_source,
                    "camera_path": camera_path,
                    "video_length": video_length,
                    "latent_count": latent_count,
                    "width": width,
                    "height": height,
                    "record": record,
                }
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc) or repr(exc)
            records.append(record)
            if _is_rank0():
                _write_record(records_path, record)

    if not pending:
        if _is_rank0():
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
        return 0

    runtime = HYWorldPlayRuntime(model_config=model_config, project_root=project_root)
    for item in pending:
        record = item["record"]
        fatal_cuda_error = False
        try:
            print(
                f"[hy-worldplay-persistent] generating {Path(item['prediction_path']).name}",
                flush=True,
            )
            payload = runtime.generate(item)
            record.update(payload)
            record["status"] = "generated"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc) or repr(exc)
            fatal_cuda_error = _is_fatal_cuda_error(exc)
            print(
                "[hy-worldplay-persistent] failed "
                f"{Path(item['prediction_path']).name}: {record['error']}",
                flush=True,
            )
            if fatal_cuda_error:
                print(
                    "[hy-worldplay-persistent] fatal CUDA error; exiting shard after current record",
                    flush=True,
                )
        records.append(record)
        if _is_rank0():
            _write_record(records_path, record)
        if fatal_cuda_error:
            break

    runtime.close()

    if _is_rank0():
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
