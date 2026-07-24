"""HY-WorldPlay model adapter for WorldAtlas Arena.

HY-WorldPlay is a camera-conditioned video world model. This adapter resolves
GT or synthetic camera trajectories into a pose JSON sidecar, then launches
``hy_worldplay_runner`` to invoke the upstream ``hyvideo/generate.py`` entrypoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.benchmark.annotations import (
    _matrix_to_pose_vector,
    _pose_vector_to_matrix,
    _resample_pose_vectors,
    load_camera_matrices,
    load_intrinsics_sequence,
)
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.annotation_index import split_annotation_reference
from worldarena.common.checkpoints import (
    project_root as worldarena_project_root,
    resolve_project_path,
)
from worldarena.common.media import probe_image
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
    run_sequential_generate_batch,
)
from worldarena.models.adapters.lingbot_world import (
    LINGBOT_FORWARD_STEP,
    LINGBOT_LATERAL_STEP,
    LINGBOT_ROTATION_DEGREES,
    _camera_delta,
    _camera_path_for_sample,
    _rigid_transform,
    _rotation_matrix_y,
)
from worldarena.models.config import ModelRuntimeConfig


WORLDPLAY_DEFAULT_WIDTH = 832
WORLDPLAY_DEFAULT_HEIGHT = 480
WORLDPLAY_DEFAULT_VIDEO_LENGTH = 125
# Default focal length scale used when synthesizing intrinsics from image size.
WORLDPLAY_DEFAULT_FOCAL_SCALE = 0.5050505


def _project_root(config: ModelRuntimeConfig) -> Path:
    del config
    return worldarena_project_root()


def _resolve_extra_path(config: ModelRuntimeConfig, value: str | None) -> Path | None:
    del config
    return resolve_project_path(value)


def _runtime_root(config: ModelRuntimeConfig) -> Path:
    configured = config.generation.get("runtime_cache_dir")
    if configured is not None:
        root = _resolve_extra_path(config, str(configured))
        if root is None:
            raise ValueError("invalid runtime_cache_dir")
    else:
        root = _project_root(config) / "cache" / "model_runtime" / "hy_worldplay"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _generation_for_suite(generation: dict[str, Any], suite: str) -> dict[str, Any]:
    """Merge suite-specific overrides from keys like ``video_length_by_suite``."""
    resolved = dict(generation)
    for key, value in generation.items():
        if not key.endswith("_by_suite") or not isinstance(value, dict):
            continue
        base_key = key[: -len("_by_suite")]
        if suite in value:
            resolved[base_key] = value[suite]
    return resolved


def _validated_video_length(value: int) -> int:
    """Enforce WorldPlay's latent-frame constraints on output video length."""
    video_length = int(value)
    if video_length <= 0:
        raise ValueError(f"video_length must be positive, got {video_length}")
    if (video_length - 1) % 4 != 0:
        raise ValueError(
            f"HY-WorldPlay requires video_length where (video_length - 1) % 4 == 0; got {video_length}"
        )
    latent_count = _latent_frame_count(video_length)
    if latent_count % 4 != 0:
        raise ValueError(
            "HY-WorldPlay requires the latent frame count to be divisible by 4; "
            f"got video_length={video_length}, latent_count={latent_count}"
        )
    return video_length


def _latent_frame_count(video_length: int) -> int:
    """Convert pixel-frame count to WorldPlay's latent temporal dimension."""
    return (int(video_length) - 1) // 4 + 1


def _interpolate_pose_segment(start_pose: np.ndarray, end_pose: np.ndarray, *, frames: int) -> np.ndarray:
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


def _worldplay_camera_delta(token: str) -> np.ndarray:
    token = str(token or "").strip().lower()
    if token == "pan_left":
        return _rigid_transform(
            rotation=_rotation_matrix_y(LINGBOT_ROTATION_DEGREES),
            translation=np.asarray([-LINGBOT_LATERAL_STEP, 0.0, LINGBOT_FORWARD_STEP], dtype=np.float32),
        )
    if token == "pan_right":
        return _rigid_transform(
            rotation=_rotation_matrix_y(-LINGBOT_ROTATION_DEGREES),
            translation=np.asarray([LINGBOT_LATERAL_STEP, 0.0, LINGBOT_FORWARD_STEP], dtype=np.float32),
        )
    if token == "orbit_left":
        return _rigid_transform(
            rotation=_rotation_matrix_y(-LINGBOT_ROTATION_DEGREES),
            translation=np.asarray([-LINGBOT_LATERAL_STEP, 0.0, -LINGBOT_FORWARD_STEP], dtype=np.float32),
        )
    if token == "orbit_right":
        return _rigid_transform(
            rotation=_rotation_matrix_y(LINGBOT_ROTATION_DEGREES),
            translation=np.asarray([LINGBOT_LATERAL_STEP, 0.0, -LINGBOT_FORWARD_STEP], dtype=np.float32),
        )
    return _camera_delta(token)


def _synthetic_image_poses(camera_path: list[str], *, latent_count: int) -> np.ndarray:
    """Build a smooth pose sequence from discrete camera_path tokens for image_static."""
    target_frames = max(int(latent_count), 1)
    if target_frames <= 1:
        return np.repeat(np.eye(4, dtype=np.float32)[None, ...], repeats=target_frames, axis=0)

    normalized_path = [token for token in camera_path if token] or ["fixed"]
    keyframes: list[np.ndarray] = [np.eye(4, dtype=np.float32)]
    current_pose = np.eye(4, dtype=np.float32)
    for token in normalized_path:
        current_pose = (current_pose @ _worldplay_camera_delta(token)).astype(np.float32)
        keyframes.append(current_pose.copy())

    segment_count = max(1, len(normalized_path))
    remaining_intervals = target_frames - 1
    base_intervals = remaining_intervals // segment_count
    remainder = remaining_intervals % segment_count

    segments: list[np.ndarray] = []
    for index in range(segment_count):
        segment_intervals = base_intervals + (1 if index < remainder else 0)
        segment_frames = segment_intervals + 1
        segment = _interpolate_pose_segment(
            keyframes[index],
            keyframes[index + 1],
            frames=segment_frames,
        )
        if index:
            segment = segment[1:]
        segments.append(segment.astype(np.float32))
    poses = np.concatenate(segments, axis=0).astype(np.float32)
    if len(poses) != target_frames:
        raise ValueError(f"expected {target_frames} synthetic poses, got {len(poses)}")
    return poses


def _synthetic_intrinsics_sequence(
    latent_count: int,
    *,
    image_width: int,
    image_height: int,
    focal_scale: float = WORLDPLAY_DEFAULT_FOCAL_SCALE,
) -> np.ndarray:
    focal_length = float(max(image_width, image_height)) * float(focal_scale)
    intrinsics = np.asarray(
        [focal_length, focal_length, image_width / 2.0, image_height / 2.0],
        dtype=np.float32,
    )
    return np.repeat(intrinsics[None, :], repeats=max(int(latent_count), 1), axis=0).astype(np.float32)


def _vector_intrinsics_to_matrix(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 4:
        raise ValueError(f"expected 4-element intrinsics vector, got shape {vector.shape}")
    fx, fy, cx, cy = vector.tolist()
    return np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _local_annotation_path(sample: BenchmarkSample) -> Path | None:
    if not sample.annotation_path:
        return None
    if split_annotation_reference(sample.annotation_path) is not None:
        return None
    candidate = Path(sample.annotation_path).expanduser().resolve()
    if not candidate.exists():
        return None
    if (candidate / "poses.npy").exists() and (candidate / "intrinsics.npy").exists():
        return candidate
    return None


def _synthetic_pose_root(
    config: ModelRuntimeConfig,
    sample: BenchmarkSample,
    *,
    latent_count: int,
    image_width: int,
    image_height: int,
) -> Path:
    camera_path = _camera_path_for_sample(sample) or ["fixed"]
    focal_scale = float(config.generation.get("trajectory_focal_scale", WORLDPLAY_DEFAULT_FOCAL_SCALE))
    cache_key_payload = {
        "sample_id": sample.sample_id,
        "relative_path": sample.relative_path,
        "annotation_path": sample.annotation_path,
        "camera_path": camera_path,
        "latent_count": int(latent_count),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "focal_scale": focal_scale,
    }
    key = hashlib.sha1(
        json.dumps(cache_key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    annotation_root = _runtime_root(config) / "synthetic_pose" / f"{sample.prediction_stem}--{key}"
    poses_path = annotation_root / "poses.npy"
    intrinsics_path = annotation_root / "intrinsics.npy"
    if poses_path.exists() and intrinsics_path.exists():
        try:
            poses = np.load(poses_path)
            intrinsics = np.load(intrinsics_path)
            if poses.shape == (latent_count, 4, 4) and intrinsics.shape == (latent_count, 4):
                return annotation_root
        except Exception:
            pass

    annotation_root.mkdir(parents=True, exist_ok=True)
    poses = _synthetic_image_poses(camera_path, latent_count=latent_count)
    intrinsics = _synthetic_intrinsics_sequence(
        latent_count,
        image_width=image_width,
        image_height=image_height,
        focal_scale=focal_scale,
    )
    np.save(poses_path, poses.astype(np.float32))
    np.save(intrinsics_path, intrinsics.astype(np.float32))
    return annotation_root


def _resolved_pose_root(
    config: ModelRuntimeConfig,
    sample: BenchmarkSample,
    *,
    latent_count: int,
    image_width: int,
    image_height: int,
) -> tuple[Path, str]:
    """Prefer on-disk annotations; fall back to synthetic poses from camera_path."""
    annotation_path = _local_annotation_path(sample)
    if annotation_path is not None:
        return annotation_path, "annotation"
    return (
        _synthetic_pose_root(
            config,
            sample,
            latent_count=latent_count,
            image_width=image_width,
            image_height=image_height,
        ),
        "synthetic",
    )


def _pose_json_payload(pose_root: Path, *, latent_count: int) -> dict[str, Any]:
    pose_matrices = load_camera_matrices(str(pose_root), target_frames=latent_count)
    intrinsics = load_intrinsics_sequence(str(pose_root), target_frames=latent_count)
    if pose_matrices is None:
        raise FileNotFoundError(f"poses.npy not found under annotation path: {pose_root}")
    if intrinsics is None:
        raise FileNotFoundError(f"intrinsics.npy not found under annotation path: {pose_root}")
    if len(pose_matrices) != latent_count or len(intrinsics) != latent_count:
        raise ValueError(
            f"expected {latent_count} pose/intrinsics rows, got {len(pose_matrices)} and {len(intrinsics)}"
        )

    payload: dict[str, Any] = {}
    for index, (pose_row, intrinsics_row) in enumerate(zip(pose_matrices, intrinsics, strict=True)):
        payload[str(index)] = {
            "extrinsic": np.asarray(pose_row, dtype=np.float32).tolist(),
            "K": _vector_intrinsics_to_matrix(intrinsics_row).tolist(),
        }
    return payload


def _resolved_pose_json_path(
    config: ModelRuntimeConfig,
    sample: BenchmarkSample,
    *,
    latent_count: int,
    image_width: int,
    image_height: int,
) -> tuple[Path, str]:
    """Materialize or reuse a cached pose JSON file consumed by the upstream model."""
    pose_root, pose_source = _resolved_pose_root(
        config,
        sample,
        latent_count=latent_count,
        image_width=image_width,
        image_height=image_height,
    )
    cache_key_payload = {
        "sample_id": sample.sample_id,
        "prediction_stem": sample.prediction_stem,
        "pose_root": str(pose_root),
        "pose_source": pose_source,
        "latent_count": int(latent_count),
        "image_width": int(image_width),
        "image_height": int(image_height),
    }
    key = hashlib.sha1(
        json.dumps(cache_key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    pose_json_path = _runtime_root(config) / "pose_json" / f"{sample.prediction_stem}--{key}.json"
    if pose_json_path.exists():
        return pose_json_path, pose_source

    pose_json_path.parent.mkdir(parents=True, exist_ok=True)
    pose_json_path.write_text(
        json.dumps(
            _pose_json_payload(pose_root, latent_count=latent_count),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return pose_json_path, pose_source


class HYWorldPlayAdapter(ModelAdapter):
    """Generate camera-guided videos via the HY-WorldPlay upstream repo."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_checkpoint_load_policy(self) -> str:
        return "reload_per_sample"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        return run_sequential_generate_batch(self, requests)

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if self.config.repo_root is None:
            raise ValueError("HY-WorldPlay adapter requires repo_root")

        generation = _generation_for_suite(self.config.generation, sample.suite)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        conditioning_image = conditioning_image.expanduser().resolve()
        if not conditioning_image.exists():
            raise FileNotFoundError(f"conditioning image not found: {conditioning_image}")

        resolution = str(generation.get("resolution", "480p")).strip()
        if resolution != "480p":
            raise ValueError(f"HY-WorldPlay currently only supports resolution=480p, got {resolution!r}")

        video_length = _validated_video_length(
            int(generation.get("video_length", WORLDPLAY_DEFAULT_VIDEO_LENGTH))
        )
        latent_count = _latent_frame_count(video_length)

        image_probe = probe_image(conditioning_image)
        image_width = int(generation.get("width", image_probe.get("width", WORLDPLAY_DEFAULT_WIDTH)))
        image_height = int(generation.get("height", image_probe.get("height", WORLDPLAY_DEFAULT_HEIGHT)))
        pose_json_path, pose_source = _resolved_pose_json_path(
            self.config,
            sample,
            latent_count=latent_count,
            image_width=image_width,
            image_height=image_height,
        )

        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()
        if not prompt_text:
            prompt_text = str(sample.prompt_target or sample.prompt_current or "").strip()
        if not prompt_text:
            raise ValueError("HY-WorldPlay requires a non-empty prompt")

        model_path = resolve_project_path(self.config.checkpoint_dir)
        if model_path is None or not model_path.is_dir():
            raise ValueError(f"HY-WorldPlay checkpoint directory not found: {self.config.checkpoint_dir}")

        action_ckpt_value = generation.get("action_ckpt")
        action_ckpt = resolve_project_path(action_ckpt_value)
        if action_ckpt is None or not action_ckpt.is_file():
            raise ValueError(f"HY-WorldPlay action checkpoint not found: {action_ckpt_value}")

        runner_payload: dict[str, Any] = dict(generation)
        runner_payload.update(
            {
                "negative_prompt": str(generation.get("negative_prompt", "")),
                "resolution": resolution,
                "aspect_ratio": str(generation.get("aspect_ratio", "16:9")),
                "num_inference_steps": int(
                    generation.get("num_inference_steps", 4 if bool(generation.get("few_step", True)) else 50)
                ),
                "video_length": video_length,
                "seed": int(generation.get("seed", 1)),
                "model_type": str(generation.get("model_type", "ar")),
                "width": image_width,
                "height": image_height,
                "dtype": str(generation.get("dtype", "bf16")),
                "launcher": str(generation.get("launcher", "python")),
                "nproc_per_node": int(generation.get("nproc_per_node", 1)),
                "gpu_index": int(generation.get("gpu_index", 0)),
                "sage_blocks_range": str(generation.get("sage_blocks_range", "0-53")),
                "quant_type": str(generation.get("quant_type", "fp8-per-block")),
                "include_patterns": str(generation.get("include_patterns", "double_blocks")),
            }
        )

        project_root = _project_root(self.config)
        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.hy_worldplay_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--entrypoint",
            str(self.config.entrypoint or "hyvideo/generate.py"),
            "--conditioning_image",
            str(conditioning_image),
            "--pose_json",
            str(pose_json_path),
            "--output_path",
            str(output_path),
            "--prompt",
            prompt_text,
            "--model_path",
            str(model_path),
            "--action_ckpt",
            str(action_ckpt),
        ]
        extend_command_with_options(
            command,
            payload=runner_payload,
            value_options={
                "negative_prompt": str,
                "resolution": str,
                "aspect_ratio": str,
                "num_inference_steps": int,
                "video_length": int,
                "seed": int,
                "model_type": str,
                "width": int,
                "height": int,
                "dtype": str,
                "launcher": str,
                "nproc_per_node": int,
                "master_port": int,
                "gpu_index": int,
                "cuda_visible_devices": str,
                "sage_blocks_range": str,
                "quant_type": str,
                "include_patterns": str,
            },
            bool_value_options=(
                "sr",
                "save_pre_sr_video",
                "rewrite",
                "offloading",
                "group_offloading",
                "enable_torch_compile",
                "few_step",
                "with_ui",
                "use_sageattn",
                "use_vae_parallel",
                "use_fp8_gemm",
                "transformer_resident_ar_rollout",
            ),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"HY-WorldPlay output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
            "pose_json_path": str(pose_json_path),
            "pose_source": pose_source,
            "video_length": video_length,
        }


__all__ = [
    "HYWorldPlayAdapter",
    "_resolved_pose_json_path",
    "_validated_video_length",
]
