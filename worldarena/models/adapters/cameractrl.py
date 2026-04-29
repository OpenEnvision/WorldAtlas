from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from worldarena.benchmark.annotations import (
    _matrix_to_pose_vector,
    _pose_vector_to_matrix,
    _resample_pose_vectors,
)
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.checkpoints import project_root as worldarena_project_root
from worldarena.common.checkpoints import resolve_project_path
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, extend_command_with_options, run_command
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample


def _rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [
            [cosine, 0.0, -sine],
            [0.0, 1.0, 0.0],
            [sine, 0.0, cosine],
        ],
        dtype=np.float32,
    )


def _rigid_transform(
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    if rotation is not None:
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float32)
    if translation is not None:
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def _cameractrl_camera_delta(token: str, generation: dict[str, Any]) -> np.ndarray:
    token = str(token or "").strip().lower()
    forward = float(generation.get("translation_step", 1.0))
    lateral = float(generation.get("lateral_step", 0.5))
    degrees = float(generation.get("rotation_degrees", 30.0))
    if not token or token == "fixed":
        return np.eye(4, dtype=np.float32)
    if token == "push_in":
        return _rigid_transform(translation=np.asarray([0.0, 0.0, -forward], dtype=np.float32))
    if token == "pull_out":
        return _rigid_transform(translation=np.asarray([0.0, 0.0, forward], dtype=np.float32))
    if token == "move_left":
        return _rigid_transform(translation=np.asarray([-lateral, 0.0, 0.0], dtype=np.float32))
    if token == "move_right":
        return _rigid_transform(translation=np.asarray([lateral, 0.0, 0.0], dtype=np.float32))
    if token == "pan_left":
        return _rigid_transform(
            rotation=_rotation_matrix_y(degrees),
            translation=np.asarray([-lateral, 0.0, forward], dtype=np.float32),
        )
    if token == "pan_right":
        return _rigid_transform(
            rotation=_rotation_matrix_y(-degrees),
            translation=np.asarray([lateral, 0.0, forward], dtype=np.float32),
        )
    if token == "orbit_left":
        return _rigid_transform(
            rotation=_rotation_matrix_y(-degrees),
            translation=np.asarray([-lateral, 0.0, -forward], dtype=np.float32),
        )
    if token == "orbit_right":
        return _rigid_transform(
            rotation=_rotation_matrix_y(degrees),
            translation=np.asarray([lateral, 0.0, -forward], dtype=np.float32),
        )
    return np.eye(4, dtype=np.float32)


def _interpolate_pose_segment(start_pose: np.ndarray, end_pose: np.ndarray, *, frames: int) -> np.ndarray:
    pose_vectors = np.stack(
        [
            _matrix_to_pose_vector(start_pose.astype(np.float32)),
            _matrix_to_pose_vector(end_pose.astype(np.float32)),
        ],
        axis=0,
    ).astype(np.float32)
    interpolated = _resample_pose_vectors(
        pose_vectors,
        np.asarray([0.0, 1.0], dtype=np.float32),
        frames,
    )
    return np.stack([_pose_vector_to_matrix(row) for row in interpolated], axis=0).astype(np.float32)


def _synthetic_c2w_poses(
    camera_path: list[str],
    *,
    frame_count: int,
    generation: dict[str, Any],
) -> np.ndarray:
    target_frames = max(int(frame_count), 1)
    if target_frames <= 1:
        return np.repeat(np.eye(4, dtype=np.float32)[None], repeats=target_frames, axis=0)

    normalized_path = [str(token).strip().lower() for token in camera_path if str(token).strip()] or ["fixed"]
    keyframes: list[np.ndarray] = [np.eye(4, dtype=np.float32)]
    current_pose = np.eye(4, dtype=np.float32)
    for token in normalized_path:
        current_pose = (current_pose @ _cameractrl_camera_delta(token, generation)).astype(np.float32)
        keyframes.append(current_pose.copy())

    remaining_intervals = target_frames - 1
    segment_count = max(1, len(normalized_path))
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
        segments.append(segment)
    return np.concatenate(segments, axis=0).astype(np.float32)


def _write_cameractrl_trajectory(
    path: Path,
    *,
    camera_path: list[str],
    frame_count: int,
    image_width: int,
    image_height: int,
    generation: dict[str, Any],
) -> None:
    poses = _synthetic_c2w_poses(camera_path, frame_count=frame_count, generation=generation)
    focal = float(max(image_width, image_height)) * float(generation.get("synthetic_focal_scale", 0.65))
    fx = focal / float(image_width)
    fy = focal / float(image_height)
    lines = ["worldarena://synthetic-camera-trajectory"]
    for index, c2w in enumerate(poses):
        w2c = np.linalg.inv(c2w).astype(np.float32)[:3, :]
        row = [
            float(index),
            fx,
            fy,
            0.5,
            0.5,
            0.0,
            0.0,
            *w2c.reshape(-1).tolist(),
        ]
        lines.append(" ".join(f"{value:.9f}" for value in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_from_generation(config_value: object, *, base: Path | None = None) -> Path | None:
    if config_value is None:
        return None
    text = str(config_value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    if base is not None:
        return (base / path).resolve()
    return resolve_project_path(text)


def _resolve_checkpoint_file(checkpoint_dir: Path, value: object, *, default: str) -> Path:
    configured = _path_from_generation(value, base=checkpoint_dir)
    path = configured or checkpoint_dir / default
    if not path.is_file():
        raise FileNotFoundError(f"CameraCtrl checkpoint file not found: {path}")
    return path.resolve()


def _request_payload_for(
    request: PreparedGenerationRequest,
    *,
    generation: dict[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    camera_path = _camera_path_for_sample(request.sample) or ["fixed"]
    frame_count = int(generation.get("video_length", 16))
    image_width = int(generation.get("image_width", 384))
    image_height = int(generation.get("image_height", 256))
    key_payload = {
        "sample_id": request.sample.sample_id,
        "prediction_stem": request.sample.prediction_stem,
        "camera_path": camera_path,
        "frame_count": frame_count,
        "image_width": image_width,
        "image_height": image_height,
        "generation": {
            key: generation.get(key)
            for key in ("synthetic_focal_scale", "translation_step", "lateral_step", "rotation_degrees")
        },
    }
    key = hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    trajectory_file = runtime_root / "trajectories" / f"{request.sample.prediction_stem}--{key}.txt"
    if not trajectory_file.exists():
        _write_cameractrl_trajectory(
            trajectory_file,
            camera_path=camera_path,
            frame_count=frame_count,
            image_width=image_width,
            image_height=image_height,
            generation=generation,
        )
    return {
        "sample_id": request.sample.sample_id,
        "sample_name": request.output_path.stem,
        "prompt": request.prompt,
        "negative_prompt": generation.get("negative_prompt"),
        "trajectory_file": str(trajectory_file),
        "output_path": str(request.output_path),
        "camera_path": camera_path,
    }


class CameraCtrlAdapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("CameraCtrl adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("CameraCtrl adapter requires checkpoint_dir")

        generation = dict(self.config.generation)
        project_root = worldarena_project_root()
        runtime_root = project_root / "cache" / "model_runtime" / "cameractrl"
        runtime_root.mkdir(parents=True, exist_ok=True)
        for request in requests:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)

        pose_adaptor_ckpt = _resolve_checkpoint_file(
            self.config.checkpoint_dir,
            generation.get("pose_adaptor_ckpt"),
            default="CameraCtrl.ckpt",
        )
        image_lora_ckpt = _path_from_generation(generation.get("image_lora_ckpt"), base=self.config.checkpoint_dir)
        model_config = _path_from_generation(generation.get("model_config"), base=repo_root)
        if model_config is None or not model_config.is_file():
            raise FileNotFoundError(f"CameraCtrl model_config not found: {model_config}")
        ori_model_path = _path_from_generation(generation.get("ori_model_path"))
        if ori_model_path is None or not ori_model_path.exists():
            raise FileNotFoundError(f"CameraCtrl SD1.5 model path not found: {ori_model_path}")
        motion_module_ckpt = _path_from_generation(generation.get("motion_module_ckpt"))
        if motion_module_ckpt is None or not motion_module_ckpt.is_file():
            raise FileNotFoundError(f"CameraCtrl motion module checkpoint not found: {motion_module_ckpt}")

        request_payloads = [
            _request_payload_for(request, generation=generation, runtime_root=runtime_root)
            for request in requests
        ]
        with TemporaryDirectory(prefix="worldarena_cameractrl_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            request_file = temp_dir / "requests.json"
            results_file = temp_dir / "results.json"
            request_file.write_text(
                json.dumps({"requests": request_payloads}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            command = [
                self.config.python_bin,
                "-m",
                "worldarena.models.adapters.cameractrl_runner",
                "--repo_root",
                str(repo_root),
                "--request_file",
                str(request_file),
                "--results_file",
                str(results_file),
                "--ori_model_path",
                str(ori_model_path),
                "--unet_subfolder",
                str(generation.get("unet_subfolder", "unet")),
                "--motion_module_ckpt",
                str(motion_module_ckpt),
                "--pose_adaptor_ckpt",
                str(pose_adaptor_ckpt),
                "--model_config",
                str(model_config),
            ]
            extend_command_with_options(
                command,
                payload=generation,
                value_options={
                    "image_lora_rank": int,
                    "image_height": int,
                    "image_width": int,
                    "video_length": int,
                    "original_pose_width": int,
                    "original_pose_height": int,
                    "target_duration_sec": float,
                    "num_inference_steps": int,
                    "guidance_scale": float,
                    "seed": int,
                    "device": str,
                },
            )
            if image_lora_ckpt is not None:
                command.extend(["--image_lora_ckpt", str(image_lora_ckpt)])
            personalized_base_model = _path_from_generation(generation.get("personalized_base_model"))
            if personalized_base_model is not None:
                command.extend(["--personalized_base_model", str(personalized_base_model)])

            env = build_env(extra_pythonpaths=[project_root], overrides=self.config.env)
            run_command(command, cwd=project_root, env=env)
            if not results_file.exists():
                raise FileNotFoundError(f"CameraCtrl runner did not write results: {results_file}")
            raw_results = json.loads(results_file.read_text(encoding="utf-8"))

        results: dict[str, dict[str, Any]] = {}
        for item in raw_results.get("results", []):
            sample_id = str(item.get("sample_id", ""))
            if sample_id:
                results[sample_id] = dict(item)
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        request = PreparedGenerationRequest(
            sample=sample,
            conditioning_image=conditioning_image,
            output_path=output_path,
            prompt=prompt,
        )
        payload = self.generate_batch([request]).get(sample.sample_id, {})
        if not payload:
            raise RuntimeError(f"CameraCtrl runner returned no result for sample {sample.sample_id}")
        if payload.get("status") == "failed":
            raise RuntimeError(str(payload.get("error", "CameraCtrl generation failed")))
        return payload


__all__ = [
    "CameraCtrlAdapter",
    "_cameractrl_camera_delta",
    "_synthetic_c2w_poses",
    "_write_cameractrl_trajectory",
]
