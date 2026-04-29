from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.checkpoints import filter_checkpoint_env_overrides
from worldarena.common.checkpoints import project_root as worldarena_project_root
from worldarena.common.checkpoints import resolve_project_path
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import extend_command_with_options, run_command
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample


_MOTIONCTRL_IMAGE_CONTRACT = "text_camera_only_no_image_conditioning"


def _rotation_matrix_y_motionctrl(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float32,
    )


def _rt_from_motionctrl_state(x: float, z: float, yaw_degrees: float) -> np.ndarray:
    rt = np.zeros((3, 4), dtype=np.float32)
    rt[:, :3] = _rotation_matrix_y_motionctrl(yaw_degrees)
    rt[:, 3] = np.asarray([x, 0.0, z], dtype=np.float32)
    return rt


def _motionctrl_camera_delta(token: str, generation: dict[str, Any]) -> np.ndarray:
    token = str(token or "").strip().lower().replace("-", "_").replace(" ", "_")
    forward = float(generation.get("translation_step", 1.5))
    lateral = float(generation.get("lateral_step", 1.0))
    degrees = float(generation.get("rotation_degrees", 30.0))
    if not token or token == "fixed":
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    if token == "push_in":
        return np.asarray([0.0, -forward, 0.0], dtype=np.float32)
    if token == "pull_out":
        return np.asarray([0.0, forward, 0.0], dtype=np.float32)
    if token == "move_left":
        return np.asarray([lateral, 0.0, 0.0], dtype=np.float32)
    if token == "move_right":
        return np.asarray([-lateral, 0.0, 0.0], dtype=np.float32)
    if token == "pan_left":
        return np.asarray([lateral, forward, degrees], dtype=np.float32)
    if token == "pan_right":
        return np.asarray([-lateral, forward, -degrees], dtype=np.float32)
    if token == "orbit_left":
        return np.asarray([lateral, -forward, -degrees], dtype=np.float32)
    if token == "orbit_right":
        return np.asarray([-lateral, -forward, degrees], dtype=np.float32)
    raise ValueError(f"Unsupported MotionCtrl camera token: {token}")


def _motionctrl_camera_pose_sequence(
    camera_path: list[str],
    *,
    frame_count: int,
    generation: dict[str, Any],
) -> np.ndarray:
    target_frames = max(int(frame_count), 1)
    normalized_path = [
        str(token).strip().lower().replace("-", "_").replace(" ", "_")
        for token in camera_path
        if str(token).strip()
    ] or ["fixed"]
    if target_frames == 1:
        return _rt_from_motionctrl_state(0.0, 0.0, 0.0).reshape(1, 12)

    keyframes = [np.asarray([0.0, 0.0, 0.0], dtype=np.float32)]
    current = keyframes[0].copy()
    for token in normalized_path:
        current = current + _motionctrl_camera_delta(token, generation)
        keyframes.append(current.copy())

    remaining_intervals = target_frames - 1
    segment_count = max(1, len(normalized_path))
    base_intervals = remaining_intervals // segment_count
    remainder = remaining_intervals % segment_count
    states: list[np.ndarray] = []
    for index in range(segment_count):
        segment_intervals = base_intervals + (1 if index < remainder else 0)
        segment_frames = segment_intervals + 1
        alpha = np.linspace(0.0, 1.0, segment_frames, dtype=np.float32)[:, None]
        segment = keyframes[index][None, :] * (1.0 - alpha) + keyframes[index + 1][None, :] * alpha
        if index:
            segment = segment[1:]
        states.append(segment.astype(np.float32))
    state_sequence = np.concatenate(states, axis=0).astype(np.float32)
    if len(state_sequence) != target_frames:
        raise ValueError(f"expected {target_frames} MotionCtrl poses, got {len(state_sequence)}")
    return np.stack(
        [_rt_from_motionctrl_state(float(x), float(z), float(yaw)).reshape(12) for x, z, yaw in state_sequence],
        axis=0,
    ).astype(np.float32)


def _write_motionctrl_camera_pose(
    path: Path,
    *,
    camera_path: list[str],
    frame_count: int,
    generation: dict[str, Any],
) -> None:
    poses = _motionctrl_camera_pose_sequence(
        camera_path,
        frame_count=frame_count,
        generation=generation,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(poses.astype(float).tolist(), indent=2),
        encoding="utf-8",
    )


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
        raise FileNotFoundError(f"MotionCtrl checkpoint file not found: {path}")
    return path.resolve()


def _build_motionctrl_env(*, extra_pythonpaths: list[Path], overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    pythonpaths = [str(path) for path in extra_pythonpaths if str(path)]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    if pythonpaths:
        env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if overrides:
        env.update(filter_checkpoint_env_overrides(overrides))
    return env


def _request_payload_for(
    request: PreparedGenerationRequest,
    *,
    generation: dict[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    camera_path = _camera_path_for_sample(request.sample) or ["fixed"]
    frame_count = int(generation.get("frames", generation.get("video_length", 16)))
    key_payload = {
        "sample_id": request.sample.sample_id,
        "prediction_stem": request.sample.prediction_stem,
        "camera_path": camera_path,
        "frame_count": frame_count,
        "translation_step": generation.get("translation_step"),
        "lateral_step": generation.get("lateral_step"),
        "rotation_degrees": generation.get("rotation_degrees"),
    }
    key = hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    pose_file = runtime_root / "camera_poses" / f"{request.sample.prediction_stem}--{key}.json"
    if not pose_file.exists():
        _write_motionctrl_camera_pose(
            pose_file,
            camera_path=camera_path,
            frame_count=frame_count,
            generation=generation,
        )
    return {
        "sample_id": request.sample.sample_id,
        "sample_name": request.output_path.stem,
        "suite": request.sample.suite,
        "prompt": request.prompt,
        "prompt_source": "prompt_target",
        "prompt_current": request.sample.prompt_current,
        "prompt_target": request.sample.prompt_target,
        "output_path": str(request.output_path),
        "camera_path": camera_path,
        "camera_pose_file": str(pose_file),
        "control_source": "camera_pose_json",
        "image_conditioning_contract": _MOTIONCTRL_IMAGE_CONTRACT,
    }


class MotionCtrlAdapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

    def _runtime_root(self) -> Path:
        configured = self.config.generation.get("runtime_cache_dir")
        if configured is not None:
            root = _path_from_generation(configured)
            if root is None:
                raise ValueError("invalid MotionCtrl runtime_cache_dir")
        else:
            root = worldarena_project_root() / "cache" / "model_runtime" / "motionctrl"
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("MotionCtrl adapter requires repo_root")

        generation = dict(self.config.generation)
        project_root = worldarena_project_root()
        runtime_root = self._runtime_root()
        for request in requests:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_base = self.config.checkpoint_dir or repo_root
        ckpt_path = _resolve_checkpoint_file(
            checkpoint_base,
            generation.get("checkpoint") or generation.get("ckpt_path"),
            default="checkpoints/motionctrl.pth",
        )
        config_path = _path_from_generation(generation.get("config_path") or generation.get("base"), base=repo_root)
        if config_path is None or not config_path.is_file():
            raise FileNotFoundError(f"MotionCtrl config not found: {config_path}")
        openclip_pretrained_path = _path_from_generation(generation.get("openclip_pretrained_path"))

        request_payloads = [
            _request_payload_for(request, generation=generation, runtime_root=runtime_root)
            for request in requests
        ]
        with TemporaryDirectory(prefix="worldarena_motionctrl_") as temp_dir_raw:
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
                "worldarena.models.adapters.motionctrl_runner",
                "--repo-root",
                str(repo_root),
                "--request-file",
                str(request_file),
                "--results-file",
                str(results_file),
                "--config-path",
                str(config_path),
                "--ckpt-path",
                str(ckpt_path),
            ]
            extend_command_with_options(
                command,
                payload=generation,
                value_options={
                    "height": int,
                    "width": int,
                    "fps": float,
                    "frames": int,
                    "n_samples": int,
                    "ddim_steps": int,
                    "ddim_eta": float,
                    "unconditional_guidance_scale": float,
                    "unconditional_guidance_scale_temporal": float,
                    "cond_T": int,
                    "seed": int,
                    "device": str,
                },
            )
            if openclip_pretrained_path is not None:
                command.extend(["--openclip-pretrained-path", str(openclip_pretrained_path)])

            env = _build_motionctrl_env(extra_pythonpaths=[project_root], overrides=self.config.env)
            run_command(command, cwd=project_root, env=env)
            if not results_file.exists():
                raise FileNotFoundError(f"MotionCtrl runner did not write results: {results_file}")
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
            raise RuntimeError(f"MotionCtrl runner returned no result for sample {sample.sample_id}")
        if payload.get("status") == "failed":
            raise RuntimeError(str(payload.get("error", "MotionCtrl generation failed")))
        return payload


__all__ = [
    "MotionCtrlAdapter",
    "_MOTIONCTRL_IMAGE_CONTRACT",
    "_motionctrl_camera_delta",
    "_motionctrl_camera_pose_sequence",
    "_request_payload_for",
    "_write_motionctrl_camera_pose",
]
