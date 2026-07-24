"""FlashWorld model adapter for WorldAtlas Arena.

FlashWorld generates camera-guided videos from a conditioning image. When GT poses
are unavailable, the adapter synthesizes trajectories from ``camera_path`` tokens.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import uuid

import numpy as np

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices
from worldarena.common.annotation_index import split_annotation_reference
from worldarena.common.media import probe_image
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, run_command
from worldarena.models.adapters.flashworld_runner import _build_request_payload, _resolve_checkpoint_path
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample
from worldarena.models.config import ModelRuntimeConfig


FLASHWORLD_DEFAULT_FOCAL_SCALE = 0.6


def _project_root(config: ModelRuntimeConfig) -> Path:
    config_path = Path(config.config_path)
    if config_path.parent.name == "models" and config_path.parent.parent.name == "config":
        return config_path.parent.parent.parent

    if config.repo_root is not None and config.repo_root.parent.name in {"thirdparty", "third_party"}:
        return config.repo_root.parent.parent

    if config.checkpoint_dir is not None and config.checkpoint_dir.parent.name == "ckpt":
        return config.checkpoint_dir.parent.parent

    return Path(__file__).parent.parent.parent.parent.absolute()


def _runtime_root(config: ModelRuntimeConfig) -> Path:
    root = _project_root(config) / "cache" / "model_runtime" / "flashworld"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _synthetic_intrinsics_sequence(
    frame_count: int,
    *,
    image_width: int,
    image_height: int,
    focal_scale: float = FLASHWORLD_DEFAULT_FOCAL_SCALE,
) -> np.ndarray:
    focal_length = float(max(image_width, image_height)) * float(focal_scale)
    intrinsics = np.asarray(
        [focal_length, focal_length, image_width / 2.0, image_height / 2.0],
        dtype=np.float32,
    )
    return np.repeat(intrinsics[None, :], repeats=max(int(frame_count), 1), axis=0).astype(np.float32)


def _synthetic_image_poses(camera_path: list[str], *, frame_count: int) -> np.ndarray:
    return synthetic_camera_matrices(
        camera_path,
        target_frames=max(int(frame_count), 1),
    ).matrices


def _synthetic_image_annotation_path(
    config: ModelRuntimeConfig,
    sample: BenchmarkSample,
    conditioning_image: Path,
    *,
    frame_count: int,
) -> Path:
    camera_path = _camera_path_for_sample(sample)
    if not camera_path:
        raise FileNotFoundError(
            f"FlashWorld image-static adaptation requires a camera_path-conditioned prompt contract for {sample.sample_id}"
        )

    image_probe = probe_image(conditioning_image)
    image_width = int(image_probe["width"])
    image_height = int(image_probe["height"])
    focal_scale = float(config.generation.get("synthetic_focal_scale", FLASHWORLD_DEFAULT_FOCAL_SCALE))
    cache_key_payload = {
        "sample_id": sample.sample_id,
        "relative_path": sample.relative_path,
        "annotation_path": sample.annotation_path,
        "camera_path": camera_path,
        "frame_count": int(frame_count),
        "image_width": image_width,
        "image_height": image_height,
        "focal_scale": focal_scale,
    }
    key = hashlib.sha1(
        json.dumps(cache_key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    annotation_root = _runtime_root(config) / "image_annotations" / f"{sample.prediction_stem}--{key}"
    poses_path = annotation_root / "poses.npy"
    intrinsics_path = annotation_root / "intrinsics.npy"
    if poses_path.exists() and intrinsics_path.exists():
        try:
            poses = np.load(poses_path)
            intrinsics = np.load(intrinsics_path)
            if poses.shape == (frame_count, 4, 4) and intrinsics.shape == (frame_count, 4):
                return annotation_root
        except Exception:
            pass

    annotation_root.mkdir(parents=True, exist_ok=True)
    poses = _synthetic_image_poses(camera_path, frame_count=frame_count)
    intrinsics = _synthetic_intrinsics_sequence(
        frame_count,
        image_width=image_width,
        image_height=image_height,
        focal_scale=focal_scale,
    )
    np.save(poses_path, poses.astype(np.float32))
    np.save(intrinsics_path, intrinsics.astype(np.float32))
    return annotation_root


def _resolved_annotation_path(
    config: ModelRuntimeConfig,
    sample: BenchmarkSample,
    conditioning_image: Path,
    *,
    frame_count: int,
) -> Path:
    if sample.annotation_path and split_annotation_reference(sample.annotation_path) is None:
        annotation_dir = Path(sample.annotation_path).expanduser().resolve()
        if (annotation_dir / "poses.npy").exists() and (annotation_dir / "intrinsics.npy").exists():
            return annotation_dir

    if sample.modality == "image" and (sample.suite == "image_static" or sample.generation_mode == "static"):
        return _synthetic_image_annotation_path(
            config,
            sample,
            conditioning_image,
            frame_count=frame_count,
        )

    raise FileNotFoundError(
        f"FlashWorld requires poses.npy/intrinsics.npy or an image_static sample with camera_path guidance for {sample.sample_id}"
    )


def _flashworld_cli_command(
    config: ModelRuntimeConfig,
    *,
    repo_root: Path,
    checkpoint_path: Path,
) -> list[str]:
    cli_path = (repo_root / "cli.py").resolve()
    if not cli_path.exists():
        raise FileNotFoundError(f"FlashWorld CLI entrypoint not found: {cli_path}")

    generation = config.generation
    command = [
        config.python_bin,
        str(cli_path),
        "--ckpt",
        str(checkpoint_path),
    ]
    for flag_name in (
        "offload_t5",
        "offload_vae",
        "offload_transformer_during_vae",
        "export_ply",
        "export_spz",
    ):
        if generation.get(flag_name, False):
            command.append(f"--{flag_name}")
    return command


def _subprocess_error_text(exc: subprocess.CalledProcessError) -> str:
    return f"exit code {exc.returncode}"


class FlashWorldAdapter(ModelAdapter):
    """Generate camera-guided videos via the FlashWorld upstream repo."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if self.config.repo_root is None:
            raise ValueError("FlashWorld adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("FlashWorld adapter requires checkpoint_dir")
        if not requests:
            return {}

        generation = self.config.generation
        project_root = _project_root(self.config)
        repo_root = self.config.repo_root.resolve()
        checkpoint_path = _resolve_checkpoint_path(str(self.config.checkpoint_dir))
        work_root = (_runtime_root(self.config) / "batch" / uuid.uuid4().hex).resolve()
        input_dir = work_root / "inputs"
        runtime_output_dir = work_root / "outputs"
        results: dict[str, dict[str, Any]] = {}
        runnable_requests: list[PreparedGenerationRequest] = []
        command = _flashworld_cli_command(
            self.config,
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
        )
        command.extend(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(runtime_output_dir),
                "--video",
                "--video_fps",
                str(int(generation.get("video_fps", 15))),
            ]
        )
        cli_error: str | None = None

        try:
            input_dir.mkdir(parents=True, exist_ok=True)
            runtime_output_dir.mkdir(parents=True, exist_ok=True)

            for request in requests:
                try:
                    annotation_path = _resolved_annotation_path(
                        self.config,
                        request.sample,
                        request.conditioning_image.expanduser().resolve(),
                        frame_count=int(generation.get("num_frames", 24)),
                    )
                    payload = _build_request_payload(
                        annotation_path=str(annotation_path),
                        conditioning_image=request.conditioning_image.expanduser().resolve(),
                        prompt=request.prompt,
                        num_frames=int(generation.get("num_frames", 24)),
                        image_height=int(generation.get("image_height", 480)),
                        image_width=int(generation.get("image_width", 704)),
                        image_index=int(generation.get("image_index", 0)),
                    )
                    request_path = input_dir / f"{request.sample.prediction_stem}.json"
                    request_path.write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    runnable_requests.append(request)
                except Exception as exc:
                    results[request.sample.sample_id] = {
                        "status": "failed",
                        "error": str(exc),
                        "prediction_path": str(request.output_path),
                        "prompt": request.prompt,
                    }

            if runnable_requests:
                env = build_env(
                    extra_pythonpaths=[project_root],
                    overrides=self.config.env,
                )
                try:
                    subprocess.run(
                        command,
                        check=True,
                        cwd=str(repo_root),
                        env=env,
                    )
                except subprocess.CalledProcessError as exc:
                    cli_error = _subprocess_error_text(exc)

                for request in runnable_requests:
                    generated_root = runtime_output_dir / request.sample.prediction_stem
                    generated_video = generated_root / "video.mp4"
                    if generated_video.exists():
                        request.output_path.parent.mkdir(parents=True, exist_ok=True)
                        request.output_path.unlink(missing_ok=True)
                        shutil.move(str(generated_video), str(request.output_path))
                        payload: dict[str, Any] = {
                            "status": "generated",
                            "command": command,
                            "prediction_path": str(request.output_path),
                            "prompt": request.prompt,
                        }
                        if generation.get("export_ply", False):
                            generated_ply = generated_root / "gaussians.ply"
                            if generated_ply.exists():
                                ply_path = request.output_path.with_suffix(".ply")
                                ply_path.unlink(missing_ok=True)
                                shutil.move(str(generated_ply), str(ply_path))
                                payload["ply_path"] = str(ply_path)
                        if generation.get("export_spz", False):
                            generated_spz = generated_root / "gaussians.spz"
                            if generated_spz.exists():
                                spz_path = request.output_path.with_suffix(".spz")
                                spz_path.unlink(missing_ok=True)
                                shutil.move(str(generated_spz), str(spz_path))
                                payload["spz_path"] = str(spz_path)
                        results[request.sample.sample_id] = payload
                    else:
                        results[request.sample.sample_id] = {
                            "status": "failed",
                            "error": cli_error
                            or f"FlashWorld output was not written: {generated_video}",
                            "prediction_path": str(request.output_path),
                            "prompt": request.prompt,
                        }
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if self.config.repo_root is None:
            raise ValueError("FlashWorld adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("FlashWorld adapter requires checkpoint_dir")

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = _project_root(self.config)
        frame_count = int(generation.get("num_frames", 24))
        annotation_path = _resolved_annotation_path(
            self.config,
            sample,
            conditioning_image.expanduser().resolve(),
            frame_count=frame_count,
        )
        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.flashworld_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--annotation_path",
            str(annotation_path),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--sample_name",
            sample.prediction_stem,
            "--prompt",
            prompt,
            "--num_frames",
            str(frame_count),
            "--image_height",
            str(int(generation.get("image_height", 480))),
            "--image_width",
            str(int(generation.get("image_width", 704))),
            "--image_index",
            str(int(generation.get("image_index", 0))),
            "--video_fps",
            str(int(generation.get("video_fps", 15))),
        ]
        command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])
        for flag_name in (
            "offload_t5",
            "offload_vae",
            "offload_transformer_during_vae",
            "export_ply",
            "export_spz",
        ):
            if generation.get(flag_name, False):
                command.append(f"--{flag_name}")

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"FlashWorld output was not written: {output_path}")

        result: dict[str, Any] = {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
        }
        if generation.get("export_ply", False):
            result["ply_path"] = str(output_path.with_suffix(".ply"))
        if generation.get("export_spz", False):
            result["spz_path"] = str(output_path.with_suffix(".spz"))
        return result


__all__ = [
    "FlashWorldAdapter",
    "_resolved_annotation_path",
    "_synthetic_image_annotation_path",
]
