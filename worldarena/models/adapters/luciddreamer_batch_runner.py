"""Batch subprocess runner for LucidDreamer inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterator

import numpy as np

from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent LucidDreamer batch runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--requests_jsonl", required=True, type=str)
    parser.add_argument("--results_jsonl", required=True, type=str)
    return parser.parse_args()


def _load_requests(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not payload.get("sample_id"):
                raise ValueError(f"missing sample_id in {path}:{line_number}")
            records.append(payload)
    return records


@contextlib.contextmanager
def _sample_work_dir(keep_work_dir: bool) -> Iterator[Path]:
    if keep_work_dir:
        path = Path(tempfile.mkdtemp(prefix="worldarena_luciddreamer_"))
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="worldarena_luciddreamer_") as temp_dir_raw:
        yield Path(temp_dir_raw)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _probe_duration_seconds(path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        text = completed.stdout.strip()
        if text:
            return float(text)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass

    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        capture.release()
        if fps > 0.0 and frame_count > 0.0:
            return frame_count / fps
    except Exception:
        return None
    return None


def _ensure_min_duration(path: Path, min_seconds: float) -> float | None:
    if min_seconds <= 0:
        return _probe_duration_seconds(path)
    duration = _probe_duration_seconds(path)
    if duration is None or duration >= min_seconds - 0.05:
        return duration
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"video duration {duration:.3f}s is shorter than required {min_seconds:.3f}s, "
            "and ffmpeg is not available to extend it"
        )

    temp_path = path.with_name(f"{path.stem}.min_duration_tmp{path.suffix}")
    command = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(path),
        "-t",
        f"{min_seconds:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    subprocess.run(command, check=True)
    shutil.move(str(temp_path), str(path))
    return _probe_duration_seconds(path)


def _prepare_repo(repo_root: Path) -> None:
    entrypoint = repo_root / "run.py"
    if not entrypoint.exists():
        raise FileNotFoundError(f"LucidDreamer entrypoint not found: {entrypoint}")
    os.chdir(repo_root)
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)


def _patch_diffusers_local_paths(repo_root: Path) -> None:
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
    )

    aliases = {
        "runwayml/stable-diffusion-inpainting": repo_root / "stablediffusion" / "SD1-5",
        "stablediffusion/SD1-5": repo_root / "stablediffusion" / "SD1-5",
        "./stablediffusion/SD1-5": repo_root / "stablediffusion" / "SD1-5",
        "lllyasviel/control_v11p_sd15_inpaint": repo_root / "stablediffusion" / "control_v11p_sd15_inpaint",
    }

    def _wrap(original):
        def _from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            key = str(pretrained_model_name_or_path)
            local_path = aliases.get(key)
            if local_path is not None and local_path.exists():
                pretrained_model_name_or_path = str(local_path)
                kwargs.pop("revision", None)
                kwargs.setdefault("use_safetensors", True)
                kwargs.setdefault("variant", "fp16")
            return original(pretrained_model_name_or_path, *args, **kwargs)

        return classmethod(_from_pretrained)

    for cls in (
        ControlNetModel,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
    ):
        cls.from_pretrained = _wrap(cls.from_pretrained)


def _new_gaussian_model(lucid: Any) -> Any:
    from scene import GaussianModel

    return GaussianModel(lucid.opt.sh_degree)


def _reset_per_sample_state(lucid: Any) -> None:
    lucid.gaussians = _new_gaussian_model(lucid)
    for attr in ("scene", "traindata"):
        if hasattr(lucid, attr):
            delattr(lucid, attr)


def _worldarena_camera_preset_data(
    camera_path: list[str],
    *,
    frame_count: int,
    translation_scale: float,
) -> dict[str, dict[str, list[dict[str, list[list[float]]]]]]:
    poses = synthetic_camera_matrices(
        camera_path,
        target_frames=max(int(frame_count), 1),
    ).matrices.copy()
    poses[:, :3, 3] *= float(translation_scale)
    frames: list[dict[str, list[list[float]]]] = []
    for pose in poses:
        # LucidDreamer's loader expects Blender axes and converts them back to
        # its internal COLMAP camera convention.
        blender_c2w = np.asarray(pose, dtype=np.float32).copy()
        blender_c2w[:3, 1:3] *= -1
        frames.append({"transform_matrix": blender_c2w.tolist()})
    return {"worldarena_camera_path": {"frames": frames}}


def _install_worldarena_camera_path(
    lucid: Any,
    camera_path: list[str],
    *,
    frame_count: int,
    translation_scale: float,
) -> None:
    from scene.dataset_readers import loadCameraPreset

    preset_data = _worldarena_camera_preset_data(
        camera_path,
        frame_count=frame_count,
        translation_scale=translation_scale,
    )
    loaded = loadCameraPreset(lucid.traindata, presetdata=preset_data)
    lucid.scene.preset_cameras["worldarena_camera_path"] = loaded[
        "worldarena_camera_path"
    ]


def _copy_outputs(
    *,
    save_dir: Path,
    output_path: Path,
    campath_render: str,
    min_seconds: float,
) -> dict[str, Any]:
    generated_video = save_dir / f"{campath_render}.mp4"
    generated_depth_video = save_dir / f"depth_{campath_render}.mp4"
    generated_ply = save_dir / "gsplat.ply"
    if not generated_video.exists():
        raise FileNotFoundError(f"LucidDreamer output video was not written: {generated_video}")
    if not generated_ply.exists():
        raise FileNotFoundError(f"LucidDreamer Gaussian output was not written: {generated_ply}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    shutil.copy2(generated_video, output_path)
    duration = _ensure_min_duration(output_path, min_seconds)

    ply_output = output_path.with_suffix(".ply")
    ply_output.unlink(missing_ok=True)
    shutil.copy2(generated_ply, ply_output)

    payload: dict[str, Any] = {
        "prediction_path": str(output_path),
        "gaussian_splat_path": str(ply_output),
    }
    if duration is not None:
        payload["video_duration_seconds"] = round(float(duration), 6)

    if generated_depth_video.exists():
        depth_output = output_path.with_name(f"{output_path.stem}.depth{output_path.suffix}")
        depth_output.unlink(missing_ok=True)
        shutil.copy2(generated_depth_video, depth_output)
        depth_duration = _ensure_min_duration(depth_output, min_seconds)
        payload["depth_video_path"] = str(depth_output)
        if depth_duration is not None:
            payload["depth_video_duration_seconds"] = round(float(depth_duration), 6)
    return payload


def _generate_one(lucid: Any, item: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image, ImageOps

    started_at = time.perf_counter()
    sample_id = str(item["sample_id"])
    output_path = Path(str(item["output_path"])).expanduser().resolve()
    conditioning_image = Path(str(item["conditioning_image"])).expanduser().resolve()
    if not conditioning_image.exists():
        raise FileNotFoundError(f"conditioning image not found: {conditioning_image}")

    keep_work_dir = bool(item.get("keep_work_dir", False))
    campath_gen = str(item.get("campath_gen", "lookdown"))
    campath_render = str(item.get("campath_render", "llff"))
    camera_path = item.get("camera_path") or []
    if not isinstance(camera_path, list):
        camera_path = [str(camera_path)]
    camera_path = [str(token) for token in camera_path]
    camera_path_primary = item.get("camera_path_primary")
    campath_render_source = str(item.get("campath_render_source", "config"))
    prompt = str(item.get("prompt", ""))
    negative_prompt = str(item.get("negative_prompt", ""))
    seed = int(item.get("seed", 1))
    diff_steps = int(item.get("diff_steps", 50))
    input_size = int(item.get("input_size", 512))
    model_name = _optional_text(item.get("model_name"))
    min_seconds = float(item.get("min_seconds", 5.0))
    camera_path_frames = int(item.get("camera_path_frames", 300))
    camera_path_translation_scale = float(item.get("camera_path_translation_scale", 1.0))

    with _sample_work_dir(keep_work_dir) as work_dir:
        save_dir = work_dir / "outputs"
        save_dir.mkdir(parents=True, exist_ok=True)
        lucid.save_dir = str(save_dir)
        _reset_per_sample_state(lucid)
        rgb_cond = Image.open(conditioning_image).convert("RGB")
        if input_size > 0:
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            rgb_cond = ImageOps.fit(rgb_cond, (input_size, input_size), method=resample)
        lucid.create(
            rgb_cond,
            prompt,
            negative_prompt,
            campath_gen,
            seed,
            diff_steps,
            model_name=model_name,
        )
        if campath_render_source == "worldarena_camera_path":
            _install_worldarena_camera_path(
                lucid,
                camera_path,
                frame_count=camera_path_frames,
                translation_scale=camera_path_translation_scale,
            )
        lucid.render_video(campath_render)
        payload = _copy_outputs(
            save_dir=save_dir,
            output_path=output_path,
            campath_render=campath_render,
            min_seconds=min_seconds,
        )
        if keep_work_dir:
            payload["work_dir"] = str(work_dir)

    payload.update(
        {
            "sample_id": sample_id,
            "status": "generated",
            "prompt": prompt,
            "camera_path": camera_path,
            "camera_path_primary": camera_path_primary,
            "campath_render_source": campath_render_source,
            "campath_gen": campath_gen,
            "campath_render": campath_render,
            "seed": seed,
            "diff_steps": diff_steps,
            "input_size": input_size,
            "model_name": model_name,
            "generation_wall_time_seconds": round(time.perf_counter() - started_at, 6),
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    requests_path = Path(args.requests_jsonl).expanduser().resolve()
    results_path = Path(args.results_jsonl).expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    _prepare_repo(repo_root)
    requests = _load_requests(requests_path)
    _patch_diffusers_local_paths(repo_root)

    from luciddreamer import LucidDreamer

    print(f"[luciddreamer-batch] loading LucidDreamer once for {len(requests)} requests", flush=True)
    lucid = LucidDreamer(for_gradio=False, save_dir=None)
    print("[luciddreamer-batch] LucidDreamer loaded", flush=True)

    with results_path.open("w", encoding="utf-8") as results_file:
        for item in requests:
            sample_id = str(item.get("sample_id", ""))
            try:
                payload = _generate_one(lucid, item)
            except Exception as exc:
                payload = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "prediction_path": str(item.get("output_path", "")),
                    "prompt": str(item.get("prompt", "")),
                }
            results_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            results_file.flush()


if __name__ == "__main__":
    main()
