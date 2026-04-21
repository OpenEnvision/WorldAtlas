from __future__ import annotations

from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import shutil
from types import ModuleType
import urllib.request
from typing import Any

import cv2
import numpy as np


PULSE_OF_MOTION_ENV = "WORLDARENA_PULSE_OF_MOTION_ROOT"
PULSE_OF_MOTION_REPO_ID = "xiangbog/Visual_Chronometer"
PULSE_OF_MOTION_CHECKPOINT = "vc_common_10_60fps.ckpt"
PULSE_OF_MOTION_CHECKPOINT_URL = (
    f"https://huggingface.co/{PULSE_OF_MOTION_REPO_ID}/resolve/main/{PULSE_OF_MOTION_CHECKPOINT}"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_pulse_of_motion_root() -> Path:
    env_root = os.environ.get(PULSE_OF_MOTION_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_project_root() / "third_party" / "Pulse-of-Motion").resolve()


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def inspect_pulse_of_motion_layout() -> dict[str, Any]:
    root = resolve_pulse_of_motion_root()
    predict_path = root / "inference" / "predict.py"
    config_path = root / "inference" / "configs" / "config_fps.yaml"
    checkpoint_path = root / "inference" / "ckpts" / PULSE_OF_MOTION_CHECKPOINT
    source_ready = predict_path.exists() and config_path.exists()
    checkpoint_present = checkpoint_path.exists()
    download_supported = True
    ready = source_ready and (checkpoint_present or download_supported)
    error = None
    if not source_ready:
        error = "Pulse-of-Motion checkout is missing predict.py or config_fps.yaml"
    return {
        "root": str(root),
        "predict_path": str(predict_path),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "source_ready": source_ready,
        "checkpoint_present": checkpoint_present,
        "download_supported": download_supported,
        "ready": ready,
        "error": error,
    }


def _module_name(root: Path) -> str:
    token = str(root).replace("/", "_").replace("-", "_")
    return f"_worldarena_pulse_predict_{token}"


@lru_cache(maxsize=2)
def _load_predict_module(root: str) -> ModuleType:
    pulse_root = Path(root)
    predict_path = pulse_root / "inference" / "predict.py"
    if not predict_path.exists():
        raise FileNotFoundError(f"Pulse-of-Motion predict.py not found: {predict_path}")
    spec = importlib.util.spec_from_file_location(_module_name(pulse_root), predict_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load Pulse-of-Motion module from: {predict_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_checkpoint(root: Path) -> Path:
    checkpoint_path = root / "inference" / "ckpts" / PULSE_OF_MOTION_CHECKPOINT
    if checkpoint_path.exists():
        return checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(".ckpt.part")
    request = urllib.request.Request(
        PULSE_OF_MOTION_CHECKPOINT_URL,
        headers={"User-Agent": "WorldArena/benchmark"},
    )
    try:
        with urllib.request.urlopen(request) as response, temporary_path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary_path.replace(checkpoint_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return checkpoint_path


def _default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load_model(root: str, device: str):
    module = _load_predict_module(root)
    pulse_root = Path(root)
    config_path = pulse_root / "inference" / "configs" / "config_fps.yaml"
    ckpt_path = _ensure_checkpoint(pulse_root)
    return module.load_model(str(config_path), str(ckpt_path), device)


def _container_fps(video_path: str) -> float | None:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    return fps if fps > 0 else None


def compute_pulse_metrics(
    video_path: str,
    *,
    runtime: dict[str, Any] | None = None,
    fallback_meta_fps: float | None = None,
) -> dict[str, Any]:
    runtime = dict(runtime or {})
    layout = inspect_pulse_of_motion_layout()
    if not layout["ready"]:
        raise RuntimeError(str(layout["error"]))

    root = str(resolve_pulse_of_motion_root())
    device = str(runtime.get("device") or _default_device())
    clip_length = int(runtime.get("clip_length", 30))
    stride = int(runtime.get("stride", 4))
    resolution = int(runtime.get("resolution", 216))

    module = _load_predict_module(root)
    model = _load_model(root, device)
    segments, avg_phyfps, total_frames = module.predict_video(
        model,
        video_path,
        device,
        clip_length=clip_length,
        stride=stride,
        resolution=resolution,
    )
    segment_phyfps = [float(item["predicted_phyfps"]) for item in segments]
    container_fps = _container_fps(video_path)
    meta_fps = container_fps if container_fps is not None else fallback_meta_fps
    meta_fps_source = "prediction_container_fps" if container_fps is not None else "fallback"
    if meta_fps is None or meta_fps <= 0:
        raise RuntimeError("meta fps is unavailable for Pulse-of-Motion")

    avg_phyfps = float(avg_phyfps)
    abs_error = abs(avg_phyfps - float(meta_fps))
    pct_error = abs_error / max(float(meta_fps), 1e-6)
    intra_video_cv = float(np.std(segment_phyfps) / max(np.mean(segment_phyfps), 1e-6))
    return {
        "backend": "pulse_of_motion",
        "metrics": {
            "physical_fps_error": float(abs_error),
            "physical_fps_pct_error": float(pct_error),
            "physical_fps_intra_video_cv": float(intra_video_cv),
        },
        "details": {
            "avg_phyfps": avg_phyfps,
            "segment_phyfps": [round(value, 4) for value in segment_phyfps],
            "segment_count": len(segment_phyfps),
            "total_frames": int(total_frames),
            "meta_fps": float(meta_fps),
            "meta_fps_source": meta_fps_source,
            "device": device,
            "clip_length": clip_length,
            "stride": stride,
            "resolution": resolution,
        },
    }


__all__ = [
    "PULSE_OF_MOTION_ENV",
    "compute_pulse_metrics",
    "inspect_pulse_of_motion_layout",
    "resolve_pulse_of_motion_root",
]
