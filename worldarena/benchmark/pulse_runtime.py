from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import ModuleType
import uuid
from typing import Any

import cv2
import numpy as np

from worldarena.common.checkpoints import (
    apply_checkpoint_env,
    checkpoint_path,
    filter_checkpoint_env_overrides,
    resolve_checkpoint_path,
)


PULSE_RUNTIME_WORKER_ENV = "WORLDARENA_PULSE_RUNTIME_WORKER"
PULSE_OF_MOTION_CHECKPOINT = "vc_common_10_60fps.ckpt"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_cache_dir(name: str) -> Path:
    path = _project_root() / ".cache" / "benchmark" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_PULSE_RUNTIME_PYTHONPATHS = (
    _project_root() / ".cache" / "pulse_runtime_overlay310",
    _project_root() / ".cache" / "pulse_runtime_overlay",
)


def resolve_pulse_of_motion_root() -> Path:
    return (_project_root() / "thirdparty" / "Pulse-of-Motion").resolve()


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _resolve_runtime_python_bin(runtime: dict[str, Any] | None) -> str | None:
    payload = dict(runtime or {})
    raw = str(payload.get("python_bin") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if raw.startswith(".") or "/" in raw:
        return str((_project_root() / path).absolute())
    return raw


def resolve_pulse_runtime_python(runtime: dict[str, Any] | None = None) -> str | None:
    return _resolve_runtime_python_bin(runtime)


def _normalize_runtime_pythonpath(value: Any) -> list[str]:
    entries: list[str] = []
    for raw_part in str(value).split(os.pathsep):
        part = raw_part.strip()
        if not part:
            continue
        path = Path(part).expanduser()
        if not path.is_absolute():
            path = _project_root() / path
        entries.append(str(path.resolve()))
    return entries


def _dedupe_paths(entries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)
    return deduped


def resolve_pulse_runtime_env(runtime: dict[str, Any] | None = None) -> dict[str, str]:
    env = apply_checkpoint_env(
        filter_checkpoint_env_overrides(dict((runtime or {}).get("env", {})))
    )
    pythonpath_entries = [str(_project_root())]
    explicit_pythonpath = env.pop("PYTHONPATH", None)
    if explicit_pythonpath:
        pythonpath_entries.extend(_normalize_runtime_pythonpath(explicit_pythonpath))
    for candidate in DEFAULT_PULSE_RUNTIME_PYTHONPATHS:
        if candidate.exists():
            pythonpath_entries.append(str(candidate.resolve()))
    env["PYTHONPATH"] = os.pathsep.join(_dedupe_paths(pythonpath_entries))
    return env


def _subprocess_runtime_env(runtime: dict[str, Any] | None) -> dict[str, str]:
    env = apply_checkpoint_env()
    env[PULSE_RUNTIME_WORKER_ENV] = "1"
    runtime_env = resolve_pulse_runtime_env(runtime)
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = _normalize_runtime_pythonpath(runtime_env.pop("PYTHONPATH", ""))
    if existing_pythonpath:
        pythonpath_entries.extend(_normalize_runtime_pythonpath(existing_pythonpath))
    env["PYTHONPATH"] = os.pathsep.join(_dedupe_paths(pythonpath_entries))
    for key, value in runtime_env.items():
        env[key] = value
    return env


def _maybe_run_in_subprocess(
    video_path: str,
    *,
    runtime: dict[str, Any] | None,
    fallback_meta_fps: float | None,
) -> dict[str, Any] | None:
    if os.getenv(PULSE_RUNTIME_WORKER_ENV) == "1":
        return None
    python_bin = _resolve_runtime_python_bin(runtime)
    if python_bin is None:
        return None

    ipc_dir = _runtime_cache_dir("pulse_runtime_ipc")
    token = uuid.uuid4().hex
    request_path = ipc_dir / f"pulse.{token}.request.json"
    response_path = ipc_dir / f"pulse.{token}.response.json"
    request_path.write_text(
        json.dumps(
            {
                "video_path": video_path,
                "runtime": dict(runtime or {}),
                "fallback_meta_fps": fallback_meta_fps,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    command = [
        python_bin,
        "-m",
        "worldarena.benchmark.pulse_runtime_worker",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(_project_root()),
            env=_subprocess_runtime_env(runtime),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = " | ".join(part for part in (stderr, stdout) if part)
        raise RuntimeError(f"pulse runtime subprocess failed: {details}") from exc
    if not response_path.exists():
        logs = " | ".join(
            part for part in ((completed.stderr or "").strip(), (completed.stdout or "").strip()) if part
        )
        raise RuntimeError(f"pulse runtime subprocess did not write a response: {logs}")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    request_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    if not bool(response.get("ok")):
        error = str(response.get("error") or "pulse runtime subprocess failed")
        traceback_text = str(response.get("traceback") or "").strip()
        stdout = str(response.get("stdout") or "").strip()
        stderr = str(response.get("stderr") or "").strip()
        raise RuntimeError(" | ".join(part for part in (error, traceback_text, stderr, stdout) if part))
    return dict(response["result"])


def resolve_pulse_of_motion_checkpoint(
    runtime: dict[str, Any] | None = None,
    *,
    required: bool = False,
) -> Path:
    configured = dict(runtime or {}).get("checkpoint_path")
    if configured is not None:
        resolved = resolve_checkpoint_path(configured, kind="file", required=required)
        if resolved is None:
            raise FileNotFoundError("Pulse-of-Motion checkpoint path is missing")
        return resolved
    return checkpoint_path(
        "Pulse-of-Motion",
        PULSE_OF_MOTION_CHECKPOINT,
        kind="file",
        required=required,
    )


def inspect_pulse_of_motion_layout(runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    root = resolve_pulse_of_motion_root()
    predict_path = root / "inference" / "predict.py"
    config_path = root / "inference" / "configs" / "config_fps.yaml"
    resolved_checkpoint_path = resolve_pulse_of_motion_checkpoint(runtime)
    source_ready = predict_path.exists() and config_path.exists()
    checkpoint_present = resolved_checkpoint_path.exists()
    ready = source_ready and checkpoint_present
    error = None
    if not source_ready:
        error = "Pulse-of-Motion checkout is missing predict.py or config_fps.yaml"
    elif not checkpoint_present:
        error = f"Pulse-of-Motion checkpoint is missing: {resolved_checkpoint_path}"
    return {
        "root": str(root),
        "predict_path": str(predict_path),
        "config_path": str(config_path),
        "checkpoint_path": str(resolved_checkpoint_path),
        "checkpoint_source": "ckpt",
        "source_ready": source_ready,
        "checkpoint_present": checkpoint_present,
        "download_supported": False,
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


def _default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load_model(root: str, device: str, checkpoint_file: str):
    module = _load_predict_module(root)
    pulse_root = Path(root)
    config_path = pulse_root / "inference" / "configs" / "config_fps.yaml"
    torch_module = getattr(module, "torch", None)
    if torch_module is None:
        return module.load_model(str(config_path), checkpoint_file, device)

    original_torch_load = torch_module.load

    def _compat_torch_load(*args, **kwargs):
        # Pulse-of-Motion checkpoints still rely on pre-2.6 torch.load behavior.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch_module.load = _compat_torch_load
    try:
        return module.load_model(str(config_path), checkpoint_file, device)
    finally:
        torch_module.load = original_torch_load


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
    delegated = _maybe_run_in_subprocess(
        video_path,
        runtime=runtime,
        fallback_meta_fps=fallback_meta_fps,
    )
    if delegated is not None:
        return delegated
    layout = inspect_pulse_of_motion_layout(runtime)
    if not layout["ready"]:
        raise RuntimeError(str(layout["error"]))

    root = str(resolve_pulse_of_motion_root())
    checkpoint_file = str(resolve_pulse_of_motion_checkpoint(runtime, required=True))
    device = str(runtime.get("device") or _default_device())
    clip_length = int(runtime.get("clip_length", 30))
    stride = int(runtime.get("stride", 4))
    resolution = int(runtime.get("resolution", 216))

    module = _load_predict_module(root)
    model = _load_model(root, device, checkpoint_file)
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
    "PULSE_RUNTIME_WORKER_ENV",
    "compute_pulse_metrics",
    "inspect_pulse_of_motion_layout",
    "resolve_pulse_of_motion_checkpoint",
    "resolve_pulse_runtime_env",
    "resolve_pulse_runtime_python",
    "resolve_pulse_of_motion_root",
]
