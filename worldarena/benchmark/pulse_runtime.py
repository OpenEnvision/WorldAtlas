"""Subprocess runtime for PULSE perceptual quality metrics."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
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
PULSE_RESULT_CACHE_SCHEMA = 2


def _project_root() -> Path:
    """Project root -> Path."""
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
    """Resolve pulse of motion root -> Path."""
    return (_project_root() / "thirdparty" / "Pulse-of-Motion").resolve()


def _pulse_import_paths(root: Path | None = None) -> list[Path]:
    pulse_root = (root or resolve_pulse_of_motion_root()).resolve()
    inference_root = pulse_root / "inference"
    return [path for path in (inference_root,) if path.exists()]


def _module_under_root(module: object, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        return Path(str(module_file)).resolve().is_relative_to(root)
    except OSError:
        return False


class _PulseImportContext:
    _GENERIC_MODULE_ROOTS = ("utils", "src")

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.paths = _pulse_import_paths(self.root)
        self._inserted_paths: list[str] = []
        self._saved_modules: dict[str, ModuleType] = {}

    def __enter__(self) -> None:
        pulse_roots = tuple(self.paths)
        for name, module in list(sys.modules.items()):
            top_level = name.split(".", 1)[0]
            if top_level not in self._GENERIC_MODULE_ROOTS:
                continue
            if any(_module_under_root(module, root) for root in pulse_roots):
                continue
            self._saved_modules[name] = module
            del sys.modules[name]
        for path in reversed(self.paths):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                self._inserted_paths.append(path_str)

    def __exit__(self, exc_type, exc, traceback) -> None:
        pulse_roots = tuple(self.paths)
        for path_str in self._inserted_paths:
            while path_str in sys.path:
                sys.path.remove(path_str)
        for name, module in list(sys.modules.items()):
            top_level = name.split(".", 1)[0]
            if top_level in self._GENERIC_MODULE_ROOTS and any(
                _module_under_root(module, root) for root in pulse_roots
            ):
                del sys.modules[name]
        sys.modules.update(self._saved_modules)


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
    for candidate in _pulse_import_paths():
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
    with _PulseImportContext(pulse_root):
        spec.loader.exec_module(module)
    return module


def _default_device() -> str:
    """Default device -> str."""
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
        with _PulseImportContext(pulse_root):
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


def _pulse_result_cache_path(
    video_path: str,
    *,
    runtime: dict[str, Any],
    fallback_meta_fps: float | None,
) -> tuple[Path, str] | None:
    if not bool(runtime.get("disk_cache", False)):
        return None
    video = Path(video_path).expanduser().resolve()
    video_stat = video.stat()
    checkpoint = resolve_pulse_of_motion_checkpoint(runtime, required=True)
    checkpoint_stat = checkpoint.stat()
    semantic_runtime = {
        key: value
        for key, value in runtime.items()
        if key not in {"cache_dir", "device", "disk_cache"}
    }
    identity = {
        "schema": PULSE_RESULT_CACHE_SCHEMA,
        "video_path": str(video),
        "video_size": video_stat.st_size,
        "video_mtime_ns": video_stat.st_mtime_ns,
        "fallback_meta_fps": fallback_meta_fps,
        "checkpoint_path": str(checkpoint),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "runtime": semantic_runtime,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    configured_root = runtime.get("cache_dir")
    if configured_root:
        cache_root = Path(str(configured_root)).expanduser()
        if not cache_root.is_absolute():
            cache_root = _project_root() / cache_root
    else:
        cache_root = _runtime_cache_dir("pulse_of_motion_results")
    path = cache_root.resolve() / digest[:2] / f"{digest}.json"
    return path, digest


def _read_pulse_result_cache(path: Path, digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if wrapper.get("schema") != PULSE_RESULT_CACHE_SCHEMA or wrapper.get("key") != digest:
        return None
    result = wrapper.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("metrics"), dict):
        return None
    copied = json.loads(json.dumps(result))
    copied.setdefault("details", {})["disk_cache_hit"] = True
    copied["details"]["disk_cache_key"] = digest[:16]
    return copied


def _write_pulse_result_cache(path: Path, digest: str, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    wrapper = {
        "schema": PULSE_RESULT_CACHE_SCHEMA,
        "key": digest,
        "result": result,
    }
    try:
        temporary.write_text(
            json.dumps(wrapper, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_analysis_frames(
    video_path: str,
    *,
    clip_length: int,
    stride: int,
    resolution: int,
    short_video_policy: str,
    short_video_min_segments: int,
) -> tuple[np.ndarray, float | None, dict[str, Any]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    container_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
    decoded: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (resolution, resolution))
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    source_count = len(decoded)
    if source_count == 0:
        raise RuntimeError(f"No decodable frames: {video_path}")

    temporal_scale = 1.0
    resampled = False
    if source_count < clip_length:
        if short_video_policy != "temporal_resample":
            raise RuntimeError(f"Video too short ({source_count} frames < {clip_length})")
        if source_count < 2:
            raise RuntimeError("At least two frames are required for temporal resampling")
        target_count = clip_length + stride * (max(short_video_min_segments, 1) - 1)
        temporal_scale = float(target_count - 1) / float(source_count - 1)
        source = np.stack(decoded, axis=0)
        positions = np.linspace(0.0, float(source_count - 1), target_count)
        lower = np.floor(positions).astype(np.int64)
        upper = np.minimum(lower + 1, source_count - 1)
        alpha = (positions - lower).astype(np.float32)[:, None, None, None]
        frames = np.rint(
            source[lower].astype(np.float32) * (1.0 - alpha)
            + source[upper].astype(np.float32) * alpha
        ).clip(0, 255).astype(np.uint8)
        resampled = True
    else:
        frames = np.stack(decoded, axis=0)

    return frames, container_fps, {
        "total_frames": source_count,
        "analysis_total_frames": int(frames.shape[0]),
        "short_video_policy": short_video_policy,
        "short_video_resampled": resampled,
        "temporal_resample_factor": temporal_scale,
    }


def _predict_phyfps_batched(
    model: Any,
    frames: np.ndarray,
    *,
    device: str,
    clip_length: int,
    stride: int,
    batch_size: int,
) -> tuple[list[float], list[int]]:
    import torch

    starts = list(range(0, int(frames.shape[0]) - clip_length + 1, stride))
    if not starts:
        raise RuntimeError(
            f"Video too short ({int(frames.shape[0])} frames < {clip_length})"
        )
    predicted: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            clips = np.stack(
                [frames[start : start + clip_length] for start in batch_starts], axis=0
            ).astype(np.float32)
            clips = clips / 127.5 - 1.0
            tensor = torch.from_numpy(clips).permute(0, 4, 1, 2, 3).to(device)
            values = torch.exp(model(tensor)).detach().float().cpu().reshape(-1).tolist()
            if len(values) != len(batch_starts):
                raise RuntimeError(
                    f"Pulse returned {len(values)} values for a batch of {len(batch_starts)} clips"
                )
            predicted.extend(round(float(value), 1) for value in values)
    return predicted, starts


def compute_pulse_metrics(
    video_path: str,
    *,
    runtime: dict[str, Any] | None = None,
    fallback_meta_fps: float | None = None,
) -> dict[str, Any]:
    runtime = dict(runtime or {})
    cache_target = _pulse_result_cache_path(
        video_path,
        runtime=runtime,
        fallback_meta_fps=fallback_meta_fps,
    )
    if cache_target is not None:
        cached = _read_pulse_result_cache(*cache_target)
        if cached is not None:
            return cached
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
    batch_size = max(int(runtime.get("batch_size", 1)), 1)
    short_video_policy = str(runtime.get("short_video_policy", "error"))
    short_video_min_segments = max(int(runtime.get("short_video_min_segments", 5)), 1)

    model = _load_model(root, device, checkpoint_file)
    frames, container_fps, preprocessing_details = _decode_analysis_frames(
        video_path,
        clip_length=clip_length,
        stride=stride,
        resolution=resolution,
        short_video_policy=short_video_policy,
        short_video_min_segments=short_video_min_segments,
    )
    analysis_segment_phyfps, segment_starts = _predict_phyfps_batched(
        model,
        frames,
        device=device,
        clip_length=clip_length,
        stride=stride,
        batch_size=batch_size,
    )
    meta_fps = container_fps if container_fps is not None else fallback_meta_fps
    meta_fps_source = "prediction_container_fps" if container_fps is not None else "fallback"
    if meta_fps is None or meta_fps <= 0:
        raise RuntimeError("meta fps is unavailable for Pulse-of-Motion")

    temporal_scale = float(preprocessing_details["temporal_resample_factor"])
    segment_phyfps = [float(value) / temporal_scale for value in analysis_segment_phyfps]
    analysis_avg_phyfps = round(float(np.mean(analysis_segment_phyfps)), 1)
    avg_phyfps = analysis_avg_phyfps / temporal_scale
    abs_error = abs(avg_phyfps - float(meta_fps))
    pct_error = abs_error / max(float(meta_fps), 1e-6)
    intra_video_cv = float(np.std(segment_phyfps) / max(np.mean(segment_phyfps), 1e-6))
    result = {
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
            "segment_start_frames": segment_starts,
            "meta_fps": float(meta_fps),
            "meta_fps_source": meta_fps_source,
            "device": device,
            "clip_length": clip_length,
            "stride": stride,
            "resolution": resolution,
            "batch_size": batch_size,
            **preprocessing_details,
        },
    }
    if preprocessing_details["short_video_resampled"]:
        result["details"].update(
            {
                "analysis_avg_phyfps": analysis_avg_phyfps,
                "analysis_segment_phyfps": analysis_segment_phyfps,
                "analysis_meta_fps": float(meta_fps) * temporal_scale,
            }
        )
    if cache_target is not None:
        cache_path, digest = cache_target
        result["details"]["disk_cache_hit"] = False
        result["details"]["disk_cache_key"] = digest[:16]
        _write_pulse_result_cache(cache_path, digest, result)
    return result


__all__ = [
    "PULSE_RUNTIME_WORKER_ENV",
    "compute_pulse_metrics",
    "inspect_pulse_of_motion_layout",
    "resolve_pulse_of_motion_checkpoint",
    "resolve_pulse_runtime_env",
    "resolve_pulse_runtime_python",
    "resolve_pulse_of_motion_root",
]
