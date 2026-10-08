"""Runtime wrapper for in-tree official metric adapters and upstream checkouts."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image

from worldatlas.benchmark.camera_alignment import (
    resample_camera_matrices,
    score_camera_trajectories,
)
from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.common.checkpoints import (
    apply_checkpoint_env,
    checkpoint_env,
    filter_checkpoint_env_overrides,
)
from worldatlas.benchmark.local_checkpoints import (
    MEGASAM_KIND,
    prepare_megasam_checkpoints,
)
from worldatlas.common.progress import log_progress, run_progress_subprocess
from worldatlas.common.video_io import declared_frame_count_off_by_one


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRDPARTY_ROOT = PROJECT_ROOT / "thirdparty"
OFFICIAL_BACKEND_ROOT = PROJECT_ROOT / "worldatlas" / "benchmark" / "official_backends"
OFFICIAL_BACKEND_PACKAGE = "worldatlas.benchmark.official_backends"
OFFICIAL_RUNTIME_WORKER_ENV = "WORLDATLAS_OFFICIAL_RUNTIME_WORKER"


@dataclass(frozen=True, slots=True)
class MetricSourceLayout:
    source_root: Path
    package_name: str
    droid_root: Path | None
    droid_build_root: Path | None
    lietorch_root: Path | None
    sea_raft_root: Path | None
    grounding_dino_root: Path | None
    sam2_root: Path | None
    vfi_mamba_root: Path | None


OFFICIAL_IMPORT_TARGETS = {
    "camera_error": (
        "camera_error",
        "CameraErrorMetric",
    ),
    "reprojection_error": (
        "reprojection_error",
        "ReprojectionErrorMetric",
    ),
    "optical_flow_aepe": (
        "flow_aepe",
        "OpticalFlowAverageEndPointErrorMetric",
    ),
    "motion_magnitude": (
        "motion_magnitude",
        "OpticalFlowMetric",
    ),
    "motion_smoothness": (
        "motion_smoothness",
        "MotionSmoothnessMetric",
    ),
    "motion_accuracy": (
        "motion_accuracy",
        "MotionAccuracyMetric",
    ),
}
OFFICIAL_METRIC_NAMES = tuple(OFFICIAL_IMPORT_TARGETS)
_METRIC_INSTANCE_CACHE: dict[tuple[str, str, str], Any] = {}


def _runtime_cache_dir(name: str) -> Path:
    cache_root = Path(os.environ.get("WORLDATLAS_RUNTIME_CACHE_ROOT", PROJECT_ROOT / ".cache" / "benchmark"))
    path = cache_root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _select_candidate_path(candidates: list[Path], *, package_prefix: str | None = None) -> Path | None:
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    if not unique_candidates:
        return None
    if package_prefix is not None:
        for candidate in unique_candidates:
            if package_prefix in str(candidate):
                return candidate
    return sorted(unique_candidates, key=lambda path: (len(path.parts), str(path)))[0]


def _existing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _torch_library_path() -> Path | None:
    """Torch library path -> Path | None."""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    lib_path = Path(torch.__file__).resolve().parent / "lib"
    return lib_path if lib_path.exists() else None


def _lietorch_build_paths(lietorch_root: Path | None) -> list[Path]:
    if lietorch_root is None:
        return []
    return sorted(
        {
            path.parent
            for path in lietorch_root.rglob("lietorch_backends*.so")
            if path.is_file()
        }
    )


def _prepend_env_path(env: dict[str, str], key: str, paths: list[Path]) -> None:
    path_values = [str(path) for path in paths if path is not None and path.exists()]
    if not path_values:
        return
    existing = env.get(key)
    if existing:
        path_values.append(existing)
    env[key] = os.pathsep.join(dict.fromkeys(path_values))


@lru_cache(maxsize=1)
def resolve_metric_source_layout() -> MetricSourceLayout:
    """Resolve in-tree metric adapters and their optional upstream checkouts."""
    missing_backends = [
        module_name
        for module_name, _ in OFFICIAL_IMPORT_TARGETS.values()
        if not (OFFICIAL_BACKEND_ROOT / f"{module_name}.py").is_file()
    ]
    if missing_backends:
        raise FileNotFoundError(
            f"official metric backends missing under {OFFICIAL_BACKEND_ROOT}: {missing_backends}"
        )

    droid_checkout = THIRDPARTY_ROOT / "DROID-SLAM"
    droid_root = _first_existing_path(
        [
            droid_checkout / "droid_slam",
        ]
    )
    droid_build_root = _select_candidate_path(
        [
            path.parent
            for path in droid_checkout.glob("droid_backends*.so")
        ],
        package_prefix="DROID-SLAM",
    )
    lietorch_root = _select_candidate_path(
        _existing_paths(
            [
                droid_checkout / "thirdparty" / "lietorch",
            ]
        ),
        package_prefix="DROID-SLAM",
    )
    sea_raft_checkout = THIRDPARTY_ROOT / "SEA-RAFT"
    sea_raft_root = (
        sea_raft_checkout.resolve()
        if (sea_raft_checkout / "core" / "raft.py").is_file()
        else None
    )
    grounding_dino_checkout = THIRDPARTY_ROOT / "GroundingDINO"
    grounding_dino_root = _select_candidate_path(
        _existing_paths([grounding_dino_checkout])
        if (grounding_dino_checkout / "groundingdino" / "__init__.py").is_file()
        else [],
        package_prefix="GroundingDINO",
    )
    sam2_checkout = THIRDPARTY_ROOT / "sam2"
    sam2_root = _select_candidate_path(
        _existing_paths([sam2_checkout])
        if (sam2_checkout / "sam2" / "__init__.py").is_file()
        else [],
    )
    vfi_mamba_checkout = THIRDPARTY_ROOT / "VFIMamba"
    vfi_mamba_root = _select_candidate_path(
        _existing_paths([vfi_mamba_checkout])
        if (vfi_mamba_checkout / "Trainer_finetune.py").is_file()
        else [],
        package_prefix="VFIMamba",
    )
    return MetricSourceLayout(
        source_root=OFFICIAL_BACKEND_ROOT,
        package_name=OFFICIAL_BACKEND_PACKAGE,
        droid_root=droid_root,
        droid_build_root=droid_build_root,
        lietorch_root=lietorch_root,
        sea_raft_root=sea_raft_root,
        grounding_dino_root=grounding_dino_root,
        sam2_root=sam2_root,
        vfi_mamba_root=vfi_mamba_root,
    )


@contextmanager
def metric_runtime_context() -> Iterator[MetricSourceLayout]:
    """Metric runtime context -> Iterator[MetricSourceLayout]."""
    layout = resolve_metric_source_layout()
    previous_cwd = Path.cwd()
    env_updates = checkpoint_env(create=True)
    previous_env = {key: os.environ.get(key) for key in env_updates}
    inserted_paths = [
        path
        for path in (
            PROJECT_ROOT,
            layout.droid_root,
            layout.droid_build_root,
            layout.lietorch_root,
            layout.sea_raft_root,
            layout.sea_raft_root / "core" if layout.sea_raft_root is not None else None,
            layout.sea_raft_root / "config" if layout.sea_raft_root is not None else None,
            layout.grounding_dino_root,
            layout.sam2_root,
            layout.vfi_mamba_root,
        )
        if path is not None
    ]
    inserted_paths.extend(_lietorch_build_paths(layout.lietorch_root))
    library_paths = [
        path
        for path in (_torch_library_path(),)
        if path is not None
    ]
    previous_ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    try:
        os.environ.update(env_updates)
        _prepend_env_path(os.environ, "LD_LIBRARY_PATH", library_paths)
        for path in reversed(inserted_paths):
            sys.path.insert(0, str(path))
        os.chdir(PROJECT_ROOT)
        yield layout
    finally:
        os.chdir(previous_cwd)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if previous_ld_library_path is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = previous_ld_library_path
        for path in inserted_paths:
            path_str = str(path)
            while path_str in sys.path:
                sys.path.remove(path_str)


def _resolve_runtime_python_bin(runtime: dict[str, Any] | None) -> str | None:
    payload = dict(runtime or {})
    raw = str(payload.get("python_bin") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if raw.startswith(".") or "/" in raw:
        return str((PROJECT_ROOT / path).absolute())
    return raw


def resolve_official_runtime_python(runtime: dict[str, Any] | None = None) -> str | None:
    return _resolve_runtime_python_bin(runtime)


def _subprocess_runtime_env(runtime: dict[str, Any] | None) -> dict[str, str]:
    env = apply_checkpoint_env()
    env[OFFICIAL_RUNTIME_WORKER_ENV] = "1"
    pythonpath_entries = [str(PROJECT_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.setdefault("PYTHONUNBUFFERED", "1")
    layout = resolve_metric_source_layout()
    _prepend_env_path(
        env,
        "PYTHONPATH",
        [
            PROJECT_ROOT,
            *[
                path
                for path in (
                    layout.droid_root,
                    layout.droid_build_root,
                    layout.lietorch_root,
                    layout.sea_raft_root,
                    layout.sea_raft_root / "core" if layout.sea_raft_root is not None else None,
                    layout.sea_raft_root / "config" if layout.sea_raft_root is not None else None,
                    layout.grounding_dino_root,
                    layout.sam2_root,
                    layout.vfi_mamba_root,
                )
                if path is not None
            ],
            *_lietorch_build_paths(layout.lietorch_root),
        ],
    )
    _prepend_env_path(
        env,
        "LD_LIBRARY_PATH",
        [
            path
            for path in (_torch_library_path(),)
            if path is not None
        ],
    )
    for key, value in filter_checkpoint_env_overrides(dict((runtime or {}).get("env", {}))).items():
        env[str(key)] = str(value)
    return env


def _maybe_run_metric_in_subprocess(
    metric_name: str,
    payload: dict[str, Any],
    runtime: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if os.getenv(OFFICIAL_RUNTIME_WORKER_ENV) == "1":
        return None
    python_bin = _resolve_runtime_python_bin(runtime)
    if python_bin is None:
        return None

    ipc_dir = _runtime_cache_dir("official_runtime_ipc")
    token = uuid.uuid4().hex
    request_path = ipc_dir / f"{metric_name}.{token}.request.json"
    response_path = ipc_dir / f"{metric_name}.{token}.response.json"
    request_path.write_text(
        json.dumps({"metric": metric_name, "kwargs": payload}, ensure_ascii=False),
        encoding="utf-8",
    )
    command = [
        python_bin,
        "-u",
        "-m",
        "worldatlas.benchmark.official_runtime_worker",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    try:
        completed = run_progress_subprocess(
            command,
            cwd=str(PROJECT_ROOT),
            env=_subprocess_runtime_env(runtime),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        error_parts = [f"official runtime subprocess failed for {metric_name}"]
        if stderr:
            error_parts.append(stderr)
        if stdout:
            error_parts.append(stdout)
        raise RuntimeError(" | ".join(error_parts)) from exc

    if not response_path.exists():
        logs = " | ".join(
            part for part in ((completed.stderr or "").strip(), (completed.stdout or "").strip()) if part
        )
        raise RuntimeError(f"official runtime subprocess did not write a response for {metric_name}: {logs}")

    response = json.loads(response_path.read_text(encoding="utf-8"))
    if not bool(response.get("ok")):
        error = str(response.get("error") or f"official runtime subprocess failed for {metric_name}")
        traceback_text = str(response.get("traceback") or "").strip()
        stderr = str(response.get("stderr") or "").strip()
        stdout = str(response.get("stdout") or "").strip()
        details = " | ".join(part for part in (error, traceback_text, stderr, stdout) if part)
        raise RuntimeError(details)

    if not _coerce_runtime_bool((runtime or {}).get("keep_ipc"), default=False):
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
    return dict(response["result"])


@lru_cache(maxsize=None)
def _load_metric_class(metric_name: str) -> type[Any]:
    with metric_runtime_context() as layout:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            pass
        module_suffix, class_name = OFFICIAL_IMPORT_TARGETS[metric_name]
        module_name = f"{layout.package_name}.{module_suffix}"
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


def _empty_cuda_cache() -> None:
    """Empty cuda cache."""
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _coerce_runtime_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_runtime_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default


def _aepe_frame_max_side(runtime: dict[str, Any] | None) -> int:
    configured = _coerce_runtime_int((runtime or {}).get("frame_max_side"))
    if configured is not None:
        return configured
    return _coerce_runtime_int(os.environ.get("WORLDATLAS_AEPE_FRAME_MAX_SIDE"), default=1920) or 1920


def _motion_sm_frame_max_side(runtime: dict[str, Any] | None) -> int:
    configured = _coerce_runtime_int((runtime or {}).get("frame_max_side"))
    if configured is not None:
        return configured
    return (
        _coerce_runtime_int(os.environ.get("WORLDATLAS_MOTION_SM_FRAME_MAX_SIDE"), default=1920)
        or 1920
    )


def _aepe_batch_size(runtime: dict[str, Any] | None) -> int:
    raw = os.environ.get("WORLDATLAS_AEPE_BATCH_SIZE")
    if raw not in (None, ""):
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _coerce_runtime_int((runtime or {}).get("batch_size"), default=2) or 2


def _resolve_runtime_device(value: Any) -> str | None:
    if value is None:
        return None
    device = str(value).strip()
    if not device or device.lower() in {"auto", "none"}:
        return None
    return device


def _configure_motion_accuracy_runtime(metric: Any, runtime: dict[str, Any] | None) -> dict[str, Any]:
    runtime = dict(runtime or {})
    details: dict[str, Any] = {}

    sam2_device = _resolve_runtime_device(runtime.get("sam2_device"))
    flow_device = _resolve_runtime_device(runtime.get("flow_device"))
    grounding_dino_device = _resolve_runtime_device(runtime.get("grounding_dino_device"))
    offload_video_to_cpu = _coerce_runtime_bool(
        runtime.get("sam2_offload_video_to_cpu"),
        default=False,
    )
    offload_state_to_cpu = _coerce_runtime_bool(
        runtime.get("sam2_offload_state_to_cpu"),
        default=False,
    )

    predictor = getattr(metric, "_predictor", None)
    if predictor is not None:
        if sam2_device is not None:
            predictor = predictor.to(sam2_device)
            metric._predictor = predictor
        details["sam2_device"] = str(predictor.device)
        if offload_video_to_cpu or offload_state_to_cpu:
            desired_offload = (offload_video_to_cpu, offload_state_to_cpu)
            original_init_state = getattr(
                predictor,
                "_worldatlas_original_init_state",
                predictor.init_state,
            )

            if getattr(predictor, "_worldatlas_offload_settings", None) != desired_offload:
                def wrapped_init_state(*args: Any, **kwargs: Any) -> Any:
                    kwargs.setdefault("offload_video_to_cpu", offload_video_to_cpu)
                    kwargs.setdefault("offload_state_to_cpu", offload_state_to_cpu)
                    kwargs.setdefault("async_loading_frames", False)
                    return original_init_state(*args, **kwargs)

                metric._predictor.init_state = wrapped_init_state
                metric._predictor._worldatlas_original_init_state = original_init_state
                metric._predictor._worldatlas_offload_settings = desired_offload
        details["sam2_offload_video_to_cpu"] = offload_video_to_cpu
        details["sam2_offload_state_to_cpu"] = offload_state_to_cpu

    optical_flow_model = getattr(metric, "_optical_flow_model", None)
    if flow_device is not None and optical_flow_model is not None:
        metric._optical_flow_model = optical_flow_model.to(flow_device)
        metric._device = flow_device
    details["flow_device"] = str(getattr(metric, "_device", "cuda"))

    grounding_dino_model = getattr(metric, "_grounding_dino_model", None)
    if grounding_dino_device is not None and grounding_dino_model is not None:
        metric._grounding_dino_model = grounding_dino_model.to(grounding_dino_device)
        if hasattr(metric, "_grounding_dino_args"):
            metric._grounding_dino_args.device = grounding_dino_device
    if hasattr(metric, "_grounding_dino_args"):
        details["grounding_dino_device"] = str(metric._grounding_dino_args.device)

    return details


def _configure_droid_runtime(metric: Any) -> None:
    args = getattr(metric, "_args", None)
    if args is None:
        return
    defaults = {
        "disable_vis": True,
        "motion_damping": 0.5,
        "frontend_device": "cuda",
        "backend_device": "cuda",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def _coerce_droid_calib(value: Any) -> list[float]:
    if value is None:
        raise ValueError("DROID calibration is missing")
    values = [float(item) for item in value]
    if len(values) != 4 or not all(np.isfinite(item) for item in values):
        raise ValueError(f"DROID calibration must be four finite values, got {value!r}")
    return values


def _first_frame_resolution(frame_paths: list[str]) -> tuple[int, int]:
    if not frame_paths:
        raise ValueError("cannot derive first-frame resolution without frame paths")
    frame = cv2.imread(str(frame_paths[0]))
    if frame is None:
        raise FileNotFoundError(f"failed to read first frame: {frame_paths[0]}")
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid first-frame resolution: {width}x{height}")
    return int(width), int(height)


def _media_resolution_and_frame_count(media_path: str) -> tuple[int, int, int | None]:
    """Read video metadata without materializing every frame on shared storage."""
    path = Path(media_path)
    if path.is_dir():
        frame_paths = [
            str(frame_path)
            for frame_path in sorted(path.iterdir())
            if frame_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        width, height = _first_frame_resolution(frame_paths)
        return width, height, len(frame_paths)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        width, height = _first_frame_resolution([str(path)])
        return width, height, 1

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0:
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(f"video contains no decodable frames: {path}")
            height, width = frame.shape[:2]
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid video resolution: {width}x{height}")
    return int(width), int(height), frame_count if frame_count > 0 else None


def _derive_droid_calib_from_runtime(
    frame_paths: list[str],
    runtime: dict[str, Any] | None,
) -> tuple[list[float], dict[str, Any]]:
    runtime = dict(runtime or {})
    explicit_calib = runtime.get("droid_calib", runtime.get("calib"))
    if explicit_calib is not None:
        calib = _coerce_droid_calib(explicit_calib)
        return calib, {
            "droid_intrinsics_policy": "explicit",
            "droid_calib": calib,
        }

    policy = str(
        runtime.get(
            "droid_intrinsics",
            runtime.get("droid_intrinsics_policy", runtime.get("intrinsics_policy", "actual_resolution")),
        )
    ).strip().lower()
    if policy in {"", "actual_resolution", "resolution_scaled", "actual"}:
        policy = "actual_resolution"
    else:
        raise ValueError(f"unsupported DROID intrinsics policy: {policy}")

    width, height = _first_frame_resolution(frame_paths)
    focal_scale = float(runtime.get("droid_focal_scale", 1.0))
    if focal_scale <= 0.0 or not np.isfinite(focal_scale):
        raise ValueError(f"DROID focal scale must be positive, got {focal_scale!r}")
    calib = [
        width * focal_scale,
        height * focal_scale,
        width / 2.0,
        height / 2.0,
    ]
    return calib, {
        "droid_intrinsics_policy": "actual_resolution",
        "droid_source_resolution": [width, height],
        "droid_focal_scale": focal_scale,
        "droid_calib": calib,
    }


def _configure_droid_intrinsics(
    metric: Any,
    frame_paths: list[str],
    runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    args = getattr(metric, "_args", None)
    if args is None:
        return details
    calib, details = _derive_droid_calib_from_runtime(frame_paths, runtime)
    args.calib = calib
    final_calib = getattr(args, "calib", None)
    if final_calib is not None and "droid_calib" not in details:
        details["droid_calib"] = _coerce_droid_calib(final_calib)
    return details


def _coerce_runtime_nonnegative_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced >= 0 else default


def _resolve_runtime_path(value: Any, default: Path | None = None) -> Path | None:
    raw = value if value not in (None, "") else default
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_runtime_executable(value: Any, default: Path | str) -> str:
    raw = str(value if value not in (None, "") else default).strip()
    if not raw:
        raise ValueError("runtime executable path is empty")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if raw.startswith((".", "~")) or "/" in raw:
        return str((PROJECT_ROOT / path).resolve())
    return raw


def _stable_megasam_scene_name(
    prediction_path: Path,
    *,
    frame_stride: int,
    start_frame: int,
    max_frames: int | None,
) -> str:
    stat = prediction_path.stat()
    frame_window = f"start={start_frame}::stride={frame_stride}::max={max_frames or 'all'}"
    payload = f"{prediction_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}::{frame_window}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", prediction_path.stem).strip("._-")
    stem = stem[:80] or "prediction"
    window_label = f"s{start_frame}_st{frame_stride}_n{max_frames or 'all'}"
    return f"{stem}_{window_label}_{digest}"


def _tail_text(path: Path, *, max_bytes: int = 6000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        return handle.read().decode("utf-8", errors="replace")


def _prepare_single_image_megasam_input(source_path: Path, input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".jpg" if source_path.suffix.lower() == ".jpeg" else source_path.suffix.lower()
    target = input_dir / f"000000{suffix}"
    if target.is_symlink() and target.resolve() == source_path.resolve():
        return
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source_path.resolve())


def _run_megasam_tracking(
    prediction_path: str,
    runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime = dict(runtime or {})
    path = Path(prediction_path)
    if not path.exists():
        raise FileNotFoundError(f"prediction path does not exist: {path}")

    default_python = PROJECT_ROOT.parent / "conda" / "envs" / "mega_sam" / "bin" / "python"
    default_hf_home = PROJECT_ROOT.parent / "conda" / "hf-home"
    script_path = _resolve_runtime_path(runtime.get("megasam_runner"), PROJECT_ROOT / "scripts" / "run_megasam_video.py")
    output_root = _resolve_runtime_path(runtime.get("megasam_output_root"), _runtime_cache_dir("megasam_camera_error"))
    mega_root = _resolve_runtime_path(runtime.get("megasam_root", runtime.get("mega_sam_root")), THIRDPARTY_ROOT / "mega-sam")
    hf_home = _resolve_runtime_path(
        runtime.get("megasam_hf_home")
        or os.environ.get("WORLDATLAS_MEGASAM_HF_HOME")
        or os.environ.get("HF_HOME"),
        default_hf_home,
    )
    if script_path is None or not script_path.exists():
        raise MetricLoadError(f"MegaSAM runner not found: {script_path}")
    if output_root is None:
        raise ValueError("MegaSAM output root is unavailable")
    if mega_root is None or not mega_root.exists():
        raise MetricLoadError(f"MegaSAM repository not found: {mega_root}")
    if hf_home is None:
        raise MetricLoadError("MegaSAM HF_HOME is unavailable")

    python_bin = _resolve_runtime_executable(
        runtime.get("megasam_python", os.environ.get("MEGASAM_PYTHON")),
        default_python,
    )
    already_localized = bool(os.environ.get("WORLDATLAS_MEGASAM_CHECKPOINT"))
    try:
        local_assets = prepare_megasam_checkpoints()
    except MetricLoadError:
        raise
    except Exception as exc:
        raise MetricLoadError(f"MegaSAM checkpoint localization failed: {type(exc).__name__}: {exc}") from exc
    os.environ.update(local_assets)
    local_hf = local_assets.get("WORLDATLAS_MEGASAM_HF_HOME") or local_assets.get("HF_HOME")
    if local_hf:
        hf_home = Path(local_hf)
        os.environ["WORLDATLAS_MEGASAM_HF_HOME"] = local_hf
    if not already_localized:
        log_progress(
            "metric_load",
            metric="camera_error",
            status="megasam_local_weights",
            megasam_checkpoint=local_assets.get("WORLDATLAS_MEGASAM_CHECKPOINT"),
            depth_anything=local_assets.get("WORLDATLAS_DEPTH_ANYTHING_CHECKPOINT"),
            hf_home=str(hf_home),
            kind=MEGASAM_KIND,
        )
    gpu = str(
        runtime.get(
            "megasam_gpu",
            os.environ.get("MEGASAM_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
        )
    )
    frame_stride = _coerce_runtime_int(runtime.get("megasam_frame_stride"), default=1) or 1
    start_frame = _coerce_runtime_nonnegative_int(runtime.get("megasam_start_frame"), default=0)
    max_frames = _coerce_runtime_int(runtime.get("megasam_max_frames"))

    scene = str(
        runtime.get("megasam_scene_name")
        or _stable_megasam_scene_name(
            path,
            frame_stride=frame_stride,
            start_frame=start_frame,
            max_frames=max_frames,
        )
    )
    run_root = output_root / scene
    run_root.mkdir(parents=True, exist_ok=True)

    if path.is_dir():
        source_args = ["--frames-dir", str(path)]
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        input_dir = run_root / "single_image_input"
        _prepare_single_image_megasam_input(path, input_dir)
        source_args = ["--frames-dir", str(input_dir)]
    else:
        source_args = ["--video", str(path)]

    command = [
        python_bin,
        str(script_path),
        *source_args,
        "--scene-name",
        scene,
        "--output-root",
        str(output_root),
        "--mega-sam-root",
        str(mega_root),
        "--python",
        python_bin,
        "--hf-home",
        str(hf_home),
        "--gpu",
        gpu,
        "--stages",
        "depth_anything,unidepth,tracking",
        "--frame-stride",
        str(frame_stride),
        "--start-frame",
        str(start_frame),
    ]
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])
    ffmpeg_path = _resolve_runtime_path(runtime.get("megasam_ffmpeg"))
    if ffmpeg_path is not None:
        command.extend(["--ffmpeg", str(ffmpeg_path)])
    if _coerce_runtime_bool(runtime.get("megasam_depth_localhub"), default=True):
        command.append("--depth-localhub")
    else:
        command.append("--no-depth-localhub")
    unidepth_device = str(
        runtime.get("megasam_unidepth_device")
        or os.environ.get("WORLDATLAS_MEGASAM_UNIDEPTH_DEVICE")
        or ""
    ).strip()
    if unidepth_device:
        command.extend(["--unidepth-device", unidepth_device])
    if "megasam_unidepth_cpu_fallback" in runtime:
        if _coerce_runtime_bool(runtime.get("megasam_unidepth_cpu_fallback"), default=True):
            command.append("--unidepth-cpu-fallback")
        else:
            command.append("--no-unidepth-cpu-fallback")
    overwrite = _coerce_runtime_bool(runtime.get("megasam_overwrite"), default=False)
    skip_existing = _coerce_runtime_bool(runtime.get("megasam_skip_existing"), default=True)
    if overwrite:
        command.append("--overwrite")
    elif skip_existing:
        command.append("--skip-existing")

    log_path = run_root / "camera_error_megasam.log"
    recon_scene = run_root / "reconstructions" / scene
    poses_path = recon_scene / "poses.npy"
    intrinsics_path = recon_scene / "intrinsics.npy"
    metadata_path = run_root / "run_config.json"
    cache_complete = (
        not overwrite
        and skip_existing
        and poses_path.is_file()
        and metadata_path.is_file()
    )
    if not cache_complete:
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write("\n" + "=" * 80 + "\n")
                log_file.write(" ".join(command) + "\n")
                log_file.flush()
                subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                )
        except subprocess.CalledProcessError as exc:
            tail = _tail_text(log_path)
            message = f"MegaSAM tracking failed for {prediction_path}; log: {log_path}"
            if tail:
                message += f" | log tail: {tail}"
            raise RuntimeError(message) from exc

    if not poses_path.exists():
        raise FileNotFoundError(f"MegaSAM poses not found: {poses_path}")
    poses = np.load(poses_path).astype(np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"MegaSAM poses must have shape (N, 7), got {poses.shape}")

    intrinsics = None
    if intrinsics_path.exists():
        intrinsics = np.load(intrinsics_path).astype(np.float32)

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "scene": scene,
        "run_root": run_root,
        "log_path": log_path,
        "poses": poses,
        "intrinsics": intrinsics,
        "metadata": metadata,
        "frame_stride": frame_stride,
        "start_frame": start_frame,
        "max_frames": max_frames,
    }


def _resample_camera_matrices(cameras: np.ndarray, target_frames: int) -> np.ndarray:
    return resample_camera_matrices(cameras, target_frames)


def _select_gt_for_megasam(
    cameras_gt: np.ndarray,
    *,
    target_frames: int,
    source_frame_count: int,
    frame_stride: int,
    start_frame: int,
    max_frames: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cameras = np.asarray(cameras_gt, dtype=np.float32)
    details: dict[str, Any] = {
        "camera_gt_source_frame_count": int(len(cameras)),
        "camera_gt_target_frame_count": int(target_frames),
    }
    if len(cameras) == source_frame_count:
        indices = list(range(start_frame, source_frame_count, max(1, frame_stride)))
        if max_frames is not None:
            indices = indices[:max_frames]
        if len(indices) == target_frames:
            selected = cameras[np.asarray(indices, dtype=np.int64)]
            details.update(
                {
                    "camera_gt_alignment": "source_indices",
                    "camera_gt_start_frame": int(start_frame),
                    "camera_gt_frame_stride": int(frame_stride),
                    "camera_gt_first_index": int(indices[0]) if indices else None,
                    "camera_gt_last_index": int(indices[-1]) if indices else None,
                }
            )
            return selected, details

    details["camera_gt_alignment"] = "resampled"
    return _resample_camera_matrices(cameras, target_frames), details


def _pose_vectors_to_camera_matrices(pose_vectors: np.ndarray) -> np.ndarray:
    poses_np = np.asarray(pose_vectors, dtype=np.float32)
    if poses_np.ndim != 2 or poses_np.shape[1] != 7:
        raise ValueError(f"pose vectors must have shape (N, 7), got {poses_np.shape}")
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
        raise ModuleNotFoundError("torch is required for MegaSAM camera_error") from exc

    with metric_runtime_context():
        from lietorch import SE3  # noqa: PLC0415

        # MegaSAM's demo exports camera-to-world poses as SE3(poses).inv().
        cameras_pred = SE3(torch.as_tensor(poses_np)).inv().matrix()
        return cameras_pred.detach().cpu().numpy().astype(np.float32)


def _score_camera_matrices(
    cameras_pred: np.ndarray,
    cameras_gt: np.ndarray,
    scale: float,
) -> tuple[list[float], dict[str, Any]]:
    components, details = score_camera_trajectories(
        cameras_pred,
        cameras_gt,
        gt_scale=float(scale),
    )
    return [float(components[0]), float(components[1])], details


def _intrinsics_details(intrinsics: np.ndarray | None) -> dict[str, Any]:
    if intrinsics is None:
        return {}
    payload = np.asarray(intrinsics)
    details: dict[str, Any] = {
        "megasam_intrinsics_shape": [int(value) for value in payload.shape],
    }
    if payload.ndim == 2 and payload.shape[1] >= 4 and len(payload) > 0:
        details["megasam_intrinsics_first"] = [float(value) for value in payload[0, :4]]
    elif payload.ndim == 1 and payload.shape[0] >= 4:
        details["megasam_intrinsics_first"] = [float(value) for value in payload[:4]]
    return details


def _video_frame_cache_key(path: Path, *, cache_variant: str = "") -> str:
    from worldatlas.benchmark.sample_identity import file_sha256
    # Decoding/resizing implementation changes must invalidate decoded pixels,
    # not only the final score cache. Include the actual decoder build as well.
    decoder = hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()
    payload = f"content_v3::{path.resolve()}::{file_sha256(path)}::{cache_variant}::{file_sha256(Path(__file__))}::{decoder}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resize_frame_to_max_side(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    if max_side is None or max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    current_max_side = max(height, width)
    if current_max_side <= max_side:
        return frame
    scale = max_side / float(current_max_side)
    target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def materialize_frame_paths(media_path: str, *, max_side: int | None = None) -> list[str]:
    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"media path does not exist: {path}")
    if path.is_dir():
        image_paths = [
            image_path
            for image_path in sorted(path.iterdir())
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        return [str(image_path) for image_path in image_paths]
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return [str(path)]

    cache_variant = "decode_complete_v3::" + (f"max_side={max_side}" if max_side else "original")
    frame_dir = _runtime_cache_dir("metric_frames") / _video_frame_cache_key(path, cache_variant=cache_variant)
    frame_dir.mkdir(parents=True, exist_ok=True)
    # Serial/parallel runs can request the same video simultaneously. Publish
    # only complete indexes under a lock; never expose a half-written cache.
    import fcntl
    with (frame_dir / '.decode.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _materialize_video_frames(path, frame_dir, max_side=max_side)


def _materialize_video_frames(path: Path, frame_dir: Path, *, max_side: int | None) -> list[str]:
    from worldatlas.benchmark.sample_identity import file_sha256
    index_path = frame_dir / "index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            count, expected = payload['frame_count'], payload['expected_frame_count']
            generation = payload['generation']
            names = payload['frames']
            valid = (type(count) is int and count > 0 and type(expected) is int
                     and (expected <= 0 or count == expected)
                     and isinstance(generation, str) and re.fullmatch(r'[a-f0-9]{32}', generation)
                     and names == [f'{generation}/{i:06d}.png' for i in range(count)]
                     and payload['max_side'] == max_side
                     and payload['source_sha256'] == file_sha256(path)
                     and set(payload['sha256']) == set(names))
            if valid and all((frame_dir / name).is_file() and payload['sha256'][name] == file_sha256(frame_dir / name)
                             for name in names):
                # SAM2 reads the containing directory, not just this list.
                actual = {p.name for p in (frame_dir / generation).iterdir()
                          if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}}
                if actual == {Path(name).name for name in names}:
                    return [str(frame_dir / name) for name in names]
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Corrupt/incomplete cache is rebuilt without deleting it.

    source_hash = file_sha256(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    expected_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    frame_names: list[str] = []
    frame_index = 0
    generation = uuid.uuid4().hex
    (frame_dir / generation).mkdir()
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame = _resize_frame_to_max_side(frame, max_side)
        frame_name = f"{generation}/{frame_index:06d}.png"
        frame_path = frame_dir / frame_name
        if not cv2.imwrite(str(frame_path), frame):
            capture.release()
            raise RuntimeError(f"failed to write decoded frame: {frame_path}")
        frame_names.append(frame_name)
        frame_index += 1
    capture.release()
    if not frame_names:
        raise RuntimeError(f"video contains no decodable frames: {path}")
    if expected_count > 0 and len(frame_names) != expected_count:
        if not declared_frame_count_off_by_one(expected_count, len(frame_names)):
            raise RuntimeError(f"Incomplete video decode: {len(frame_names)}/{expected_count} frames: {path}")
        expected_count = len(frame_names)
    if file_sha256(path) != source_hash:
        raise RuntimeError(f"Video changed during frame decoding: {path}")
    temporary_index = frame_dir / f'index.{generation}.tmp'
    temporary_index.write_text(
        json.dumps({"frames": frame_names, "max_side": max_side, "fps": fps,
                    "frame_count": len(frame_names), "expected_frame_count": expected_count,
                    "generation": generation, "source_sha256": source_hash,
                    "sha256": {name: file_sha256(frame_dir / name) for name in frame_names}}),
        encoding="utf-8",
    )
    temporary_index.replace(index_path)
    return [str(frame_dir / frame_name) for frame_name in frame_names]


def _normalize_empirical(score: float, spec: dict[str, Any]) -> float:
    empirical_max = float(spec["empirical_max"])
    empirical_min = float(spec.get("empirical_min", 0.0))
    if not all(math.isfinite(v) for v in (score, empirical_min, empirical_max)) or empirical_max <= empirical_min:
        raise ValueError('Empirical normalization requires finite values and max > min')
    normalized = (max(empirical_min, min(empirical_max, score)) - empirical_min) / (
        empirical_max - empirical_min
    )
    if not bool(spec.get("higher_is_better", True)):
        normalized = 1.0 - normalized
    return max(0.0, min(1.0, float(normalized)))


def _normalize_empirical_components(score: list[float], spec: dict[str, Any]) -> float:
    minima = spec.get("empirical_min", [0.0] * len(score))
    if not score or any(not isinstance(v, (list, tuple)) or len(v) != len(score)
                        for v in (spec['empirical_max'], minima, spec['higher_is_better'])):
        raise ValueError('Normalization component counts differ; truncation is forbidden')
    normalized_components: list[float] = []
    for score_i, empirical_max, empirical_min, higher_is_better in zip(
        score,
        spec["empirical_max"],
        minima,
        spec["higher_is_better"],
    ):
        normalized_component = _normalize_empirical(
            float(score_i),
            {
                "empirical_max": float(empirical_max),
                "empirical_min": float(empirical_min),
                "higher_is_better": bool(higher_is_better),
            },
        )
        normalized_components.append(normalized_component)
    reducer = str(spec.get("reducer", "geometric_mean"))
    if reducer not in {'geometric_mean', 'arithmetic_mean'}:
        raise ValueError(f'Unknown normalization reducer: {reducer}')
    if reducer == "arithmetic_mean":
        return float(sum(normalized_components) / len(normalized_components))
    product = 1.0
    for value in normalized_components:
        product *= value
    return float(product ** (1.0 / len(normalized_components)))


def _normalize_zscore(score: float, spec: dict[str, Any]) -> float:
    avg = float(spec["avg"])
    std = float(spec["std"])
    z_max = float(spec["z_max"])
    z_min = float(spec["z_min"])
    score_range = spec.get("range", [0.25, 0.75])
    if (len(score_range) != 2 or not all(math.isfinite(float(v)) for v in (score, avg, std, z_max, z_min, *score_range))
            or std <= 0 or z_max <= z_min or not 0 <= float(score_range[0]) < float(score_range[1]) <= 1):
        raise ValueError('Z-score normalization requires finite values, positive std and increasing valid ranges')
    x_prime = ((float(score) - avg) / std - z_min) / (z_max - z_min)
    if not bool(spec.get("higher_is_better", True)):
        x_prime = 1.0 - x_prime
    normalized = float(score_range[0]) + (float(score_range[1]) - float(score_range[0])) * x_prime
    return max(0.0, min(1.0, normalized))


def normalize_official_score(score: Any, spec: dict[str, Any]) -> float:
    from worldatlas.benchmark.metrics.base import apply_display_error_ref

    if "avg" in spec:
        return _normalize_zscore(float(score), spec)
    if isinstance(score, (list, tuple)):
        return _normalize_empirical_components([float(value) for value in score], spec)
    if spec.get("error_ref") is not None and spec.get("error_from") == "raw":
        return apply_display_error_ref(float(score), spec)
    return _normalize_empirical(float(score), spec)


def _native_score_details(score: Any) -> dict[str, Any]:
    if isinstance(score, (list, tuple)):
        return {"native_components": [round(float(value), 4) for value in score]}
    return {"native_score": round(float(score), 4)}


def _run_metric(metric_name: str, *args: Any, **kwargs: Any) -> Any:
    metric_class = _load_metric_class(metric_name)
    try:
        return metric_class(*args, **kwargs)
    except Exception:
        _empty_cuda_cache()
        raise


def _run_metric_instance(
    metric_name: str,
    *args: Any,
    runtime: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    if not _coerce_runtime_bool((runtime or {}).get("cache_instance"), default=False):
        return _run_metric(metric_name, *args, **kwargs)
    key = (metric_name, repr(args), repr(sorted(kwargs.items())))
    metric = _METRIC_INSTANCE_CACHE.get(key)
    if metric is None:
        metric = _run_metric(metric_name, *args, **kwargs)
        _METRIC_INSTANCE_CACHE[key] = metric
    return metric


def inspect_metric_source_layout() -> dict[str, Any]:
    """Inspect metric source layout -> dict[str, Any]."""
    try:
        layout = resolve_metric_source_layout()
    except Exception as exc:
        return {
            "ready": False,
            "error": str(exc),
        }

    return {
        "ready": True,
        "source_root": str(layout.source_root),
        "package_name": layout.package_name,
        "droid_root": str(layout.droid_root) if layout.droid_root is not None else None,
        "droid_build_root": str(layout.droid_build_root) if layout.droid_build_root is not None else None,
        "lietorch_root": str(layout.lietorch_root) if layout.lietorch_root is not None else None,
        "sea_raft_root": str(layout.sea_raft_root) if layout.sea_raft_root is not None else None,
        "grounding_dino_root": (
            str(layout.grounding_dino_root) if layout.grounding_dino_root is not None else None
        ),
        "sam2_root": str(layout.sam2_root) if layout.sam2_root is not None else None,
        "vfi_mamba_root": str(layout.vfi_mamba_root) if layout.vfi_mamba_root is not None else None,
    }


def inspect_official_metric_import(metric_name: str) -> dict[str, Any]:
    if metric_name not in OFFICIAL_IMPORT_TARGETS:
        raise KeyError(f"unsupported official metric: {metric_name}")

    try:
        layout = resolve_metric_source_layout()
    except Exception as exc:
        return {
            "ready": False,
            "error": str(exc),
        }

    module_suffix, class_name = OFFICIAL_IMPORT_TARGETS[metric_name]
    module_name = f"{layout.package_name}.{module_suffix}"
    module_path = layout.source_root / Path(module_suffix.replace(".", "/")).with_suffix(".py")
    if not module_path.exists():
        return {
            "ready": False,
            "module": module_name,
            "class_name": class_name,
            "source_file": str(module_path),
            "error": f"metric source file not found: {module_path}",
        }

    try:
        module_ast = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    except (OSError, SyntaxError) as exc:
        return {
            "ready": False,
            "module": module_name,
            "class_name": class_name,
            "source_file": str(module_path),
            "error": str(exc),
        }

    if not any(
        isinstance(node, ast.ClassDef) and node.name == class_name for node in module_ast.body
    ):
        return {
            "ready": False,
            "module": module_name,
            "class_name": class_name,
            "source_file": str(module_path),
            "error": f"class {class_name} not declared in {module_path}",
        }

    return {
        "ready": True,
        "class_name": class_name,
        "module": module_name,
        "source_file": str(module_path),
    }


def compute_camera_error(
    prediction_path: str,
    cameras_gt: np.ndarray,
    scale: float,
    normalization: dict[str, Any],
    *,
    frame_paths: list[str] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delegated = _maybe_run_metric_in_subprocess(
        "camera_error",
        {
            "prediction_path": prediction_path,
            "cameras_gt": np.asarray(cameras_gt).tolist(),
            "scale": float(scale),
            "normalization": normalization,
            "frame_paths": frame_paths,
            "runtime": runtime,
        },
        runtime,
    )
    if delegated is not None:
        return delegated
    frame_paths = list(frame_paths) if frame_paths is not None else materialize_frame_paths(prediction_path)
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
        raise ModuleNotFoundError("torch is required for camera_error") from exc

    with metric_runtime_context():
        metric = _run_metric("camera_error")
        _configure_droid_runtime(metric)
        droid_details = _configure_droid_intrinsics(metric, frame_paths, runtime)
        try:
            score = metric._compute_scores(
                frame_paths,
                torch.as_tensor(np.asarray(cameras_gt)),
                scale=scale,
            )
            alignment_details = dict(getattr(metric, "_alignment_details", {}))
        finally:
            del metric
            _empty_cuda_cache()
    raw_components = [float(score[0]), float(score[1])]
    return {
        "raw": {
            "rotation_error": round(raw_components[0], 4),
            "translation_error": round(raw_components[1], 4),
        },
        "normalized": normalize_official_score(raw_components, normalization),
        "backend": "droid_slam",
        "details": {
            "frame_count": len(frame_paths),
            "camera_scale": float(scale),
            **droid_details,
            **alignment_details,
            **_native_score_details(raw_components),
        },
    }


def compute_camera_error_megasam(
    prediction_path: str,
    cameras_gt: np.ndarray,
    scale: float,
    normalization: dict[str, Any],
    *,
    frame_paths: list[str] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame_paths = list(frame_paths) if frame_paths is not None else materialize_frame_paths(prediction_path)
    source_width, source_height = _first_frame_resolution(frame_paths)

    megasam = _run_megasam_tracking(prediction_path, runtime)
    poses = np.asarray(megasam["poses"], dtype=np.float32)
    pose_count = int(len(poses))
    if pose_count == 0:
        raise ValueError("MegaSAM produced zero poses")

    metadata = dict(megasam.get("metadata") or {})
    prepared_frame_count = int(metadata.get("frame_count") or pose_count)
    gt_selected, gt_details = _select_gt_for_megasam(
        np.asarray(cameras_gt, dtype=np.float32),
        target_frames=pose_count,
        source_frame_count=len(frame_paths),
        frame_stride=int(megasam["frame_stride"]),
        start_frame=int(megasam["start_frame"]),
        max_frames=megasam["max_frames"],
    )
    cameras_pred = _pose_vectors_to_camera_matrices(poses)
    raw_components, alignment_details = _score_camera_matrices(
        cameras_pred,
        gt_selected,
        scale,
    )

    details = {
        "frame_count": pose_count,
        "source_frame_count": len(frame_paths),
        "camera_scale": float(scale),
        "megasam_scene": str(megasam["scene"]),
        "megasam_run_root": str(megasam["run_root"]),
        "megasam_log_path": str(megasam["log_path"]),
        "megasam_prepared_frame_count": prepared_frame_count,
        "megasam_pose_count": pose_count,
        "megasam_frame_stride": int(megasam["frame_stride"]),
        "megasam_start_frame": int(megasam["start_frame"]),
        "megasam_max_frames": megasam["max_frames"],
        "megasam_intrinsics_policy": "unidepth_actual_resolution",
        "megasam_source_resolution": [source_width, source_height],
        **gt_details,
        **alignment_details,
        **_intrinsics_details(megasam.get("intrinsics")),
        **_native_score_details(raw_components),
    }
    return {
        "raw": {
            "rotation_error": round(raw_components[0], 4),
            "translation_error": round(raw_components[1], 4),
        },
        "normalized": normalize_official_score(raw_components, normalization),
        "backend": "megasam",
        "details": details,
    }


def compute_camera_error_megasam_native_intended(
    prediction_path: str,
    sample: Any,
    normalization: dict[str, Any],
    *,
    frame_paths: list[str] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from worldatlas.benchmark.metrics.base import apply_display_error_ref
    from worldatlas.benchmark.native_camera import score_native_intended_cameras  # noqa: PLC0415

    if frame_paths is None:
        source_width, source_height, source_frame_count = _media_resolution_and_frame_count(
            prediction_path
        )
    else:
        frame_paths = list(frame_paths)
        source_width, source_height = _first_frame_resolution(frame_paths)
        source_frame_count = len(frame_paths)

    megasam = _run_megasam_tracking(prediction_path, runtime)
    poses = np.asarray(megasam["poses"], dtype=np.float32)
    pose_count = int(len(poses))
    if pose_count == 0:
        raise ValueError("MegaSAM produced zero poses")

    metadata = dict(megasam.get("metadata") or {})
    prepared_frame_count = int(metadata.get("frame_count") or pose_count)
    if source_frame_count is None:
        source_frame_count = prepared_frame_count
    cameras_pred = _pose_vectors_to_camera_matrices(poses)
    native_result = score_native_intended_cameras(
        sample,
        prediction_path,
        cameras_pred,
    )

    details = {
        "frame_count": pose_count,
        "source_frame_count": source_frame_count,
        "megasam_scene": str(megasam["scene"]),
        "megasam_run_root": str(megasam["run_root"]),
        "megasam_log_path": str(megasam["log_path"]),
        "megasam_prepared_frame_count": prepared_frame_count,
        "megasam_pose_count": pose_count,
        "megasam_frame_stride": int(megasam["frame_stride"]),
        "megasam_start_frame": int(megasam["start_frame"]),
        "megasam_max_frames": megasam["max_frames"],
        "megasam_intrinsics_policy": "unidepth_actual_resolution",
        "megasam_source_resolution": [source_width, source_height],
        **_intrinsics_details(megasam.get("intrinsics")),
        **dict(native_result.get("details") or {}),
    }
    return {
        "raw": native_result["raw"],
        "normalized": apply_display_error_ref(native_result["normalized"], normalization),
        "backend": "megasam",
        "details": details,
    }


def compute_reprojection_error(
    prediction_path: str,
    normalization: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delegated = _maybe_run_metric_in_subprocess(
        "reprojection_error",
        {
            "prediction_path": prediction_path,
            "normalization": normalization,
            "runtime": runtime,
        },
        runtime,
    )
    if delegated is not None:
        return delegated
    frame_paths = materialize_frame_paths(prediction_path)
    with metric_runtime_context():
        metric = _run_metric("reprojection_error")
        _configure_droid_runtime(metric)
        droid_details = _configure_droid_intrinsics(metric, frame_paths, runtime)
        droid_details["droid_max_factors"] = int(getattr(metric._args, "max_factors", 0))
        try:
            score = float(metric._compute_scores(frame_paths))
            # The backend chooses its stride from the actual decoded length.
            droid_details["droid_frame_stride"] = int(getattr(metric._args, "stride", 1))
            droid_details["droid_tracked_frames"] = len(frame_paths[::droid_details["droid_frame_stride"]])
        finally:
            del metric
            _empty_cuda_cache()
    return {
        "raw": score,
        "normalized": normalize_official_score(score, normalization),
        "backend": "droid_slam",
        "details": {
            "frame_count": len(frame_paths),
            **droid_details,
            **_native_score_details(score),
        },
    }


def compute_optical_flow_aepe(
    prediction_path: str,
    normalization: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delegated = _maybe_run_metric_in_subprocess(
        "optical_flow_aepe",
        {
            "prediction_path": prediction_path,
            "normalization": normalization,
            "runtime": runtime,
        },
        runtime,
    )
    if delegated is not None:
        return delegated
    # SEA-RAFT's Spring-M configuration is designed around a 960-pixel model
    # input (the metric applies its own 0.5 scale to 1920-pixel source frames).
    # Honour the existing runtime frame limit here as the other flow metrics
    # do, otherwise multi-thousand-pixel image-static videos can require a
    # correlation tensor larger than an 80 GiB GPU.
    frame_max_side = _aepe_frame_max_side(runtime)
    frame_paths = materialize_frame_paths(prediction_path, max_side=frame_max_side)
    cache_instance = _coerce_runtime_bool((runtime or {}).get("cache_instance"), default=True)
    batch_size = _aepe_batch_size(runtime)
    with metric_runtime_context():
        metric = _run_metric_instance("optical_flow_aepe", runtime=runtime)
        metric._batch_size = batch_size
        try:
            score = float(metric._compute_scores(frame_paths))
        except MetricLoadError:
            raise
        finally:
            if not cache_instance:
                del metric
                _empty_cuda_cache()
    return {
        "raw": score,
        "normalized": normalize_official_score(score, normalization),
        "backend": "sea_raft",
        "details": {
            "frame_count": len(frame_paths),
            "frame_max_side": frame_max_side,
            "batch_size": batch_size,
            **_native_score_details(score),
        },
    }


def compute_motion_magnitude(
    prediction_path: str,
    normalization: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delegated = _maybe_run_metric_in_subprocess(
        "motion_magnitude",
        {
            "prediction_path": prediction_path,
            "normalization": normalization,
            "runtime": runtime,
        },
        runtime,
    )
    if delegated is not None:
        return delegated
    # Universal input policy, not a LongCat-specific fallback. SEA-RAFT then
    # applies its configured 0.5 scale (at most 960 px on the network input).
    frame_max_side = _coerce_runtime_int((runtime or {}).get("frame_max_side"), default=1920)
    source_width, source_height, _ = _media_resolution_and_frame_count(prediction_path)
    frame_paths = materialize_frame_paths(prediction_path, max_side=frame_max_side)
    eval_width, eval_height = _first_frame_resolution(frame_paths)
    flow_pixel_scale = (source_width / eval_width, source_height / eval_height)
    cache_instance = _coerce_runtime_bool((runtime or {}).get("cache_instance"), default=True)
    with metric_runtime_context():
        metric = _run_metric_instance("motion_magnitude", runtime=runtime)
        try:
            score = float(metric._compute_scores(frame_paths, flow_pixel_scale=flow_pixel_scale))
        except MetricLoadError:
            raise
        finally:
            if not cache_instance:
                del metric
                _empty_cuda_cache()
    return {
        "raw": score,
        "normalized": normalize_official_score(score, normalization),
        "backend": "sea_raft",
        "details": {
            "frame_count": len(frame_paths),
            "frame_max_side": frame_max_side,
            "source_resolution": [source_width, source_height],
            "evaluation_resolution": [eval_width, eval_height],
            "flow_pixel_scale": list(flow_pixel_scale),
            "flow_units": "source_pixels_per_frame_pair",
            "resolution_policy": f"max_side_{frame_max_side}_source_pixel_flow",
            **_native_score_details(score),
        },
    }


def compute_motion_smoothness(
    prediction_path: str,
    normalization: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delegated = _maybe_run_metric_in_subprocess(
        "motion_smoothness",
        {
            "prediction_path": prediction_path,
            "normalization": normalization,
            "runtime": runtime,
        },
        runtime,
    )
    if delegated is not None:
        return delegated
    # VFIMamba + LPIPS at native 8K (longcat image_dynamic) OOMs on H800 when
    # several workers share a GPU. Match AEPE: default 1920, Hope/env override.
    frame_max_side = _motion_sm_frame_max_side(runtime)
    frame_paths = materialize_frame_paths(prediction_path, max_side=frame_max_side)
    cache_instance = _coerce_runtime_bool((runtime or {}).get("cache_instance"), default=True)
    with metric_runtime_context():
        metric = _run_metric_instance("motion_smoothness", runtime=runtime)
        try:
            score = metric._compute_scores(frame_paths)
        except MetricLoadError:
            raise
        finally:
            if not cache_instance:
                del metric
                _empty_cuda_cache()
    components = [float(score[0]), float(score[1]), float(score[2])]
    return {
        "raw": {
            "mse": round(components[0], 4),
            "ssim": round(components[1], 4),
            "lpips": round(components[2], 4),
        },
        "normalized": normalize_official_score(components, normalization),
        "backend": "vfi_mamba",
        "details": {
            "frame_count": len(frame_paths),
            "frame_max_side": frame_max_side,
            "motion_smoothness_scoring_version": "v2_float32_mse",
            "native_components": {
                "mse": round(components[0], 4),
                "ssim": round(components[1], 4),
                "lpips": round(components[2], 4),
            },
        },
    }


def compute_motion_accuracy(
    prediction_path: str,
    mask_dir: str,
    normalization: dict[str, Any],
    *,
    generate_type: str = "i2v",
    runtime: dict[str, Any] | None = None,
    source_path: str | None = None,
    relative_path: str | None = None,
    conditioning_geometry: dict | None = None,
) -> dict[str, Any]:
    delegated = _maybe_run_metric_in_subprocess(
        "motion_accuracy",
        {
            "prediction_path": prediction_path,
            "mask_dir": mask_dir,
            "normalization": normalization,
            "generate_type": generate_type,
            "runtime": runtime,
            "source_path": source_path,
            "relative_path": relative_path,
            "conditioning_geometry": conditioning_geometry,
        },
        runtime,
    )
    if delegated is not None:
        return delegated
    frame_max_side = _coerce_runtime_int((runtime or {}).get("frame_max_side"))
    frame_paths = materialize_frame_paths(prediction_path, max_side=frame_max_side)
    mask_paths = [
        str(path)
        for path in sorted(Path(mask_dir).iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not mask_paths:
        raise FileNotFoundError(f"no masks found in {mask_dir}")
    if not source_path or not relative_path or not conditioning_geometry:
        raise ValueError("Motion accuracy requires verified source path and conditioning geometry")
    if len(frame_paths) < 2:
        raise ValueError("Motion accuracy requires at least two decoded video frames")
    from worldatlas.benchmark.mask_geometry import aligned_mask_paths
    with Image.open(frame_paths[0]) as first_frame:
        target_size = first_frame.size
    capture = cv2.VideoCapture(prediction_path)
    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    cache_instance = _coerce_runtime_bool((runtime or {}).get("cache_instance"), default=False)
    with metric_runtime_context():
        metric = _run_metric_instance("motion_accuracy", generate_type, runtime=runtime)
        runtime_details = _configure_motion_accuracy_runtime(metric, runtime)
        try:
            with aligned_mask_paths(mask_dir, source_path=source_path, relative_path=relative_path,
                                    geometry=conditioning_geometry, target_size=target_size) as aligned:
                score = metric._compute_scores(frame_paths, aligned, [])
            score_details = dict(getattr(metric, "_last_score_details", {}))
        finally:
            if not cache_instance:
                del metric
                _empty_cuda_cache()
    details = {
        "scoring_version": "aligned_mask_contrast_v2",
        "conditioning_geometry": conditioning_geometry,
        "decoded_width": target_size[0],
        "decoded_height": target_size[1],
        "native_fps": native_fps,
        **score_details,
        "frame_count": len(frame_paths),
        "frame_max_side": frame_max_side,
        "mask_count": len(mask_paths),
        "mask_dir": mask_dir,
        "generate_type": generate_type,
        **runtime_details,
    }
    if score is None:
        from worldatlas.benchmark.mask_geometry import PROPAGATED_MASK_NA_REASON
        reason = score_details.get("not_applicable_reason") or PROPAGATED_MASK_NA_REASON
        details["not_applicable_reason"] = reason
        return {
            "raw": None,
            "normalized": None,
            "backend": "sam2_sea_raft",
            "eligibility_status": "not_applicable",
            "error": reason,
            "details": details,
        }
    details.update(_native_score_details(score))
    return {
        "raw": float(score),
        "normalized": normalize_official_score(score, normalization),
        "backend": "sam2_sea_raft",
        "details": details,
    }


__all__ = [
    "OFFICIAL_METRIC_NAMES",
    "compute_camera_error",
    "compute_camera_error_megasam",
    "compute_camera_error_megasam_native_intended",
    "compute_motion_accuracy",
    "compute_motion_magnitude",
    "compute_motion_smoothness",
    "compute_optical_flow_aepe",
    "compute_reprojection_error",
    "inspect_metric_source_layout",
    "inspect_official_metric_import",
    "materialize_frame_paths",
    "metric_runtime_context",
    "normalize_official_score",
    "resolve_official_runtime_python",
    "resolve_metric_source_layout",
]
