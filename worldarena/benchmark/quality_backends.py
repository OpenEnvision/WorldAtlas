from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys
import types
from typing import Any, Sequence

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

import numpy as np

from worldarena.common.checkpoints import checkpoint_env, checkpoint_path, resolve_checkpoint_path
from worldarena.benchmark.pkg_resources_compat import (
    ensure_openai_clip_cache_compat,
    ensure_pkg_resources_compat,
    resolve_openai_clip_cache_dir,
)


QUALITY_COMPONENT_SPECS = {
    "clip_iqa": ("clipiqa_plus", "clipiqa+", 1.0),
    "aesthetic": ("laion_aes", "laion_aes", 10.0),
}
REQUIRED_REFERENCE_QUALITY_COMPONENTS = ("clip_iqa", "aesthetic")
MUSIQ_NATIVE_SCORE_SCALE = 100.0


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _round(values: Sequence[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def _quality_cache_dir() -> Path:
    base = checkpoint_path("quality")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _quality_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_numpy_compat_for_imgaug() -> None:
    if hasattr(np, "sctypes"):
        return
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_, np.str_, np.bytes_, np.void],
        "inexact": [np.float16, np.float32, np.float64, np.complex64, np.complex128],
    }


def _ensure_timm_compat_for_pyiqa() -> None:
    try:
        from timm.models.helpers import build_model_with_cfg
        import timm.models.layers as timm_layers
    except Exception:
        return

    if "timm.models._builder" not in sys.modules:
        builder_module = types.ModuleType("timm.models._builder")
        builder_module.build_model_with_cfg = build_model_with_cfg
        sys.modules["timm.models._builder"] = builder_module

    if not hasattr(timm_layers, "_assert"):
        def _assert(condition: bool, message: str = "") -> None:
            assert condition, message

        timm_layers._assert = _assert
    sys.modules.setdefault("timm.layers", timm_layers)


def _patch_pyiqa_clip_cache() -> None:
    cache_dir = resolve_openai_clip_cache_dir()
    if cache_dir is None:
        return
    try:
        from pyiqa.archs import clipiqa_arch
    except Exception:
        return

    original_load = getattr(clipiqa_arch, "load", None)
    if original_load is None or getattr(original_load, "__worldarena_cache_patched__", False):
        return

    def patched_load(*args: Any, **kwargs: Any) -> Any:
        args_list = list(args)
        if "download_root" in kwargs:
            if kwargs["download_root"] is None:
                kwargs["download_root"] = str(cache_dir)
        elif len(args_list) >= 4:
            if args_list[3] is None:
                args_list[3] = str(cache_dir)
        else:
            kwargs["download_root"] = str(cache_dir)
        return original_load(*tuple(args_list), **kwargs)

    patched_load.__worldarena_cache_patched__ = True
    clipiqa_arch.load = patched_load


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _prepare_quality_runtime_compat(*, enable_clip_cache_compat: bool) -> None:
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    os.environ.update(checkpoint_env())
    torch_home = Path(os.environ["TORCH_HOME"])
    (torch_home / "hub").mkdir(parents=True, exist_ok=True)
    _ensure_numpy_compat_for_imgaug()
    ensure_pkg_resources_compat()
    _ensure_timm_compat_for_pyiqa()
    if enable_clip_cache_compat:
        ensure_openai_clip_cache_compat()


def _configure_pyiqa_cache() -> None:
    try:
        from pyiqa.utils import download_util
    except Exception:
        return

    pyiqa_cache_dir = Path(os.environ["TORCH_HOME"]).expanduser() / "hub" / "pyiqa"
    pyiqa_cache_dir.mkdir(parents=True, exist_ok=True)
    download_util.DEFAULT_CACHE_DIR = str(pyiqa_cache_dir)


def _frame_to_float_tensor(frame: np.ndarray, device: str) -> Any:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(frame))
    tensor = tensor.permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device)


@lru_cache(maxsize=len(QUALITY_COMPONENT_SPECS))
def load_quality_component(
    component_name: str,
) -> tuple[str, Any, str, float] | None:
    component_spec = QUALITY_COMPONENT_SPECS.get(component_name)
    if component_spec is None:
        raise KeyError(f"unsupported quality component: {component_name}")

    try:
        _prepare_quality_runtime_compat(enable_clip_cache_compat=True)
        import pyiqa
    except (ImportError, ModuleNotFoundError):
        return None

    _patch_pyiqa_clip_cache()
    _configure_pyiqa_cache()
    backend_name, metric_name, scale = component_spec
    device = _quality_device()
    metric = pyiqa.create_metric(metric_name).to(device)
    metric.eval()
    return backend_name, metric, device, scale


def quality_component_available(component_name: str) -> bool:
    try:
        return load_quality_component(component_name) is not None
    except Exception:
        return False


def compute_quality_component_scores(
    frames: Sequence[np.ndarray],
    *,
    component_name: str,
) -> dict[str, Any]:
    payload = load_quality_component(component_name)
    if payload is None:
        raise ModuleNotFoundError(f"{component_name} quality backend is unavailable")

    backend_name, metric, device, scale = payload
    if not frames:
        return {
            "backend": backend_name,
            "raw": None,
            "frame_scores": [],
            "native_frame_scores": [],
            "native_score_range": [0.0, scale],
        }

    native_scores: list[float] = []
    normalized_scores: list[float] = []
    for frame in frames:
        native_score = _to_float(metric(_frame_to_float_tensor(frame, device)))
        normalized_score = clamp01(native_score / scale)
        native_scores.append(native_score)
        normalized_scores.append(normalized_score)

    return {
        "backend": backend_name,
        "raw": _mean(normalized_scores),
        "frame_scores": _round(normalized_scores),
        "native_frame_scores": _round(native_scores),
        "native_score_range": [0.0, scale],
    }


def resolve_musiq_spaq_weights(model_path: str | os.PathLike[str] | None = None) -> str:
    if model_path is not None:
        resolved = resolve_checkpoint_path(model_path, kind="file", required=True)
        if resolved is None:
            raise FileNotFoundError("MUSIQ-SPAQ weights path is missing")
        return str(resolved)

    weights_path = checkpoint_path(
        "musiq",
        "musiq_spaq_ckpt-358bb6af.pth",
        kind="file",
        required=True,
    )
    return str(weights_path)


def _musiq_transform_frames(
    frames: Sequence[np.ndarray],
    *,
    preprocess_mode: str,
) -> Any:
    import torch
    from torchvision import transforms

    if not frames:
        return torch.empty((0, 3, 0, 0), dtype=torch.float32)

    images = torch.from_numpy(np.stack([np.ascontiguousarray(frame) for frame in frames]))
    images = images.permute(0, 3, 1, 2).float()
    if preprocess_mode.startswith("shorter"):
        _, _, height, width = images.size()
        if min(height, width) > 512:
            scale = 512.0 / min(height, width)
            images = transforms.Resize(
                size=(int(scale * height), int(scale * width)),
                antialias=False,
            )(images)
            if preprocess_mode == "shorter_centercrop":
                images = transforms.CenterCrop(512)(images)
    elif preprocess_mode == "longer":
        _, _, height, width = images.size()
        if max(height, width) > 512:
            scale = 512.0 / max(height, width)
            images = transforms.Resize(
                size=(int(scale * height), int(scale * width)),
                antialias=False,
            )(images)
    elif preprocess_mode == "None":
        return images / 255.0
    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.0


@lru_cache(maxsize=2)
def load_musiq_backend(
    model_path: str | None = None,
) -> tuple[str, Any, str, str] | None:
    try:
        _prepare_quality_runtime_compat(enable_clip_cache_compat=False)
        _configure_pyiqa_cache()
        from pyiqa.archs.musiq_arch import MUSIQ
    except (ImportError, ModuleNotFoundError):
        return None

    resolved_model_path = resolve_musiq_spaq_weights(model_path)
    device = _quality_device()
    metric = MUSIQ(pretrained_model_path=resolved_model_path).to(device)
    metric.eval()
    return "musiq", metric, device, resolved_model_path


def compute_musiq_scores(
    frames: Sequence[np.ndarray],
    *,
    model_path: str | None = None,
    preprocess_mode: str = "longer",
) -> dict[str, Any]:
    payload = load_musiq_backend(model_path)
    if payload is None:
        raise ModuleNotFoundError("MUSIQ backend is unavailable")

    backend_name, metric, device, resolved_model_path = payload
    if not frames:
        return {
            "backend": backend_name,
            "raw": None,
            "frame_scores": [],
            "native_frame_scores": [],
            "native_score_range": [0.0, MUSIQ_NATIVE_SCORE_SCALE],
            "preprocess_mode": preprocess_mode,
            "weights_path": resolved_model_path,
        }

    import torch

    images = _musiq_transform_frames(frames, preprocess_mode=preprocess_mode)
    native_scores: list[float] = []
    with torch.inference_mode():
        for image in images:
            native_score = _to_float(metric(image.unsqueeze(0).to(device)))
            native_scores.append(native_score)
    normalized_scores = [
        clamp01(native_score / MUSIQ_NATIVE_SCORE_SCALE)
        for native_score in native_scores
    ]
    return {
        "backend": backend_name,
        "raw": _mean(normalized_scores),
        "frame_scores": _round(normalized_scores),
        "native_frame_scores": _round(native_scores),
        "native_score_range": [0.0, MUSIQ_NATIVE_SCORE_SCALE],
        "preprocess_mode": preprocess_mode,
        "weights_path": resolved_model_path,
    }


__all__ = [
    "QUALITY_COMPONENT_SPECS",
    "REQUIRED_REFERENCE_QUALITY_COMPONENTS",
    "compute_musiq_scores",
    "compute_quality_component_scores",
    "load_musiq_backend",
    "load_quality_component",
    "quality_component_available",
    "resolve_musiq_spaq_weights",
]
