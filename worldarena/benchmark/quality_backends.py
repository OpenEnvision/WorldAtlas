"""Backend implementations for image and video quality metrics."""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from functools import lru_cache
import gc
import os
from pathlib import Path
import sys
import types
from typing import Any, Sequence

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

import numpy as np
from PIL import Image

from worldarena.common.checkpoints import (
    checkpoint_env,
    checkpoint_path,
    project_root,
    resolve_checkpoint_path,
    resolve_project_path,
)
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
HPSV3_DEFAULT_P1 = 5.21
HPSV3_DEFAULT_P99 = 8.66


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _round(values: Sequence[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def _quality_cache_dir() -> Path:
    """Quality cache dir -> Path."""
    base = checkpoint_path("quality")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _quality_device() -> str:
    """Quality device -> str."""
    override = os.environ.get("WORLDARENA_QUALITY_DEVICE")
    if override in {"cpu", "cuda"}:
        return override
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_local_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    resolved = resolve_project_path(path)
    return resolved if resolved is not None else project_root() / path


def _ensure_numpy_compat_for_imgaug() -> None:
    """Ensure numpy compat for imgaug."""
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
    """Ensure timm compat for pyiqa."""
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
    """Patch pyiqa clip cache."""
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


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "reshape") and hasattr(value, "cpu"):
        return [float(item) for item in value.reshape(-1).cpu().tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [_to_float(value)]


def _quality_batch_size() -> int:
    value = os.environ.get("WORLDARENA_QUALITY_BATCH_SIZE", "8")
    try:
        return max(1, int(value))
    except ValueError:
        return 8


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


def _prepare_hpsv3_runtime_compat() -> None:
    """Prepare hpsv3 runtime compat."""
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    os.environ.update(checkpoint_env())
    ensure_pkg_resources_compat()


def _configure_pyiqa_cache() -> None:
    """Configure pyiqa cache."""
    try:
        from pyiqa.utils import download_util
    except Exception:
        return

    pyiqa_cache_dir = Path(os.environ["TORCH_HOME"]).expanduser() / "hub" / "pyiqa"
    pyiqa_cache_dir.mkdir(parents=True, exist_ok=True)
    download_util.DEFAULT_CACHE_DIR = str(pyiqa_cache_dir)


@contextmanager
def _force_laion_clip_cpu_load(component_name: str):
    """Avoid hard exits from OpenAI CLIP ViT-L/14 direct CUDA construction."""
    if component_name != "aesthetic":
        yield
        return
    try:
        import clip
    except Exception:
        yield
        return

    original_load = clip.load

    def load_with_cpu_default(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ViT-L/14" and len(args) == 0 and "device" not in kwargs:
            kwargs["device"] = "cpu"
        return original_load(name, *args, **kwargs)

    clip.load = load_with_cpu_default
    try:
        yield
    finally:
        clip.load = original_load


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
    with _force_laion_clip_cpu_load(component_name):
        metric = pyiqa.create_metric(metric_name, device=device).to(device)
    metric.eval()
    return backend_name, metric, device, scale


def quality_component_available(component_name: str) -> bool:
    try:
        return load_quality_component(component_name) is not None
    except Exception:
        return False


def clear_quality_component_cache() -> None:
    """Release cached PyIQA component models before loading another CLIP stack."""
    load_quality_component.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


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
    import torch

    batch_size = _quality_batch_size()
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            batch = torch.cat([_frame_to_float_tensor(frame, device) for frame in chunk], dim=0)
            chunk_scores = _to_float_list(metric(batch))
            if len(chunk_scores) != len(chunk):
                chunk_scores = [
                    _to_float(metric(_frame_to_float_tensor(frame, device)))
                    for frame in chunk
                ]
            for native_score in chunk_scores:
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
        batch_size = _quality_batch_size()
        for image_batch in images.split(batch_size):
            chunk_scores = _to_float_list(metric(image_batch.to(device)))
            if len(chunk_scores) != len(image_batch):
                chunk_scores = [
                    _to_float(metric(image.unsqueeze(0).to(device)))
                    for image in image_batch
                ]
            native_scores.extend(chunk_scores)
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


def _resolve_hpsv3_root(root_path: str | os.PathLike[str] | None = None) -> Path | None:
    explicit = _resolve_local_path(root_path or os.environ.get("HPSV3_ROOT"))
    if explicit is not None:
        return explicit
    for candidate in (
        project_root() / "thirdparty" / "HPSv3",
        project_root() / "third_party" / "HPSv3",
    ):
        if candidate.is_dir():
            return candidate
    return None


def _resolve_hpsv3_checkpoint(checkpoint: str | os.PathLike[str] | None = None) -> Path:
    explicit = _resolve_local_path(checkpoint)
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"HPSv3 checkpoint not found: {explicit}")
        return explicit
    return checkpoint_path("HPSv3", "HPSv3.safetensors", kind="file", required=True)


def resolve_hpsv3_checkpoint(checkpoint: str | os.PathLike[str] | None = None) -> Path:
    return _resolve_hpsv3_checkpoint(checkpoint)


def _resolve_hpsv3_config(config_path: str | os.PathLike[str] | None = None) -> Path:
    explicit = _resolve_local_path(config_path)
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"HPSv3 config not found: {explicit}")
        return explicit
    return checkpoint_path("HPSv3", "HPSv3_7B_local.yaml", kind="file", required=True)


def resolve_hpsv3_config(config_path: str | os.PathLike[str] | None = None) -> Path:
    return _resolve_hpsv3_config(config_path)


def _resolved_hpsv3_config(
    config_path: Path,
    *,
    qwen_model_path: str | os.PathLike[str] | None = None,
) -> Path:
    qwen_path = _resolve_local_path(qwen_model_path)
    if qwen_path is None:
        default_qwen = checkpoint_path("Qwen2-VL-7B-Instruct", kind="dir", required=False)
        qwen_path = default_qwen if default_qwen.is_dir() else None
    if qwen_path is None:
        return config_path
    if not qwen_path.is_dir():
        raise FileNotFoundError(f"HPSv3 Qwen model directory not found: {qwen_path}")

    try:
        import yaml
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError("PyYAML is required to rewrite HPSv3 config") from exc

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload["model_name_or_path"] = str(qwen_path)
    resolved_config = checkpoint_path("HPSv3", "_HPSv3_7B_resolved.yaml")
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    # Multiple evaluation shards can initialize HPSv3 at the same time.  A
    # direct write lets another process observe a truncated YAML file, which
    # then surfaces as seemingly unrelated missing TrainingConfig fields.
    temporary_config = resolved_config.with_name(
        f".{resolved_config.name}.{os.getpid()}.tmp"
    )
    temporary_config.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    temporary_config.replace(resolved_config)
    return resolved_config


def _prepare_hpsv3_import(root_path: Path | None) -> None:
    try:
        import transformers.image_utils as image_utils
    except Exception:
        image_utils = None
    if image_utils is not None and not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = list[np.ndarray]

    if root_path is not None:
        root_text = str(root_path)
        if root_path.is_dir() and root_text not in sys.path:
            sys.path.insert(0, root_text)


@lru_cache(maxsize=16)
def load_hpsv3_backend(
    root_path: str | None = None,
    checkpoint: str | None = None,
    config_path: str | None = None,
    qwen_model_path: str | None = None,
    device: str | None = None,
) -> tuple[str, Any, str, str, str] | None:
    try:
        _prepare_hpsv3_runtime_compat()
        root = _resolve_hpsv3_root(root_path)
        if root is None and importlib.util.find_spec("hpsv3") is None:
            return None
        _prepare_hpsv3_import(root)
        from hpsv3.inference import HPSv3RewardInferencer
    except (ImportError, ModuleNotFoundError):
        return None

    checkpoint_path_resolved = _resolve_hpsv3_checkpoint(checkpoint)
    config_path_resolved = _resolved_hpsv3_config(
        _resolve_hpsv3_config(config_path),
        qwen_model_path=qwen_model_path,
    )
    device_name = device or _quality_device()
    inferencer = HPSv3RewardInferencer(
        config_path=str(config_path_resolved),
        checkpoint_path=str(checkpoint_path_resolved),
        device=device_name,
    )
    return (
        "hpsv3",
        inferencer,
        device_name,
        str(checkpoint_path_resolved),
        str(config_path_resolved),
    )


def compute_hpsv3_norm_scores(
    frames: Sequence[np.ndarray],
    *,
    prompt: str = "",
    p1: float = HPSV3_DEFAULT_P1,
    p99: float = HPSV3_DEFAULT_P99,
    batch_size: int = 8,
    root_path: str | None = None,
    checkpoint: str | None = None,
    config_path: str | None = None,
    qwen_model_path: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    payload = load_hpsv3_backend(
        root_path=root_path,
        checkpoint=checkpoint,
        config_path=config_path,
        qwen_model_path=qwen_model_path,
        device=device,
    )
    if payload is None:
        raise ModuleNotFoundError("HPSv3 backend is unavailable")
    if p99 <= p1:
        raise ValueError(f"HPSv3 normalization requires p99 > p1, got p1={p1}, p99={p99}")
    if batch_size <= 0:
        raise ValueError(f"HPSv3 batch_size must be positive, got {batch_size}")

    backend_name, inferencer, device_name, checkpoint_path_resolved, config_path_resolved = payload
    if not frames:
        return {
            "backend": backend_name,
            "raw": None,
            "raw_reward_mean": None,
            "raw_rewards": [],
            "normalization_percentiles": [float(p1), float(p99)],
            "score_0_100": None,
            "checkpoint_path": checkpoint_path_resolved,
            "config_path": config_path_resolved,
            "device": device_name,
        }

    prompts = [prompt] * len(frames)
    raw_rewards: list[float] = []
    # HPSv3's fetch_image accepts PIL images directly.  Keeping lossless RGB
    # frames in memory avoids writing and reopening up to 20 large PNG files
    # per video while preserving exactly the same pixels.
    image_inputs = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]

    import torch

    with torch.inference_mode():
        for start in range(0, len(image_inputs), batch_size):
            stop = start + batch_size
            rewards = inferencer.reward(
                prompts[start:stop],
                image_inputs[start:stop],
            )
            for reward in rewards:
                if hasattr(reward, "reshape"):
                    value = reward.reshape(-1)[0]
                elif isinstance(reward, (list, tuple)):
                    value = reward[0]
                else:
                    value = reward
                raw_rewards.append(_to_float(value))

    raw_reward_mean = float(np.mean(raw_rewards))
    normalized = clamp01((raw_reward_mean - float(p1)) / (float(p99) - float(p1)))
    return {
        "backend": backend_name,
        "raw": normalized,
        "raw_reward_mean": raw_reward_mean,
        "raw_rewards": _round(raw_rewards),
        "normalization_percentiles": [float(p1), float(p99)],
        "score_0_100": round(normalized * 100.0, 4),
        "checkpoint_path": checkpoint_path_resolved,
        "config_path": config_path_resolved,
        "device": device_name,
    }


__all__ = [
    "HPSV3_DEFAULT_P1",
    "HPSV3_DEFAULT_P99",
    "QUALITY_COMPONENT_SPECS",
    "REQUIRED_REFERENCE_QUALITY_COMPONENTS",
    "compute_hpsv3_norm_scores",
    "compute_musiq_scores",
    "compute_quality_component_scores",
    "clear_quality_component_cache",
    "load_hpsv3_backend",
    "load_musiq_backend",
    "load_quality_component",
    "quality_component_available",
    "resolve_hpsv3_checkpoint",
    "resolve_hpsv3_config",
    "resolve_musiq_spaq_weights",
]
