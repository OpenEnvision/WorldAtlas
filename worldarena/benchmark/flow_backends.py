"""Optical-flow computation backends used by consistency metrics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.common.checkpoints import checkpoint_path, resolve_checkpoint_path
from worldarena.benchmark.flow_sources import (
    amt_runtime_context,
    inspect_flow_source_layout,
    raft_runtime_context,
)
from worldarena.benchmark.metrics.base import clamp01
from worldarena.common.video_io import probe_video_fps, read_video_frames


class _AttrDict(dict):
    """Dict subclass that exposes keys as attributes for RAFT CLI-style args."""

    def __getattr__(self, key: str) -> Any:
        """Read a pseudo-attribute from the underlying mapping."""
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        """Write pseudo-attribute updates back into the mapping."""
        self[key] = value


def _normalize_device(device: str | Any | None) -> str:
    """Return a torch device string, defaulting to the first CUDA device when available."""
    import torch

    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


def resolve_amt_config_path(config_path: str | os.PathLike[str] | None = None) -> str:
    """Locate the AMT network config YAML from flow-source layout or an explicit path."""
    if config_path is None:
        layout = inspect_flow_source_layout()
        resolved = Path(str(layout["amt_config"])).resolve()
    else:
        resolved = Path(config_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"AMT config not found: {resolved}")
    return str(resolved)


def resolve_amt_s_checkpoint(ckpt_path: str | os.PathLike[str] | None = None) -> str:
    """Resolve the AMT-S interpolation checkpoint under ``ckpt/amt_model/amt-s.pth``."""
    if ckpt_path is not None:
        resolved = resolve_checkpoint_path(ckpt_path, kind="file", required=True)
        if resolved is None:
            raise FileNotFoundError("AMT-S checkpoint path is missing")
        return str(resolved)
    return str(
        checkpoint_path(
            "amt_model",
            "amt-s.pth",
            kind="file",
            required=True,
        )
    )


def resolve_raft_things_checkpoint(model_path: str | os.PathLike[str] | None = None) -> str:
    """Resolve the RAFT-things flow checkpoint under ``ckpt/raft_model/models/raft-things.pth``."""
    if model_path is not None:
        resolved = resolve_checkpoint_path(model_path, kind="file", required=True)
        if resolved is None:
            raise FileNotFoundError("RAFT checkpoint path is missing")
        return str(resolved)
    return str(
        checkpoint_path(
            "raft_model",
            "models",
            "raft-things.pth",
            kind="file",
            required=True,
        )
    )


@dataclass
class AMTMotionSmoothnessBackend:
    """Cached AMT-S backend used by the motion_smoothness metric."""

    model: Any
    device: str
    embt: Any
    anchor_resolution: int
    anchor_memory: int
    anchor_memory_bias: int
    vram_avail: int
    input_padder_cls: Any
    check_dim_and_resize: Any
    img_to_tensor: Any
    tensor_to_image: Any

    def score_video(self, video_path: str | Path) -> dict[str, Any]:
        """Interpolate every other frame with AMT-S and score mean absolute reconstruction error."""
        import torch

        frames = read_video_frames(Path(video_path))
        sampled_frames = frames[::2]
        if len(sampled_frames) < 2:
            return {
                "backend": "amt_s",
                "raw": None,
                "frame_count": len(frames),
                "sampled_frame_count": len(sampled_frames),
                "error": "not enough frames for AMT motion smoothness",
            }

        inputs = [self.img_to_tensor(frame).to(self.device) for frame in sampled_frames]
        inputs = self.check_dim_and_resize(inputs)
        height, width = inputs[0].shape[-2:]
        scale = self.anchor_resolution / (height * width)
        scale *= np.sqrt(max(self.vram_avail - self.anchor_memory_bias, 1) / self.anchor_memory)
        scale = 1.0 if scale > 1.0 else scale
        scale = 1.0 / np.floor(1.0 / np.sqrt(scale) * 16.0) * 16.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        padder = self.input_padder_cls(inputs[0].shape, int(16 / scale))
        inputs = padder.pad(*inputs)
        outputs = [inputs[0]]
        with torch.inference_mode():
            for left, right in zip(inputs[:-1], inputs[1:]):
                prediction = self.model(
                    left.to(self.device),
                    right.to(self.device),
                    self.embt,
                    scale_factor=scale,
                    eval=True,
                )["imgt_pred"]
                outputs += [prediction.cpu(), right.cpu()]

        outputs = padder.unpad(*outputs)
        interpolated_frames = [self.tensor_to_image(frame) for frame in outputs][1::2]
        target_frames = frames[1::2]
        vfi_differences = [
            float(np.mean(np.abs(target.astype(np.float32) - predicted.astype(np.float32))))
            for target, predicted in zip(target_frames, interpolated_frames)
        ]
        mean_difference = float(np.mean(vfi_differences))
        return {
            "backend": "amt_s",
            "raw": clamp01((255.0 - mean_difference) / 255.0),
            "frame_count": len(frames),
            "sampled_frame_count": len(sampled_frames),
            "vfi_difference": round(mean_difference, 4),
            "vfi_differences": [round(value, 4) for value in vfi_differences],
        }


@dataclass
class RAFTDynamicDegreeBackend:
    """Cached RAFT-things backend used by the dynamic_degree metric."""

    model: Any
    device: str
    input_padder_cls: Any

    def score_video(self, video_path: str | Path) -> dict[str, Any]:
        """Estimate whether a video has sufficient motion via RAFT flow magnitude thresholds."""
        import torch

        path = Path(video_path)
        frames = read_video_frames(path)
        interval = max(1, round(probe_video_fps(path, default=8.0) / 8.0))
        sampled_frames = frames[::interval]
        if len(sampled_frames) < 2:
            return {
                "backend": "raft_things",
                "raw": None,
                "frame_count": len(frames),
                "sampled_frame_count": len(sampled_frames),
                "sampling_interval": interval,
                "error": "not enough frames for RAFT dynamic degree",
            }

        tensor_frames = [
            torch.from_numpy(np.ascontiguousarray(frame))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)
            for frame in sampled_frames
        ]
        scale = min(tensor_frames[0].shape[-2:])
        threshold = 6.0 * (scale / 256.0)
        count_threshold = max(int(round(4 * (len(tensor_frames) / 16.0))), 1)
        motion_scores: list[float] = []
        active_count = 0
        with torch.inference_mode():
            for image1, image2 in zip(tensor_frames[:-1], tensor_frames[1:]):
                padder = self.input_padder_cls(image1.shape)
                image1_padded, image2_padded = padder.pad(image1, image2)
                _, flow_up = self.model(image1_padded, image2_padded, iters=20, test_mode=True)
                max_motion = _raft_max_motion_score(flow_up)
                motion_scores.append(max_motion)
                if max_motion > threshold:
                    active_count += 1

        raw = 1.0 if active_count >= count_threshold else 0.0
        return {
            "backend": "raft_things",
            "raw": raw,
            "frame_count": len(frames),
            "sampled_frame_count": len(sampled_frames),
            "sampling_interval": interval,
            "motion_threshold": round(threshold, 4),
            "count_threshold": count_threshold,
            "active_transition_count": active_count,
            "motion_scores": [round(value, 4) for value in motion_scores],
        }


def _raft_max_motion_score(flow: Any) -> float:
    """Aggregate the top 5% of RAFT flow magnitudes into a single motion score."""
    flow_np = flow[0].permute(1, 2, 0).detach().cpu().numpy()
    rad = np.sqrt(np.square(flow_np[:, :, 0]) + np.square(flow_np[:, :, 1]))
    rad_flat = rad.reshape(-1)
    cut_index = max(1, int(rad_flat.size * 0.05))
    top_scores = np.sort(rad_flat)[-cut_index:]
    return float(np.mean(top_scores))


@lru_cache(maxsize=4)
def load_amt_motion_smoothness_backend(
    device: str | Any | None = None,
    *,
    config_path: str | None = None,
    ckpt_path: str | None = None,
) -> AMTMotionSmoothnessBackend | None:
    """Load and cache the AMT-S motion-smoothness backend, or None when deps are missing."""
    layout = inspect_flow_source_layout()
    if not bool(layout["amt_ready"]):
        return None
    try:
        import torch
        from omegaconf import OmegaConf
    except (ImportError, ModuleNotFoundError):
        return None

    device_name = _normalize_device(device)
    with amt_runtime_context():
        try:
            from utils.build_utils import build_from_cfg
            from utils.utils import InputPadder, check_dim_and_resize, img2tensor, tensor2img
        except (ImportError, ModuleNotFoundError):
            return None
        network_cfg = OmegaConf.load(resolve_amt_config_path(config_path)).network
        model = build_from_cfg(network_cfg)
    try:
        checkpoint = torch.load(resolve_amt_s_checkpoint(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolve_amt_s_checkpoint(ckpt_path), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device_name)
    model.eval()

    if torch.device(device_name).type == "cuda":
        total_memory = int(torch.cuda.get_device_properties(torch.device(device_name)).total_memory)
        anchor_resolution = 1024 * 512
        anchor_memory = 1500 * 1024**2
        anchor_memory_bias = 2500 * 1024**2
    else:
        total_memory = 1
        anchor_resolution = 8192 * 8192
        anchor_memory = 1
        anchor_memory_bias = 0

    return AMTMotionSmoothnessBackend(
        model=model,
        device=device_name,
        embt=torch.tensor(0.5).float().view(1, 1, 1, 1).to(device_name),
        anchor_resolution=anchor_resolution,
        anchor_memory=anchor_memory,
        anchor_memory_bias=anchor_memory_bias,
        vram_avail=total_memory,
        input_padder_cls=InputPadder,
        check_dim_and_resize=check_dim_and_resize,
        img_to_tensor=img2tensor,
        tensor_to_image=tensor2img,
    )


@lru_cache(maxsize=4)
def load_raft_dynamic_degree_backend(
    device: str | Any | None = None,
    *,
    model_path: str | None = None,
) -> RAFTDynamicDegreeBackend | None:
    """Load and cache the RAFT dynamic-degree backend, or None when deps are missing."""
    layout = inspect_flow_source_layout()
    if not bool(layout["raft_ready"]):
        return None
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return None

    device_name = _normalize_device(device)
    with raft_runtime_context():
        try:
            from raft import RAFT
            from utils.utils import InputPadder
        except (ImportError, ModuleNotFoundError):
            return None
    args = _AttrDict(
        model=resolve_raft_things_checkpoint(model_path),
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    )
    model = RAFT(args)
    checkpoint = torch.load(args.model, map_location="cpu")
    model.load_state_dict({key.replace("module.", ""): value for key, value in checkpoint.items()})
    model.to(device_name)
    model.eval()
    return RAFTDynamicDegreeBackend(model=model, device=device_name, input_padder_cls=InputPadder)


def compute_amt_motion_smoothness(
    video_path: str | Path,
    *,
    device: str | Any | None = None,
    config_path: str | None = None,
    ckpt_path: str | None = None,
) -> dict[str, Any]:
    """Score motion smoothness for one video using the AMT-S backend."""
    backend = load_amt_motion_smoothness_backend(
        device,
        config_path=config_path,
        ckpt_path=ckpt_path,
    )
    if backend is None:
        raise ModuleNotFoundError("AMT motion-smoothness backend is unavailable")
    return backend.score_video(video_path)


def compute_raft_dynamic_degree(
    video_path: str | Path,
    *,
    device: str | Any | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Score dynamic degree for one video using the RAFT-things backend."""
    backend = load_raft_dynamic_degree_backend(
        device,
        model_path=model_path,
    )
    if backend is None:
        raise ModuleNotFoundError("RAFT dynamic-degree backend is unavailable")
    return backend.score_video(video_path)


__all__ = [
    "compute_amt_motion_smoothness",
    "compute_raft_dynamic_degree",
    "load_amt_motion_smoothness_backend",
    "load_raft_dynamic_degree_backend",
    "resolve_amt_config_path",
    "resolve_amt_s_checkpoint",
    "resolve_raft_things_checkpoint",
]
