"""Subprocess runner for Lyra-2 inference inside the upstream model repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

from worldatlas.benchmark.annotations import load_camera_matrices, load_intrinsics_sequence
from worldatlas.common.checkpoints import (
    apply_checkpoint_env,
    hf_local_dir,
    resolve_checkpoint_path,
)


HF_ALIAS_SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...], bool]] = {
    "xlm-roberta-large": (
        ("xlm-roberta-large", "FacebookAI/xlm-roberta-large", "FacebookAI--xlm-roberta-large"),
        ("config.json", "sentencepiece.bpe.model"),
        False,
    ),
    "google/umt5-xxl": (
        ("google/umt5-xxl", "google--umt5-xxl", "umt5-xxl"),
        ("spiece.model", "tokenizer_config.json"),
        False,
    ),
    "Ruicheng/moge-vitl": (
        ("Ruicheng/moge-vitl", "Ruicheng--moge-vitl", "moge-vitl"),
        ("model.pt",),
        True,
    ),
}
UMT5_CONFIG_SHIM = '{"model_type": "umt5"}\n'
FVCORE_INIT_SHIM = "from __future__ import annotations\n"
FVCORE_COMMON_INIT_SHIM = "from .registry import Registry\n"
FVCORE_REGISTRY_SHIM = textwrap.dedent(
    """
    from __future__ import annotations


    class Registry:
        def __init__(self, name: str) -> None:
            self._name = name
            self._obj_map = {}

        def _do_register(self, name: str, obj):
            if name in self._obj_map:
                raise KeyError(f"An object named {name!r} was already registered in {self._name!r}")
            self._obj_map[name] = obj

        def register(self, obj=None, *, name: str | None = None):
            if obj is None:
                def decorator(func_or_class):
                    self._do_register(name or func_or_class.__name__, func_or_class)
                    return func_or_class

                return decorator
            self._do_register(name or obj.__name__, obj)
            return obj

        def get(self, name: str):
            obj = self._obj_map.get(name)
            if obj is None:
                raise KeyError(f"No object named {name!r} found in {self._name!r}")
            return obj

        def __contains__(self, name: str) -> bool:
            return name in self._obj_map

        def __repr__(self) -> str:
            return f"Registry(name={self._name!r}, items={list(self._obj_map)})"
    """
)
FTFY_SHIM = textwrap.dedent(
    """
    from __future__ import annotations


    def fix_text(text, *args, **kwargs):
        del args, kwargs
        return str(text)
    """
)
ADDICT_SHIM = textwrap.dedent(
    """
    from __future__ import annotations


    class Dict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.update(*args, **kwargs)

        @classmethod
        def _hook(cls, value):
            if isinstance(value, dict) and not isinstance(value, Dict):
                return cls(value)
            if isinstance(value, list):
                return [cls._hook(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cls._hook(item) for item in value)
            return value

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            try:
                return self[name]
            except KeyError:
                value = type(self)()
                dict.__setitem__(self, name, value)
                return value

        def __setattr__(self, name, value):
            if name.startswith("_"):
                object.__setattr__(self, name, value)
            else:
                self[name] = value

        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setitem__(self, key, value):
            dict.__setitem__(self, key, type(self)._hook(value))

        def update(self, *args, **kwargs):
            values = dict(*args, **kwargs)
            for key, value in values.items():
                self[key] = value
            return None

        def copy(self):
            return type(self)(self)

        def to_dict(self):
            def unwrap(value):
                if isinstance(value, Dict):
                    return {k: unwrap(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [unwrap(item) for item in value]
                if isinstance(value, tuple):
                    return tuple(unwrap(item) for item in value)
                return value

            return {key: unwrap(value) for key, value in self.items()}
    """
)
EVO_INIT_SHIM = "from __future__ import annotations\n"
EVO_CORE_INIT_SHIM = "from __future__ import annotations\n"
EVO_TRAJECTORY_SHIM = textwrap.dedent(
    """
    from __future__ import annotations

    import numpy as np


    class PosePath3D:
        def __init__(self, poses_se3):
            self.poses_se3 = [np.array(pose, dtype=np.float64, copy=True) for pose in poses_se3]

        def align(self, traj_ref, correct_scale=False, *args, **kwargs):
            del args, kwargs
            src = np.stack([pose[:3, 3] for pose in self.poses_se3], axis=0)
            dst = np.stack([pose[:3, 3] for pose in traj_ref.poses_se3], axis=0)
            if src.shape != dst.shape:
                raise ValueError(f"trajectory shape mismatch: {src.shape} != {dst.shape}")
            rot, trans, scale = _umeyama(src, dst, with_scale=bool(correct_scale))
            aligned = []
            for pose in self.poses_se3:
                out = np.array(pose, dtype=np.float64, copy=True)
                out[:3, :3] = rot @ out[:3, :3]
                out[:3, 3] = scale * (rot @ out[:3, 3]) + trans
                aligned.append(out)
            self.poses_se3 = aligned
            return rot, trans, scale


    def _umeyama(src, dst, *, with_scale):
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        if src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 1:
            raise ValueError(f"expected Nx3 source points, got {src.shape}")
        mean_src = src.mean(axis=0)
        mean_dst = dst.mean(axis=0)
        src_centered = src - mean_src
        dst_centered = dst - mean_dst
        var_src = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
        if var_src <= np.finfo(np.float64).eps:
            return np.eye(3), mean_dst - mean_src, 1.0
        cov = (dst_centered.T @ src_centered) / src.shape[0]
        u, singular_values, vt = np.linalg.svd(cov)
        sign = np.ones(3)
        if np.linalg.det(u @ vt) < 0:
            sign[-1] = -1.0
        rot = u @ np.diag(sign) @ vt
        scale = 1.0
        if with_scale:
            scale = float(np.sum(singular_values * sign) / var_src)
        trans = mean_dst - scale * (rot @ mean_src)
        return rot, trans, scale
    """
)
FLASH_ATTN_INIT_SHIM = "from __future__ import annotations\n"
FLASH_ATTN_LAYERS_INIT_SHIM = "from __future__ import annotations\n"
FLASH_ATTN_ROTARY_SHIM = textwrap.dedent(
    """
    from __future__ import annotations

    import torch


    def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False):
        del inplace
        rotary_dim = int(cos.shape[-1]) * 2
        if rotary_dim == 0 or rotary_dim > x.shape[-1]:
            raise ValueError(
                f"rotary dim {rotary_dim} is incompatible with last dim {x.shape[-1]}"
            )
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        cos_b = cos
        sin_b = sin
        if cos_b.ndim == 2:
            cos_b = cos_b[None, :, None, :]
            sin_b = sin_b[None, :, None, :]
        elif cos_b.ndim == x_rot.ndim - 1:
            cos_b = cos_b.unsqueeze(-2)
            sin_b = sin_b.unsqueeze(-2)
        if interleaved:
            x1 = x_rot[..., ::2]
            x2 = x_rot[..., 1::2]
            o1 = x1 * cos_b - x2 * sin_b
            o2 = x1 * sin_b + x2 * cos_b
            rotated = torch.stack((o1, o2), dim=-1).flatten(-2)
        else:
            x1, x2 = x_rot.chunk(2, dim=-1)
            o1 = x1 * cos_b - x2 * sin_b
            o2 = x1 * sin_b + x2 * cos_b
            rotated = torch.cat((o1, o2), dim=-1)
        if x_pass.numel() == 0:
            return rotated
        return torch.cat((rotated, x_pass), dim=-1)
    """
)
TRANSFORMER_ENGINE_INIT_SHIM = textwrap.dedent(
    """
    from __future__ import annotations

    from . import common, pytorch

    __version__ = "0.0.0"
    __all__ = ["__version__", "common", "pytorch"]
    """
)
TRANSFORMER_ENGINE_RECIPE_SHIM = textwrap.dedent(
    """
    from __future__ import annotations


    class Format:
        E4M3 = "E4M3"
        HYBRID = "HYBRID"


    class DelayedScaling:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs


    class Float8CurrentScaling(DelayedScaling):
        pass


    class MXFP8BlockScaling(DelayedScaling):
        pass
    """
)
TRANSFORMER_ENGINE_DISTRIBUTED_SHIM = textwrap.dedent(
    """
    from __future__ import annotations


    class CudaRNGStatesTracker:
        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            self._states = {}

        def set_states(self, states) -> None:
            self._states = states

        def add(self, name, seed) -> None:
            self._states[name] = seed


    def checkpoint(forward_func, *args, **kwargs):
        return forward_func(*args)
    """
)
TRANSFORMER_ENGINE_PYTORCH_SHIM = textwrap.dedent(
    """
    from __future__ import annotations

    from contextlib import nullcontext

    import torch
    from torch import nn

    from . import distributed
    from .attention import DotProductAttention


    class LayerNorm(nn.LayerNorm):
        def __init__(self, hidden_size: int, eps: float = 1e-5, **_: object) -> None:
            super().__init__(hidden_size, eps=eps)


    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-5, **_: object) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = float(eps)
            self.fwd_rmsnorm_sm_margin = 0
            self.bwd_rmsnorm_sm_margin = 0
            self.inf_rmsnorm_sm_margin = 0
            self.zero_centered_gamma = False
            self.activation_dtype = None

        def forward(self, x):
            norm = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x * norm.to(dtype=x.dtype) * self.weight


    class _RMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            inp,
            weight,
            eps,
            fwd_rmsnorm_sm_margin=0,
            bwd_rmsnorm_sm_margin=0,
            inf_rmsnorm_sm_margin=0,
            zero_centered_gamma=False,
            is_grad_enabled=False,
            activation_dtype=None,
        ):
            del fwd_rmsnorm_sm_margin, bwd_rmsnorm_sm_margin, inf_rmsnorm_sm_margin
            del zero_centered_gamma, is_grad_enabled, activation_dtype
            if ctx is not None:
                ctx.save_for_backward(inp, weight)
                ctx.eps = eps
            norm = torch.rsqrt(inp.float().pow(2).mean(dim=-1, keepdim=True) + float(eps))
            return inp * norm.to(dtype=inp.dtype) * weight

        @staticmethod
        def backward(ctx, grad_output):
            inp, weight = ctx.saved_tensors
            grad_input = grad_output * weight
            grad_weight = (grad_output * inp).sum(dim=tuple(range(grad_output.ndim - 1)))
            return grad_input, grad_weight, None, None, None, None, None, None, None


    class Linear(nn.Linear):
        def __init__(
            self,
            *args: object,
            in_features: int | None = None,
            out_features: int | None = None,
            bias: bool = True,
            return_bias: bool = False,
            **_: object,
        ) -> None:
            if args:
                if len(args) < 2:
                    raise TypeError("Linear fallback requires in_features and out_features")
                in_features = int(args[0])
                out_features = int(args[1])
                if len(args) >= 3:
                    bias = bool(args[2])
            if in_features is None or out_features is None:
                raise TypeError("Linear fallback requires in_features and out_features")
            super().__init__(in_features, out_features, bias=bias)
            self.return_bias = bool(return_bias)

        def forward(self, x, **_: object):
            output = super().forward(x)
            if self.return_bias and self.bias is not None:
                return output, self.bias
            return output


    class Conv1d(nn.Conv1d):
        pass


    class LayerNormLinear(nn.Module):
        def __init__(
            self,
            *,
            in_features: int,
            out_features: int,
            bias: bool = True,
            return_bias: bool = False,
            normalization: str = "LayerNorm",
            eps: float = 1e-5,
            **kwargs: object,
        ) -> None:
            super().__init__()
            norm_cls = RMSNorm if normalization == "RMSNorm" else LayerNorm
            self.norm = norm_cls(in_features, eps=eps, **kwargs)
            self.linear = Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                return_bias=return_bias,
                **kwargs,
            )

        def forward(self, x, **kwargs: object):
            return self.linear(self.norm(x), **kwargs)


    class GroupedLinear(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.return_bias = bool(kwargs.get("return_bias", False))

        def forward(self, x, **_: object):
            if self.return_bias:
                return x, None
            return x


    class TransformerLayer(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.args = args
            self.kwargs = kwargs

        def forward(self, hidden_states, *args: object, **kwargs: object):
            del args, kwargs
            return hidden_states


    def fp8_autocast(*args: object, **kwargs: object):
        return nullcontext()


    def fp8_model_init(*args: object, **kwargs: object):
        return nullcontext()


    __all__ = [
        "distributed",
        "DotProductAttention",
        "LayerNorm",
        "RMSNorm",
        "_RMSNorm",
        "Linear",
        "Conv1d",
        "LayerNormLinear",
        "GroupedLinear",
        "TransformerLayer",
        "fp8_autocast",
        "fp8_model_init",
    ]
    """
)
TRANSFORMER_ENGINE_MODULE_INIT_SHIM = textwrap.dedent(
    """
    from __future__ import annotations

    from .. import Linear, RMSNorm, TransformerLayer, _RMSNorm
    from .base import TransformerEngineBaseModule

    __all__ = ["Linear", "RMSNorm", "_RMSNorm", "TransformerLayer", "TransformerEngineBaseModule"]
    """
)
TRANSFORMER_ENGINE_MODULE_LINEAR_SHIM = "from .. import Linear\n"
TRANSFORMER_ENGINE_MODULE_RMSNORM_SHIM = "from .. import RMSNorm, _RMSNorm\n"
TRANSFORMER_ENGINE_MODULE_BASE_SHIM = textwrap.dedent(
    """
    from __future__ import annotations


    class TransformerEngineBaseModule:
        @staticmethod
        def set_activation_dtype(module, inp) -> None:
            module.activation_dtype = getattr(inp, "dtype", None)
    """
)
TRANSFORMER_ENGINE_JIT_SHIM = textwrap.dedent(
    """
    from __future__ import annotations


    def no_torch_dynamo(*decorator_args, **decorator_kwargs):
        del decorator_kwargs
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorator_args[0]

        def _decorator(func):
            return func

        return _decorator
    """
)
TRANSFORMER_ENGINE_CONSTANTS_SHIM = textwrap.dedent(
    """
    from __future__ import annotations

    AttnBiasTypes = ("no_bias", "pre_scale_bias", "post_scale_bias", "alibi")
    """
)
TRANSFORMER_ENGINE_FLOAT8_TENSOR_SHIM = textwrap.dedent(
    """
    from __future__ import annotations


    class Float8Tensor:
        pass
    """
)
TRANSFORMER_ENGINE_ATTENTION_SHIM = textwrap.dedent(
    """
    from __future__ import annotations

    import torch
    import torch.nn.functional as F
    from torch import nn


    class _SplitAlongDim(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor, split_size_or_sections, dim):
            ctx.dim = dim
            return torch.split(tensor, split_size_or_sections, dim=dim)

        @staticmethod
        def backward(ctx, *grad_outputs):
            return torch.cat(grad_outputs, dim=ctx.dim), None, None


    class DotProductAttention(nn.Module):
        def __init__(
            self,
            num_attention_heads: int,
            kv_channels: int,
            *,
            num_gqa_groups: int | None = None,
            attention_dropout: float = 0.0,
            qkv_format: str = "bshd",
            attn_mask_type: str = "no_mask",
            **_: object,
        ) -> None:
            super().__init__()
            if qkv_format not in {"bshd", "sbhd"}:
                raise NotImplementedError(f"unsupported qkv_format for fallback: {qkv_format}")
            if attn_mask_type != "no_mask":
                raise NotImplementedError(f"unsupported attn_mask_type for fallback: {attn_mask_type}")
            self.num_attention_heads = int(num_attention_heads)
            self.kv_channels = int(kv_channels)
            self.num_gqa_groups = int(num_gqa_groups or num_attention_heads)
            self.attention_dropout = float(attention_dropout)
            self.qkv_format = qkv_format

        def forward(self, q_B_L_H_D, k_B_L_H_D, v_B_L_H_D, *_args, **_kwargs):
            if self.qkv_format == "bshd":
                q = q_B_L_H_D.permute(0, 2, 1, 3).contiguous()
                k = k_B_L_H_D.permute(0, 2, 1, 3).contiguous()
                v = v_B_L_H_D.permute(0, 2, 1, 3).contiguous()
            else:
                q = q_B_L_H_D.permute(1, 2, 0, 3).contiguous()
                k = k_B_L_H_D.permute(1, 2, 0, 3).contiguous()
                v = v_B_L_H_D.permute(1, 2, 0, 3).contiguous()
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            if self.qkv_format == "bshd":
                output = output.permute(0, 2, 1, 3).contiguous()
                return output.flatten(-2)
            output = output.permute(2, 0, 1, 3).contiguous()
            return output.flatten(-2)

        def set_context_parallel_group(self, *_args, **_kwargs) -> None:
            return None


    def apply_rotary_pos_emb(tensor, rotary_pos_emb, tensor_format: str = "bshd", fused: bool = False):
        del fused
        if rotary_pos_emb is None:
            return tensor
        if isinstance(rotary_pos_emb, (tuple, list)):
            cos, sin = rotary_pos_emb
        else:
            emb = rotary_pos_emb.to(device=tensor.device, dtype=tensor.dtype)
            cos = torch.cos(emb)
            sin = torch.sin(emb)

        if tensor_format == "bshd":
            seq_dim = 1
        elif tensor_format == "sbhd":
            seq_dim = 0
        else:
            raise NotImplementedError(f"unsupported tensor_format for rotary fallback: {tensor_format}")

        def _match_shape(value):
            value = value.to(device=tensor.device, dtype=tensor.dtype)
            if value.shape[seq_dim] != tensor.shape[seq_dim]:
                value = value.narrow(seq_dim, 0, tensor.shape[seq_dim])
            while value.ndim < tensor.ndim:
                value = value.unsqueeze(0)
            return value

        cos = _match_shape(cos)
        sin = _match_shape(sin)
        rotary_dim = min(tensor.shape[-1], cos.shape[-1])
        x_rot = tensor[..., :rotary_dim]
        x_pass = tensor[..., rotary_dim:]
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        output = x_rot * cos[..., :rotary_dim] + rotated * sin[..., :rotary_dim]
        if x_pass.numel() == 0:
            return output
        return torch.cat((output, x_pass), dim=-1)


    def check_set_window_size(attn_mask_type, window_size):
        del attn_mask_type
        return window_size
    """
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Lyra-2 subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--annotation_path", required=True, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--experiment", default="lyra2", type=str)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=832, type=int)
    parser.add_argument("--num_frames", default=81, type=int)
    parser.add_argument("--guidance", default=5.0, type=float)
    parser.add_argument("--shift", default=5.0, type=float)
    parser.add_argument("--num_sampling_step", default=35, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--fps", default=16, type=int)
    parser.add_argument("--pose_scale", default=1.0, type=float)
    parser.add_argument("--context_parallel_size", default=1, type=int)
    parser.add_argument("--prompt_suffix", default="", type=str)
    parser.add_argument("--use_moge_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--da3_model_name", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1", type=str)
    parser.add_argument("--da3_model_path_custom", default=None, type=str)
    parser.add_argument("--da3_frame_interval", default=8, type=int)
    parser.add_argument("--da3_max_history_frames", default=10, type=int)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--offload_when_prompt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--da3_include_ar_chunk_last_frames", action="store_true")
    parser.add_argument("--da3_use_predicted_pose", action="store_true")
    parser.add_argument("--da3_predicted_pose_continuation", action="store_true")
    parser.add_argument("--ablate_same_t5", action="store_true")
    parser.add_argument("--use_dmd_scheduler", action="store_true")
    parser.add_argument("--disable_cache_update", action="store_true")
    parser.add_argument("--offload_da3_diffusion", action="store_true")
    return parser.parse_args()


def _intrinsics_matrix(values: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(values, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        return intrinsics.astype(np.float32)
    if intrinsics.shape != (4,):
        raise ValueError(f"unsupported Lyra-2 intrinsics shape: {intrinsics.shape}")
    fx, fy, cx, cy = [float(value) for value in intrinsics]
    return np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _trajectory_payload(
    *,
    annotation_path: str,
    num_frames: int,
    height: int,
    width: int,
) -> dict[str, np.ndarray | int]:
    camera_c2w = load_camera_matrices(annotation_path, target_frames=num_frames)
    if camera_c2w is None:
        raise FileNotFoundError(f"poses.npy not found under annotation path: {annotation_path}")

    intrinsics = load_intrinsics_sequence(annotation_path, target_frames=len(camera_c2w))
    if intrinsics is None:
        raise FileNotFoundError(f"intrinsics.npy not found under annotation path: {annotation_path}")

    camera_w2c = np.linalg.inv(camera_c2w.astype(np.float64)).astype(np.float32)
    intrinsics_33 = np.stack(
        [_intrinsics_matrix(entry) for entry in np.asarray(intrinsics, dtype=np.float32)],
        axis=0,
    ).astype(np.float32)
    return {
        "w2c": camera_w2c,
        "intrinsics": intrinsics_33,
        "image_height": int(height),
        "image_width": int(width),
    }


def _checkpoint_tree(checkpoint_dir: Path) -> Path:
    resolved = checkpoint_dir.expanduser().resolve()
    if (resolved / "checkpoints").is_dir():
        return (resolved / "checkpoints").resolve()
    required_dirs = ("image_encoder", "lora", "model", "recon", "text_encoder", "vae")
    if all((resolved / name).exists() for name in required_dirs):
        return resolved
    raise FileNotFoundError(
        "Lyra-2 checkpoint tree not found. Expected either a base directory with "
        f"`checkpoints/` under it or a direct checkpoint tree at: {resolved}"
    )


def _resolve_local_da3_model_name(value: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError("Lyra-2 DA3 model name is empty")
    if token.startswith(("ckpt/", "./", "../", "/", "~")):
        resolved = resolve_checkpoint_path(token, kind="dir", required=True)
        if resolved is None:
            raise FileNotFoundError(f"Lyra-2 DA3 model directory is missing: {token}")
        return str(resolved)
    return token


def _repo_id_candidates(repo_id: str) -> tuple[str, ...]:
    token = str(repo_id).strip()
    if not token:
        return ()
    candidates = [token]
    if "/" in token:
        candidates.append(token.replace("/", "--"))
        candidates.append(token.split("/")[-1])
    return tuple(dict.fromkeys(candidates))


def _local_hf_dir_candidates(repo_id: str) -> tuple[Path, ...]:
    """Return direct-download candidates for an HF repo across supported ckpt layouts."""
    token = str(repo_id).strip()
    if not token:
        return ()
    relative_tokens = [token]
    if "/" in token:
        relative_tokens.extend([token.replace("/", "--"), token.split("/")[-1]])
    candidates: list[Path] = []
    for relative_token in dict.fromkeys(relative_tokens):
        resolved = resolve_checkpoint_path(
            Path("ckpt") / Path(relative_token),
            kind="dir",
            required=False,
        )
        if resolved is not None:
            candidates.append(resolved)
    return tuple(dict.fromkeys(candidates))


def _hf_dir_status(source_path: Path, required_files: tuple[str, ...]) -> tuple[bool, str]:
    if not source_path.is_dir():
        return False, f"missing directory {source_path}"
    missing = []
    empty = []
    for filename in required_files:
        candidate = source_path / filename
        if not candidate.exists():
            missing.append(filename)
        elif candidate.is_file() and candidate.stat().st_size <= 0:
            empty.append(filename)
    if not missing and not empty:
        return True, ""
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if empty:
        details.append(f"empty {', '.join(empty)}")
    return False, f"{source_path} ({'; '.join(details)})"


def _complete_local_hf_dir(
    repo_ids: tuple[str, ...],
    *,
    required_files: tuple[str, ...],
) -> Path:
    errors: list[str] = []
    for repo_id in repo_ids:
        source_path = hf_local_dir(repo_id, required=False)
        complete, detail = _hf_dir_status(source_path, required_files)
        if complete:
            return source_path
        errors.append(f"{repo_id}: {detail}")
    for repo_id in repo_ids:
        for source_path in _local_hf_dir_candidates(repo_id):
            complete, detail = _hf_dir_status(source_path, required_files)
            if complete:
                return source_path
            errors.append(f"{repo_id}: {detail}")
    raise FileNotFoundError(
        "Unable to find a complete local Hugging Face dependency. Tried: "
        + " | ".join(errors)
    )


def _hf_repo_aliases(da3_model_name: str, *, use_moge_scale: bool = True) -> dict[str, Path]:
    aliases: dict[str, Path] = {}
    for repo_id, (local_repo_ids, required_files, link_file) in HF_ALIAS_SPECS.items():
        if repo_id == "Ruicheng/moge-vitl" and not use_moge_scale:
            continue
        source_path = _complete_local_hf_dir(
            local_repo_ids,
            required_files=required_files,
        )
        aliases[repo_id] = (source_path / required_files[0]) if link_file else source_path
    if "/" in da3_model_name and not Path(da3_model_name).exists():
        source_path = _complete_local_hf_dir(
            _repo_id_candidates(da3_model_name),
            required_files=("config.json", "model.safetensors"),
        )
        aliases[da3_model_name] = source_path
    return aliases


def _link_hf_repo_aliases(
    workspace_root: Path,
    da3_model_name: str,
    *,
    use_moge_scale: bool = True,
) -> None:
    for repo_id, source_path in _hf_repo_aliases(
        da3_model_name,
        use_moge_scale=use_moge_scale,
    ).items():
        link_path = workspace_root / repo_id
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if repo_id == "google/umt5-xxl":
            if link_path.exists() or link_path.is_symlink():
                if link_path.is_dir() and not link_path.is_symlink():
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()
            link_path.mkdir(parents=True, exist_ok=True)
            for source_file in source_path.iterdir():
                if source_file.is_file():
                    (link_path / source_file.name).symlink_to(source_file)
            if not (link_path / "config.json").exists():
                (link_path / "config.json").write_text(UMT5_CONFIG_SHIM, encoding="utf-8")
            continue
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() and link_path.resolve() == source_path:
                continue
            if link_path.is_dir():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.symlink_to(source_path, target_is_directory=source_path.is_dir())


def _prepare_fvcore_registry_shim(workspace_root: Path) -> None:
    try:
        from fvcore.common.registry import Registry as _Registry

        del _Registry
        return
    except ImportError:
        pass
    package_root = workspace_root / "fvcore"
    common_root = package_root / "common"
    common_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text(FVCORE_INIT_SHIM, encoding="utf-8")
    (common_root / "__init__.py").write_text(FVCORE_COMMON_INIT_SHIM, encoding="utf-8")
    (common_root / "registry.py").write_text(FVCORE_REGISTRY_SHIM, encoding="utf-8")


def _prepare_ftfy_shim(workspace_root: Path) -> None:
    try:
        import ftfy as _ftfy

        del _ftfy
        return
    except ImportError:
        pass
    (workspace_root / "ftfy.py").write_text(FTFY_SHIM, encoding="utf-8")


def _prepare_addict_shim(workspace_root: Path) -> None:
    try:
        from addict import Dict as _Dict

        del _Dict
        return
    except ImportError:
        pass
    (workspace_root / "addict.py").write_text(ADDICT_SHIM, encoding="utf-8")


def _prepare_evo_trajectory_shim(workspace_root: Path) -> None:
    try:
        from evo.core.trajectory import PosePath3D as _PosePath3D

        del _PosePath3D
        return
    except ImportError:
        pass
    package_root = workspace_root / "evo"
    core_root = package_root / "core"
    core_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text(EVO_INIT_SHIM, encoding="utf-8")
    (core_root / "__init__.py").write_text(EVO_CORE_INIT_SHIM, encoding="utf-8")
    (core_root / "trajectory.py").write_text(EVO_TRAJECTORY_SHIM, encoding="utf-8")


def _prepare_flash_attn_rotary_shim(workspace_root: Path) -> None:
    try:
        from flash_attn.layers.rotary import apply_rotary_emb as _apply_rotary_emb

        del _apply_rotary_emb
        return
    except ImportError:
        pass
    package_root = workspace_root / "flash_attn"
    layers_root = package_root / "layers"
    layers_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text(FLASH_ATTN_INIT_SHIM, encoding="utf-8")
    (layers_root / "__init__.py").write_text(FLASH_ATTN_LAYERS_INIT_SHIM, encoding="utf-8")
    (layers_root / "rotary.py").write_text(FLASH_ATTN_ROTARY_SHIM, encoding="utf-8")


def _prepare_transformer_engine_shim(workspace_root: Path) -> None:
    package_root = workspace_root / "transformer_engine"
    pytorch_root = package_root / "pytorch"
    common_root = package_root / "common"
    module_root = pytorch_root / "module"
    pytorch_root.mkdir(parents=True, exist_ok=True)
    common_root.mkdir(parents=True, exist_ok=True)
    module_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text(TRANSFORMER_ENGINE_INIT_SHIM, encoding="utf-8")
    (common_root / "__init__.py").write_text("from . import recipe\n", encoding="utf-8")
    (common_root / "recipe.py").write_text(TRANSFORMER_ENGINE_RECIPE_SHIM, encoding="utf-8")
    (pytorch_root / "__init__.py").write_text(TRANSFORMER_ENGINE_PYTORCH_SHIM, encoding="utf-8")
    (pytorch_root / "attention.py").write_text(TRANSFORMER_ENGINE_ATTENTION_SHIM, encoding="utf-8")
    (pytorch_root / "constants.py").write_text(TRANSFORMER_ENGINE_CONSTANTS_SHIM, encoding="utf-8")
    (pytorch_root / "distributed.py").write_text(
        TRANSFORMER_ENGINE_DISTRIBUTED_SHIM,
        encoding="utf-8",
    )
    (pytorch_root / "float8_tensor.py").write_text(TRANSFORMER_ENGINE_FLOAT8_TENSOR_SHIM, encoding="utf-8")
    (pytorch_root / "jit.py").write_text(TRANSFORMER_ENGINE_JIT_SHIM, encoding="utf-8")
    (module_root / "__init__.py").write_text(TRANSFORMER_ENGINE_MODULE_INIT_SHIM, encoding="utf-8")
    (module_root / "base.py").write_text(TRANSFORMER_ENGINE_MODULE_BASE_SHIM, encoding="utf-8")
    (module_root / "linear.py").write_text(TRANSFORMER_ENGINE_MODULE_LINEAR_SHIM, encoding="utf-8")
    (module_root / "rmsnorm.py").write_text(TRANSFORMER_ENGINE_MODULE_RMSNORM_SHIM, encoding="utf-8")


def _use_transformer_engine_shim() -> bool:
    """Use transformer engine shim -> bool."""
    backend = os.environ.get("WORLDATLAS_LYRA2_ATTENTION_BACKEND", "native").strip().lower()
    return backend in {"shim", "pytorch", "torch", "sdpa"}


def _prepend_runtime_library_path(env: dict[str, str], executable: str) -> None:
    env_root = Path(executable).expanduser().resolve().parent.parent
    lib_dir = env_root / "lib"
    if not (lib_dir / "libstdc++.so.6").exists():
        return
    existing = env.get("LD_LIBRARY_PATH")
    parts = [str(lib_dir)]
    if existing:
        parts.extend(part for part in existing.split(os.pathsep) if part and part != str(lib_dir))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)


def _configure_lyra2_runtime_env(env: dict[str, str], executable: str) -> None:
    _prepend_runtime_library_path(env, executable)
    if not _use_transformer_engine_shim():
        env.setdefault("NVTE_FLASH_ATTN", "1")
        env.setdefault("NVTE_FUSED_ATTN", "0")
        env.setdefault("NVTE_UNFUSED_ATTN", "1")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("Lyra-2 runner requires checkpoint_dir")
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_tree = _checkpoint_tree(checkpoint_dir)
    da3_model_name = _resolve_local_da3_model_name(args.da3_model_name)
    da3_model_path_custom = None
    if args.da3_model_path_custom:
        da3_model_path_custom = resolve_checkpoint_path(
            args.da3_model_path_custom,
            kind="file",
            required=True,
        )
        if da3_model_path_custom is None:
            raise FileNotFoundError(
                f"Lyra-2 DA3 checkpoint path is missing: {args.da3_model_path_custom}"
            )

    with tempfile.TemporaryDirectory(prefix="worldatlas_lyra2_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        workspace_root = temp_dir / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)

        if _use_transformer_engine_shim():
            _prepare_transformer_engine_shim(workspace_root)
        _prepare_fvcore_registry_shim(workspace_root)
        _prepare_ftfy_shim(workspace_root)
        _prepare_addict_shim(workspace_root)
        _prepare_evo_trajectory_shim(workspace_root)
        _prepare_flash_attn_rotary_shim(workspace_root)
        _link_hf_repo_aliases(
            workspace_root,
            da3_model_name,
            use_moge_scale=args.use_moge_scale,
        )

        (workspace_root / "lyra_2").symlink_to(repo_root / "lyra_2", target_is_directory=True)
        (workspace_root / "checkpoints").symlink_to(checkpoint_tree, target_is_directory=True)

        input_root = temp_dir / "input"
        input_root.mkdir(parents=True, exist_ok=True)
        conditioning_source = Path(args.conditioning_image).expanduser().resolve()
        conditioning_copy = input_root / f"{args.sample_name}{conditioning_source.suffix or '.png'}"
        shutil.copy2(conditioning_source, conditioning_copy)

        trajectory_path = temp_dir / "trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            **_trajectory_payload(
                annotation_path=args.annotation_path,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
            ),
        )

        output_root = temp_dir / "outputs"
        command = [
            sys.executable,
            "-m",
            "lyra_2._src.inference.lyra2_custom_traj_inference",
            "--input_image_path",
            str(conditioning_copy),
            "--trajectory_path",
            str(trajectory_path),
            "--checkpoint_dir",
            "checkpoints/model",
            "--experiment",
            args.experiment,
            "--output_path",
            str(output_root),
            "--prompt",
            args.prompt,
            "--guidance",
            str(args.guidance),
            "--shift",
            str(args.shift),
            "--num_sampling_step",
            str(args.num_sampling_step),
            "--seed",
            str(args.seed),
            "--fps",
            str(args.fps),
            "--num_frames",
            str(args.num_frames),
            "--pose_scale",
            str(args.pose_scale),
            "--resolution",
            f"{args.height},{args.width}",
            "--context_parallel_size",
            str(args.context_parallel_size),
            "--prompt_suffix",
            args.prompt_suffix,
            "--da3_model_name",
            da3_model_name,
            "--da3_frame_interval",
            str(args.da3_frame_interval),
            "--da3_max_history_frames",
            str(args.da3_max_history_frames),
        ]
        if args.use_moge_scale:
            command.append("--use_moge_scale")
        else:
            command.append("--no-use_moge_scale")
        if da3_model_path_custom is not None:
            command.extend(["--da3_model_path_custom", str(da3_model_path_custom)])
        for flag_name in (
            "offload",
            "offload_when_prompt",
            "debug",
            "da3_include_ar_chunk_last_frames",
            "da3_use_predicted_pose",
            "da3_predicted_pose_continuation",
            "ablate_same_t5",
            "use_dmd_scheduler",
            "disable_cache_update",
            "offload_da3_diffusion",
        ):
            if getattr(args, flag_name):
                command.append(f"--{flag_name}")

        env = apply_checkpoint_env()
        # Transformers still honors TRANSFORMERS_CACHE before HF_HUB_CACHE.
        env["TRANSFORMERS_CACHE"] = env["HF_HUB_CACHE"]
        _configure_lyra2_runtime_env(env, sys.executable)
        pythonpath = [str(workspace_root), str(repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        subprocess.run(
            command,
            check=True,
            cwd=str(workspace_root),
            env=env,
        )

        generated_video = output_root / f"{args.sample_name}.mp4"
        if not generated_video.exists():
            raise FileNotFoundError(f"Lyra-2 output video was not written: {generated_video}")
        shutil.copy2(generated_video, output_path)


if __name__ == "__main__":
    main()
