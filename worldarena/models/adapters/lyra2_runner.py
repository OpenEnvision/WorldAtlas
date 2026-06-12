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

from worldarena.benchmark.annotations import load_camera_matrices, load_intrinsics_sequence
from worldarena.common.checkpoints import (
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
    parser = argparse.ArgumentParser(description="WorldArena Lyra-2 subprocess runner.")
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


def _complete_local_hf_dir(
    repo_ids: tuple[str, ...],
    *,
    required_files: tuple[str, ...],
) -> Path:
    errors: list[str] = []
    for repo_id in repo_ids:
        source_path = hf_local_dir(repo_id, required=False)
        if not source_path.is_dir():
            errors.append(f"{repo_id}: missing directory {source_path}")
            continue
        missing = []
        empty = []
        for filename in required_files:
            candidate = source_path / filename
            if not candidate.exists():
                missing.append(filename)
            elif candidate.is_file() and candidate.stat().st_size <= 0:
                empty.append(filename)
        if not missing and not empty:
            return source_path
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if empty:
            details.append(f"empty {', '.join(empty)}")
        errors.append(f"{repo_id}: {source_path} ({'; '.join(details)})")
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
    backend = os.environ.get("WORLDARENA_LYRA2_ATTENTION_BACKEND", "native").strip().lower()
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

    with tempfile.TemporaryDirectory(prefix="worldarena_lyra2_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        workspace_root = temp_dir / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)

        if _use_transformer_engine_shim():
            _prepare_transformer_engine_shim(workspace_root)
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
