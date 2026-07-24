"""SDPA-based fallback implementations of flash-attn public APIs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    # flash-attn uses [batch, seq, heads, dim] or [seq, heads, dim].
    squeezed = q.dim() == 3
    if squeezed:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    output = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    ).transpose(1, 2)
    return output.squeeze(0) if squeezed else output


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *args,
    **kwargs,
) -> torch.Tensor:
    return _sdpa(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *args,
    **kwargs,
) -> torch.Tensor:
    return flash_attn_func(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *args,
    **kwargs,
) -> torch.Tensor:
    return flash_attn_func(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *args,
    **kwargs,
) -> torch.Tensor:
    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    if (
        q_lengths.numel() > 0
        and torch.all(q_lengths == max_seqlen_q).item()
        and torch.all(k_lengths == max_seqlen_k).item()
        and int(cu_seqlens_q[-1].item()) == int(q.shape[0])
        and int(cu_seqlens_k[-1].item()) == int(k.shape[0])
    ):
        batch_size = int(q_lengths.numel())
        q_batched = q.view(batch_size, int(max_seqlen_q), *q.shape[1:])
        k_batched = k.view(batch_size, int(max_seqlen_k), *k.shape[1:])
        v_batched = v.view(batch_size, int(max_seqlen_k), *v.shape[1:])
        return _sdpa(
            q_batched,
            k_batched,
            v_batched,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        ).reshape_as(q)

    outputs = torch.empty_like(q)
    cu_q = cu_seqlens_q.detach().cpu().tolist()
    cu_k = cu_seqlens_k.detach().cpu().tolist()
    for index in range(len(cu_q) - 1):
        q_start, q_end = int(cu_q[index]), int(cu_q[index + 1])
        k_start, k_end = int(cu_k[index]), int(cu_k[index + 1])
        if q_end <= q_start:
            continue
        outputs[q_start:q_end] = _sdpa(
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    return outputs


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *args,
    **kwargs,
) -> torch.Tensor:
    return flash_attn_varlen_func(
        q,
        kv[:, 0],
        kv[:, 1],
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )


__all__ = [
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
]
