"""Minimal bert-padding helpers for the flash-attn fallback package."""

from __future__ import annotations

import torch


def index_first_axis(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return input.index_select(0, indices)


__all__ = ["index_first_axis"]
