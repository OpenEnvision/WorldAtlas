"""WorldArena compatibility hooks for the upstream SEA-RAFT checkout."""

from __future__ import annotations

from typing import Any


def _corr_block_init(self: Any, fmap1: Any, fmap2: Any, args: Any) -> None:
    import torch.nn.functional as functional

    self.num_levels = args.corr_levels
    self.radius = args.corr_radius
    self.args = args
    self.corr_pyramid = []
    for index in range(self.num_levels):
        corr = type(self).corr(fmap1, fmap2, 1)
        batch, height, width, dimension, height2, width2 = corr.shape
        corr = corr.reshape(batch * height * width, dimension, height2, width2)
        self.corr_pyramid.append(corr)
        if index + 1 < self.num_levels:
            fmap2 = functional.interpolate(
                fmap2,
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            )


def _bilinear_sampler(
    image: Any,
    coords: Any,
    mode: str = "bilinear",
    mask: bool = False,
) -> Any:
    import torch
    import torch.nn.functional as functional

    height, width = image.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = torch.zeros_like(xgrid) if width == 1 else 2 * xgrid / (width - 1) - 1
    ygrid = torch.zeros_like(ygrid) if height == 1 else 2 * ygrid / (height - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    sampled = functional.grid_sample(image, grid, mode=mode, align_corners=True)
    if mask:
        valid = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return sampled, valid.float()
    return sampled


def install_sea_raft_compat() -> None:
    """Install small-input fixes without modifying the upstream submodule."""
    import corr
    import utils.utils as sea_raft_utils

    corr.CorrBlock.__init__ = _corr_block_init
    corr.bilinear_sampler = _bilinear_sampler
    sea_raft_utils.bilinear_sampler = _bilinear_sampler


__all__ = ["install_sea_raft_compat"]
