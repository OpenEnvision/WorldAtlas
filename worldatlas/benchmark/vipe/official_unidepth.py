"""Load memory-track UniDepth from WorldAtlas ``thirdparty/UniDepth``.

Official NVIDIA ViPE vendors a UniDepth whose decoder layout no longer matches
the ``lpiccinelli/unidepth-v2-vitl14`` snapshot (``level_embeds`` ``[4, 512]``
vs ``[1, 1, 4, 512]``, ResidualConv ``gamma`` ``[512]`` vs ``[1, C, 1, 1]``).
This module puts the official clone on ``sys.path``, fills the slim HF config,
reshapes compatible tensors, and replaces ``make_depth_model("unidepth-*")``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.common.checkpoints import project_root
from worldatlas.benchmark.vipe.depth_contract import pinhole_supported_camera_types
from worldatlas.benchmark.vipe.memory_weights import unidepth_snapshot_dir

UNIDEPTH_REPO = project_root() / "thirdparty" / "UniDepth"


def official_unidepth_root() -> Path:
    explicit = os.environ.get("WORLDATLAS_UNIDEPTH_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return UNIDEPTH_REPO


def ensure_unidepth_on_path() -> Path:
    root = official_unidepth_root()
    package = root / "unidepth"
    if not package.is_dir():
        raise MetricLoadError(
            f"Official UniDepth is missing at {root}. Clone "
            "https://github.com/lpiccinelli-eth/UniDepth into WorldAtlas/thirdparty/UniDepth."
        )
    inserted = str(root)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    return root


def normalize_unidepth_v2_hf_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill fields the older HF snapshot omits but Decoder.build / infer need."""
    model = config.setdefault("model", {})
    model.setdefault("layer_scale", 1.0)
    decoder = model.setdefault("pixel_decoder", {})
    decoder.setdefault("out_dim", 64)

    data = config.setdefault("data", {})
    augmentations = data.setdefault("augmentations", {})
    constraints = augmentations.get("shape_constraints") or dict(data.get("shape_constraints") or {})
    bounds = constraints.get("pixels_bounds")
    if bounds and len(bounds) >= 2:
        constraints.setdefault("pixels_min", bounds[0])
        constraints.setdefault("pixels_max", bounds[1])
    constraints.setdefault("pixels_min", 1400)
    constraints.setdefault("pixels_max", 2400)
    constraints.setdefault("ratio_bounds", [0.66, 2.0])
    augmentations["shape_constraints"] = constraints
    return config


def reshape_unidepth_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """View older HF tensors onto the official module layout when numel matches."""
    target = model.state_dict()
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        dest = target.get(key)
        if dest is not None and dest.shape != value.shape and dest.numel() == value.numel():
            remapped[key] = value.reshape(dest.shape)
        else:
            remapped[key] = value
    return remapped


def load_official_unidepth_v2(source: str | Path) -> nn.Module:
    """Build official UniDepthV2 from a local snapshot directory."""
    from safetensors.torch import load_file

    ensure_unidepth_on_path()
    from unidepth.models.unidepthv2.unidepthv2 import UniDepthV2

    root = Path(source)
    config_path = root / "config.json"
    weight_path = root / "model.safetensors"
    if not config_path.is_file():
        raise MetricLoadError(f"UniDepth config missing: {config_path}")
    if not weight_path.is_file():
        raise MetricLoadError(f"UniDepth weights missing: {weight_path}")
    config = normalize_unidepth_v2_hf_config(json.loads(config_path.read_text(encoding="utf-8")))
    # Official UniDepthV2.__init__ always builds training losses. The HF
    # snapshot ships ``training: {}``; those heads are unused at infer time.
    original_build_losses = UniDepthV2.build_losses

    def _skip_losses(self, _config):
        self.losses = {}

    UniDepthV2.build_losses = _skip_losses
    try:
        model = UniDepthV2(config)
    finally:
        UniDepthV2.build_losses = original_build_losses
    remapped = reshape_unidepth_state_dict(model, load_file(str(weight_path), device="cpu"))
    model.load_state_dict(remapped, strict=False)
    if not hasattr(model, "resolution_level"):
        model.resolution_level = 9
    return model


class ArenaUniDepth2Model:
    """Official-ViPE UniDepth wrapper backed by the official clone."""

    def __init__(self, type: Literal["s", "b", "l"] = "l", model_path: str | None = None) -> None:
        explicit = (model_path or os.environ.get("WORLDATLAS_UNIDEPTH_MODEL", "")).strip()
        source = Path(explicit) if explicit else unidepth_snapshot_dir()
        self.model = load_official_unidepth_v2(source)
        self.model.interpolation_mode = "bilinear"
        self.model = self.model.cuda().eval()

    @property
    def depth_type(self):
        from vipe.priors.depth.base import DepthType

        return DepthType.MODEL_METRIC_DEPTH

    @property
    def supported_camera_types(self):
        return pinhole_supported_camera_types()

    def estimate(self, src):
        from vipe.priors.depth.base import DepthEstimationResult
        from vipe.utils.cameras import CameraType
        from vipe.utils.misc import unpack_optional
        from unidepth.utils.camera import Pinhole

        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.camera_type == CameraType.PINHOLE, "UniDepth only supports pinhole cameras"
        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb = torch.clamp(rgb.moveaxis(-1, 1) * 255.0, max=255.0).byte()
        k = torch.tensor(
            [
                [focal_length, 0, rgb.shape[-1] / 2],
                [0, focal_length, rgb.shape[-2] / 2],
                [0, 0, 1],
            ],
            device=rgb.device,
        ).float()
        camera = Pinhole(K=k[None].repeat(rgb.shape[0], 1, 1))
        predictions = self.model.infer(rgb, camera)
        pred_depth = predictions["depth"].squeeze(1)
        confidence = predictions["confidence"].squeeze(1)
        if not batch_dim:
            pred_depth, confidence = pred_depth[0], confidence[0]
        return DepthEstimationResult(
            metric_depth=pred_depth,
            confidence=confidence,
        )


def install_official_unidepth_override() -> None:
    """Install Arena official UniDepth/VDA/PriorDA before ViPE imports depth models."""
    from worldatlas.benchmark.vipe.official_depth import install_official_depth_override

    install_official_depth_override()
