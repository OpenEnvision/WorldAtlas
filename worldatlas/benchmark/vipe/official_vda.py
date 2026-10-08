"""Load memory-track Video-Depth-Anything through official NVIDIA ViPE.

Official ViPE vendors a VDA adapter whose ``infer_video_depth`` already
expects ViPE ``[0, 1]`` RGB (the standalone DepthAnything repo still
divides by 255). Weights come from WorldAtlas ``ckpt/`` so Hope stays offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.common.checkpoints import project_root
from worldatlas.benchmark.vipe.depth_contract import pinhole_supported_camera_types
from worldatlas.benchmark.vipe.memory_weights import vda_checkpoint_path

VDA_REPO = project_root() / "thirdparty" / "Video-Depth-Anything"
MODEL_CONFIG = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def official_vda_root() -> Path:
    explicit = os.environ.get("WORLDATLAS_VDA_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return VDA_REPO


def ensure_vda_on_path() -> Path:
    root = official_vda_root()
    package = root / "video_depth_anything"
    if not package.is_dir():
        raise MetricLoadError(
            f"Official Video-Depth-Anything is missing at {root}. Clone "
            "https://github.com/DepthAnything/Video-Depth-Anything into "
            "WorldAtlas/thirdparty/Video-Depth-Anything."
        )
    inserted = str(root)
    if inserted in sys.path:
        sys.path.remove(inserted)
    sys.path.insert(0, inserted)
    return root


class ArenaVideoDepthAnythingModel:
    """Official-ViPE VDA wrapper backed by a local weight file."""

    def __init__(
        self,
        model: Literal["vits", "vitl"] = "vits",
        input_size: int = 518,
        weights_path: str | None = None,
    ) -> None:
        if model not in MODEL_CONFIG:
            raise ValueError(f"Video-Depth-Anything model {model} not supported")
        from vipe.priors.depth.videodepthanything.video_depth import VideoDepthAnything

        self.input_size = input_size
        self.use_fp32 = model == "vits"
        self.model = VideoDepthAnything(**MODEL_CONFIG[model])
        weight = Path(weights_path).expanduser() if weights_path else vda_checkpoint_path()
        if not weight.is_file() or weight.stat().st_size <= 0:
            raise MetricLoadError(f"Video-Depth-Anything weight missing: {weight}")
        state = torch.load(str(weight), map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.cuda().eval()

    @property
    def depth_type(self):
        from vipe.priors.depth.base import DepthType

        return DepthType.AFFINE_DISP

    @property
    def supported_camera_types(self):
        return pinhole_supported_camera_types()

    def estimate(self, src):
        from vipe.priors.depth.base import DepthEstimationResult
        from vipe.utils.misc import unpack_optional

        frame_list = unpack_optional(src.video_frame_list)
        frames = list(frame_list) if not isinstance(frame_list, list) else frame_list
        depths = self.model.infer_video_depth(
            frames,
            input_size=self.input_size,
            fp32=self.use_fp32,
        )
        return DepthEstimationResult(relative_inv_depth=torch.from_numpy(np.asarray(depths)).float().cuda())
