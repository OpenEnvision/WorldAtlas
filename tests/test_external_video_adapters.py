from __future__ import annotations

from pathlib import Path

from worldarena.models.adapters.cogvideo import CogVideoAdapter
from worldarena.models.adapters.cosmos_framework import CosmosFrameworkAdapter
from worldarena.models.adapters.hunyuan_video import HunyuanVideoAdapter
from worldarena.models.adapters.longcat_video import LongCatVideoAdapter
from worldarena.models.adapters.ltx23 import LTX23Adapter
from worldarena.models.adapters.sana_wm import SanaWMAdapter
from worldarena.models.adapters.wan22 import Wan22Adapter
from worldarena.models.adapters.warp_as_history import WarpAsHistoryAdapter
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


def test_external_video_model_configs_build_batch_adapters() -> None:
    expected = {
        "config/models/wan22.yaml": Wan22Adapter,
        "config/models/warp_as_history.yaml": WarpAsHistoryAdapter,
        "config/models/sana_wm.yaml": SanaWMAdapter,
        "config/models/longcat_video.yaml": LongCatVideoAdapter,
        "config/models/ltx23.yaml": LTX23Adapter,
        "config/models/hunyuan_video.yaml": HunyuanVideoAdapter,
        "config/models/cosmos_framework.yaml": CosmosFrameworkAdapter,
        "config/models/cogvideo.yaml": CogVideoAdapter,
    }

    for config_path, adapter_type in expected.items():
        adapter = build_model_adapter(load_model_config(Path(config_path)))
        assert isinstance(adapter, adapter_type)
        assert adapter.supports_batch_generation()
