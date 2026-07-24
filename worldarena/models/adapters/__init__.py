"""Lazy-loaded registry of all supported world model adapters.

Adapters are imported on first access so ``import worldarena.models.adapters``
does not pull in every model's heavy dependencies at once.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ADAPTER_MODULES = {
    "ABotWorldAdapter": "worldarena.models.adapters.abot_world",
    "AlayaWorldAdapter": "worldarena.models.adapters.alayaworld",
    "ApiModelAdapter": "worldarena.models.adapters.api",
    "CamI2VAdapter": "worldarena.models.adapters.cami2v",
    "CosmosPredict25Adapter": "worldarena.models.adapters.cosmos_predict25",
    "FlashWorldAdapter": "worldarena.models.adapters.flashworld",
    "Gen3CAdapter": "worldarena.models.adapters.gen3c",
    "HYWorldPlayAdapter": "worldarena.models.adapters.hy_worldplay",
    "HunyuanGameCraftAdapter": "worldarena.models.adapters.hunyuan_gamecraft",
    "HunyuanWorldAdapter": "worldarena.models.adapters.hunyuanworld",
    "InSpatioWorldAdapter": "worldarena.models.adapters.inspatio_world",
    "InfiniteWorldAdapter": "worldarena.models.adapters.infinite_world",
    "LingBotVideoAdapter": "worldarena.models.adapters.lingbot_video",
    "LingBotWorldAdapter": "worldarena.models.adapters.lingbot_world",
    "LucidDreamerAdapter": "worldarena.models.adapters.luciddreamer",
    "Lyra1Adapter": "worldarena.models.adapters.lyra1",
    "Lyra2Adapter": "worldarena.models.adapters.lyra2",
    "MatrixGameAdapter": "worldarena.models.adapters.matrix_game",
    "MatrixGame35Adapter": "worldarena.models.adapters.matrix_game_35",
    "ModelAdapter": "worldarena.models.adapters.base",
    "NeoVerseAdapter": "worldarena.models.adapters.neoverse",
    "RealCamI2VAdapter": "worldarena.models.adapters.realcam_i2v",
    "StableVirtualCameraAdapter": "worldarena.models.adapters.stable_virtual_camera",
    "VideoApiAdapter": "worldarena.models.adapters.video_api",
    "WonderJourneyAdapter": "worldarena.models.adapters.wonderjourney",
    "WonderWorldAdapter": "worldarena.models.adapters.wonderworld",
    "WorldCamAdapter": "worldarena.models.adapters.worldcam",
    "WorldFMAdapter": "worldarena.models.adapters.worldfm",
    "YumeAdapter": "worldarena.models.adapters.yume",
}


def __getattr__(name: str) -> Any:
    module_name = _ADAPTER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = sorted(_ADAPTER_MODULES)
