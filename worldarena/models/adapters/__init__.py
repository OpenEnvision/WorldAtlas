from __future__ import annotations

from importlib import import_module
from typing import Any

_ADAPTER_MODULES = {
    "AstraAdapter": "worldarena.models.adapters.astra",
    "ApiModelAdapter": "worldarena.models.adapters.api",
    "CameraCtrlAdapter": "worldarena.models.adapters.cameractrl",
    "FlashWorldAdapter": "worldarena.models.adapters.flashworld",
    "Gen3CAdapter": "worldarena.models.adapters.gen3c",
    "HYWorldPlayAdapter": "worldarena.models.adapters.hy_worldplay",
    "HunyuanGameCraftAdapter": "worldarena.models.adapters.hunyuan_gamecraft",
    "HunyuanWorldAdapter": "worldarena.models.adapters.hunyuanworld",
    "InSpatioWorldAdapter": "worldarena.models.adapters.inspatio_world",
    "InfiniteWorldAdapter": "worldarena.models.adapters.infinite_world",
    "LingBotWorldAdapter": "worldarena.models.adapters.lingbot_world",
    "LucidDreamerAdapter": "worldarena.models.adapters.luciddreamer",
    "Lyra1Adapter": "worldarena.models.adapters.lyra1",
    "Lyra2Adapter": "worldarena.models.adapters.lyra2",
    "MatrixGameAdapter": "worldarena.models.adapters.matrix_game",
    "ModelAdapter": "worldarena.models.adapters.base",
    "MotionCtrlAdapter": "worldarena.models.adapters.motionctrl",
    "NeoVerseAdapter": "worldarena.models.adapters.neoverse",
    "ReCamMasterAdapter": "worldarena.models.adapters.recammaster",
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
