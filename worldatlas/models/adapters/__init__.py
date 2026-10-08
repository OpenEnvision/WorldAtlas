"""Lazy-loaded registry of all supported world model adapters.

Adapters are imported on first access so ``import worldatlas.models.adapters``
does not pull in every model's heavy dependencies at once.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ADAPTER_MODULES = {
    "ABotWorldAdapter": "worldatlas.models.adapters.abot_world.abot_world",
    "AlayaWorldAdapter": "worldatlas.models.adapters.alayaworld.alayaworld",
    "AlayaWorldV11Adapter": "worldatlas.models.adapters.alayaworld.alayaworld_v11",
    "AstronexWorldAdapter": "worldatlas.models.adapters.astronex_world.astronex_world",
    "ApiModelAdapter": "worldatlas.models.adapters.api",
    "Cosmos3Adapter": "worldatlas.models.adapters.cosmos3.cosmos3",
    "DreamXWorldAdapter": "worldatlas.models.adapters.dreamx_world.dreamx_world",
    "EchoWMAdapter": "worldatlas.models.adapters.echo_wm.echo_wm",
    "EvokeAdapter": "worldatlas.models.adapters.evoke.evoke",
    "FlashWorldAdapter": "worldatlas.models.adapters.flashworld.flashworld",
    "Gen3CAdapter": "worldatlas.models.adapters.gen3c.gen3c",
    "HeliosAdapter": "worldatlas.models.adapters.helios.helios",
    "HYWorldPlayAdapter": "worldatlas.models.adapters.hy_worldplay.hy_worldplay",
    "InfiniteWorldAdapter": "worldatlas.models.adapters.infinite_world.infinite_world",
    "LingBotVideoAdapter": "worldatlas.models.adapters.lingbot_video.lingbot_video",
    "LingBotWorldAdapter": "worldatlas.models.adapters.lingbot_world.lingbot_world",
    "LingBotWorldV2Adapter": "worldatlas.models.adapters.lingbot_world.lingbot_world_v2",
    "LongCatVideoAdapter": "worldatlas.models.adapters.longcat_video.longcat_video",
    "LTX23Adapter": "worldatlas.models.adapters.ltx23.ltx23",
    "LTX25Adapter": "worldatlas.models.adapters.ltx25.ltx25",
    "LucidDreamerAdapter": "worldatlas.models.adapters.luciddreamer.luciddreamer",
    "Lyra2Adapter": "worldatlas.models.adapters.lyra2.lyra2",
    "MatrixGameAdapter": "worldatlas.models.adapters.matrix_game.matrix_game",
    "MatrixGame35Adapter": "worldatlas.models.adapters.matrix_game.matrix_game_35",
    "MiniMaxH3Adapter": "worldatlas.models.adapters.minimax_h3.minimax_h3",
    "ModelAdapter": "worldatlas.models.adapters.base",
    "RealCamI2VAdapter": "worldatlas.models.adapters.realcam_i2v.realcam_i2v",
    "SanaWMAdapter": "worldatlas.models.adapters.sana_wm.sana_wm",
    "SolarWMAdapter": "worldatlas.models.adapters.solarwm.solarwm",
    "StableVirtualCameraAdapter": "worldatlas.models.adapters.stable_virtual_camera.stable_virtual_camera",
    "VideoApiAdapter": "worldatlas.models.adapters.video_api",
    "Wan22Adapter": "worldatlas.models.adapters.wan22.wan22",
    "WarpAsHistoryAdapter": "worldatlas.models.adapters.warp_as_history.warp_as_history",
    "WorldCrafterAdapter": "worldatlas.models.adapters.worldcrafter.worldcrafter",
    "YumeAdapter": "worldatlas.models.adapters.yume.yume",
    "ZingWorldAdapter": "worldatlas.models.adapters.zing_world.zing_world",
}


def __getattr__(name: str) -> Any:
    module_name = _ADAPTER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = sorted(_ADAPTER_MODULES)
