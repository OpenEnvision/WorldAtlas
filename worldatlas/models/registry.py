"""Factory that maps ``ModelRuntimeConfig.family`` to a concrete ``ModelAdapter``."""

from __future__ import annotations

from worldatlas.models.adapters import (
    ABotWorldAdapter,
    AlayaWorldAdapter,
    AlayaWorldV11Adapter,
    AstronexWorldAdapter,
    Cosmos3Adapter,
    DreamXWorldAdapter,
    EchoWMAdapter,
    EvokeAdapter,
    FlashWorldAdapter,
    Gen3CAdapter,
    HeliosAdapter,
    HYWorldPlayAdapter,
    InfiniteWorldAdapter,
    LingBotVideoAdapter,
    LingBotWorldAdapter,
    LingBotWorldV2Adapter,
    LongCatVideoAdapter,
    LTX23Adapter,
    LTX25Adapter,
    LucidDreamerAdapter,
    Lyra2Adapter,
    MatrixGame35Adapter,
    MatrixGameAdapter,
    MiniMaxH3Adapter,
    ModelAdapter,
    RealCamI2VAdapter,
    SanaWMAdapter,
    SolarWMAdapter,
    StableVirtualCameraAdapter,
    VideoApiAdapter,
    Wan22Adapter,
    WarpAsHistoryAdapter,
    WorldCrafterAdapter,
    YumeAdapter,
    ZingWorldAdapter,
)
from worldatlas.models.config import ModelRuntimeConfig


def build_model_adapter(config: ModelRuntimeConfig) -> ModelAdapter:
    """Instantiate the adapter class registered for ``config.family``."""
    if config.backend_type == "http_api":
        return VideoApiAdapter(config)
    if config.backend_type == "sdk_api":
        raise ValueError(
            "sdk_api adapters are not wired yet; use backend_type=http_api with submit_mode=http_polling for now"
        )

    family = config.family.lower()
    if family in {"zing", "zing_world", "zing-world", "zing-world-model", "zing_0_5", "zing-0.5"}:
        return ZingWorldAdapter(config)
    if family in {"helios", "helios_distilled"}:
        return HeliosAdapter(config)
    if family in {"abot", "abot_world", "abot-world"}:
        return ABotWorldAdapter(config)
    if family in {"alaya", "alaya_world", "alaya-world", "alayaworld"}:
        return AlayaWorldAdapter(config)
    if family in {"alayaworld_v11", "alayaworld-v1.1"}:
        return AlayaWorldV11Adapter(config)
    if family in {"astronex_world", "astronex-world", "astronex"}:
        return AstronexWorldAdapter(config)
    if family in {"realcam_i2v", "realcam-i2v", "realcami2v"}:
        return RealCamI2VAdapter(config)
    if family in {"sana_wm", "sana-wm", "sanawm", "sana"}:
        return SanaWMAdapter(config)
    if family in {
        "cosmos3",
        "cosmos-3",
        "cosmos3_nano",
        "cosmos3-nano",
    }:
        return Cosmos3Adapter(config)
    if family in {"dreamx_world", "dreamx-world", "dreamx"}:
        return DreamXWorldAdapter(config)
    if family in {"echo_wm", "echo-wm", "echowm"}:
        return EchoWMAdapter(config)
    if family in {"evoke", "alaya_evoke", "alaya-evoke", "alayaevoke"}:
        return EvokeAdapter(config)
    if family in {"lingbot_world_v2", "lingbot-world-v2"}:
        return LingBotWorldV2Adapter(config)
    if family in {"lingbot", "lingbot_world", "lingbot-world"}:
        return LingBotWorldAdapter(config)
    if family in {"lingbot_video", "lingbot-video"}:
        return LingBotVideoAdapter(config)
    if family in {"gen3c"}:
        return Gen3CAdapter(config)
    if family in {"luciddreamer", "lucid_dreamer", "lucid-dreamer"}:
        return LucidDreamerAdapter(config)
    if family in {"hy_worldplay", "hy-worldplay", "worldplay"}:
        return HYWorldPlayAdapter(config)
    if family in {"longcat_video", "longcat-video", "longcat"}:
        return LongCatVideoAdapter(config)
    if family in {"ltx2.3", "ltx23", "ltx_2_3", "ltx-2.3", "ltx-23", "ltx_video"}:
        return LTX23Adapter(config)
    if family in {"ltx2.5", "ltx25", "ltx_2_5", "ltx-2.5", "ltx-25"}:
        return LTX25Adapter(config)
    if family in {
        "matrix_game",
        "matrix-game",
        "matrix_game_1",
        "matrix-game-1",
        "matrix_game_2",
        "matrix-game-2",
        "matrix_game_3",
        "matrix-game-3",
    }:
        return MatrixGameAdapter(config)
    if family in {
        "matrix_game_35",
        "matrix-game-35",
        "matrix_game_3_5",
        "matrix-game-3.5",
    }:
        return MatrixGame35Adapter(config)
    if family in {"minimax_h3", "minimax-h3", "minimaxh3", "minimax_h3_t2va"}:
        return MiniMaxH3Adapter(config)
    if family in {"lyra2", "lyra_2", "lyra-2"}:
        return Lyra2Adapter(config)
    if family in {"flashworld", "flash_world", "flash-world"}:
        return FlashWorldAdapter(config)
    if family in {"infinite_world", "infinite-world"}:
        return InfiniteWorldAdapter(config)
    if family in {"stable_virtual_camera", "stable-virtual-camera"}:
        return StableVirtualCameraAdapter(config)
    if family in {"solarwm", "solar_wm", "solar-wm"}:
        return SolarWMAdapter(config)
    if family in {"solarwm_h3", "solarwm-h3"}:
        from worldatlas.models.adapters.solarwm.solarwm_h3 import SolarWMH3Adapter

        return SolarWMH3Adapter(config)
    if family in {"worldcrafter", "worldcrafter_fast", "worldcrafter-fast"}:
        return WorldCrafterAdapter(config)
    if family in {"yume", "yume1_5", "yume_1_5", "yume-1.5", "yume1.5"}:
        return YumeAdapter(config)
    if family in {"wan", "wan22", "wan2.2", "wan_2_2", "wan2_2"}:
        return Wan22Adapter(config)
    if family in {"warp_as_history", "warp-as-history", "warpas_history", "wah"}:
        return WarpAsHistoryAdapter(config)
    raise ValueError(f"Unsupported model family: {config.family}")
