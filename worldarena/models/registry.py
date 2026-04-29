from __future__ import annotations

from worldarena.models.adapters import (
    AstraAdapter,
    FlashWorldAdapter,
    CameraCtrlAdapter,
    Gen3CAdapter,
    HYWorldPlayAdapter,
    HunyuanGameCraftAdapter,
    HunyuanWorldAdapter,
    InSpatioWorldAdapter,
    InfiniteWorldAdapter,
    LingBotWorldAdapter,
    LucidDreamerAdapter,
    Lyra1Adapter,
    Lyra2Adapter,
    MatrixGameAdapter,
    NeoVerseAdapter,
    ModelAdapter,
    MotionCtrlAdapter,
    ReCamMasterAdapter,
    VideoApiAdapter,
    WonderJourneyAdapter,
    WonderWorldAdapter,
    WorldCamAdapter,
    WorldFMAdapter,
    YumeAdapter,
)
from worldarena.models.config import ModelRuntimeConfig


def build_model_adapter(config: ModelRuntimeConfig) -> ModelAdapter:
    if config.backend_type == "http_api":
        return VideoApiAdapter(config)
    if config.backend_type == "sdk_api":
        raise ValueError(
            "sdk_api adapters are not wired yet; use backend_type=http_api with submit_mode=http_polling for now"
        )

    family = config.family.lower()
    if family in {"cameractrl", "camera_ctrl", "camera-ctrl"}:
        return CameraCtrlAdapter(config)
    if family in {"astra"}:
        return AstraAdapter(config)
    if family in {"lingbot", "lingbot_world", "lingbot-world"}:
        return LingBotWorldAdapter(config)
    if family in {"gen3c"}:
        return Gen3CAdapter(config)
    if family in {"luciddreamer", "lucid_dreamer", "lucid-dreamer"}:
        return LucidDreamerAdapter(config)
    if family in {"hy_worldplay", "hy-worldplay", "worldplay"}:
        return HYWorldPlayAdapter(config)
    if family in {
        "hunyuan_gamecraft",
        "hunyuan-gamecraft",
        "hunyuan_gamecraft_1_0",
        "hunyuan-gamecraft-1.0",
    }:
        return HunyuanGameCraftAdapter(config)
    if family in {"hunyuanworld", "hunyuan_world", "hunyuanworld_1_0", "hunyuanworld-1.0"}:
        return HunyuanWorldAdapter(config)
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
    if family in {"motionctrl", "motion_ctrl", "motion-ctrl"}:
        return MotionCtrlAdapter(config)
    if family in {"recammaster", "recam_master", "recam-master"}:
        return ReCamMasterAdapter(config)
    if family in {"lyra1", "lyra_1", "lyra-1"}:
        return Lyra1Adapter(config)
    if family in {"lyra2", "lyra_2", "lyra-2"}:
        return Lyra2Adapter(config)
    if family in {"flashworld", "flash_world", "flash-world"}:
        return FlashWorldAdapter(config)
    if family in {"inspatio_world", "inspatio-world"}:
        return InSpatioWorldAdapter(config)
    if family in {"infinite_world", "infinite-world"}:
        return InfiniteWorldAdapter(config)
    if family in {"neoverse", "neo_verse", "neo-verse"}:
        return NeoVerseAdapter(config)
    if family in {"wonderworld", "wonder_world", "wonder-world"}:
        return WonderWorldAdapter(config)
    if family in {"wonderjourney", "wonder_journey", "wonder-journey"}:
        return WonderJourneyAdapter(config)
    if family in {"worldcam"}:
        return WorldCamAdapter(config)
    if family in {"worldfm", "world_fm", "world-fm"}:
        return WorldFMAdapter(config)
    if family in {"yume", "yume1_5", "yume_1_5", "yume-1.5", "yume1.5"}:
        return YumeAdapter(config)
    raise ValueError(f"Unsupported model family: {config.family}")
