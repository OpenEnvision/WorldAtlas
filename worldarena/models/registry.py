from __future__ import annotations

from worldarena.models.adapters import (
    HunyuanWorldAdapter,
    InfiniteWorldAdapter,
    LingBotWorldAdapter,
    Lyra1Adapter,
    Lyra2Adapter,
    MatrixGameAdapter,
    ModelAdapter,
    VideoApiAdapter,
    WorldCamAdapter,
    WorldFMAdapter,
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
    if family in {"lingbot", "lingbot_world", "lingbot-world"}:
        return LingBotWorldAdapter(config)
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
    if family in {"lyra1", "lyra_1", "lyra-1"}:
        return Lyra1Adapter(config)
    if family in {"lyra2", "lyra_2", "lyra-2"}:
        return Lyra2Adapter(config)
    if family in {"infinite_world", "infinite-world"}:
        return InfiniteWorldAdapter(config)
    if family in {"worldcam"}:
        return WorldCamAdapter(config)
    if family in {"worldfm", "world_fm", "world-fm"}:
        return WorldFMAdapter(config)
    raise ValueError(f"Unsupported model family: {config.family}")
