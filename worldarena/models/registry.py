from __future__ import annotations

from worldarena.models.adapters import LingBotWorldAdapter, MatrixGameAdapter, ModelAdapter
from worldarena.models.config import ModelRuntimeConfig


def build_model_adapter(config: ModelRuntimeConfig) -> ModelAdapter:
    family = config.family.lower()
    if family in {"lingbot", "lingbot_world", "lingbot-world"}:
        return LingBotWorldAdapter(config)
    if family in {"matrix_game", "matrix-game", "matrix_game_3", "matrix-game-3"}:
        return MatrixGameAdapter(config)
    raise ValueError(f"Unsupported model family: {config.family}")
