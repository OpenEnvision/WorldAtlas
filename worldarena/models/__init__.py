"""Public model loading and adapter factory entry points."""

from __future__ import annotations

from worldarena.models.config import ModelRuntimeConfig, load_model_config


def build_model_adapter(config: ModelRuntimeConfig):
    from worldarena.models.registry import build_model_adapter as _build_model_adapter

    return _build_model_adapter(config)

__all__ = [
    "ModelRuntimeConfig",
    "build_model_adapter",
    "load_model_config",
]
