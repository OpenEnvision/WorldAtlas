from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from worldarena.models.config import ModelRuntimeConfig
from worldarena.benchmark_core.config import BenchmarkConfig, BenchmarkPaths, load_config as load_core_config

__all__ = [
    "BenchmarkConfig",
    "BenchmarkPaths",
    "apply_model_metric_overrides",
    "load_config",
]


def load_config(path: Path) -> BenchmarkConfig:
    return load_core_config(path)


def apply_model_metric_overrides(
    config: BenchmarkConfig,
    model_config: ModelRuntimeConfig | None,
) -> BenchmarkConfig:
    if model_config is None or not model_config.metric_overrides:
        return config

    patched = deepcopy(config)
    patched.metrics_by_suite = deepcopy(config.metrics_by_suite)
    for suite, metrics in model_config.metric_overrides.items():
        patched.metrics_by_suite[suite] = list(metrics)
    return patched
