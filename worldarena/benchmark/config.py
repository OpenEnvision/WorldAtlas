from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from worldarena.models.config import ModelRuntimeConfig


@dataclass(slots=True)
class BenchmarkPaths:
    image_root: Path
    video_root: Path
    inventory_dir: Path
    manifest_dir: Path
    reports_dir: Path
    cache_dir: Path


DEFAULT_DERIVED_AGGREGATION_WEIGHTS: dict[str, float] = {
    "image_static": 0.15,
    "image_dynamic": 0.15,
    "video_static": 0.30,
    "video_dynamic": 0.40,
}


def _default_derived_aggregation_weights() -> dict[str, float]:
    return dict(DEFAULT_DERIVED_AGGREGATION_WEIGHTS)


@dataclass(slots=True)
class BenchmarkAggregationConfig:
    enabled: bool = False
    suite_weights: dict[str, float] = field(default_factory=_default_derived_aggregation_weights)


@dataclass(slots=True)
class BenchmarkProtocolConfig:
    official: str = "anchor_compat"
    compatibility: str = "anchor_compat"
    seed: int = 17
    segment_count: int = 4


@dataclass(slots=True)
class BenchmarkConfig:
    config_path: Path
    paths: BenchmarkPaths
    snapshot_name: str
    target_fps: int
    anchor_positions: list[float]
    motion_frame_count: int
    split_ratios: dict[str, float]
    metrics_by_suite: dict[str, list[str]]
    metric_backends: dict[str, str]
    normalization: dict[str, dict[str, Any]]
    aggregation: BenchmarkAggregationConfig = field(default_factory=BenchmarkAggregationConfig)
    planned_counts: dict[str, int | None] = field(default_factory=dict)
    protocol: BenchmarkProtocolConfig = field(default_factory=BenchmarkProtocolConfig)


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(path: Path) -> BenchmarkConfig:
    config_path = path.resolve()
    config_root = config_path.parent
    project_root = config_root.parent
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    path_payload = payload.get("paths", {})
    benchmark_payload = payload.get("benchmark", {})
    aggregation_payload = benchmark_payload.get("aggregation") or {}
    protocol_payload = benchmark_payload.get("protocol") or {}

    paths = BenchmarkPaths(
        image_root=_resolve(project_root, path_payload["image_root"]),
        video_root=_resolve(project_root, path_payload["video_root"]),
        inventory_dir=_resolve(project_root, path_payload["inventory_dir"]),
        manifest_dir=_resolve(project_root, path_payload["manifest_dir"]),
        reports_dir=_resolve(project_root, path_payload["reports_dir"]),
        cache_dir=_resolve(project_root, path_payload["cache_dir"]),
    )

    return BenchmarkConfig(
        config_path=config_path,
        paths=paths,
        snapshot_name=benchmark_payload.get("snapshot_name", "worldarena"),
        target_fps=int(benchmark_payload.get("target_fps", 24)),
        anchor_positions=[
            float(value)
            for value in benchmark_payload.get("anchor_positions", [0.0, 0.3333, 0.6667, 1.0])
        ],
        motion_frame_count=int(benchmark_payload.get("motion_frame_count", 12)),
        split_ratios={
            key: float(value)
            for key, value in benchmark_payload.get(
                "split_ratios",
                {"dev_public": 0.2, "dev_private": 0.1, "test_hidden": 0.7},
            ).items()
        },
        metrics_by_suite={
            key: [str(metric) for metric in value]
            for key, value in benchmark_payload.get("metrics_by_suite", {}).items()
        },
        metric_backends={
            key: str(value)
            for key, value in benchmark_payload.get("metric_backends", {}).items()
        },
        normalization={
            key: dict(value)
            for key, value in benchmark_payload.get("normalization", {}).items()
        },
        aggregation=BenchmarkAggregationConfig(
            enabled=bool(aggregation_payload.get("enabled", False)),
            suite_weights={
                key: float(value)
                for key, value in aggregation_payload.get(
                    "suite_weights",
                    DEFAULT_DERIVED_AGGREGATION_WEIGHTS,
                ).items()
            },
        ),
        planned_counts={
            key: (None if value is None else int(value))
            for key, value in benchmark_payload.get("planned_counts", {}).items()
        },
        protocol=BenchmarkProtocolConfig(
            official=str(protocol_payload.get("official", "anchor_compat")),
            compatibility=str(protocol_payload.get("compatibility", "anchor_compat")),
            seed=int(protocol_payload.get("seed", 17)),
            segment_count=int(protocol_payload.get("segment_count", 4)),
        ),
    )


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


__all__ = [
    "BenchmarkAggregationConfig",
    "BenchmarkConfig",
    "BenchmarkPaths",
    "BenchmarkProtocolConfig",
    "apply_model_metric_overrides",
    "load_config",
]
