"""Public benchmark entry points.

Imports are resolved lazily so generation-only environments do not need the
heavier evaluation/build dependencies such as pandas.
"""

from importlib import import_module


_EXPORTS = {
    "BenchmarkAggregationConfig": ("worldarena.benchmark.config", "BenchmarkAggregationConfig"),
    "BenchmarkConfig": ("worldarena.benchmark.config", "BenchmarkConfig"),
    "BenchmarkPaths": ("worldarena.benchmark.config", "BenchmarkPaths"),
    "BenchmarkProtocolConfig": ("worldarena.benchmark.config", "BenchmarkProtocolConfig"),
    "aggregate_results": ("worldarena.benchmark.aggregates", "aggregate_results"),
    "build_inventory": ("worldarena.benchmark.inventory", "build_inventory"),
    "build_manifest": ("worldarena.benchmark.manifest", "build_manifest"),
    "build_wmbench_physics_judge_manifest": (
        "worldarena.benchmark.physics_judge_manifest",
        "build_wmbench_physics_judge_manifest",
    ),
    "build_worldarena_physics_seed_manifest": (
        "worldarena.benchmark.physics_manifest",
        "build_worldarena_physics_seed_manifest",
    ),
    "build_worldarena_physics_manifest": (
        "worldarena.benchmark.physics_manifest",
        "build_worldarena_physics_manifest",
    ),
    "load_config": ("worldarena.benchmark.config", "load_config"),
    "load_manifest": ("worldarena.benchmark.manifest", "load_manifest"),
    "run_benchmark": ("worldarena.benchmark.runner", "run_benchmark"),
    "write_physics_judge_manifest_artifacts": (
        "worldarena.benchmark.physics_judge_manifest",
        "write_physics_judge_manifest_artifacts",
    ),
    "write_physics_manifest_artifacts": (
        "worldarena.benchmark.physics_manifest",
        "write_physics_manifest_artifacts",
    ),
    "write_inventory_artifacts": ("worldarena.benchmark.inventory", "write_inventory_artifacts"),
    "write_manifest_artifacts": ("worldarena.benchmark.manifest", "write_manifest_artifacts"),
}

__all__ = [
    "BenchmarkAggregationConfig",
    "BenchmarkConfig",
    "BenchmarkPaths",
    "BenchmarkProtocolConfig",
    "aggregate_results",
    "build_inventory",
    "build_manifest",
    "build_wmbench_physics_judge_manifest",
    "build_worldarena_physics_seed_manifest",
    "build_worldarena_physics_manifest",
    "load_config",
    "load_manifest",
    "run_benchmark",
    "write_physics_judge_manifest_artifacts",
    "write_physics_manifest_artifacts",
    "write_inventory_artifacts",
    "write_manifest_artifacts",
]


def __getattr__(name: str):
    """Lazy attribute accessor."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
