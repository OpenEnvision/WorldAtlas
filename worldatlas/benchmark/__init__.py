"""Public benchmark entry points.

Imports are resolved lazily so generation-only environments do not need the
heavier evaluation/build dependencies such as pandas.
"""

from importlib import import_module


_EXPORTS = {
    "BenchmarkAggregationConfig": ("worldatlas.benchmark.config", "BenchmarkAggregationConfig"),
    "BenchmarkConfig": ("worldatlas.benchmark.config", "BenchmarkConfig"),
    "BenchmarkPaths": ("worldatlas.benchmark.config", "BenchmarkPaths"),
    "BenchmarkProtocolConfig": ("worldatlas.benchmark.config", "BenchmarkProtocolConfig"),
    "aggregate_results": ("worldatlas.benchmark.aggregates", "aggregate_results"),
    "build_inventory": ("worldatlas.benchmark.inventory", "build_inventory"),
    "build_manifest": ("worldatlas.benchmark.manifest", "build_manifest"),
    "load_config": ("worldatlas.benchmark.config", "load_config"),
    "load_manifest": ("worldatlas.benchmark.manifest", "load_manifest"),
    "run_benchmark": ("worldatlas.benchmark.runner", "run_benchmark"),
    "write_inventory_artifacts": ("worldatlas.benchmark.inventory", "write_inventory_artifacts"),
    "write_manifest_artifacts": ("worldatlas.benchmark.manifest", "write_manifest_artifacts"),
}

__all__ = [
    "BenchmarkAggregationConfig",
    "BenchmarkConfig",
    "BenchmarkPaths",
    "BenchmarkProtocolConfig",
    "aggregate_results",
    "build_inventory",
    "build_manifest",
    "load_config",
    "load_manifest",
    "run_benchmark",
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
