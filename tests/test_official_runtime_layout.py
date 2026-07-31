import pytest

from worldarena.benchmark import official_runtime
from worldarena.benchmark.config import load_config
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.registry import METRIC_ALIASES, build_metrics


def test_official_backends_are_integrated_in_tree():
    official_runtime.resolve_metric_source_layout.cache_clear()
    layout = official_runtime.resolve_metric_source_layout()

    assert layout.source_root == official_runtime.OFFICIAL_BACKEND_ROOT
    assert layout.package_name == official_runtime.OFFICIAL_BACKEND_PACKAGE
    assert not (official_runtime.PROJECT_ROOT / "worldarena" / "_vendor").exists()
    assert not (
        official_runtime.PROJECT_ROOT / "worldarena" / "benchmark" / "official_source"
    ).exists()
    assert not hasattr(official_runtime, "VENDOR_METRICS_ROOT")
    if layout.droid_root is not None:
        assert layout.droid_root.is_relative_to(official_runtime.THIRDPARTY_ROOT)


def test_all_official_metric_adapters_are_inspectable():
    official_runtime.resolve_metric_source_layout.cache_clear()

    for metric_name in official_runtime.OFFICIAL_METRIC_NAMES:
        result = official_runtime.inspect_official_metric_import(metric_name)
        assert result["ready"] is True, result
        assert result["module"].startswith(f"{official_runtime.OFFICIAL_BACKEND_PACKAGE}.")
        assert result["source_file"].startswith(str(official_runtime.OFFICIAL_BACKEND_ROOT))


def test_object_detection_metric_is_removed_from_active_surface():
    config = load_config(official_runtime.PROJECT_ROOT / "config" / "benchmark.yaml")

    assert all(
        "object_detection" not in metric_names
        for metric_names in config.metrics_by_suite.values()
    )
    assert "object_detection" not in config.metric_backends
    assert "object_detection" not in config.normalization
    assert "object_detection" not in official_runtime.OFFICIAL_METRIC_NAMES
    assert "object_detection" not in METRIC_DIMENSIONS
    assert "object_control" not in METRIC_ALIASES
    assert not (official_runtime.OFFICIAL_BACKEND_ROOT / "object_detection.py").exists()

    with pytest.raises(KeyError, match="unsupported metric: object_detection"):
        build_metrics(["object_detection"], config)
