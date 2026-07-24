from worldarena.benchmark import official_runtime


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
