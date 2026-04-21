from __future__ import annotations

from pathlib import Path

import pytest

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.metrics.registry import build_metrics


def _make_config(tmp_path: Path, metric_backends: dict[str, str]) -> BenchmarkConfig:
    return BenchmarkConfig(
        config_path=tmp_path / "benchmark.yaml",
        paths=BenchmarkPaths(
            image_root=tmp_path / "image",
            video_root=tmp_path / "video",
            inventory_dir=tmp_path / "inventory",
            manifest_dir=tmp_path / "manifest",
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / ".cache",
        ),
        snapshot_name="test_snapshot",
        target_fps=24,
        anchor_positions=[0.0, 0.5, 1.0],
        motion_frame_count=8,
        split_ratios={"dev_public": 0.2, "dev_private": 0.1, "test_hidden": 0.7},
        metrics_by_suite={},
        metric_backends=metric_backends,
        normalization={},
    )


@pytest.mark.parametrize(
    ("metric_names", "metric_backends"),
    [
        (["scene_alignment"], {"scene_alignment": "reference_proxy"}),
        (["style_consistency"], {"style_consistency": "histogram"}),
        (["subjective_quality"], {"subjective_quality": "heuristic"}),
        (["motion_magnitude"], {"motion_magnitude": "pixel_diff"}),
        (["rollout_quality"], {"rollout_quality": "reference_proxy"}),
    ],
)
def test_build_metrics_rejects_disabled_fallback_backends(
    tmp_path: Path,
    metric_names: list[str],
    metric_backends: dict[str, str],
) -> None:
    config = _make_config(tmp_path, metric_backends)

    with pytest.raises(ValueError, match="fallback metrics are not allowed"):
        build_metrics(metric_names, config)
