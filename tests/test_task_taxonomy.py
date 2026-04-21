from __future__ import annotations

from pathlib import Path

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.manifest import build_manifest
from worldarena.benchmark.schemas import DiscoveredAsset


def _config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        config_path=tmp_path / "benchmark.yaml",
        paths=BenchmarkPaths(
            image_root=tmp_path / "image",
            video_root=tmp_path / "video",
            inventory_dir=tmp_path / "inventory",
            manifest_dir=tmp_path / "manifest",
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        ),
        snapshot_name="test",
        target_fps=24,
        anchor_positions=[0.0, 1.0],
        motion_frame_count=4,
        split_ratios={"dev_public": 1.0, "dev_private": 0.0, "test_hidden": 0.0},
        metrics_by_suite={},
        metric_backends={},
        normalization={},
    )


def test_manifest_derives_task_and_control_metadata(tmp_path: Path) -> None:
    asset = DiscoveredAsset(
        asset_level="video_dynamic",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        path=str(tmp_path / "video.mp4"),
        relative_path="photorealistic/video.mp4",
        category_path="dynamic/photorealistic/video",
        source_name="unit_test",
        source_type="curated_video",
        source_group_id="group-1",
        license_bucket="cc-by",
        style="photorealistic",
        motion_category="fluid",
        has_pose=True,
    )

    sample = build_manifest([asset], _config(tmp_path))[0]

    assert sample.benchmark_family == "worldarena"
    assert sample.task_family == "video_generation"
    assert sample.artifact_type == "video"
    assert sample.control_signals == ["text", "image", "camera_pose"]
