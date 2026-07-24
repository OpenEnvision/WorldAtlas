from __future__ import annotations

from pathlib import Path

import numpy as np

from worldarena.benchmark import depth_collision_backend
from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.depth_collision import DepthCollisionMetric
from worldarena.benchmark.metrics.registry import build_metrics
from worldarena.benchmark.schemas import BenchmarkSample


def _frame(value: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def _media(frames: list[np.ndarray]) -> LoadedMedia:
    return LoadedMedia(
        modality="video",
        path="/tmp/prediction.mp4",
        frames=frames,
        anchor_frames=frames,
        sampled_frame_indices=list(range(len(frames))),
        protocol_frame_indices=list(range(len(frames))),
        native_frame_count=len(frames),
    )


def _sample(suite: str = "image_static") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-001",
        suite=suite,
        split="test",
        asset_level=suite,
        modality="image" if suite.startswith("image") else "video",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/passageways",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/indoor/passageways/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem=f"{suite}-001",
        conditioning_strategy="reference_image",
        prompt_current="A hallway.",
        prompt_target="Move forward without colliding.",
        style="photorealistic",
        environment="indoor",
        scene="passageways",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
    )


def _config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        config_path=tmp_path / "benchmark.yaml",
        paths=BenchmarkPaths(
            image_root=tmp_path / "images",
            video_root=tmp_path / "videos",
            inventory_dir=tmp_path / "inventory",
            manifest_dir=tmp_path / "manifest",
            predictions_dir=tmp_path / "predictions",
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        ),
        snapshot_name="test",
        target_fps=24,
        anchor_positions=[0.0, 1.0],
        motion_frame_count=4,
        metrics_by_suite={
            "image_static": ["depth_collision"],
            "image_dynamic": ["depth_collision"],
            "video_static": ["depth_collision"],
            "video_dynamic": ["depth_collision"],
        },
        metric_backends={"depth_collision": "video_depth_anything"},
        metric_runtime={
            "depth_collision": {
                "depth_threshold": 1.0,
                "center_crop_ratio": 0.5,
            }
        },
        normalization={
            "depth_collision": {
                "lower": 0.0,
                "upper": 1.0,
                "higher_is_better": False,
            }
        },
    )


def test_depth_collision_backend_flags_center_crop_collision(monkeypatch) -> None:
    depths = np.full((3, 4, 4), 2.0, dtype=np.float32)
    depths[1, 1:3, 1:3] = 0.4

    monkeypatch.setattr(
        depth_collision_backend,
        "predict_metric_depths",
        lambda frames, *, runtime: (
            depths,
            {"backend": "video_depth_anything", "encoder": "vits"},
        ),
    )

    result = depth_collision_backend.compute_depth_collision(
        [_frame(), _frame(), _frame()],
        runtime={"depth_threshold": 1.0, "center_crop_ratio": 0.5},
    )

    assert result["raw"] == 1.0
    assert result["details"]["collision"] is True
    assert result["details"]["first_collision_frame"] == 1
    assert result["details"]["center_crop_mean_depths"] == [2.0, 0.4, 2.0]


def test_depth_collision_metric_is_diagnostic_for_image_suite(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.depth_collision.compute_depth_collision",
        lambda frames, *, runtime: {
            "raw": 0.0,
            "backend": "video_depth_anything",
            "details": {"frame_count": len(frames), "collision": False},
            "error": None,
        },
    )

    metric = DepthCollisionMetric(
        backend="video_depth_anything",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": False},
    )
    media = _media([_frame(), _frame()])
    result = metric.compute(_sample("image_static"), media, media)

    assert result.raw == 0.0
    assert result.normalized == 1.0
    assert result.eligibility_status == "diagnostic"
    assert result.details["frame_count"] == 2


def test_depth_collision_registered_for_image_and_video_suites(tmp_path: Path) -> None:
    config = _config(tmp_path)

    for suite in ("image_static", "image_dynamic", "video_static", "video_dynamic"):
        metrics = build_metrics(config.metrics_by_suite[suite], config)
        assert [metric.name for metric in metrics] == ["depth_collision"]

    assert METRIC_DIMENSIONS["depth_collision"] == "consistency_3d_4d"
