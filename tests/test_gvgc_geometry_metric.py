from __future__ import annotations

from pathlib import Path

import numpy as np

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.consistency import GvgcGeometryMetric
from worldarena.benchmark.metrics.registry import build_metrics
from worldarena.benchmark.schemas import BenchmarkSample


def _frame(value: int = 0) -> np.ndarray:
    frame = np.full((16, 16, 3), value, dtype=np.uint8)
    frame[2:6, 3:7] = 255 - value
    frame[9:13, 10:14] = 128
    return frame


def _media(frames: list[np.ndarray], modality: str = "video") -> LoadedMedia:
    return LoadedMedia(
        modality=modality,
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
        prompt_target="Move forward without changing the layout.",
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
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        ),
        snapshot_name="test",
        target_fps=24,
        anchor_positions=[0.0, 1.0],
        motion_frame_count=4,
        metrics_by_suite={
            "image_static": ["epipolar_error", "epipolar_inlier_rate", "feature_match_coverage"],
        },
        metric_backends={
            "epipolar_error": "opencv_epipolar",
            "epipolar_inlier_rate": "opencv_epipolar",
            "feature_match_coverage": "opencv_epipolar",
        },
        metric_runtime={"gvgc_geometry": {"feature_detector": "orb"}},
        normalization={
            "epipolar_error": {"lower": 0.0, "upper": 5.0, "higher_is_better": False},
            "epipolar_inlier_rate": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "feature_match_coverage": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        },
    )


def test_gvgc_geometry_metric_is_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.consistency.compute_gvgc_geometry",
        lambda frames, runtime: {
            "backend": "opencv_epipolar",
            "raw": {
                "epipolar_error": 1.25,
                "epipolar_inlier_rate": 0.8,
                "feature_match_coverage": 0.4,
            },
            "details": {"frame_count": len(frames), "valid_pair_count": 1},
            "error": None,
        },
    )

    metric = GvgcGeometryMetric(
        metric_name="epipolar_error",
        backend="opencv_epipolar",
        normalization={"lower": 0.0, "upper": 5.0, "higher_is_better": False},
    )
    media = _media([_frame(), _frame(1)])
    result = metric.compute(_sample(), media, media)

    assert result.raw == 1.25
    assert result.normalized == 0.75
    assert result.eligibility_status == "diagnostic"
    assert result.backend == "opencv_epipolar"
    assert result.details["valid_pair_count"] == 1


def test_gvgc_geometry_metric_requires_video_prediction() -> None:
    metric = GvgcGeometryMetric(
        metric_name="epipolar_inlier_rate",
        backend="opencv_epipolar",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    media = _media([_frame()], modality="image")
    result = metric.compute(_sample(), media, media)

    assert result.raw is None
    assert result.normalized is None
    assert result.eligibility_status == "diagnostic"
    assert "requires a video prediction" in str(result.error)


def test_gvgc_geometry_metrics_are_registered(tmp_path: Path) -> None:
    config = _config(tmp_path)

    metrics = build_metrics(config.metrics_by_suite["image_static"], config)

    assert [metric.name for metric in metrics] == [
        "epipolar_error",
        "epipolar_inlier_rate",
        "feature_match_coverage",
    ]
    assert all(METRIC_DIMENSIONS[metric.name] == "consistency_3d_4d" for metric in metrics)
