from __future__ import annotations

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.registry import canonical_metric_name
from worldarena.benchmark.metrics.temporal_flickering import (
    TemporalFlickeringMetric,
    compute_temporal_flickering_score,
)
from worldarena.benchmark.schemas import BenchmarkSample


def _solid(rgb: tuple[int, int, int], size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


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


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="video_dynamic-flicker",
        suite="video_dynamic",
        split="test",
        asset_level="video_dynamic",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="dynamic/photorealistic/outdoor/test",
        path="/tmp/reference.mp4",
        relative_path="dynamic/photorealistic/outdoor/test/reference.mp4",
        reference_path="/tmp/reference.mp4",
        prediction_stem="video_dynamic-flicker",
        conditioning_strategy="reference_video",
        prompt_current="A scene.",
        prompt_target="Keep a stable scene.",
        style="photorealistic",
        environment="outdoor",
        scene="test",
        motion_category="camera",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-flicker",
        license_bucket="internal",
    )


def test_identical_frames_score_one() -> None:
    frame = _solid((128, 64, 32))
    result = compute_temporal_flickering_score([frame, frame, frame])
    assert result["raw"] == 1.0
    assert result["details"]["mean_pairwise_mae"] == 0.0


def test_opposite_frames_score_zero() -> None:
    black = _solid((0, 0, 0))
    white = _solid((255, 255, 255))
    result = compute_temporal_flickering_score([black, white])
    assert result["raw"] == 0.0
    assert result["details"]["mean_pairwise_mae"] == 255.0


def test_metric_registry_and_dimension() -> None:
    assert canonical_metric_name("Sflick") == "temporal_flickering"
    assert METRIC_DIMENSIONS["temporal_flickering"] == "quality"
    metric = TemporalFlickeringMetric(
        backend="pixel_mae",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    assert metric.name == "temporal_flickering"


def test_metric_compute_on_stable_video() -> None:
    frame = _solid((90, 120, 150))
    metric = TemporalFlickeringMetric(
        backend="pixel_mae",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    output = metric.compute(_sample(), _media([frame, frame, frame]), _media([frame]))
    assert output.error is None
    assert output.raw == 1.0
    assert output.normalized == 1.0
    assert output.backend == "pixel_mae"


def test_metric_requires_two_frames() -> None:
    metric = TemporalFlickeringMetric(
        backend="pixel_mae",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    output = metric.compute(_sample(), _media([_solid((1, 2, 3))]), _media([_solid((1, 2, 3))]))
    assert output.raw is None
    assert output.error == "not enough frames for temporal flickering metric"
