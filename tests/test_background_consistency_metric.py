from __future__ import annotations

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.background_consistency import (
    BackgroundConsistencyMetric,
    compute_background_consistency_score,
)
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.registry import canonical_metric_name
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
        sample_id="video_dynamic-background",
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
        prediction_stem="video_dynamic-background",
        conditioning_strategy="reference_video",
        prompt_current="A scene.",
        prompt_target="Keep a stable scene.",
        style="photorealistic",
        environment="outdoor",
        scene="test",
        motion_category="camera",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-background",
        license_bucket="internal",
    )


def test_identical_features_score_one(monkeypatch) -> None:
    features = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        "worldarena.benchmark.metrics.background_consistency.clip_image_features",
        lambda frames, model_name: (features[: len(frames)], "test_loader"),
    )

    result = compute_background_consistency_score([_solid((1, 2, 3)), _solid((1, 2, 3)), _solid((1, 2, 3))])
    assert result["raw"] == 1.0
    assert result["details"]["mean_pairwise_cosine"] == 1.0


def test_orthogonal_features_score_zero(monkeypatch) -> None:
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(
        "worldarena.benchmark.metrics.background_consistency.clip_image_features",
        lambda frames, model_name: (features, "test_loader"),
    )

    result = compute_background_consistency_score([_solid((1, 2, 3)), _solid((4, 5, 6))])
    assert result["raw"] == 0.0


def test_metric_registry_and_dimension() -> None:
    assert canonical_metric_name("S_bg") == "background_consistency"
    assert METRIC_DIMENSIONS["background_consistency"] == "quality"
    metric = BackgroundConsistencyMetric(
        backend="clip_vit_b32",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    assert metric.name == "background_consistency"


def test_metric_compute_on_stable_video(monkeypatch) -> None:
    features = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.background_consistency.clip_image_features",
        lambda frames, model_name: (features, "test_loader"),
    )

    metric = BackgroundConsistencyMetric(
        backend="clip_vit_b32",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    output = metric.compute(_sample(), _media([_solid((1, 2, 3)), _solid((1, 2, 3))]), _media([_solid((1, 2, 3))]))
    assert output.error is None
    assert output.raw == 1.0
    assert output.backend == "clip_vit_b32"


def test_metric_requires_two_frames() -> None:
    metric = BackgroundConsistencyMetric(
        backend="clip_vit_b32",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    output = metric.compute(_sample(), _media([_solid((1, 2, 3))]), _media([_solid((1, 2, 3))]))
    assert output.raw is None
    assert output.error == "not enough frames for background consistency metric"
