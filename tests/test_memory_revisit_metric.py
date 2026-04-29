from __future__ import annotations

import numpy as np
import pytest

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.memory import MemoryRevisitConsistencyMetric
from worldarena.benchmark.schemas import BenchmarkSample


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="memory-001",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/passageways",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/indoor/passageways/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem="memory-001",
        conditioning_strategy="reference_image",
        prompt_current="A hallway.",
        prompt_target="Rotate in place 360 degrees and return to the starting view.",
        style="photorealistic",
        environment="indoor",
        scene="passageways",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
    )


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


def test_memory_revisit_consistency_rewards_closed_loop_with_progress(monkeypatch) -> None:
    start = np.zeros((16, 16, 3), dtype=np.uint8)
    mid = np.full((16, 16, 3), 180, dtype=np.uint8)
    prediction = _media([start, mid, 255 - mid, mid, start.copy()])
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.memory.dinov2_image_features",
        lambda frames: features,
    )

    metric = MemoryRevisitConsistencyMetric(backend="dinov2", normalization={})
    result = metric.compute(_sample(), prediction, prediction)

    assert result.raw == 1.0
    assert result.normalized == 1.0
    assert result.details["closure_score"] == 1.0
    assert result.details["progress_score"] == 1.0


def test_memory_revisit_consistency_penalizes_no_rotation(monkeypatch) -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    prediction = _media([frame.copy() for _ in range(5)])
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.memory.dinov2_image_features",
        lambda frames: np.asarray([[1.0, 0.0] for _ in frames], dtype=np.float32),
    )

    metric = MemoryRevisitConsistencyMetric(backend="dinov2", normalization={})
    result = metric.compute(_sample(), prediction, prediction)

    assert result.raw == pytest.approx(0.65)
    assert result.details["closure_score"] == 1.0
    assert result.details["progress_score"] == 0.0
