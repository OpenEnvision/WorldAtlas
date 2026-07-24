from __future__ import annotations

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.registry import canonical_metric_name
from worldarena.benchmark.metrics.segment_continuity import (
    SegmentContinuityMetric,
    compute_segment_continuity_score,
    detect_scene_cuts,
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
        sample_id="video_dynamic-segment",
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
        prediction_stem="video_dynamic-segment",
        conditioning_strategy="reference_video",
        prompt_current="A scene.",
        prompt_target="Keep a stable scene.",
        style="photorealistic",
        environment="outdoor",
        scene="test",
        motion_category="camera",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-segment",
        license_bucket="internal",
    )


def test_detect_scene_cuts_filters_short_scenes() -> None:
    predictions = np.array([0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    result = detect_scene_cuts(predictions, threshold=0.5, min_scene_len=10)
    assert result["candidate_cut_count"] == 1
    assert result["cut_count"] == 0


def test_detect_scene_cuts_keeps_valid_cut() -> None:
    predictions = np.zeros(25, dtype=np.float32)
    predictions[12] = 0.9
    result = detect_scene_cuts(predictions, threshold=0.5, min_scene_len=10)
    assert result["cut_count"] == 1
    assert result["cut_frame_indices"] == [12]


def test_compute_segment_continuity_score_cut_free(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.segment_continuity.predict_transnet_single_frame_probabilities",
        lambda frames, runtime=None: {
            "probabilities": np.zeros(len(frames), dtype=np.float32),
            "repo_root": "/tmp/transnet",
            "weights_dir": "/tmp/weights",
            "frame_count": len(frames),
        },
    )
    result = compute_segment_continuity_score([_solid((1, 2, 3)), _solid((4, 5, 6))])
    assert result["raw"] == 1.0
    assert result["details"]["cut_free"] is True


def test_compute_segment_continuity_score_with_cut(monkeypatch) -> None:
    predictions = np.zeros(25, dtype=np.float32)
    predictions[12] = 0.9
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.segment_continuity.predict_transnet_single_frame_probabilities",
        lambda frames, runtime=None: {
            "probabilities": predictions[: len(frames)],
            "repo_root": "/tmp/transnet",
            "weights_dir": "/tmp/weights",
            "frame_count": len(frames),
        },
    )
    frames = [_solid((index, 0, 0)) for index in range(25)]
    result = compute_segment_continuity_score(frames)
    assert result["raw"] == 0.0
    assert result["details"]["cut_free"] is False


def test_compute_segment_continuity_adapts_min_scene_len_for_short_video(monkeypatch) -> None:
    predictions = np.zeros(16, dtype=np.float32)
    predictions[8] = 0.9
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.segment_continuity.predict_transnet_single_frame_probabilities",
        lambda frames, runtime=None: {
            "probabilities": predictions[: len(frames)],
            "repo_root": "/tmp/transnet",
            "weights_dir": "/tmp/weights",
            "frame_count": len(frames),
        },
    )
    frames = [_solid((index, 0, 0)) for index in range(16)]
    result = compute_segment_continuity_score(frames, min_scene_len=10)
    assert result["raw"] == 0.0
    assert result["details"]["configured_min_scene_len"] == 10
    assert result["details"]["min_scene_len"] == 8
    assert result["details"]["adaptive_min_scene_len"] is True


def test_metric_registry_and_dimension() -> None:
    assert canonical_metric_name("S_seg") == "segment_continuity"
    assert METRIC_DIMENSIONS["segment_continuity"] == "quality"
    metric = SegmentContinuityMetric(
        backend="transnet_v2",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    assert metric.name == "segment_continuity"


def test_metric_compute(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.segment_continuity.predict_transnet_single_frame_probabilities",
        lambda frames, runtime=None: {
            "probabilities": np.zeros(len(frames), dtype=np.float32),
            "repo_root": "/tmp/transnet",
            "weights_dir": "/tmp/weights",
            "frame_count": len(frames),
        },
    )
    metric = SegmentContinuityMetric(
        backend="transnet_v2",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    output = metric.compute(_sample(), _media([_solid((1, 2, 3)), _solid((4, 5, 6))]), _media([_solid((1, 2, 3))]))
    assert output.error is None
    assert output.raw == 1.0
    assert output.backend == "transnet_v2"
