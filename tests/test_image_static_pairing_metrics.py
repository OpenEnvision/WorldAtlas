from __future__ import annotations

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.style import StyleConsistencyMetric
from worldarena.benchmark.schemas import BenchmarkSample


def _frame(value: int, size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _media(frames: list[np.ndarray], *, modality: str = "video") -> LoadedMedia:
    return LoadedMedia(
        modality=modality,
        path="/tmp/prediction.mp4" if modality == "video" else "/tmp/reference.png",
        frames=frames,
        anchor_frames=frames,
        sampled_frame_indices=list(range(len(frames))),
        protocol_frame_indices=list(range(len(frames))),
        native_frame_count=len(frames),
    )


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="image-static-pairing",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/test",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/indoor/test/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem="image-static-pairing",
        conditioning_strategy="reference_image",
        prompt_current="A room.",
        prompt_target="Move the camera while preserving the room.",
        style="photorealistic",
        environment="indoor",
        scene="test",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-pairing",
        license_bucket="internal",
        task_family="video_generation",
        artifact_type="image",
    )


def test_style_consistency_uses_prediction_first_anchor(monkeypatch) -> None:
    calls: list[tuple[float, list[float]]] = []

    def _fake_gram_distances(
        reference_frame: np.ndarray,
        rendered_frames: list[np.ndarray],
    ):
        reference_mean = float(np.mean(reference_frame))
        rendered_means = [float(np.mean(frame)) for frame in rendered_frames]
        calls.append((reference_mean, rendered_means))
        distances = [(rendered_mean - reference_mean) / 10000.0 for rendered_mean in rendered_means]
        return distances, [[distance] for distance in distances]

    monkeypatch.setattr(
        "worldarena.benchmark.metrics.style._gram_distances",
        _fake_gram_distances,
    )

    reference = _media([_frame(250)], modality="image")
    prediction = _media([_frame(0), _frame(10), _frame(20), _frame(30)], modality="video")
    result = StyleConsistencyMetric(
        backend="gram_matrix",
        normalization={"lower": 0.0, "upper": 0.007, "higher_is_better": False},
    ).compute(_sample(), prediction, reference)

    assert result.backend == "gram_matrix"
    assert result.details["style_reference_policy"] == "prediction_first_anchor"
    assert result.details["pairing_policy"] == "first_anchor_to_later_anchors"
    assert result.details["prediction_frame_count"] == 4
    assert result.details["reference_frame_count"] == 1
    assert result.details["ground_truth_reference_frame_count"] == 1
    assert result.details["paired_frame_count"] == 3
    assert result.details["worldscore_metric"] == "gram_matrix"
    assert len(result.details["frame_scores"]) == 3
    assert calls == [(0.0, [10.0, 20.0, 30.0])]
    assert np.isclose(result.raw, np.mean([0.001, 0.002, 0.003]))
    assert np.isclose(result.normalized, np.clip((0.007 - result.raw) / 0.007, 0.0, 1.0))
