from __future__ import annotations

import numpy as np
import pytest

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.memory import (
    EntityReappearanceConsistencyMetric,
    MemoryRevisitConsistencyMetric,
    MemorySymmetryMetric,
    RevisitReturnGateMetric,
    compute_entity_reappearance_consistency,
    compute_memory_symmetry,
    compute_revisit_return_gate,
)
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


def _loop_poses() -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=7, axis=0)
    poses[:, 0, 3] = np.asarray([0.0, 0.4, 0.8, 1.0, 0.6, 0.15, 0.01])
    angles = np.deg2rad([0.0, 20.0, 50.0, 90.0, 45.0, 10.0, 1.0])
    poses[:, 0, 0] = np.cos(angles)
    poses[:, 0, 2] = np.sin(angles)
    poses[:, 2, 0] = -np.sin(angles)
    poses[:, 2, 2] = np.cos(angles)
    return poses


def test_revisit_return_gate_requires_departure_and_return() -> None:
    result = compute_revisit_return_gate(_loop_poses())
    static = compute_revisit_return_gate(
        np.repeat(np.eye(4, dtype=np.float32)[None], repeats=7, axis=0)
    )

    assert result["passed"] is True
    assert result["revisit_index"] == 6
    assert result["max_departure_translation"] == pytest.approx(1.0)
    assert static["passed"] is False
    assert static["departure_passed"] is False


def test_entity_reappearance_is_hard_gated_by_actual_return() -> None:
    passed = compute_entity_reappearance_consistency(
        np.asarray([1.0, 0.0]),
        np.asarray([0.8, 0.2]),
        anchor_area=0.10,
        revisit_area=0.09,
        return_gate_passed=True,
    )
    failed = compute_entity_reappearance_consistency(
        np.asarray([1.0, 0.0]),
        np.asarray([0.8, 0.2]),
        anchor_area=0.10,
        revisit_area=0.09,
        return_gate_passed=False,
    )

    assert passed["conditional_consistency"] > 0.8
    assert failed["conditional_consistency"] == pytest.approx(
        passed["conditional_consistency"]
    )
    assert failed["entity_reappearance_consistency"] == 0.0


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


def test_memory_symmetry_rewards_symmetric_frame_pairs() -> None:
    start = np.zeros((16, 16, 3), dtype=np.uint8)
    middle = np.full((16, 16, 3), 128, dtype=np.uint8)

    symmetric_score = compute_memory_symmetry([start, middle, middle.copy(), start.copy()])["raw"]
    asymmetric_score = compute_memory_symmetry([start, middle, middle.copy(), np.full_like(start, 255)])["raw"]

    assert symmetric_score == pytest.approx(1.0)
    assert asymmetric_score < symmetric_score


def test_memory_symmetry_metric_outputs_diagnostic() -> None:
    start = np.zeros((16, 16, 3), dtype=np.uint8)
    middle = np.full((16, 16, 3), 128, dtype=np.uint8)
    prediction = _media([start, middle, middle.copy(), start.copy()])
    metric = MemorySymmetryMetric(
        backend="mse_symmetry",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )

    result = metric.compute(_sample(), prediction, prediction)

    assert result.raw == pytest.approx(1.0)
    assert result.normalized == pytest.approx(1.0)
    assert result.eligibility_status == "diagnostic"


def test_return_gate_and_entity_metrics_read_sidecars(tmp_path) -> None:
    prediction_path = tmp_path / "prediction.mp4"
    prediction_path.write_bytes(b"placeholder")
    np.save(tmp_path / "prediction_poses_c2w.npy", _loop_poses())
    np.savez(
        tmp_path / "prediction_entity_revisit.npz",
        anchor_descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
        revisit_descriptor=np.asarray([0.9, 0.1], dtype=np.float32),
        anchor_mask_area=np.asarray([0.10], dtype=np.float32),
        revisit_mask_area=np.asarray([0.09], dtype=np.float32),
    )
    prediction = _media([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)])
    prediction.path = str(prediction_path)

    gate_metric = RevisitReturnGateMetric(
        backend="pose_gate",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    entity_metric = EntityReappearanceConsistencyMetric(
        backend="masked_dinov2",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )

    gate_result = gate_metric.compute(_sample(), prediction, prediction)
    entity_result = entity_metric.compute(_sample(), prediction, prediction)

    assert gate_result.raw == 1.0
    assert entity_result.raw is not None and entity_result.raw > 0.8
    assert entity_result.details["metric_uses_vlm"] is False
    assert entity_result.details["metric_uses_llm_as_judge"] is False
