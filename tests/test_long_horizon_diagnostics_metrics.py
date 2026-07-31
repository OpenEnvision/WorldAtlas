from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.long_horizon_diagnostics import (
    KeystrokeActionMetric,
    SceneMemoryF1Metric,
    VisualDriftMetric,
    _VISUAL_CACHE,
    compute_visual_drift_metrics,
    rollout_visual_score_from_components,
    scene_memory_f1,
    segment_drift,
    strict_partial_accuracy,
    trimmed_icp_align,
)
from worldarena.benchmark.schemas import BenchmarkSample


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="long-horizon-001",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/test_scene",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/outdoor/test_scene/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem="prediction",
        conditioning_strategy="reference_image",
        prompt_current="A test scene.",
        prompt_target="Move forward.",
        style="photorealistic",
        environment="outdoor",
        scene="test_scene",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
    )


def _media(path: Path, *, frame_count: int = 8) -> LoadedMedia:
    return LoadedMedia(
        modality="video",
        path=str(path),
        frames=[np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(min(frame_count, 2))],
        anchor_frames=[],
        sampled_frame_indices=list(range(min(frame_count, 2))),
        protocol_frame_indices=list(range(min(frame_count, 2))),
        native_frame_count=frame_count,
    )


def test_segment_drift_uses_best_and_worst_timeline_segments() -> None:
    payload = segment_drift([1.0, 1.0, 0.5, 0.5], segment_count=2)

    assert payload["drift"] == pytest.approx(0.5)
    assert payload["best_segment_mean"] == pytest.approx(1.0)
    assert payload["worst_segment_mean"] == pytest.approx(0.5)


def test_strict_partial_accuracy_scores_compound_actions() -> None:
    predicted = [frozenset({"forward"}), frozenset({"camera_l"}), frozenset()]
    ground_truth = [
        frozenset({"forward", "camera_r"}),
        frozenset({"camera_r"}),
        frozenset(),
    ]

    result = strict_partial_accuracy(predicted, ground_truth)

    assert result["strict_accuracy"] == pytest.approx(1 / 3)
    assert result["partial_accuracy"] == pytest.approx(2 / 3)


def test_keystroke_action_metric_uses_action_and_pose_sidecars(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.mp4"
    prediction_path.write_bytes(b"placeholder")
    action_payload = {
        "keyboard_condition": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(8)],
        "mouse_condition": [[0.0, 0.0] for _ in range(8)],
        "total_frames": 8,
    }
    (tmp_path / "prediction_actions.json").write_text(json.dumps(action_payload), encoding="utf-8")
    sidecar = tmp_path / "annotations_vipe" / "prediction_vipe_ann"
    sidecar.mkdir(parents=True)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=9, axis=0)
    poses[:, 2, 3] = np.linspace(0.0, 0.8, num=9, dtype=np.float32)
    np.save(sidecar / "poses_c2w.npy", poses)

    metric = KeystrokeActionMetric(
        metric_name="keystroke_action_score",
        backend="pose_action",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        runtime={"latent_stride": 4},
    )

    result = metric.compute(_sample(), _media(prediction_path, frame_count=8), _media(prediction_path))

    assert result.raw == pytest.approx(1.0)
    assert result.normalized == pytest.approx(1.0)
    assert result.details["strict_accuracy"] == pytest.approx(1.0)
    assert result.details["partial_accuracy"] == pytest.approx(1.0)
    assert result.details["traj_score"] == pytest.approx(1.0)


def test_visual_drift_metrics_compute_drift_components(monkeypatch, tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.mp4"
    prediction_path.write_bytes(b"placeholder")
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]
    _VISUAL_CACHE.clear()
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.long_horizon_diagnostics.read_video_frames",
        lambda path: frames,
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.long_horizon_diagnostics.compute_quality_component_scores",
        lambda frames, *, component_name: {
            "backend": "laion_aes",
            "raw": 0.75,
            "frame_scores": [1.0, 1.0, 0.5, 0.5],
        },
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.long_horizon_diagnostics.compute_musiq_scores",
        lambda frames, *, model_path, preprocess_mode: {
            "backend": "musiq",
            "raw": 0.75,
            "frame_scores": [1.0, 0.5, 1.0, 0.5],
        },
    )

    result = compute_visual_drift_metrics(
        prediction_path,
        runtime={"segment_count": 2, "max_frames": 4},
    )

    assert result["aesthetic_drift"]["drift"] == pytest.approx(0.5)
    assert result["imaging_drift"]["drift"] == pytest.approx(0.0)
    assert result["visual_score"] == pytest.approx(0.75)


def test_rollout_visual_score_matches_worldodyssey_formula() -> None:
    score = rollout_visual_score_from_components(0.75, 0.75, 0.5, 0.0)
    assert score == pytest.approx(0.75)


def test_scene_memory_f1_balances_retention_and_hallucination() -> None:
    result = scene_memory_f1(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.01], [2.0, 0.0, 0.0]], dtype=np.float32),
        distance_threshold=0.1,
    )

    assert result["retention"] == pytest.approx(0.5)
    assert result["hallucination"] == pytest.approx(0.5)
    assert result["scene_memory_f1"] == pytest.approx(0.5)


def test_trimmed_icp_accepts_small_rigid_drift_and_rejects_large_correction() -> None:
    target = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    small = target + np.asarray([0.03, 0.0, 0.0], dtype=np.float32)
    large = target + np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    aligned, accepted = trimmed_icp_align(small, target, max_translation_ratio=0.10)
    unchanged, rejected = trimmed_icp_align(large, target, max_translation_ratio=0.10)

    assert accepted["alignment_accepted"] is True
    assert np.mean(np.linalg.norm(aligned - target, axis=1)) < 1e-5
    assert rejected["alignment_accepted"] is False
    assert np.allclose(unchanged, large)


def test_scene_memory_metric_reads_explicit_npz_sidecar(tmp_path: Path) -> None:
    prediction_path = tmp_path / "prediction.mp4"
    prediction_path.write_bytes(b"placeholder")
    np.savez(
        tmp_path / "prediction_scene_memory.npz",
        pobs=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        prev=np.asarray([[0.0, 0.0, 0.01], [2.0, 0.0, 0.0]], dtype=np.float32),
    )
    metric = SceneMemoryF1Metric(
        backend="point_cloud_f1",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        runtime={"distance_threshold": 0.1, "centroid_align": False},
    )

    result = metric.compute(_sample(), _media(prediction_path, frame_count=2), _media(prediction_path))

    assert result.raw == pytest.approx(0.5)
    assert result.normalized == pytest.approx(0.5)
    assert result.details["retention"] == pytest.approx(0.5)
    assert result.details["hallucination"] == pytest.approx(0.5)
