from __future__ import annotations

import json
from pathlib import Path

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.realtime import (
    InteractionLatencyMetric,
    RealtimeThroughputMetric,
)
from worldarena.benchmark.schemas import BenchmarkSample


def _sample(prediction_stem: str = "sample") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="sample-001",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/test",
        path="/tmp/reference.png",
        relative_path="static/test/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem=prediction_stem,
        conditioning_strategy="reference_image",
        prompt_current="A scene.",
        prompt_target="Camera moves right. A scene.",
        style="synthetic",
        environment="test",
        scene="test_scene",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group",
        license_bucket="internal",
        fps=10.0,
    )


def _media(path: Path) -> LoadedMedia:
    return LoadedMedia(
        modality="video",
        path=str(path),
        frames=[],
        anchor_frames=[],
        native_frame_count=20,
    )


def test_realtime_throughput_metrics_read_generation_records(monkeypatch, tmp_path: Path) -> None:
    prediction_path = tmp_path / "sample.mp4"
    prediction_path.write_bytes(b"placeholder")
    (tmp_path / "generation_records.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-001",
                "suite": "image_static",
                "prediction_path": str(prediction_path),
                "generation_wall_time_seconds": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.realtime._video_probe",
        lambda path: {"frame_count": 20, "fps": 10.0, "duration_seconds": 2.0},
    )

    sample = _sample()
    prediction = _media(prediction_path)
    reference = _media(prediction_path)
    fps_metric = RealtimeThroughputMetric(
        metric_name="effective_generation_fps",
        backend="generation_metadata",
        normalization={},
        runtime={"target_fps": 10},
    )
    factor_metric = RealtimeThroughputMetric(
        metric_name="realtime_factor",
        backend="generation_metadata",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        runtime={"target_fps": 10},
    )

    fps_output = fps_metric.compute(sample=sample, prediction=prediction, reference=reference)
    factor_output = factor_metric.compute(sample=sample, prediction=prediction, reference=reference)

    assert fps_output.raw == 10.0
    assert fps_output.normalized == 1.0
    assert factor_output.raw == 1.0
    assert factor_output.normalized == 1.0


def test_interaction_latency_metrics_use_action_payload(monkeypatch, tmp_path: Path) -> None:
    prediction_path = tmp_path / "sample.mp4"
    prediction_path.write_bytes(b"placeholder")
    action_path = tmp_path / "sample_actions.json"
    action_path.write_text(
        json.dumps(
            {
                "actions": ["idle", "camera_r", "camera_l"],
                "total_frames": 20,
                "keyboard_condition": [[0, 0, 0, 0]] * 5
                + [[0, 0, 0, 0]] * 5
                + [[0, 0, 0, 0]] * 10,
                "mouse_condition": [[0.0, 0.0]] * 5
                + [[0.0, 0.1]] * 5
                + [[0.0, -0.1]] * 10,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "generation_records.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-001",
                "suite": "image_static",
                "prediction_path": str(prediction_path),
                "action_spec_path": str(action_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.realtime._video_probe",
        lambda path: {"frame_count": 20, "fps": 10.0, "duration_seconds": 2.0},
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.realtime._measure_action_latencies",
        lambda **kwargs: {
            "latencies_seconds": [0.05, 0.2],
            "missing_response_count": 0,
            "events": [{"latency_seconds": 0.05}, {"latency_seconds": 0.2}],
        },
    )

    sample = _sample()
    prediction = _media(prediction_path)
    reference = _media(prediction_path)
    latency_metric = InteractionLatencyMetric(
        metric_name="action_response_latency",
        backend="action_flow",
        normalization={"lower": 0.0, "upper": 0.1, "higher_is_better": False},
        runtime={"latency_threshold_seconds": 0.1},
    )
    pass_rate_metric = InteractionLatencyMetric(
        metric_name="latency_budget_pass_rate",
        backend="action_flow",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        runtime={"latency_threshold_seconds": 0.1},
    )

    latency_output = latency_metric.compute(sample=sample, prediction=prediction, reference=reference)
    pass_rate_output = pass_rate_metric.compute(sample=sample, prediction=prediction, reference=reference)

    assert latency_output.raw == 0.125
    assert latency_output.normalized == 0.0
    assert pass_rate_output.raw == 0.5
    assert pass_rate_output.normalized == 0.5
