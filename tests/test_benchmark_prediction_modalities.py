from __future__ import annotations

from pathlib import Path

import numpy as np

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.manifest import build_manifest
from worldarena.benchmark.media import LoadedMedia, protocol_frame_details
from worldarena.benchmark.runner import run_benchmark
from worldarena.benchmark.schemas import BenchmarkSample, DiscoveredAsset, MetricOutput
from worldarena.benchmark.taxonomy import prediction_modality_for_task


def _make_config(tmp_path: Path) -> BenchmarkConfig:
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
        metrics_by_suite={"image_static": ["dummy_metric"]},
        metric_backends={},
        normalization={},
    )


def test_prediction_modality_for_image_suite_is_video() -> None:
    assert (
        prediction_modality_for_task(
            suite="image_static",
            source_modality="image",
            task_family="video_generation",
            artifact_type="image",
        )
        == "video"
    )


def test_protocol_frame_details_reports_native_indices() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    media = LoadedMedia(
        modality="video",
        path="/tmp/prediction.mp4",
        frames=[frame, frame, frame, frame],
        anchor_frames=[frame, frame],
        evaluation_protocol="segment_random",
        sampled_frame_indices=[0, 4, 9, 13],
        protocol_frame_indices=[1, 3],
        native_frame_count=16,
    )

    details = protocol_frame_details(media)

    assert details["evaluation_protocol"] == "segment_random"
    assert details["frame_indices"] == [1, 3]
    assert details["sampled_frame_indices"] == [0, 4, 9, 13]
    assert details["protocol_native_frame_indices"] == [4, 13]
    assert details["protocol_frame_count"] == 2
    assert details["loaded_frame_count"] == 4
    assert details["native_frame_count"] == 16


def test_build_manifest_marks_image_suite_as_video_artifact(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    asset = DiscoveredAsset(
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        path=str(tmp_path / "reference.png"),
        relative_path="static/photorealistic/indoor/dining_spaces/reference.png",
        category_path="static/photorealistic/indoor/dining_spaces",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        style="photorealistic",
        environment="indoor",
        scene="dining_spaces",
        fine_class="dining_spaces",
        has_annotation=False,
        has_instruction=False,
        has_pose=False,
        size_bytes=1,
    )

    manifest = build_manifest([asset], config)

    assert len(manifest) == 1
    sample = manifest[0]
    assert sample.modality == "image"
    assert sample.task_family == "video_generation"
    assert sample.artifact_type == "video"


def test_run_benchmark_uses_video_predictions_for_image_suite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _make_config(tmp_path)
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"png")

    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir(parents=True)
    prediction_path = predictions_root / "image_static_sample.mp4"
    prediction_path.write_bytes(b"mp4")

    sample = BenchmarkSample(
        sample_id="image-static-001",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path=str(reference_path),
        relative_path="static/photorealistic/indoor/dining_spaces/reference.png",
        reference_path=str(reference_path),
        prediction_stem="image_static_sample",
        conditioning_strategy="reference_image",
        prompt_current="A dining room.",
        prompt_target="Camera move left. A dining room remains still.",
        style="photorealistic",
        environment="indoor",
        scene="dining_spaces",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        task_family="video_generation",
        artifact_type="image",
        has_annotation=True,
        has_instruction=True,
    )

    class _DummyMetric:
        name = "dummy_metric"

        def compute(self, sample, prediction, reference) -> MetricOutput:
            del sample
            del prediction
            del reference
            return MetricOutput(raw=1.0, normalized=1.0, backend="dummy")

    load_calls: list[tuple[str, str]] = []

    def _fake_load_media(path: Path, modality: str, *args, **kwargs) -> LoadedMedia:
        del args
        del kwargs
        load_calls.append((str(path), modality))
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        return LoadedMedia(
            modality=modality,
            path=str(path),
            frames=[frame],
            anchor_frames=[frame],
            evaluation_protocol="segment_random",
            sampled_frame_indices=[0],
            protocol_frame_indices=[0],
            native_frame_count=1,
        )

    monkeypatch.setattr("worldarena.benchmark.runner.build_metrics", lambda metric_names, config: [_DummyMetric()])
    monkeypatch.setattr("worldarena.benchmark.runner.load_media", _fake_load_media)
    monkeypatch.setattr(
        "worldarena.benchmark.runner.aggregate_results",
        lambda sample_scores: {
            "per_metric_summary": {"image_static": {"dummy_metric": {}}},
            "per_dimension_summary": {},
            "diagnostic_metric_summary": {},
            "failure_summary": {},
        },
    )
    monkeypatch.setattr("worldarena.benchmark.runner.write_run_outputs", lambda **kwargs: None)

    result = run_benchmark(
        config=config,
        manifest=[sample],
        predictions_root=predictions_root,
        output_dir=tmp_path / "reports",
        model_name="flashworld-image-static",
    )

    assert result["run_manifest"]["missing_predictions"] == 0
    assert result["sample_scores"][0].prediction_path == str(prediction_path)
    assert load_calls == [
        (str(reference_path), "image"),
        (str(prediction_path), "video"),
    ]
