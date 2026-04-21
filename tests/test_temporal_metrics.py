from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.registry import build_metrics
from worldarena.benchmark.metrics.temporal import TemporalCalibrationMetric, _PULSE_RESULT_CACHE
from worldarena.benchmark.pulse_runtime import inspect_pulse_of_motion_layout
from worldarena.benchmark.schemas import BenchmarkSample


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (32, 24))
    assert writer.isOpened()
    for color in ((10, 20, 30), (40, 50, 60), (70, 80, 90)):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :] = color
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _sample(video_path: Path) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="video_dynamic_photorealistic_fluid_0001",
        suite="video_dynamic",
        split="dev_public",
        asset_level="video_dynamic",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="dynamic/photorealistic/videos/fluid",
        path=str(video_path),
        relative_path="dynamic/photorealistic/videos/fluid/0001.avi",
        reference_path=str(video_path),
        prediction_stem="video_dynamic_photorealistic_fluid_0001",
        conditioning_strategy="first_frame",
        prompt_current="A photorealistic scene with fluid motion.",
        prompt_target="Generate a photorealistic video with fluid motion.",
        style="photorealistic",
        environment=None,
        scene=None,
        motion_category="fluid",
        source_name="worldarena_formal_video",
        source_type="curated_video",
        source_group_id="dynamic/photorealistic/videos/fluid::0001",
        license_bucket="C",
        fps=24.0,
    )


def _loaded_media(video_path: Path) -> LoadedMedia:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    return LoadedMedia(
        modality="video",
        path=str(video_path),
        frames=[frame, frame, frame],
        anchor_frames=[frame],
        sampled_frame_indices=[0, 1, 2],
        protocol_frame_indices=[0],
    )


def test_temporal_calibration_metric_uses_pulse_runtime(monkeypatch, tmp_path: Path) -> None:
    _PULSE_RESULT_CACHE.clear()
    video_path = tmp_path / "prediction.avi"
    _write_video(video_path)

    def fake_compute(video_path_arg, *, runtime=None, fallback_meta_fps=None):
        assert video_path_arg == str(video_path)
        assert fallback_meta_fps == 24.0
        return {
            "backend": "pulse_of_motion",
            "metrics": {
                "physical_fps_error": 2.5,
                "physical_fps_pct_error": 0.1,
                "physical_fps_intra_video_cv": 0.2,
            },
            "details": {
                "avg_phyfps": 21.5,
                "meta_fps": 24.0,
                "segment_phyfps": [20.0, 21.0, 23.5],
                "segment_count": 3,
            },
        }

    monkeypatch.setattr("worldarena.benchmark.metrics.temporal.compute_pulse_metrics", fake_compute)

    metric = TemporalCalibrationMetric(
        metric_name="physical_fps_error",
        backend="pulse_of_motion",
        normalization={"lower": 0.0, "upper": 30.0, "higher_is_better": False},
    )
    output = metric.compute(_sample(video_path), _loaded_media(video_path), _loaded_media(video_path))

    assert output.backend == "pulse_of_motion"
    assert output.raw == 2.5
    assert output.normalized == 0.9166666666666666
    assert output.details["avg_phyfps"] == 21.5
    assert output.details["segment_count"] == 3


def test_metric_registry_builds_temporal_calibration_metrics(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        config_path=tmp_path / "benchmark.yaml",
        paths=BenchmarkPaths(
            image_root=tmp_path / "image",
            video_root=tmp_path / "video",
            inventory_dir=tmp_path / "inventory",
            manifest_dir=tmp_path / "manifest",
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        ),
        snapshot_name="test",
        target_fps=24,
        anchor_positions=[0.0, 1.0],
        motion_frame_count=4,
        split_ratios={"dev_public": 1.0, "dev_private": 0.0, "test_hidden": 0.0},
        metrics_by_suite={},
        metric_runtime={
            "temporal_calibration": {
                "device": "cuda:2",
                "clip_length": 30,
                "stride": 4,
            }
        },
        metric_backends={
            "physical_fps_error": "pulse_of_motion",
            "physical_fps_pct_error": "pulse_of_motion",
            "physical_fps_intra_video_cv": "pulse_of_motion",
        },
        normalization={
            "physical_fps_error": {"lower": 0.0, "upper": 30.0, "higher_is_better": False},
            "physical_fps_pct_error": {"lower": 0.0, "upper": 1.0, "higher_is_better": False},
            "physical_fps_intra_video_cv": {"lower": 0.0, "upper": 1.0, "higher_is_better": False},
        },
    )

    metrics = build_metrics(
        [
            "physical_fps_error",
            "physical_fps_pct_error",
            "physical_fps_intra_video_cv",
        ],
        config,
    )

    assert [metric.name for metric in metrics] == [
        "physical_fps_error",
        "physical_fps_pct_error",
        "physical_fps_intra_video_cv",
    ]
    assert all(isinstance(metric, TemporalCalibrationMetric) for metric in metrics)
    assert all(metric.runtime["device"] == "cuda:2" for metric in metrics)
    assert all(metric.runtime["clip_length"] == 30 for metric in metrics)


def test_inspect_pulse_layout_allows_auto_download_without_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "Pulse-of-Motion"
    predict_path = root / "inference" / "predict.py"
    config_path = root / "inference" / "configs" / "config_fps.yaml"
    predict_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    predict_path.write_text("print('ok')\n", encoding="utf-8")
    config_path.write_text("model: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        "worldarena.benchmark.pulse_runtime.resolve_pulse_of_motion_root",
        lambda: root,
    )
    monkeypatch.setattr(
        "worldarena.benchmark.pulse_runtime._module_available",
        lambda module_name: module_name == "huggingface_hub",
    )

    result = inspect_pulse_of_motion_layout()

    assert result["source_ready"] is True
    assert result["checkpoint_present"] is False
    assert result["download_supported"] is True
    assert result["ready"] is True
    assert result["error"] is None
