from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.jedi_backend import compute_jedi_distance
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.registry import build_metrics, split_suite_level_metrics
from worldarena.benchmark.runner import run_benchmark
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.video_io import write_video_frames


def _write_video(path: Path, *, base: int, frame_count: int = 4) -> Path:
    frames = [
        np.full((12, 12, 3), base + index, dtype=np.uint8)
        for index in range(frame_count)
    ]
    return write_video_frames(path, frames, fps=4.0)


def _sample(reference_path: Path) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="video-dynamic-jedi-001",
        suite="video_dynamic",
        split="test",
        asset_level="video_dynamic",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="dynamic/test",
        path=str(reference_path),
        relative_path="dynamic/test/reference.mp4",
        reference_path=str(reference_path),
        prediction_stem="prediction_001",
        conditioning_strategy="reference_video",
        prompt_current="A moving scene.",
        prompt_target="Preserve the motion.",
        style="synthetic",
        environment="indoor",
        scene="test",
        motion_category="camera",
        source_name="unit",
        source_type="synthetic",
        source_group_id="group-jedi",
        license_bucket="internal",
    )


def _config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        config_path=tmp_path / "benchmark.yaml",
        paths=BenchmarkPaths(
            image_root=tmp_path / "images",
            video_root=tmp_path / "videos",
            inventory_dir=tmp_path / "inventory",
            manifest_dir=tmp_path / "manifest",
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        ),
        snapshot_name="test",
        target_fps=24,
        anchor_positions=[0.0, 1.0],
        motion_frame_count=4,
        metrics_by_suite={"video_dynamic": ["jedi"]},
        metric_backends={"jedi": "jedi"},
        normalization={"jedi": {"lower": 0.0, "upper": 100.0, "higher_is_better": False}},
        metric_runtime={"jedi": {"clip_frame_count": 4, "analysis_size": 8, "min_clip_count": 2}},
    )


def test_jedi_backend_exports_clip_sets_and_normalizes(monkeypatch, tmp_path: Path) -> None:
    pairs: list[tuple[Path, Path, str]] = []
    for index in range(2):
        prediction = _write_video(tmp_path / f"pred_{index}.mp4", base=20 + index)
        reference = _write_video(tmp_path / f"ref_{index}.mp4", base=40 + index)
        pairs.append((prediction, reference, f"sample-{index}"))

    class FakeJEDiMetric:
        def __init__(self, feature_path: str, model_dir: str, config_path: str | None) -> None:
            self.feature_path = feature_path
            self.model_dir = model_dir
            self.config_path = config_path

        def load_features(self, train_loader: Any, test_loader: Any, num_samples: int) -> None:
            self.train_batches = list(train_loader)
            self.test_batches = list(test_loader)
            self.num_samples = num_samples

        def compute_metric(self) -> float:
            train_batch, _ = self.train_batches[0]
            test_batch, _ = self.test_batches[0]
            assert tuple(train_batch.shape) == (2, 4, 3, 8, 8)
            assert tuple(test_batch.shape) == (2, 4, 3, 8, 8)
            assert self.num_samples == 2
            return 25.0

    monkeypatch.setattr(
        "worldarena.benchmark.jedi_backend._load_jedi_metric_class",
        lambda runtime: (FakeJEDiMetric, tmp_path),
    )

    result = compute_jedi_distance(
        prediction_reference_pairs=pairs,
        normalization={"lower": 0.0, "upper": 100.0, "higher_is_better": False},
        runtime={"clip_frame_count": 4, "analysis_size": 8, "min_clip_count": 2, "batch_size": 2},
        output_dir=tmp_path / "jedi",
    )

    assert result["raw"] == 25.0
    assert result["normalized"] == 0.75
    assert result["details"]["prediction_clip_count"] == 2
    assert result["details"]["reference_clip_count"] == 2


def test_runner_computes_jedi_as_suite_level_metric(monkeypatch, tmp_path: Path) -> None:
    reference_path = _write_video(tmp_path / "reference.mp4", base=40)
    prediction_path = tmp_path / "predictions" / "video_dynamic" / "prediction_001.mp4"
    _write_video(prediction_path, base=20)
    config = _config(tmp_path)

    def fake_jedi(
        *,
        prediction_reference_pairs: list[tuple[Path, Path, str]],
        normalization: dict[str, Any],
        runtime: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        assert prediction_reference_pairs == [
            (prediction_path, reference_path, "video-dynamic-jedi-001")
        ]
        assert normalization["higher_is_better"] is False
        assert runtime["clip_frame_count"] == 4
        assert output_dir.name == "jedi"
        return {
            "raw": 25.0,
            "normalized": 0.75,
            "backend": "jedi",
            "details": {"used_pair_count": 1},
        }

    monkeypatch.setattr(
        "worldarena.benchmark.runner.compute_jedi_distance",
        fake_jedi,
    )

    result = run_benchmark(
        config=config,
        manifest=[_sample(reference_path)],
        predictions_root=tmp_path / "predictions",
        output_dir=tmp_path / "report",
        model_name="unit-model",
    )

    aggregate = result["aggregate_payload"]
    assert aggregate["per_metric_summary"] == {"video_dynamic": {"jedi": 0.75}}
    assert aggregate["per_dimension_summary"] == {"video_dynamic": {"quality": 0.75}}
    assert aggregate["leaderboard_row"] == {"video_dynamic__jedi": 0.75}
    assert result["run_manifest"]["reported_metrics"] == {"video_dynamic": ["jedi"]}

    details = json.loads((tmp_path / "report" / "distribution_metric_details.json").read_text())
    assert details["video_dynamic"]["jedi"]["raw"] == 25.0


def test_jedi_registry_is_suite_level_quality_metric(tmp_path: Path) -> None:
    per_sample, suite_level = split_suite_level_metrics(["image_quality", "jepa_embedding_distance", "jedi"])
    assert per_sample == ["image_quality"]
    assert suite_level == ["jedi"]
    assert build_metrics(["jedi"], _config(tmp_path)) == []
    assert METRIC_DIMENSIONS["jedi"] == "quality"
