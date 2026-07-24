from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.fvmd_backend import compute_frechet_video_motion_distance
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
        sample_id="video-dynamic-001",
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
        source_group_id="group-fvmd",
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
            predictions_dir=tmp_path / "predictions",
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / "cache",
        ),
        snapshot_name="test",
        target_fps=24,
        anchor_positions=[0.0, 1.0],
        motion_frame_count=4,
        metrics_by_suite={"video_dynamic": ["fvmd"]},
        metric_backends={"fvmd": "fvmd"},
        normalization={"fvmd": {"lower": 0.0, "upper": 10000.0, "higher_is_better": False}},
        metric_runtime={"fvmd": {"clip_frame_count": 4, "analysis_size": 8}},
    )


def test_fvmd_backend_exports_clip_sets_and_normalizes(monkeypatch, tmp_path: Path) -> None:
    pairs: list[tuple[Path, Path, str]] = []
    for index in range(2):
        prediction = _write_video(tmp_path / f"pred_{index}.mp4", base=20 + index)
        reference = _write_video(tmp_path / f"ref_{index}.mp4", base=40 + index)
        pairs.append((prediction, reference, f"sample-{index}"))

    def fake_loader(runtime: dict[str, Any]):
        def fake_fvmd(*, log_dir: str, gen_path: str, gt_path: str) -> float:
            assert Path(log_dir).is_dir()
            generated = np.load(gen_path)
            reference = np.load(gt_path)
            assert generated.shape == (2, 4, 8, 8, 3)
            assert reference.shape == (2, 4, 8, 8, 3)
            return 100.0

        return fake_fvmd, tmp_path / "FVMD"

    monkeypatch.setattr("worldarena.benchmark.fvmd_backend._load_fvmd_callable", fake_loader)

    result = compute_frechet_video_motion_distance(
        prediction_reference_pairs=pairs,
        normalization={"lower": 0.0, "upper": 10000.0, "higher_is_better": False},
        runtime={"clip_frame_count": 4, "analysis_size": 8, "min_clip_count": 2},
        output_dir=tmp_path / "fvmd",
    )

    assert result["raw"] == 100.0
    assert result["normalized"] == 0.99
    assert result["details"]["prediction_clip_count"] == 2
    assert result["details"]["reference_clip_count"] == 2


def test_runner_computes_fvmd_as_suite_level_metric(monkeypatch, tmp_path: Path) -> None:
    reference_path = _write_video(tmp_path / "reference.mp4", base=40)
    prediction_path = tmp_path / "predictions" / "video_dynamic" / "prediction_001.mp4"
    _write_video(prediction_path, base=20)
    config = _config(tmp_path)

    def fake_fvmd(
        *,
        prediction_reference_pairs: list[tuple[Path, Path, str]],
        normalization: dict[str, Any],
        runtime: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        assert prediction_reference_pairs == [
            (prediction_path, reference_path, "video-dynamic-001")
        ]
        assert normalization["higher_is_better"] is False
        assert runtime["clip_frame_count"] == 4
        assert output_dir.name == "fvmd"
        return {
            "raw": 2500.0,
            "normalized": 0.75,
            "backend": "fvmd",
            "details": {"used_pair_count": 1},
        }

    monkeypatch.setattr(
        "worldarena.benchmark.runner.compute_frechet_video_motion_distance",
        fake_fvmd,
    )

    result = run_benchmark(
        config=config,
        manifest=[_sample(reference_path)],
        predictions_root=tmp_path / "predictions",
        output_dir=tmp_path / "report",
        model_name="unit-model",
    )

    aggregate = result["aggregate_payload"]
    assert aggregate["per_metric_summary"] == {"video_dynamic": {"fvmd": 0.75}}
    assert aggregate["per_dimension_summary"] == {"video_dynamic": {"physics": 0.75}}
    assert aggregate["leaderboard_row"] == {"video_dynamic__fvmd": 0.75}
    assert result["run_manifest"]["reported_metrics"] == {"video_dynamic": ["fvmd"]}

    details = json.loads((tmp_path / "report" / "distribution_metric_details.json").read_text())
    assert details["video_dynamic"]["fvmd"]["raw"] == 2500.0


def test_fvmd_registry_is_suite_level_physics_metric(tmp_path: Path) -> None:
    per_sample, suite_level = split_suite_level_metrics(
        ["pmf", "frechet_video_motion_distance", "fvmd"]
    )
    assert per_sample == ["pmf"]
    assert suite_level == ["fvmd"]
    assert build_metrics(["fvmd"], _config(tmp_path)) == []
    assert METRIC_DIMENSIONS["fvmd"] == "physics"
