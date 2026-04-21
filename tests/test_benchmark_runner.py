from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

import worldarena.benchmark_core.metrics.alignment as alignment_metrics
import worldarena.benchmark_core.metrics.quality as quality_metrics
import worldarena.benchmark_core.metrics.style as style_metrics
from worldarena.benchmark import run_benchmark
from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.schemas import BenchmarkSample


def _make_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        config_path=tmp_path / "benchmark.yaml",
        paths=BenchmarkPaths(
            image_root=tmp_path / "image",
            video_root=tmp_path / "videos",
            inventory_dir=tmp_path / "inventory",
            manifest_dir=tmp_path / "manifest",
            reports_dir=tmp_path / "reports_root",
            cache_dir=tmp_path / ".cache" / "benchmark",
        ),
        snapshot_name="test_snapshot",
        target_fps=24,
        anchor_positions=[0.0, 0.5, 1.0],
        motion_frame_count=8,
        split_ratios={"dev_public": 0.2, "dev_private": 0.1, "test_hidden": 0.7},
        metrics_by_suite={
            "image_static": ["prompt_alignment", "style_consistency", "perceptual_quality"],
            "image_dynamic": [],
            "video_static": [],
            "video_dynamic": [],
            "auxiliary": [],
            "experimental": [],
        },
        metric_backends={
            "prompt_alignment": "reference",
            "style_consistency": "reference",
            "perceptual_quality": "reference",
        },
        normalization={
            "prompt_alignment": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "style_consistency": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "perceptual_quality": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        },
    )


def _make_sample(
    reference_path: Path,
    *,
    conditioning_strategy: str,
    has_instruction: bool,
) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="image_static_photorealistic_living_spaces_0001",
        suite="image_static",
        split="dev_public",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/living_spaces",
        path=str(reference_path),
        relative_path="static/photorealistic/indoor/living_spaces/0001.png",
        reference_path=str(reference_path),
        prediction_stem="image_static_photorealistic_living_spaces_0001",
        conditioning_strategy=conditioning_strategy,
        prompt_current="A photorealistic indoor living spaces scene.",
        prompt_target="Generate a photorealistic still image of a indoor living spaces scene.",
        style="photorealistic",
        environment="indoor",
        scene="living_spaces",
        motion_category=None,
        source_name="worldarena_formal_image",
        source_type="curated_image",
        source_group_id="static/photorealistic/indoor/living_spaces::0001",
        license_bucket="C",
        has_annotation=False,
        has_instruction=has_instruction,
        has_pose=False,
        has_mask=False,
        mask_count=0,
        width=64,
        height=48,
        fps=None,
        duration_seconds=None,
        duration_bucket=None,
    )


def _prepare_prediction(tmp_path: Path) -> tuple[Path, Path, Path]:
    reference_path = tmp_path / "reference.png"
    prediction_root = tmp_path / "predictions" / "image_static"
    prediction_path = prediction_root / "image_static_photorealistic_living_spaces_0001.png"
    output_dir = tmp_path / "reports"
    Image.new("RGB", (64, 48), (80, 120, 180)).save(reference_path)
    prediction_root.mkdir(parents=True, exist_ok=True)
    Image.open(reference_path).save(prediction_path)
    return reference_path, prediction_root.parent, output_dir


def _mock_reference_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        alignment_metrics,
        "compute_reference_prompt_alignment",
        lambda frames, prompt: {
            "raw": 0.92,
            "backend": "clipscore_torchmetrics",
            "details": {
                "frame_scores": [0.92 for _ in frames],
                "native_frame_scores": [92.0 for _ in frames],
                "native_score_range": [0.0, 100.0],
            },
        },
    )
    monkeypatch.setattr(
        style_metrics,
        "compute_reference_style_consistency",
        lambda reference_frames, prediction_frames: {
            "raw": 0.88,
            "backend": "vgg19_gram",
            "details": {
                "frame_scores": [0.88 for _ in reference_frames],
                "native_frame_distances": [0.1364 for _ in reference_frames],
                "native_layer_distances": [[0.1364, 0.1364] for _ in reference_frames],
                "native_direction": "lower_is_better",
            },
        },
    )
    monkeypatch.setattr(
        quality_metrics,
        "compute_reference_perceptual_quality",
        lambda frames: {
            "raw": 0.81,
            "backend": "pyiqa_stack",
            "details": {
                "frame_scores": [0.81 for _ in frames],
                "required_components": ["clip_iqa", "aesthetic"],
                "components": {
                    "clip_iqa": {
                        "backend": "clipiqa_plus",
                        "frame_scores": [0.76 for _ in frames],
                        "native_frame_scores": [0.76 for _ in frames],
                        "native_score_range": [0.0, 1.0],
                    },
                    "aesthetic": {
                        "backend": "laion_aes",
                        "frame_scores": [0.86 for _ in frames],
                        "native_frame_scores": [8.6 for _ in frames],
                        "native_score_range": [0.0, 10.0],
                    },
                },
            },
        },
    )


def test_benchmark_runner_reports_official_metrics_for_instruction_conditioned_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _mock_reference_metrics(monkeypatch)
    reference_path, predictions_root, output_dir = _prepare_prediction(tmp_path)
    config = _make_config(tmp_path)
    manifest = [_make_sample(reference_path, conditioning_strategy="text", has_instruction=True)]

    result = run_benchmark(
        config=config,
        manifest=manifest,
        predictions_root=predictions_root,
        output_dir=output_dir,
        model_name="toy_model",
        suites=None,
        splits={"dev_public"},
        limit=None,
    )

    assert result["run_manifest"]["missing_predictions"] == 0
    assert result["aggregate_payload"]["per_metric_summary"]["image_static"]["prompt_alignment"] is not None
    assert result["aggregate_payload"]["per_metric_summary"]["image_static"]["prompt_alignment"] > 0.7
    assert result["run_manifest"]["reported_metrics"]["image_static"] == [
        "perceptual_quality",
        "prompt_alignment",
        "style_consistency",
    ]
    assert result["run_manifest"]["reported_diagnostics"] == {}
    assert result["run_manifest"]["evaluation_protocol"] == "anchor_compat"
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "per_sample_scores.jsonl").exists()
    assert (output_dir / "leaderboard.csv").exists()
    assert not (output_dir / "diagnostic_metric_summary.json").exists()

    manifest_payload = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["reported_metrics"] == result["run_manifest"]["reported_metrics"]

    first_row = json.loads(
        (output_dir / "per_sample_scores.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "score" not in first_row
    assert set(first_row["metrics"].keys()) == {
        "prompt_alignment",
        "style_consistency",
        "perceptual_quality",
    }
    assert first_row["metrics"]["prompt_alignment"]["eligibility_status"] == "official"

    with (output_dir / "leaderboard.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "model_name",
            "image_static__perceptual_quality",
            "image_static__prompt_alignment",
            "image_static__style_consistency",
        ]
        rows = list(reader)

    assert rows == [
        {
            "model_name": "toy_model",
            "image_static__perceptual_quality": str(
                result["aggregate_payload"]["leaderboard_row"]["image_static__perceptual_quality"]
            ),
            "image_static__prompt_alignment": str(
                result["aggregate_payload"]["leaderboard_row"]["image_static__prompt_alignment"]
            ),
            "image_static__style_consistency": str(
                result["aggregate_payload"]["leaderboard_row"]["image_static__style_consistency"]
            ),
        }
    ]


def test_benchmark_runner_separates_weak_prompt_alignment_into_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _mock_reference_metrics(monkeypatch)
    reference_path, predictions_root, output_dir = _prepare_prediction(tmp_path)
    config = _make_config(tmp_path)
    manifest = [_make_sample(reference_path, conditioning_strategy="self", has_instruction=False)]

    result = run_benchmark(
        config=config,
        manifest=manifest,
        predictions_root=predictions_root,
        output_dir=output_dir,
        model_name="toy_model",
        suites=None,
        splits={"dev_public"},
        limit=None,
    )

    sample_payload = result["sample_scores"][0].to_dict()
    assert sample_payload["metrics"]["prompt_alignment"]["eligibility_status"] == "weak_supervision"
    assert "prompt_alignment" not in result["aggregate_payload"]["per_metric_summary"]["image_static"]
    assert (
        result["aggregate_payload"]["diagnostic_metric_summary"]["image_static"]["prompt_alignment"]
        is not None
    )
    assert result["run_manifest"]["reported_metrics"]["image_static"] == [
        "perceptual_quality",
        "style_consistency",
    ]
    assert result["run_manifest"]["reported_diagnostics"]["image_static"] == ["prompt_alignment"]
    assert (output_dir / "diagnostic_metric_summary.json").exists()

    with (output_dir / "leaderboard.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "model_name",
            "image_static__perceptual_quality",
            "image_static__style_consistency",
        ]
