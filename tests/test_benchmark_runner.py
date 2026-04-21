from __future__ import annotations

from pathlib import Path

from PIL import Image

from worldarena.benchmark import run_benchmark
from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.schemas import BenchmarkSample


def test_benchmark_runner_scores_identical_image_prediction(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.png"
    prediction_root = tmp_path / "predictions" / "image_static"
    prediction_path = prediction_root / "image_static_photorealistic_living_spaces_0001.png"
    output_dir = tmp_path / "reports"
    Image.new("RGB", (64, 48), (80, 120, 180)).save(reference_path)
    prediction_root.mkdir(parents=True, exist_ok=True)
    Image.open(reference_path).save(prediction_path)

    config = BenchmarkConfig(
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
        aggregate_weights={
            "image_static": 0.15,
            "image_dynamic": 0.15,
            "video_static": 0.30,
            "video_dynamic": 0.40,
        },
        metrics_by_suite={
            "image_static": ["scene_alignment", "style_consistency", "subjective_quality"],
            "image_dynamic": [],
            "video_static": [],
            "video_dynamic": [],
            "auxiliary": [],
            "experimental": [],
        },
        metric_backends={
            "scene_alignment": "reference_proxy",
            "style_consistency": "histogram",
            "subjective_quality": "heuristic",
        },
        normalization={
            "scene_alignment": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "style_consistency": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "subjective_quality": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        },
    )
    manifest = [
        BenchmarkSample(
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
            conditioning_strategy="self",
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
            has_instruction=False,
            has_pose=False,
            has_mask=False,
            mask_count=0,
            width=64,
            height=48,
            fps=None,
            duration_seconds=None,
            duration_bucket=None,
        )
    ]

    result = run_benchmark(
        config=config,
        manifest=manifest,
        predictions_root=tmp_path / "predictions",
        output_dir=output_dir,
        model_name="toy_model",
        suites=None,
        splits={"dev_public"},
        limit=None,
    )

    assert result["run_manifest"]["missing_predictions"] == 0
    assert result["aggregate_payload"]["overall_scores"]["S_image_static"] is not None
    assert result["aggregate_payload"]["overall_scores"]["S_image_static"] > 0.7
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "per_sample_scores.jsonl").exists()
    assert (output_dir / "leaderboard.csv").exists()
