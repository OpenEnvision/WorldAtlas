from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.appearance import (
    BrightnessConsistencyMetric,
    BrightnessDistributionConsistencyMetric,
    ColorTemperatureConstraintMetric,
    ColorTemperatureConsistencyMetric,
    SharpnessRetentionMetric,
    SharpnessVectorRetentionMetric,
    compute_brightness_consistency,
    compute_brightness_distribution_consistency,
    compute_color_temperature_consistency,
    compute_color_temperature_constraint,
    compute_sharpness_retention,
    compute_sharpness_vector_retention,
)
from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.metrics.quality import ImageQualityMetric
from worldarena.benchmark.metrics.registry import build_metrics
from worldarena.benchmark.schemas import BenchmarkSample


def _solid(rgb: tuple[int, int, int], size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def _checkerboard(size: int = 32, block: int = 2) -> np.ndarray:
    grid = np.indices((size, size)).sum(axis=0) // block
    image = ((grid % 2) * 255).astype(np.uint8)
    return np.repeat(image[:, :, None], 3, axis=2)


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


def _sample(suite: str = "video_dynamic") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-appearance",
        suite=suite,
        split="test",
        asset_level=suite,
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="dynamic/photorealistic/outdoor/test",
        path="/tmp/reference.mp4",
        relative_path="dynamic/photorealistic/outdoor/test/reference.mp4",
        reference_path="/tmp/reference.mp4",
        prediction_stem=f"{suite}-appearance",
        conditioning_strategy="reference_video",
        prompt_current="A scene.",
        prompt_target="Keep a stable scene.",
        style="photorealistic",
        environment="outdoor",
        scene="test",
        motion_category="camera",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-appearance",
        license_bucket="internal",
    )


def _config(tmp_path: Path) -> BenchmarkConfig:
    metric_names = [
        "image_quality",
        "brightness_consistency",
        "brightness_distribution_consistency",
        "color_temperature_consistency",
        "color_temperature_constraint",
        "sharpness_retention",
        "sharpness_vector_retention",
    ]
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
        metrics_by_suite={
            "image_static": metric_names,
            "image_dynamic": metric_names,
            "video_static": metric_names,
            "video_dynamic": metric_names,
        },
        metric_backends={
            "image_quality": "musiq",
            "brightness_consistency": "luminance_temporal",
            "brightness_distribution_consistency": "distribution",
            "color_temperature_consistency": "hsv_hue_drift",
            "color_temperature_constraint": "hue_distribution",
            "sharpness_retention": "tenengrad",
            "sharpness_vector_retention": "tenengrad_vector",
        },
        normalization={
            "image_quality": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "brightness_consistency": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "brightness_distribution_consistency": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "color_temperature_consistency": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "color_temperature_constraint": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "sharpness_retention": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "sharpness_vector_retention": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        },
    )


def test_brightness_consistency_penalizes_temporal_luminance_flicker() -> None:
    stable = [_solid((100, 100, 100)), _solid((102, 102, 102)), _solid((101, 101, 101))]
    flicker = [_solid((20, 20, 20)), _solid((235, 235, 235)), _solid((20, 20, 20))]

    stable_score = compute_brightness_consistency(stable)["raw"]
    flicker_score = compute_brightness_consistency(flicker)["raw"]

    assert stable_score > 0.95
    assert flicker_score < 0.05


def test_brightness_distribution_consistency_penalizes_distribution_shift() -> None:
    stable = [_solid((100, 100, 100)), _solid((102, 102, 102)), _solid((101, 101, 101))]
    drifting = [_solid((100, 100, 100)), _solid((240, 240, 240)), _solid((245, 245, 245))]

    stable_score = compute_brightness_distribution_consistency(stable)["raw"]
    drifting_score = compute_brightness_distribution_consistency(drifting)["raw"]

    assert stable_score > 0.95
    assert drifting_score < 0.05


def test_color_temperature_consistency_penalizes_hue_drift() -> None:
    stable = [_solid((220, 40, 40)), _solid((225, 42, 42)), _solid((218, 40, 40))]
    drifting = [_solid((220, 40, 40)), _solid((40, 40, 220)), _solid((220, 40, 40))]

    stable_score = compute_color_temperature_consistency(stable)["raw"]
    drifting_score = compute_color_temperature_consistency(drifting)["raw"]

    assert stable_score > 0.95
    assert drifting_score < 0.01


def test_color_temperature_constraint_penalizes_hue_distribution_drift() -> None:
    stable = [_solid((220, 40, 40)), _solid((225, 42, 42)), _solid((218, 40, 40))]
    drifting = [_solid((220, 40, 40)), _solid((40, 40, 220)), _solid((40, 220, 40))]

    stable_score = compute_color_temperature_constraint(stable)["raw"]
    drifting_score = compute_color_temperature_constraint(drifting)["raw"]

    assert stable_score > 0.95
    assert drifting_score < 0.2


def test_sharpness_retention_penalizes_blur_after_initial_frames() -> None:
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (9, 9), 2.0)

    stable_score = compute_sharpness_retention(
        [sharp, sharp, sharp],
        use_brisque_gate=False,
        baseline_frame_count=1,
    )["raw"]
    blurred_score = compute_sharpness_retention(
        [sharp, blurred, blurred],
        use_brisque_gate=False,
        baseline_frame_count=1,
    )["raw"]

    assert stable_score > 0.99
    assert blurred_score < 0.75


def test_sharpness_vector_retention_penalizes_edge_direction_change() -> None:
    vertical = np.zeros((32, 32, 3), dtype=np.uint8)
    vertical[:, 8:24, :] = 255
    horizontal = np.zeros((32, 32, 3), dtype=np.uint8)
    horizontal[8:24, :, :] = 255

    stable_score = compute_sharpness_vector_retention([vertical, vertical, vertical])["raw"]
    shifted_score = compute_sharpness_vector_retention([vertical, horizontal, horizontal])["raw"]

    assert stable_score > 0.99
    assert shifted_score < 0.8


def test_sharpness_retention_brisque_gate_is_strict(monkeypatch) -> None:
    sharp = _checkerboard()
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.appearance._brisque_frame_scores",
        lambda frames: [90.0 for _ in frames],
    )

    result = compute_sharpness_retention(
        [sharp, sharp],
        use_brisque_gate=True,
        baseline_frame_count=1,
        brisque_good=30.0,
        brisque_bad=80.0,
    )

    assert result["raw"] == 0.0
    assert result["details"]["brisque_gate"] == 0.0


def test_sharpness_retention_rejects_nonfinite_brisque_scores(monkeypatch) -> None:
    sharp = _checkerboard()
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.appearance._brisque_frame_scores",
        lambda frames: [float("nan") for _ in frames],
    )

    metric = SharpnessRetentionMetric(
        backend="tenengrad_brisque",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    result = metric.compute(_sample(), _media([sharp, sharp]), _media([sharp, sharp]))

    assert result.raw is None
    assert result.normalized is None
    assert result.eligibility_status == "diagnostic"
    assert "non-finite" in str(result.error)


def test_appearance_metrics_are_registered_as_quality_diagnostics(tmp_path: Path) -> None:
    config = _config(tmp_path)

    for suite in ("image_static", "image_dynamic", "video_static", "video_dynamic"):
        metrics = build_metrics(config.metrics_by_suite[suite], config)
        assert [metric.name for metric in metrics] == [
            "image_quality",
            "brightness_consistency",
            "brightness_distribution_consistency",
            "color_temperature_consistency",
            "color_temperature_constraint",
            "sharpness_retention",
            "sharpness_vector_retention",
        ]

    assert METRIC_DIMENSIONS["image_quality"] == "quality"
    assert METRIC_DIMENSIONS["brightness_consistency"] == "quality"
    assert METRIC_DIMENSIONS["brightness_distribution_consistency"] == "quality"
    assert METRIC_DIMENSIONS["color_temperature_consistency"] == "quality"
    assert METRIC_DIMENSIONS["color_temperature_constraint"] == "quality"
    assert METRIC_DIMENSIONS["sharpness_retention"] == "quality"
    assert METRIC_DIMENSIONS["sharpness_vector_retention"] == "quality"


def test_appearance_metric_outputs_are_diagnostic() -> None:
    media = _media([_solid((120, 120, 120)), _solid((121, 121, 121))])
    sample = _sample()
    brightness = BrightnessConsistencyMetric(
        backend="luminance_temporal",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    output = brightness.compute(sample, media, media)

    assert output.normalized is not None
    assert output.eligibility_status == "diagnostic"

    color = ColorTemperatureConsistencyMetric(
        backend="hsv_hue_drift",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    color_output = color.compute(
        sample,
        _media([_solid((220, 40, 40)), _solid((222, 40, 40))]),
        media,
    )

    assert color_output.normalized is not None
    assert color_output.eligibility_status == "diagnostic"

    sharp = _checkerboard()
    sharpness = SharpnessRetentionMetric(
        backend="tenengrad",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    sharpness_output = sharpness.compute(sample, _media([sharp, sharp]), media)

    assert sharpness_output.normalized is not None
    assert sharpness_output.eligibility_status == "diagnostic"

    distribution = BrightnessDistributionConsistencyMetric(
        backend="distribution",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    distribution_output = distribution.compute(sample, media, media)
    assert distribution_output.normalized is not None

    constraint = ColorTemperatureConstraintMetric(
        backend="hue_distribution",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    constraint_output = constraint.compute(
        sample,
        _media([_solid((220, 40, 40)), _solid((222, 40, 40))]),
        media,
    )
    assert constraint_output.normalized is not None

    vector_retention = SharpnessVectorRetentionMetric(
        backend="tenengrad_vector",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )
    vector_output = vector_retention.compute(sample, _media([sharp, sharp]), media)
    assert vector_output.normalized is not None


def test_image_quality_metric_uses_musiq_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.quality.compute_musiq_scores",
        lambda frames, **kwargs: {
            "backend": "musiq",
            "raw": 0.75,
            "frame_scores": [0.75 for _ in frames],
            "native_frame_scores": [75.0 for _ in frames],
            "native_score_range": [0.0, 100.0],
            "preprocess_mode": kwargs["preprocess_mode"],
            "weights_path": "/tmp/musiq.pth",
        },
    )
    media = _media([_solid((120, 120, 120)), _solid((121, 121, 121))])
    metric = ImageQualityMetric(
        backend="musiq",
        normalization={"lower": 0.0, "upper": 1.0, "higher_is_better": True},
    )

    output = metric.compute(_sample(), media, media)

    assert output.raw == 0.75
    assert output.normalized == 0.75
    assert output.eligibility_status == "diagnostic"
