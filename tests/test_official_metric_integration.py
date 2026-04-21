from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.consistency import (
    OpticalFlowAepeMetric,
    ReprojectionErrorMetric,
)
from worldarena.benchmark.metrics.control import (
    CameraErrorMetric,
    MotionAccuracyMetric,
    ObjectDetectionMetric,
)
from worldarena.benchmark.metrics.motion import MotionMagnitudeMetric, MotionSmoothnessMetric
from worldarena.benchmark.metrics.registry import build_metrics
from worldarena.benchmark.official_runtime import _install_base_metric_shim
from worldarena.benchmark.schemas import BenchmarkSample


def _make_sample(
    *,
    suite: str = "video_static",
    conditioning_strategy: str = "first_frame",
    annotation_path: str | None = "/tmp/sample_ann",
    mask_path: str | None = None,
    has_pose: bool = True,
    has_mask: bool = False,
) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}_sample_0001",
        suite=suite,
        split="dev_public",
        asset_level=suite,
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="synthetic/test",
        path="/tmp/reference.mp4",
        relative_path="synthetic/test/reference.mp4",
        reference_path="/tmp/reference.mp4",
        prediction_stem=f"{suite}_sample_0001",
        conditioning_strategy=conditioning_strategy,
        prompt_current="A test scene.",
        prompt_target="Generate a test scene.",
        style="photorealistic",
        environment="outdoor",
        scene="reef",
        motion_category="fluid" if suite == "video_dynamic" else None,
        source_name="synthetic",
        source_type="unit_test",
        source_group_id=f"{suite}::0001",
        license_bucket="C",
        annotation_path=annotation_path,
        mask_path=mask_path,
        has_annotation=annotation_path is not None,
        has_instruction=True,
        has_pose=has_pose,
        has_mask=has_mask,
        mask_count=4 if has_mask else 0,
        width=64,
        height=64,
        fps=8.0,
        duration_seconds=1.0,
        duration_bucket="0-5s",
    )


def _make_loaded_media(path: str = "/tmp/prediction.mp4") -> LoadedMedia:
    return LoadedMedia(
        modality="video",
        path=path,
        frames=[],
        anchor_frames=[],
        sampled_frame_indices=[0, 3, 6],
        protocol_frame_indices=[0, 3, 6],
    )


def _make_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        config_path=tmp_path / "benchmark.yaml",
        paths=BenchmarkPaths(
            image_root=tmp_path / "images",
            video_root=tmp_path / "videos",
            inventory_dir=tmp_path / "inventory",
            manifest_dir=tmp_path / "manifest",
            reports_dir=tmp_path / "reports",
            cache_dir=tmp_path / ".cache",
        ),
        snapshot_name="test_snapshot",
        target_fps=24,
        anchor_positions=[0.0, 0.5, 1.0],
        motion_frame_count=8,
        split_ratios={"dev_public": 0.2, "dev_private": 0.1, "test_hidden": 0.7},
        metrics_by_suite={},
        metric_backends={
            "camera_control": "droid_slam",
            "photometric_consistency": "sea_raft",
            "motion_accuracy": "sam2_sea_raft",
        },
        normalization={},
    )


def test_registry_builds_new_official_metrics_from_aliases(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    metrics = build_metrics(
        ["camera_control", "photometric_consistency", "motion_accuracy"],
        config,
    )

    assert [metric.name for metric in metrics] == [
        "camera_error",
        "optical_flow_aepe",
        "motion_accuracy",
    ]


def test_camera_error_metric_uses_official_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.load_camera_matrices",
        lambda annotation_path, target_frames=None: object(),
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.compute_camera_error",
        lambda prediction_path, cameras_gt, scale, normalization: {
            "raw": {"rotation_error": 1.2, "translation_error": 0.03},
            "normalized": 0.91,
            "backend": "droid_slam",
            "details": {"frame_count": 16},
        },
    )

    metric = CameraErrorMetric(
        backend="droid_slam",
        normalization={"empirical_min": [0.0, 0.0], "empirical_max": [15.0, 0.5], "higher_is_better": [False, False]},
    )
    output = metric.compute(_make_sample(), _make_loaded_media(), _make_loaded_media("/tmp/reference.mp4"))

    assert output.normalized == 0.91
    assert output.backend == "droid_slam"
    assert output.raw == {"rotation_error": 1.2, "translation_error": 0.03}


def test_object_detection_metric_marks_caption_derived_prompts_as_weak(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.derive_object_prompts",
        lambda annotation_path: (["coral", "fish"], "caption_derived"),
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.compute_object_detection",
        lambda prediction_path, prompt_string, normalization: {
            "raw": 0.75,
            "normalized": 0.75,
            "backend": "grounding_dino",
            "details": {"prompt_string": prompt_string},
        },
    )

    metric = ObjectDetectionMetric(
        backend="grounding_dino",
        normalization={"empirical_min": 0.0, "empirical_max": 1.0, "higher_is_better": True},
    )
    output = metric.compute(_make_sample(), _make_loaded_media(), _make_loaded_media("/tmp/reference.mp4"))

    assert output.normalized == 0.75
    assert output.eligibility_status == "weak_supervision"
    assert output.details["prompt_candidates"] == ["coral", "fish"]


def test_static_heavy_metrics_route_through_official_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.consistency.compute_reprojection_error",
        lambda prediction_path, normalization: {
            "raw": 0.2,
            "normalized": 0.81,
            "backend": "droid_slam",
            "details": {"frame_count": 12},
        },
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.consistency.compute_optical_flow_aepe",
        lambda prediction_path, normalization: {
            "raw": 0.15,
            "normalized": 0.88,
            "backend": "sea_raft",
            "details": {"frame_count": 12},
        },
    )

    reprojection_metric = ReprojectionErrorMetric(
        backend="droid_slam",
        normalization={"empirical_min": 0.0, "empirical_max": 1.0719, "higher_is_better": False},
    )
    aepe_metric = OpticalFlowAepeMetric(
        backend="sea_raft",
        normalization={"empirical_min": 0.0, "empirical_max": 1.192, "higher_is_better": False},
    )

    reprojection_output = reprojection_metric.compute(
        _make_sample(),
        _make_loaded_media(),
        _make_loaded_media("/tmp/reference.mp4"),
    )
    aepe_output = aepe_metric.compute(
        _make_sample(),
        _make_loaded_media(),
        _make_loaded_media("/tmp/reference.mp4"),
    )

    assert reprojection_output.backend == "droid_slam"
    assert reprojection_output.normalized == 0.81
    assert aepe_output.backend == "sea_raft"
    assert aepe_output.normalized == 0.88


def test_motion_accuracy_metric_requires_masks(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.compute_motion_accuracy",
        lambda prediction_path, mask_dir, normalization, generate_type, runtime: captured.update({"runtime": runtime}) or {
            "raw": 1.4,
            "normalized": 0.67,
            "backend": "sam2_sea_raft",
            "details": {"mask_dir": mask_dir, "generate_type": generate_type},
        },
    )

    metric = MotionAccuracyMetric(
        backend="sam2_sea_raft",
        normalization={"avg": -0.1965, "std": 2.3687, "z_max": 1.518, "z_min": -1.6147, "range": [0.25, 0.75], "higher_is_better": True},
        runtime={"sam2_offload_state_to_cpu": True},
    )
    sample = _make_sample(
        suite="video_dynamic",
        mask_path="/tmp/masks",
        has_mask=True,
    )
    output = metric.compute(sample, _make_loaded_media(), _make_loaded_media("/tmp/reference.mp4"))

    assert output.backend == "sam2_sea_raft"
    assert output.normalized == 0.67
    assert output.details["generate_type"] == "i2v"
    assert captured["runtime"] == {"sam2_offload_state_to_cpu": True}


def test_motion_metrics_support_official_backends(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.motion.compute_official_motion_magnitude",
        lambda prediction_path, normalization: {
            "raw": 3.2,
            "normalized": 0.7,
            "backend": "sea_raft",
            "details": {"frame_count": 18},
        },
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.motion.compute_official_motion_smoothness",
        lambda prediction_path, normalization: {
            "raw": {"mse": 8.0, "ssim": 0.97, "lpips": 0.01},
            "normalized": 0.82,
            "backend": "vfi_mamba",
            "details": {"frame_count": 18},
        },
    )

    magnitude_metric = MotionMagnitudeMetric(
        backend="sea_raft",
        normalization={"avg": 3.2425, "std": 3.4505, "z_max": 2.7498, "z_min": -0.8638, "range": [0.25, 0.75], "higher_is_better": True},
    )
    smoothness_metric = MotionSmoothnessMetric(
        backend="vfi_mamba",
        normalization={"empirical_min": [0.0, 0.9224, 0.0], "empirical_max": [82.4014, 1.0, 0.0228], "higher_is_better": [False, True, False]},
    )
    sample = _make_sample(suite="video_dynamic", has_pose=False)

    magnitude_output = magnitude_metric.compute(sample, _make_loaded_media(), _make_loaded_media("/tmp/reference.mp4"))
    smoothness_output = smoothness_metric.compute(sample, _make_loaded_media(), _make_loaded_media("/tmp/reference.mp4"))

    assert magnitude_output.backend == "sea_raft"
    assert magnitude_output.normalized == 0.7
    assert smoothness_output.backend == "vfi_mamba"
    assert smoothness_output.normalized == 0.82


def test_official_base_metric_shim_supports_numpy_tensor_conversion() -> None:
    package_name = "worldarena_shim_test"
    module_name = f"{package_name}.benchmark.metrics.base_metrics"
    sys.modules.pop(module_name, None)

    _install_base_metric_shim(package_name)
    module = sys.modules[module_name]
    metric = module.BaseMetric()

    left = np.zeros((8, 8, 3), dtype=np.uint8)
    right = np.ones((8, 8, 3), dtype=np.uint8) * 255
    left_tensor, right_tensor = metric._process_np_to_tensor(left, right)

    assert tuple(left_tensor.shape) == (1, 3, 8, 8)
    assert tuple(right_tensor.shape) == (1, 3, 8, 8)
