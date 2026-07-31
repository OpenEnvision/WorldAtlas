from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from PIL import Image

from worldarena.benchmark import official_runtime
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.control import CameraErrorMetric
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.synthetic_camera import (
    _rotation_matrix_y,
    synthetic_camera_matrices,
    synthetic_camera_matrices_for_sample,
)


def _sample(*, suite: str, modality: str = "image", generation_mode: str = "static") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}_sample",
        suite=suite,
        split="test",
        asset_level=suite,
        modality=modality,
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/aquatic_landscape",
        path="/tmp/input.png",
        relative_path="static/photorealistic/outdoor/aquatic_landscape/000.png",
        reference_path="/tmp/input.png",
        prediction_stem=f"{suite}_sample",
        conditioning_strategy="reference_image",
        prompt_current="A coastal scene.",
        prompt_target="Camera orbit left. A coastal scene.",
        style="photorealistic",
        environment="outdoor",
        scene="aquatic_landscape",
        motion_category=None,
        source_name="unit",
        source_type="unit",
        source_group_id="unit",
        license_bucket="unit",
        prompt_sequence=["Camera orbit left. A coastal scene."],
        camera_path=["orbit_left"],
        generation_mode=generation_mode,
        task_family="video_generation",
        artifact_type="video",
        control_signals=["text", "image"],
        annotation_path=None,
        has_annotation=False,
        has_instruction=True,
        has_pose=False,
    )


def _media() -> LoadedMedia:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return LoadedMedia(
        modality="video",
        path="/tmp/prediction.mp4",
        frames=[frame, frame, frame],
        anchor_frames=[frame],
        sampled_frame_indices=[0, 1, 2],
        protocol_frame_indices=[0, 1, 2],
        native_frame_count=3,
    )


def test_synthetic_camera_matrices_only_for_image_static() -> None:
    image_static = _sample(suite="image_static")
    trajectory = synthetic_camera_matrices_for_sample(image_static, target_frames=5)

    assert trajectory is not None
    assert trajectory.source == "synthetic_camera_path"
    assert trajectory.camera_path == ("orbit_left",)
    assert trajectory.matrices.shape == (5, 4, 4)
    assert not np.allclose(trajectory.matrices[0], trajectory.matrices[-1])

    image_dynamic = _sample(suite="image_dynamic", generation_mode="dynamic")
    assert synthetic_camera_matrices_for_sample(image_dynamic, target_frames=5) is None


def test_camera_error_uses_synthetic_static_fallback(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_materialize_frame_paths(path: str) -> list[str]:
        assert path == "/tmp/prediction.mp4"
        return ["/tmp/000.png", "/tmp/001.png", "/tmp/002.png"]

    def fake_compute_camera_error(prediction_path: str, **kwargs):
        assert prediction_path == "/tmp/prediction.mp4"
        captured.update(kwargs)
        cameras_gt = np.asarray(kwargs["cameras_gt"])
        assert cameras_gt.shape == (3, 4, 4)
        assert not np.allclose(cameras_gt[0], cameras_gt[-1])
        return {
            "raw": {"rotation_error": 1.0, "translation_error": 0.2},
            "normalized": 0.8,
            "backend": "droid_slam",
            "details": {"frame_count": 3, "camera_scale": kwargs["scale"]},
        }

    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.materialize_frame_paths",
        fake_materialize_frame_paths,
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.compute_camera_error",
        fake_compute_camera_error,
    )

    metric = CameraErrorMetric(backend="droid_slam", normalization={})
    output = metric.compute(sample=_sample(suite="image_static"), prediction=_media(), reference=_media())

    assert output.raw == {"rotation_error": 1.0, "translation_error": 0.2}
    assert output.normalized == 0.8
    assert output.details["camera_pose_source"] == "synthetic_camera_path"
    assert output.details["camera_path"] == ["orbit_left"]
    assert captured["scale"] == 1.0


def test_camera_error_dispatches_megasam_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_materialize_frame_paths(path: str) -> list[str]:
        assert path == "/tmp/prediction.mp4"
        return ["/tmp/000.png", "/tmp/001.png", "/tmp/002.png"]

    def fake_compute_camera_error_megasam(prediction_path: str, **kwargs):
        assert prediction_path == "/tmp/prediction.mp4"
        captured.update(kwargs)
        cameras_gt = np.asarray(kwargs["cameras_gt"])
        assert cameras_gt.shape == (3, 4, 4)
        assert not np.allclose(cameras_gt[0], cameras_gt[-1])
        return {
            "raw": {"rotation_error": 0.5, "translation_error": 0.1},
            "normalized": 0.9,
            "backend": "megasam",
            "details": {
                "frame_count": 3,
                "camera_scale": kwargs["scale"],
                "megasam_pose_count": 3,
            },
        }

    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.materialize_frame_paths",
        fake_materialize_frame_paths,
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.compute_camera_error_megasam",
        fake_compute_camera_error_megasam,
    )

    runtime = {"megasam_skip_existing": True}
    metric = CameraErrorMetric(backend="megasam", normalization={}, runtime=runtime)
    output = metric.compute(sample=_sample(suite="image_static"), prediction=_media(), reference=_media())

    assert output.raw == {"rotation_error": 0.5, "translation_error": 0.1}
    assert output.normalized == 0.9
    assert output.backend == "megasam"
    assert output.details["camera_pose_source"] == "synthetic_camera_path"
    assert output.details["camera_path"] == ["orbit_left"]
    assert output.details["megasam_pose_count"] == 3
    assert captured["runtime"] == runtime
    assert captured["frame_paths"] == ["/tmp/000.png", "/tmp/001.png", "/tmp/002.png"]
    assert captured["scale"] == 1.0


def test_camera_error_dispatches_megasam_native_intended_policy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fail_materialize_frame_paths(path: str) -> list[str]:
        raise AssertionError(f"native-intended MegaSAM must not materialize frames: {path}")

    def fake_compute_camera_error_megasam_native_intended(prediction_path: str, **kwargs):
        assert prediction_path == "/tmp/prediction.mp4"
        captured.update(kwargs)
        return {
            "raw": {
                "direction_score": 0.8,
                "magnitude_score": 0.7,
                "stability_score": 0.9,
            },
            "normalized": 0.8,
            "backend": "megasam",
            "details": {
                "frame_count": 3,
                "megasam_pose_count": 3,
                "native_camera_policy": "intended_camera_path",
            },
        }

    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.materialize_frame_paths",
        fail_materialize_frame_paths,
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.compute_camera_error_megasam_native_intended",
        fake_compute_camera_error_megasam_native_intended,
    )

    runtime = {"camera_gt_policy": "native_intended", "megasam_skip_existing": True}
    sample = _sample(suite="image_static")
    metric = CameraErrorMetric(backend="megasam", normalization={}, runtime=runtime)
    output = metric.compute(sample=sample, prediction=_media(), reference=_media())

    assert output.backend == "megasam"
    assert output.normalized == 0.8
    assert output.details["camera_pose_source"] == "native_intended"
    assert output.details["native_camera_policy"] == "intended_camera_path"
    assert captured["sample"] is sample
    assert captured["runtime"] == runtime
    assert "frame_paths" not in captured


def test_camera_error_does_not_synthesize_dynamic(monkeypatch) -> None:
    def fake_materialize_frame_paths(path: str) -> list[str]:
        return ["/tmp/000.png", "/tmp/001.png", "/tmp/002.png"]

    def fail_compute_camera_error(**kwargs):
        raise AssertionError("dynamic samples must not synthesize camera_error GT")

    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.materialize_frame_paths",
        fake_materialize_frame_paths,
    )
    monkeypatch.setattr(
        "worldarena.benchmark.metrics.control.compute_camera_error",
        fail_compute_camera_error,
    )

    metric = CameraErrorMetric(backend="droid_slam", normalization={})
    output = metric.compute(
        sample=_sample(suite="image_dynamic", generation_mode="dynamic"),
        prediction=_media(),
        reference=_media(),
    )

    assert output.raw is None
    assert output.error == "camera poses are unavailable for this sample"


def test_synthetic_camera_matrices_exact_frame_count() -> None:
    trajectory = synthetic_camera_matrices(["push_in", "move_left"], target_frames=6)
    assert trajectory.matrices.shape == (6, 4, 4)
    np.testing.assert_allclose(trajectory.matrices[0], np.eye(4, dtype=np.float32))


def test_synthetic_y_rotation_uses_standard_right_handed_convention() -> None:
    angle = np.deg2rad(30.0)
    expected = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(_rotation_matrix_y(30.0), expected, atol=1e-6)

    pan_left = synthetic_camera_matrices(["pan_left"], target_frames=3).matrices[-1]
    pan_right = synthetic_camera_matrices(["pan_right"], target_frames=3).matrices[-1]
    orbit_left = synthetic_camera_matrices(["orbit_left"], target_frames=3).matrices[-1]
    orbit_right = synthetic_camera_matrices(["orbit_right"], target_frames=3).matrices[-1]
    assert pan_left[0, 2] < 0.0 < pan_right[0, 2]
    assert orbit_left[0, 3] < 0.0 < orbit_right[0, 3]


def test_official_camera_error_scales_droid_intrinsics_to_frame_resolution(
    tmp_path,
    monkeypatch,
) -> None:
    frame_path = tmp_path / "000.png"
    Image.new("RGB", (1024, 512), color=(0, 0, 0)).save(frame_path)
    captured: dict[str, object] = {}

    class DummyMetric:
        def __init__(self) -> None:
            self._args = SimpleNamespace(calib=[500.0, 500.0, 256.0, 256.0])

        def _compute_scores(self, frame_paths, cameras_gt, scale):
            captured["frame_paths"] = list(frame_paths)
            captured["calib"] = list(self._args.calib)
            captured["scale"] = scale
            captured["cameras_shape"] = tuple(cameras_gt.shape)
            return (1.0, 0.2)

    monkeypatch.setattr(official_runtime, "metric_runtime_context", lambda: nullcontext())
    monkeypatch.setattr(official_runtime, "_run_metric", lambda name: DummyMetric())
    monkeypatch.setattr(official_runtime, "_empty_cuda_cache", lambda: None)

    cameras_gt = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=2, axis=0)
    output = official_runtime.compute_camera_error(
        "prediction.mp4",
        cameras_gt=cameras_gt,
        scale=1.0,
        normalization={
            "empirical_min": [0.0, 0.0],
            "empirical_max": [15.0, 0.5],
            "higher_is_better": [False, False],
        },
        frame_paths=[str(frame_path)],
        runtime={"droid_intrinsics": "actual_resolution"},
    )

    assert captured["frame_paths"] == [str(frame_path)]
    assert captured["calib"] == [1024.0, 512.0, 512.0, 256.0]
    assert captured["scale"] == 1.0
    assert captured["cameras_shape"] == (2, 4, 4)
    assert output["details"]["droid_intrinsics_policy"] == "actual_resolution"
    assert output["details"]["droid_source_resolution"] == [1024, 512]
    assert output["details"]["droid_focal_scale"] == 1.0
    assert output["details"]["droid_calib"] == [1024.0, 512.0, 512.0, 256.0]


def test_official_camera_error_megasam_selects_gt_frame_window(
    tmp_path,
    monkeypatch,
) -> None:
    frame_paths = []
    for index in range(4):
        frame_path = tmp_path / f"{index:06d}.png"
        Image.new("RGB", (64, 32), color=(index, index, index)).save(frame_path)
        frame_paths.append(str(frame_path))

    cameras_gt = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=4, axis=0)
    cameras_gt[:, 2, 3] = np.arange(4, dtype=np.float32)
    captured: dict[str, object] = {}

    def fake_run_megasam_tracking(prediction_path: str, runtime):
        captured["prediction_path"] = prediction_path
        captured["runtime"] = dict(runtime)
        return {
            "scene": "unit_scene",
            "run_root": tmp_path / "run",
            "log_path": tmp_path / "run" / "camera_error_megasam.log",
            "poses": np.zeros((2, 7), dtype=np.float32),
            "intrinsics": np.asarray([[50.0, 51.0, 32.0, 16.0]], dtype=np.float32),
            "metadata": {"frame_count": 2},
            "frame_stride": 2,
            "start_frame": 1,
            "max_frames": 2,
        }

    def fake_pose_vectors_to_camera_matrices(poses):
        captured["pose_shape"] = tuple(np.asarray(poses).shape)
        return np.repeat(np.eye(4, dtype=np.float32)[None], repeats=2, axis=0)

    def fake_score_camera_matrices(cameras_pred, cameras_selected, scale):
        captured["pred_shape"] = tuple(np.asarray(cameras_pred).shape)
        captured["selected_z"] = list(np.asarray(cameras_selected)[:, 2, 3])
        captured["scale"] = scale
        return [2.0, 0.1], {"translation_alignment": "unit_test"}

    monkeypatch.setattr(official_runtime, "_run_megasam_tracking", fake_run_megasam_tracking)
    monkeypatch.setattr(
        official_runtime,
        "_pose_vectors_to_camera_matrices",
        fake_pose_vectors_to_camera_matrices,
    )
    monkeypatch.setattr(official_runtime, "_score_camera_matrices", fake_score_camera_matrices)

    runtime = {"megasam_frame_stride": 2, "megasam_start_frame": 1, "megasam_max_frames": 2}
    output = official_runtime.compute_camera_error_megasam(
        "prediction.mp4",
        cameras_gt=cameras_gt,
        scale=2.0,
        normalization={
            "empirical_min": [0.0, 0.0],
            "empirical_max": [10.0, 1.0],
            "higher_is_better": [False, False],
        },
        frame_paths=frame_paths,
        runtime=runtime,
    )

    assert output["backend"] == "megasam"
    assert output["raw"] == {"rotation_error": 2.0, "translation_error": 0.1}
    assert output["normalized"] > 0.0
    assert output["details"]["megasam_scene"] == "unit_scene"
    assert output["details"]["megasam_pose_count"] == 2
    assert output["details"]["megasam_frame_stride"] == 2
    assert output["details"]["megasam_start_frame"] == 1
    assert output["details"]["megasam_max_frames"] == 2
    assert output["details"]["megasam_source_resolution"] == [64, 32]
    assert output["details"]["camera_gt_alignment"] == "source_indices"
    assert output["details"]["camera_gt_first_index"] == 1
    assert output["details"]["camera_gt_last_index"] == 3
    assert output["details"]["megasam_intrinsics_shape"] == [1, 4]
    assert output["details"]["megasam_intrinsics_first"] == [50.0, 51.0, 32.0, 16.0]
    assert captured["runtime"] == runtime
    assert captured["pose_shape"] == (2, 7)
    assert captured["pred_shape"] == (2, 4, 4)
    assert captured["selected_z"] == [1.0, 3.0]
    assert captured["scale"] == 2.0
