from __future__ import annotations

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.control import CameraErrorMetric
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.synthetic_camera import (
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
