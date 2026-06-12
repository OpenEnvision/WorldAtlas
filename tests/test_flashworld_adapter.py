from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.video_io import probe_video_fps, probe_video_frame_count, read_video_frame, write_video_frames
from worldarena.models.adapters.flashworld import _resolved_annotation_path
from worldarena.models.adapters.flashworld_runner import (
    _build_request_payload,
    flashworld_center_crop_resize_frame,
    flashworld_render_frame_count,
    write_flashworld_aligned_reference_video,
)
from worldarena.models.config import ModelRuntimeConfig


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "models" / "flashworld.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    return ModelRuntimeConfig(
        config_path=config_path,
        name="flashworld-static",
        family="flashworld",
        python_bin="python",
        repo_root=tmp_path,
        checkpoint_dir=tmp_path,
        entrypoint="cli.py",
        output_ext=".mp4",
        supported_suites=["video_static", "image_static"],
        prompt_mode="prompt_current",
        reference_mode="reference_image",
        control_signals=["text", "image", "camera_pose"],
        generation={
            "num_frames": 24,
            "image_height": 480,
            "image_width": 704,
            "synthetic_focal_scale": 0.6,
        },
    )


def _make_image_sample(*, suite: str, generation_mode: str | None, camera_path: list[str]) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-001",
        suite=suite,
        split="test",
        asset_level="image",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path=f"/dataset/{suite}.png",
        relative_path=f"{suite}/{suite}.png",
        reference_path=f"/dataset/{suite}.png",
        prediction_stem=suite,
        conditioning_strategy="reference_image",
        prompt_current="A reference image of a dining room.",
        prompt_target="Camera push in. A dining room scene. Everything in the scene remains completely motionless.",
        style="photorealistic",
        environment="indoor",
        scene="dining space",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=camera_path,
        generation_mode=generation_mode,
        annotation_path=f"/tmp/image_annotations.json::{suite}/sample.png",
        has_annotation=True,
        has_pose=False,
    )


def _make_video_sample(annotation_path: Path) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="video-static-001",
        suite="video_static",
        split="test",
        asset_level="video",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path=str(annotation_path.parent / "sample.mp4"),
        relative_path="video_static/sample.mp4",
        reference_path=str(annotation_path.parent / "sample.mp4"),
        prediction_stem="video_static",
        conditioning_strategy="first_frame",
        prompt_current="A static dining room scene.",
        prompt_target="A dining room scene.",
        style="photorealistic",
        environment="indoor",
        scene="dining space",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        annotation_path=str(annotation_path),
        has_annotation=True,
        has_pose=True,
    )


def test_resolved_annotation_path_synthesizes_image_controls(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    sample = _make_image_sample(
        suite="image_static",
        generation_mode="static",
        camera_path=["push_in", "move_right"],
    )
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (800, 600), color="white").save(conditioning_image)

    resolved = _resolved_annotation_path(config, sample, conditioning_image, frame_count=24)

    assert str(resolved).startswith(str(tmp_path / "cache"))
    poses = np.load(resolved / "poses.npy")
    intrinsics = np.load(resolved / "intrinsics.npy")
    assert poses.shape == (24, 4, 4)
    assert intrinsics.shape == (24, 4)
    np.testing.assert_allclose(poses[0], np.eye(4, dtype=np.float32))
    assert not np.allclose(poses[-1], poses[0])
    np.testing.assert_allclose(
        intrinsics[0],
        np.asarray([480.0, 480.0, 400.0, 300.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_resolved_annotation_path_preserves_real_video_annotation(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    annotation_path = tmp_path / "annotations" / "sample_ann"
    annotation_path.mkdir(parents=True)
    np.save(annotation_path / "poses.npy", np.repeat(np.eye(4, dtype=np.float32)[None, ...], repeats=24, axis=0))
    np.save(
        annotation_path / "intrinsics.npy",
        np.repeat(np.asarray([[500.0, 500.0, 352.0, 240.0]], dtype=np.float32), repeats=24, axis=0),
    )
    sample = _make_video_sample(annotation_path)
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (704, 480), color="white").save(conditioning_image)

    resolved = _resolved_annotation_path(config, sample, conditioning_image, frame_count=24)

    assert resolved == annotation_path


def test_build_request_payload_denormalizes_spatialvid_intrinsics(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations" / "sample_ann"
    annotation_path.mkdir(parents=True)
    poses = np.repeat(
        np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        repeats=19,
        axis=0,
    )
    intrinsics = np.repeat(
        np.asarray([[0.44, 0.7822222, 0.5, 0.5]], dtype=np.float32),
        repeats=19,
        axis=0,
    )
    np.save(annotation_path / "poses.npy", poses)
    np.save(annotation_path / "intrinsics.npy", intrinsics)
    (annotation_path / "indexes.txt").write_text(
        "\n".join(f"{index} {index * 5}" for index in range(19)),
        encoding="utf-8",
    )
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (1280, 720), color="white").save(conditioning_image)

    payload = _build_request_payload(
        annotation_path=str(annotation_path),
        conditioning_image=conditioning_image,
        prompt="A dining room.",
        num_frames=24,
        image_height=480,
        image_width=704,
        image_index=0,
    )

    assert payload["resolution"] == [24, 480, 704]
    assert payload["image_prompt"] == str(conditioning_image)
    cameras = payload["cameras"]
    assert isinstance(cameras, list)
    assert len(cameras) == 24
    first_camera = cameras[0]
    assert first_camera["fx"] == pytest.approx(563.2, abs=1e-4)
    assert first_camera["fy"] == pytest.approx(563.2, abs=1e-4)
    assert first_camera["cx"] == pytest.approx(640.0, abs=1e-4)
    assert first_camera["cy"] == pytest.approx(360.0, abs=1e-4)


def test_flashworld_center_crop_resize_matches_official_crop() -> None:
    frame = np.zeros((6, 10, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(10, dtype=np.uint8)[None, :]

    aligned = flashworld_center_crop_resize_frame(
        frame,
        image_height=6,
        image_width=6,
    )

    assert aligned.shape == (6, 6, 3)
    np.testing.assert_array_equal(aligned[0, :, 0], np.arange(2, 8, dtype=np.uint8))


def test_write_flashworld_aligned_reference_video_uses_official_frame_count(tmp_path: Path) -> None:
    input_video = tmp_path / "source.mp4"
    output_video = tmp_path / "aligned.mp4"
    frames = [
        np.full((60, 80, 3), fill_value=index * 40, dtype=np.uint8)
        for index in range(4)
    ]
    write_video_frames(input_video, frames, fps=25.0)
    target_frame_count = flashworld_render_frame_count(num_frames=3, video_fps=3)

    summary = write_flashworld_aligned_reference_video(
        input_video=input_video,
        output_path=output_video,
        image_height=40,
        image_width=40,
        target_frame_count=target_frame_count,
        video_fps=3,
    )

    assert summary["source_frame_count"] == 4
    assert summary["target_frame_count"] == 7
    assert summary["codec"] == "h264"
    assert probe_video_frame_count(output_video) == 7
    assert probe_video_fps(output_video) == pytest.approx(3.0, abs=0.1)
    assert read_video_frame(output_video).shape == (40, 40, 3)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "h264"


def test_resolved_annotation_path_rejects_dynamic_image_samples(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    sample = _make_image_sample(
        suite="image_dynamic",
        generation_mode="dynamic",
        camera_path=["fixed"],
    )
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (800, 600), color="white").save(conditioning_image)

    with pytest.raises(FileNotFoundError):
        _resolved_annotation_path(config, sample, conditioning_image, frame_count=24)
