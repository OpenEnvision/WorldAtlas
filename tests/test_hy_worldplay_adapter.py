from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common import checkpoints
from worldarena.models.adapters.hy_worldplay import (
    HYWorldPlayAdapter,
    _resolved_pose_json_path,
    _validated_video_length,
    _worldplay_camera_delta,
)
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter


def _patch_workspace(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    workspace_root = tmp_path / "WorldArena"
    ckpt_root = workspace_root / "ckpt"
    monkeypatch.setattr(checkpoints, "PROJECT_ROOT", workspace_root)
    monkeypatch.setattr(checkpoints, "CANONICAL_CHECKPOINT_ROOT", ckpt_root)
    return workspace_root, ckpt_root


def _make_config(tmp_path: Path, monkeypatch) -> ModelRuntimeConfig:
    workspace_root, ckpt_root = _patch_workspace(monkeypatch, tmp_path)
    config_path = workspace_root / "config" / "models" / "hy_worldplay.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")

    repo_root = workspace_root / "thirdparty" / "HY-WorldPlay"
    repo_root.mkdir(parents=True, exist_ok=True)
    weights_root = ckpt_root / "HY-WorldPlay"
    weights_root.mkdir(parents=True, exist_ok=True)
    action_ckpt = (
        ckpt_root
        / "HY-WorldPlay"
        / "ar_distilled_action_model"
        / "model.safetensors"
    )
    action_ckpt.parent.mkdir(parents=True, exist_ok=True)
    action_ckpt.write_bytes(b"ckpt")

    return ModelRuntimeConfig(
        config_path=config_path,
        name="hy-worldplay",
        family="hy_worldplay",
        python_bin="python",
        repo_root=repo_root,
        checkpoint_dir=weights_root,
        entrypoint="hyvideo/generate.py",
        output_ext=".mp4",
        supported_suites=["image_static", "image_dynamic", "video_static", "video_dynamic"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        control_signals=["text", "image", "trajectory", "camera_pose"],
        generation={
            "action_ckpt": str(action_ckpt),
            "video_length": 125,
            "width": 832,
            "height": 480,
            "resolution": "480p",
            "model_type": "ar",
            "few_step": True,
            "num_inference_steps": 4,
            "launcher": "python",
            "gpu_index": 0,
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
        prompt_current="A still dining room image.",
        prompt_target="Camera push in. A still dining room scene. Everything in the scene remains completely motionless.",
        style="photorealistic",
        environment="indoor",
        scene="dining space",
        motion_category=None if suite == "image_static" else "articulated",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=camera_path,
        generation_mode=generation_mode,
        annotation_path=f"/tmp/annotations.json::{suite}/sample.png",
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
        prompt_target="Camera fixed. A static dining room scene.",
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


def test_resolved_pose_json_path_uses_video_annotation(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)
    annotation_path = tmp_path / "annotations" / "sample_ann"
    annotation_path.mkdir(parents=True)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], repeats=32, axis=0)
    intrinsics = np.repeat(
        np.asarray([[500.0, 500.0, 416.0, 240.0]], dtype=np.float32),
        repeats=32,
        axis=0,
    )
    np.save(annotation_path / "poses.npy", poses)
    np.save(annotation_path / "intrinsics.npy", intrinsics)
    sample = _make_video_sample(annotation_path)

    pose_json_path, pose_source = _resolved_pose_json_path(
        config,
        sample,
        latent_count=32,
        image_width=832,
        image_height=480,
    )

    payload = json.loads(pose_json_path.read_text(encoding="utf-8"))
    assert pose_source == "annotation"
    assert len(payload) == 32
    assert payload["0"]["K"] == [
        [500.0, 0.0, 416.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
    assert payload["0"]["extrinsic"] == np.eye(4, dtype=np.float32).tolist()


def test_resolved_pose_json_path_synthesizes_image_camera_path(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)
    sample = _make_image_sample(
        suite="image_static",
        generation_mode="static",
        camera_path=["push_in", "move_right"],
    )

    pose_json_path, pose_source = _resolved_pose_json_path(
        config,
        sample,
        latent_count=32,
        image_width=832,
        image_height=480,
    )

    payload = json.loads(pose_json_path.read_text(encoding="utf-8"))
    first_pose = np.asarray(payload["0"]["extrinsic"], dtype=np.float32)
    last_pose = np.asarray(payload[str(len(payload) - 1)]["extrinsic"], dtype=np.float32)

    assert pose_source == "synthetic"
    assert len(payload) == 32
    np.testing.assert_allclose(first_pose, np.eye(4, dtype=np.float32))
    assert not np.allclose(last_pose, first_pose)
    assert float(last_pose[0, 3]) > 0.0
    assert abs(float(last_pose[2, 3])) > 0.0


@pytest.mark.parametrize(
    ("camera_token", "expected_x_sign", "expected_z_sign", "expected_yaw_sign"),
    [
        ("pan_left", -1, 1, 1),
        ("pan_right", 1, 1, -1),
        ("orbit_left", -1, -1, -1),
        ("orbit_right", 1, -1, 1),
    ],
)
def test_worldplay_camera_delta_matches_user_direction_mapping(
    camera_token: str,
    expected_x_sign: int,
    expected_z_sign: int,
    expected_yaw_sign: int,
) -> None:
    delta = _worldplay_camera_delta(camera_token)
    assert np.sign(float(delta[0, 3])) == expected_x_sign
    assert np.sign(float(delta[2, 3])) == expected_z_sign
    assert np.sign(float(delta[2, 0])) == expected_yaw_sign


def test_build_model_adapter_returns_hy_worldplay_adapter(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)

    adapter = build_model_adapter(config)

    assert isinstance(adapter, HYWorldPlayAdapter)


def test_validated_video_length_rejects_invalid_worldplay_lengths() -> None:
    with pytest.raises(ValueError):
        _validated_video_length(81)


def test_generate_builds_runner_command_and_writes_output(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)
    adapter = HYWorldPlayAdapter(config)
    sample = _make_image_sample(
        suite="image_dynamic",
        generation_mode="dynamic",
        camera_path=["fixed"],
    )
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (832, 480), color="white").save(conditioning_image)
    output_path = tmp_path / "predictions" / "sample.mp4"
    observed: dict[str, object] = {}

    def _fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd
        del env
        observed["command"] = command
        output_index = command.index("--output_path") + 1
        Path(command[output_index]).write_bytes(b"mp4")

    monkeypatch.setattr("worldarena.models.adapters.hy_worldplay.run_command", _fake_run_command)

    payload = adapter.generate(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_target,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert output_path.exists()
    assert "-m" in command
    assert "worldarena.models.adapters.hy_worldplay_runner" in command
    assert "--pose_json" in command
    assert "--model_path" in command
    assert "--action_ckpt" in command
    assert payload["prediction_path"] == str(output_path)
    assert payload["pose_source"] == "synthetic"
