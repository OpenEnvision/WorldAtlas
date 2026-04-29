from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.recammaster import (
    ReCamMasterAdapter,
    _RECAMMASTER_IMAGE_CONTRACT,
    _recammaster_camera_pose_sequence,
    _request_payload_for,
    _write_recammaster_camera_pose,
)
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


def _make_sample(camera_path: list[str]) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="image-static-recam-001",
        suite="image_static",
        split="test_hidden",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/test_scene",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/outdoor/test_scene/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem="image_static_recammaster_sample",
        conditioning_strategy="reference_image",
        prompt_current="A static outdoor scene.",
        prompt_target="Camera orbit left. A static outdoor scene.",
        style="photorealistic",
        environment="outdoor",
        scene="test scene",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=camera_path,
    )


@pytest.mark.parametrize(
    ("camera_token", "expected_x_sign", "expected_z_sign", "expected_yaw_sign"),
    [
        ("push_in", 0, 1, 0),
        ("pull_out", 0, -1, 0),
        ("move_left", -1, 0, 0),
        ("move_right", 1, 0, 0),
        ("pan_left", -1, -1, -1),
        ("pan_right", 1, -1, 1),
        ("orbit_left", -1, 1, 1),
        ("orbit_right", 1, 1, -1),
    ],
)
def test_recammaster_camera_path_maps_to_user_definition(
    camera_token: str,
    expected_x_sign: int,
    expected_z_sign: int,
    expected_yaw_sign: int,
) -> None:
    poses = _recammaster_camera_pose_sequence(
        [camera_token],
        pose_count=21,
        generation={"forward_step": 2.0, "lateral_step": 2.0, "rotation_degrees": 30.0},
    ).reshape(21, 3, 4)
    final = poses[-1]
    yaw_proxy = final[0, 2]

    assert np.sign(final[0, 3]) == expected_x_sign
    assert np.sign(final[2, 3]) == expected_z_sign
    assert np.sign(yaw_proxy) == expected_yaw_sign


def test_write_recammaster_camera_pose_uses_multi_segment_path(tmp_path: Path) -> None:
    pose_path = tmp_path / "pose.json"

    _write_recammaster_camera_pose(
        pose_path,
        camera_path=["pan_left", "move_left", "pull_out"],
        pose_count=21,
        generation={"forward_step": 2.0, "lateral_step": 2.0, "rotation_degrees": 30.0},
    )

    payload = np.asarray(json.loads(pose_path.read_text(encoding="utf-8")), dtype=np.float32)
    poses = payload.reshape(21, 3, 4)
    assert payload.shape == (21, 12)
    assert poses[-1, 0, 3] < 0
    assert poses[-1, 2, 3] < 0
    assert poses[-1, 0, 2] < 0


def test_recammaster_config_and_registry_load() -> None:
    config = load_model_config(Path("config/models/recammaster.yaml"))
    adapter = build_model_adapter(config)

    assert isinstance(adapter, ReCamMasterAdapter)
    assert config.prompt_mode == "prompt_target"
    assert config.reference_mode == "reference_video"
    assert config.output_ext == ".mp4"
    assert config.supported_suites == ["image_static"]
    assert int(config.generation["num_frames"]) == 81
    assert int(config.generation["conditioning_video_frames"]) == 81
    assert int(config.generation["pose_frames"]) == 21
    assert config.metadata["image_conditioning_contract"] == _RECAMMASTER_IMAGE_CONTRACT


def test_recammaster_request_payload_records_prompt_target_and_source_video(tmp_path: Path) -> None:
    sample = _make_sample(["orbit_left"])
    request = PreparedGenerationRequest(
        sample=sample,
        conditioning_image=tmp_path / "source.mp4",
        output_path=tmp_path / "prediction.mp4",
        prompt=sample.prompt_target,
    )

    payload = _request_payload_for(
        request,
        generation={"pose_frames": 21, "forward_step": 2.0, "lateral_step": 2.0, "rotation_degrees": 30.0},
        runtime_root=tmp_path,
    )

    assert payload["prompt"] == sample.prompt_target
    assert payload["prompt_source"] == "prompt_target"
    assert payload["source_video_path"] == str(request.conditioning_image)
    assert payload["camera_path"] == ["orbit_left"]
    assert payload["control_source"] == "camera_pose_json"
    assert payload["image_conditioning_contract"] == _RECAMMASTER_IMAGE_CONTRACT
    pose = np.asarray(json.loads(Path(payload["camera_pose_file"]).read_text(encoding="utf-8")))
    assert pose.shape == (21, 12)
