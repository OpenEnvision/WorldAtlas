from __future__ import annotations

from pathlib import Path

import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.matrix_game import MatrixGameAdapter, _build_action_payload
from worldarena.models.config import load_model_config


def _make_image_static_sample(*, camera_token: str) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"image-static-{camera_token}",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/test_scene",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/outdoor/test_scene/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem=f"image_static_{camera_token}",
        conditioning_strategy="reference_image",
        prompt_current="A static outdoor scene.",
        prompt_target=f"Camera {camera_token.replace('_', ' ')}. A static outdoor scene.",
        style="photorealistic",
        environment="outdoor",
        scene="test_scene",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=[camera_token],
    )


@pytest.mark.parametrize(
    ("camera_token", "expected_action"),
    [
        ("push_in", "forward"),
        ("pull_out", "back"),
        ("move_left", "left"),
        ("move_right", "right"),
        ("pan_left", "back_left_camera_l"),
        ("pan_right", "back_right_camera_r"),
        ("orbit_left", "forward_left_camera_r"),
        ("orbit_right", "forward_right_camera_l"),
    ],
)
def test_matrix_game_camera_path_maps_to_user_action_definition(
    camera_token: str,
    expected_action: str,
) -> None:
    config = load_model_config(Path("config/models/matrix_game_2.yaml"))
    sample = _make_image_static_sample(camera_token=camera_token)

    payload = _build_action_payload(
        config.generation,
        sample.suite,
        family=config.family,
        sample=sample,
    )

    assert payload["actions"] == [expected_action]
    assert payload["camera_token"] == camera_token
    assert payload["control_source"] == "camera_path"
    assert payload["prompt_target"] == sample.prompt_target


def test_active_matrix_game_2_config_uses_prompt_target() -> None:
    config = load_model_config(Path("config/models/matrix_game_2.yaml"))
    sample = _make_image_static_sample(camera_token="pan_left")
    adapter = MatrixGameAdapter(config)

    assert adapter.prompt_for(sample) == sample.prompt_target
    assert int(config.generation["num_output_frames"]) == 150
    assert int(config.generation["output_fps"]) == 12


def test_matrix_game_2_multi_segment_camera_path_spans_full_control_sequence() -> None:
    config = load_model_config(Path("config/models/matrix_game_2.yaml"))
    sample = _make_image_static_sample(camera_token="pan_right")
    sample.camera_path = ["pan_right", "move_right", "pull_out"]

    payload = _build_action_payload(
        config.generation,
        sample.suite,
        family=config.family,
        sample=sample,
    )

    assert payload["actions"] == ["back_right_camera_r", "right", "back"]
    assert payload["total_frames"] == 597
    assert len(payload["keyboard_condition"]) == 597
    assert len(payload["mouse_condition"]) == 597
    assert payload["keyboard_condition"][0][1] == 1.0
    assert payload["keyboard_condition"][0][3] == 1.0
    assert payload["mouse_condition"][0] == pytest.approx([0.0, 0.1])
    assert payload["keyboard_condition"][199][1] == 0.0
    assert payload["keyboard_condition"][199][3] == 1.0
    assert payload["mouse_condition"][199] == pytest.approx([0.0, 0.0])
    assert payload["keyboard_condition"][398][1] == 1.0
    assert payload["keyboard_condition"][398][3] == 0.0
    assert payload["keyboard_condition"][596][1] == 1.0
