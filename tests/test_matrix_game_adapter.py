from __future__ import annotations

from pathlib import Path

import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.matrix_game import MatrixGameAdapter, _build_action_payload, _matrix_game_3_command
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
@pytest.mark.parametrize(
    "config_path",
    [
        "config/models/matrix_game_2_5s.yaml",
        "config/models/matrix_game_3_5s.yaml",
    ],
)
def test_matrix_game_camera_path_maps_to_user_action_definition(
    config_path: str,
    camera_token: str,
    expected_action: str,
) -> None:
    config = load_model_config(Path(config_path))
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


@pytest.mark.parametrize(
    ("config_path", "expected_frames", "expected_fps"),
    [
        ("config/models/matrix_game_2_5s.yaml", 60, 12),
        ("config/models/matrix_game_3_5s.yaml", 85, 17),
    ],
)
def test_matrix_game_5s_configs_use_prompt_target(
    config_path: str,
    expected_frames: int,
    expected_fps: int,
) -> None:
    config = load_model_config(Path(config_path))
    sample = _make_image_static_sample(camera_token="pan_left")
    adapter = MatrixGameAdapter(config)

    assert adapter.prompt_for(sample) == sample.prompt_target
    assert int(config.generation["save_num_frames"]) == expected_frames
    fps_key = "output_fps" if config.family == "matrix_game_2" else "fps"
    assert int(config.generation[fps_key]) == expected_fps
    assert expected_frames / expected_fps == 5.0


def test_matrix_game3_multi_segment_camera_path_fits_saved_5s_frames() -> None:
    config = load_model_config(Path("config/models/matrix_game_3_5s.yaml"))
    sample = _make_image_static_sample(camera_token="pan_right")
    sample.camera_path = ["pan_right", "move_right", "pull_out"]

    payload = _build_action_payload(
        config.generation,
        sample.suite,
        family=config.family,
        sample=sample,
    )

    assert payload["actions"] == ["back_right_camera_r", "right", "back"]
    assert payload["total_frames"] == 97
    assert len(payload["keyboard_condition"]) == 97
    assert len(payload["mouse_condition"]) == 97
    assert payload["keyboard_condition"][0][1] == 1.0
    assert payload["keyboard_condition"][0][3] == 1.0
    assert payload["mouse_condition"][0] == pytest.approx([0.0, 0.1])
    assert payload["keyboard_condition"][28][1] == 0.0
    assert payload["keyboard_condition"][28][3] == 1.0
    assert payload["mouse_condition"][28] == pytest.approx([0.0, 0.0])
    assert payload["keyboard_condition"][56][1] == 1.0
    assert payload["keyboard_condition"][56][3] == 0.0
    assert payload["keyboard_condition"][84][1] == 1.0


def test_matrix_game3_command_passes_save_num_frames() -> None:
    config = load_model_config(Path("config/models/matrix_game_3_5s.yaml"))
    command = _matrix_game_3_command(
        python_bin=config.python_bin,
        repo_root=Path("/repo"),
        checkpoint_dir=Path("/ckpt"),
        action_spec_path=Path("/tmp/actions.json"),
        conditioning_image=Path("/tmp/image.png"),
        output_path=Path("/tmp/out/sample.mp4"),
        prompt="Camera fixed. A static scene.",
        generation=config.generation,
        action_payload={"num_iterations": 2},
    )

    flag_index = command.index("--save_num_frames")
    assert command[flag_index + 1] == "85"
