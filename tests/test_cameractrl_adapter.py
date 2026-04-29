from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.cameractrl import (
    CameraCtrlAdapter,
    _cameractrl_camera_delta,
    _write_cameractrl_trajectory,
)
from worldarena.models.config import ModelRuntimeConfig, load_model_config
from worldarena.models.registry import build_model_adapter


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    repo_root = tmp_path / "CameraCtrl"
    repo_root.mkdir()
    model_config = repo_root / "config.yaml"
    model_config.write_text("model: {}\n", encoding="utf-8")
    ckpt_root = tmp_path / "ckpt"
    ckpt_root.mkdir()
    (ckpt_root / "CameraCtrl.ckpt").write_bytes(b"ckpt")
    sd_root = tmp_path / "sd15"
    sd_root.mkdir()
    motion = tmp_path / "motion.ckpt"
    motion.write_bytes(b"motion")
    return ModelRuntimeConfig(
        config_path=tmp_path / "cameractrl.yaml",
        name="cameractrl",
        family="cameractrl",
        python_bin="python",
        repo_root=repo_root,
        checkpoint_dir=ckpt_root,
        entrypoint="inference.py",
        output_ext=".mp4",
        supported_suites=["image_static"],
        prompt_mode="prompt_target",
        reference_mode="none",
        control_signals=["text", "camera_pose"],
        generation={
            "ori_model_path": str(sd_root),
            "unet_subfolder": "unet_webvidlora_v3",
            "motion_module_ckpt": str(motion),
            "pose_adaptor_ckpt": "CameraCtrl.ckpt",
            "model_config": str(model_config),
            "image_height": 256,
            "image_width": 384,
            "video_length": 16,
            "original_pose_width": 1280,
            "original_pose_height": 720,
            "target_duration_sec": None,
        },
    )


def _make_sample(camera_path: list[str]) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="image-static-001",
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
        prediction_stem="image_static_sample",
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
        ("pan_left", -1, 1, 1),
        ("pan_right", 1, 1, -1),
        ("orbit_left", -1, -1, -1),
        ("orbit_right", 1, -1, 1),
    ],
)
def test_cameractrl_camera_delta_matches_user_mapping(
    camera_token: str,
    expected_x_sign: int,
    expected_z_sign: int,
    expected_yaw_sign: int,
) -> None:
    delta = _cameractrl_camera_delta(camera_token, {})

    assert np.sign(delta[0, 3]) == expected_x_sign
    assert np.sign(delta[2, 3]) == expected_z_sign
    assert np.sign(delta[2, 0]) == expected_yaw_sign


def test_write_cameractrl_trajectory_uses_worldarena_camera_path(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.txt"

    _write_cameractrl_trajectory(
        trajectory,
        camera_path=["pan_left", "orbit_right"],
        frame_count=16,
        image_width=384,
        image_height=256,
        generation={},
    )

    lines = trajectory.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "worldarena://synthetic-camera-trajectory"
    assert len(lines) == 17
    assert all(len(line.split()) == 19 for line in lines[1:])


def test_cameractrl_config_and_registry_load() -> None:
    config = load_model_config(Path("config/models/cameractrl.yaml"))
    adapter = build_model_adapter(config)

    assert isinstance(adapter, CameraCtrlAdapter)
    assert config.prompt_mode == "prompt_target"
    assert config.reference_mode == "none"
    assert config.output_ext == ".mp4"
    assert config.generation["unet_subfolder"] == "unet_webvidlora_v3"
    assert config.generation["original_pose_width"] == 1280
    assert config.generation["original_pose_height"] == 720
    assert config.generation["target_duration_sec"] is None


def test_cameractrl_batch_invokes_single_persistent_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    adapter = CameraCtrlAdapter(config)
    sample = _make_sample(["orbit_left"])
    output_path = tmp_path / "out.mp4"
    calls: list[list[str]] = []

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd, env
        calls.append(command)
        results_file = Path(command[command.index("--results_file") + 1])
        request_file = Path(command[command.index("--request_file") + 1])
        assert command[command.index("--unet_subfolder") + 1] == "unet_webvidlora_v3"
        assert command[command.index("--original_pose_width") + 1] == "1280"
        assert command[command.index("--original_pose_height") + 1] == "720"
        assert "--target_duration_sec" not in command
        payload = json.loads(request_file.read_text(encoding="utf-8"))
        assert len(payload["requests"]) == 1
        assert payload["requests"][0]["camera_path"] == ["orbit_left"]
        output_path.write_bytes(b"mp4")
        results_file.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "sample_id": sample.sample_id,
                            "status": "generated",
                            "prediction_path": str(output_path),
                            "prompt": sample.prompt_target,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("worldarena.models.adapters.cameractrl.run_command", fake_run_command)

    results = adapter.generate_batch(
        [
            type(
                "Request",
                (),
                {
                    "sample": sample,
                    "conditioning_image": tmp_path / "unused",
                    "output_path": output_path,
                    "prompt": sample.prompt_target,
                },
            )()
        ]
    )

    assert len(calls) == 1
    assert results[sample.sample_id]["status"] == "generated"
