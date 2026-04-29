from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.motionctrl import (
    MotionCtrlAdapter,
    _MOTIONCTRL_IMAGE_CONTRACT,
    _motionctrl_camera_pose_sequence,
    _write_motionctrl_camera_pose,
)
from worldarena.models.config import ModelRuntimeConfig, load_model_config
from worldarena.models.registry import build_model_adapter


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    repo_root = tmp_path / "MotionCtrl"
    repo_root.mkdir()
    config_path = repo_root / "configs" / "inference" / "config_both.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    ckpt_root = repo_root / "checkpoints"
    ckpt_root.mkdir()
    (ckpt_root / "motionctrl.pth").write_bytes(b"ckpt")
    return ModelRuntimeConfig(
        config_path=tmp_path / "motionctrl.yaml",
        name="motionctrl",
        family="motionctrl",
        python_bin="python",
        repo_root=repo_root,
        checkpoint_dir=ckpt_root,
        entrypoint="main/evaluation/motionctrl_inference.py",
        output_ext=".mp4",
        supported_suites=["image_static", "image_dynamic"],
        prompt_mode="prompt_target",
        reference_mode="none",
        control_signals=["text", "camera_pose"],
        generation={
            "config_path": "configs/inference/config_both.yaml",
            "checkpoint": "motionctrl.pth",
            "height": 256,
            "width": 256,
            "frames": 16,
            "fps": 10.0,
            "n_samples": 1,
            "ddim_steps": 50,
            "ddim_eta": 1.0,
            "unconditional_guidance_scale": 7.5,
            "cond_T": 800,
            "seed": 1234,
            "translation_step": 1.5,
            "lateral_step": 1.0,
            "rotation_degrees": 30.0,
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
        ("push_in", 0, -1, 0),
        ("pull_out", 0, 1, 0),
        ("move_left", 1, 0, 0),
        ("move_right", -1, 0, 0),
        ("pan_left", 1, 1, 1),
        ("pan_right", -1, 1, -1),
        ("orbit_left", 1, -1, -1),
        ("orbit_right", -1, -1, 1),
    ],
)
def test_motionctrl_camera_path_maps_to_user_definition(
    camera_token: str,
    expected_x_sign: int,
    expected_z_sign: int,
    expected_yaw_sign: int,
) -> None:
    poses = _motionctrl_camera_pose_sequence(
        [camera_token],
        frame_count=16,
        generation={"translation_step": 1.5, "lateral_step": 1.0, "rotation_degrees": 30.0},
    ).reshape(16, 3, 4)
    final = poses[-1]
    yaw_proxy = final[0, 2]

    assert np.sign(final[0, 3]) == expected_x_sign
    assert np.sign(final[2, 3]) == expected_z_sign
    assert np.sign(yaw_proxy) == expected_yaw_sign


def test_write_motionctrl_camera_pose_uses_worldarena_camera_path(tmp_path: Path) -> None:
    pose_path = tmp_path / "pose.json"

    _write_motionctrl_camera_pose(
        pose_path,
        camera_path=["pan_left", "orbit_right"],
        frame_count=16,
        generation={"translation_step": 1.5, "lateral_step": 1.0, "rotation_degrees": 30.0},
    )

    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    assert len(payload) == 16
    assert all(len(row) == 12 for row in payload)
    assert payload[-1][3] == pytest.approx(0.0)
    assert payload[-1][11] == pytest.approx(0.0)


def test_motionctrl_config_and_registry_load() -> None:
    config = load_model_config(Path("config/models/motionctrl.yaml"))
    adapter = build_model_adapter(config)

    assert isinstance(adapter, MotionCtrlAdapter)
    assert config.prompt_mode == "prompt_target"
    assert config.reference_mode == "none"
    assert config.output_ext == ".mp4"
    assert config.supported_suites == ["image_static", "image_dynamic"]
    assert config.generation["frames"] == 16
    assert config.generation["fps"] == 10.0
    assert config.generation["ddim_steps"] == 50
    assert config.generation["unconditional_guidance_scale"] == 7.5
    assert "batch_size" not in config.generation
    assert config.metadata["image_conditioning_contract"] == _MOTIONCTRL_IMAGE_CONTRACT


def test_motionctrl_batch_invokes_single_runner_and_records_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    adapter = MotionCtrlAdapter(config)
    sample = _make_sample(["orbit_left"])
    output_path = tmp_path / "out.mp4"
    calls: list[list[str]] = []

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd, env
        calls.append(command)
        request_file = Path(command[command.index("--request-file") + 1])
        results_file = Path(command[command.index("--results-file") + 1])
        assert command[command.index("--ddim_steps") + 1] == "50"
        assert command[command.index("--unconditional_guidance_scale") + 1] == "7.5"
        payload = json.loads(request_file.read_text(encoding="utf-8"))
        assert len(payload["requests"]) == 1
        request = payload["requests"][0]
        assert request["prompt"] == sample.prompt_target
        assert request["prompt_source"] == "prompt_target"
        assert request["prompt_target"] == sample.prompt_target
        assert request["output_path"] == str(output_path)
        assert request["camera_path"] == ["orbit_left"]
        assert request["control_source"] == "camera_pose_json"
        assert request["image_conditioning_contract"] == _MOTIONCTRL_IMAGE_CONTRACT
        pose = np.asarray(json.loads(Path(request["camera_pose_file"]).read_text(encoding="utf-8")))
        assert pose.shape == (16, 12)
        assert pose[-1, 3] > 0
        assert pose[-1, 11] < 0
        output_path.write_bytes(b"mp4")
        results_file.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "sample_id": sample.sample_id,
                            "status": "generated",
                            "prediction_path": str(output_path),
                            "output_path": str(output_path),
                            "prompt": sample.prompt_target,
                            "prompt_source": "prompt_target",
                            "prompt_target": sample.prompt_target,
                            "camera_path": ["orbit_left"],
                            "camera_pose_file": request["camera_pose_file"],
                            "control_source": "camera_pose_json",
                            "image_conditioning_contract": _MOTIONCTRL_IMAGE_CONTRACT,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("worldarena.models.adapters.motionctrl.run_command", fake_run_command)

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
