from __future__ import annotations

from pathlib import Path

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.hunyuan_gamecraft import (
    HunyuanGameCraftAdapter,
    _action_payload_for_sample,
    _action_plan_for_suite,
)
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "models" / "hunyuan_gamecraft.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    return ModelRuntimeConfig(
        config_path=config_path,
        name="hunyuan-gamecraft",
        family="hunyuan_gamecraft",
        python_bin="python",
        repo_root=tmp_path / "thirdparty" / "Hunyuan-GameCraft-1.0",
        checkpoint_dir=tmp_path / "weights",
        entrypoint="hymm_sp/sample_batch.py",
        output_ext=".mp4",
        supported_suites=["image_static", "video_static", "video_dynamic"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        control_signals=["text", "image", "actions"],
        generation={
            "num_gpus": 1,
            "cpu_offload": True,
            "disable_sp": True,
            "action_speed": 0.3,
            "action_plan": {
                "default": ["w", "d", "w"],
                "by_suite": {
                    "video_dynamic": ["w", "a", "w", "d"],
                },
                "speed_default": [0.2],
                "speed_by_suite": {
                    "video_dynamic": [0.1, 0.2, 0.3, 0.4],
                },
            },
        },
    )


def _make_sample(suite: str = "video_dynamic", camera_path: list[str] | None = None) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-001",
        suite=suite,
        split="test",
        asset_level="video",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="dynamic/photorealistic/videos/gameplay",
        path=f"/dataset/{suite}.mp4",
        relative_path=f"{suite}/sample.mp4",
        reference_path=f"/dataset/{suite}.mp4",
        prediction_stem=f"{suite}_sample",
        conditioning_strategy="first_frame",
        prompt_current="A still gameplay frame.",
        prompt_target="The player moves through the game world.",
        style="photorealistic",
        environment="outdoor",
        scene="gameplay",
        motion_category="camera",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=list(camera_path or []),
    )


def test_action_plan_for_suite_uses_suite_specific_speeds() -> None:
    actions, speeds = _action_plan_for_suite(
        {
            "action_speed": 0.3,
            "action_plan": {
                "default": ["w", "d", "w"],
                "by_suite": {"video_dynamic": ["w", "a", "w", "d"]},
                "speed_default": [0.2],
                "speed_by_suite": {"video_dynamic": [0.1, 0.2, 0.3, 0.4]},
            },
        },
        "video_dynamic",
    )

    assert actions == ["w", "a", "w", "d"]
    assert speeds == [0.1, 0.2, 0.3, 0.4]


def test_gamecraft_camera_path_maps_to_native_action_space(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.generation["action_source"] = "worldarena_camera_path"
    sample = _make_sample(
        suite="image_static",
        camera_path=["push_in", "pan_right", "orbit_left", "fixed"],
    )

    payload = _action_payload_for_sample(config.generation, sample)

    assert payload["actions"] == ["w", "right_rot", "right_rot", "w"]
    assert payload["action_speeds"] == [0.3, 0.3, 0.3, 0.0]
    assert payload["camera_tokens"] == ["push_in", "pan_right", "orbit_left", "fixed"]
    assert payload["control_source"] == "coarse_camera_path"
    assert payload["approximations"] == [
        {
            "camera_token": "orbit_left",
            "native_action": "right_rot",
            "type": "coarse_orbit_to_counter_rotation",
        }
    ]


def test_registry_builds_hunyuan_gamecraft_adapter(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    adapter = build_model_adapter(config)

    assert isinstance(adapter, HunyuanGameCraftAdapter)


def test_hunyuan_gamecraft_generate_builds_runner_command(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.repo_root.mkdir(parents=True, exist_ok=True)
    conditioning_image = tmp_path / "conditioning.png"
    conditioning_image.write_bytes(b"image")
    output_path = tmp_path / "prediction.mp4"
    sample = _make_sample()
    recorded: dict[str, object] = {}

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["env"] = env
        output_path.write_bytes(b"video")

    monkeypatch.setattr("worldarena.models.adapters.hunyuan_gamecraft.run_command", fake_run_command)

    adapter = HunyuanGameCraftAdapter(config)
    result = adapter.generate(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_target,
    )

    command = recorded["command"]
    assert isinstance(command, list)
    assert "worldarena.models.adapters.hunyuan_gamecraft_runner" in command
    assert "--action_list" in command
    assert "--action_speed_list" in command
    assert result["actions"] == ["w", "a", "w", "d"]
    assert result["action_speeds"] == [0.1, 0.2, 0.3, 0.4]
    assert Path(result["action_spec_path"]).exists()
    assert result["prediction_path"] == str(output_path)
