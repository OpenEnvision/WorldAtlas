from __future__ import annotations

from pathlib import Path

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.yume import YumeAdapter, _aligned_prompt_for_sample
from worldarena.models.config import ModelRuntimeConfig


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "models" / "yume.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    return ModelRuntimeConfig(
        config_path=config_path,
        name="yume",
        family="yume",
        python_bin="python",
        repo_root=tmp_path / "thirdparty" / "YUME",
        checkpoint_dir=tmp_path / "weights",
        entrypoint="webapp_single_gpu.py",
        output_ext=".mp4",
        supported_suites=["image_static", "image_dynamic"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        control_signals=["text", "image", "actions"],
        generation={
            "mode": "I2V",
            "action_source": "worldarena_camera_path",
            "worldarena_action_speed_cue": "The motion speed is moderate.",
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
        category_path="static/photorealistic/indoor/test_scene",
        path="/dataset/reference.png",
        relative_path="static/photorealistic/indoor/test_scene/reference.png",
        reference_path="/dataset/reference.png",
        prediction_stem="image_static_001",
        conditioning_strategy="reference_image",
        prompt_current="A static indoor scene.",
        prompt_target="Camera orbit left. A static indoor scene remains still.",
        style="photorealistic",
        environment="indoor",
        scene="test_scene",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=list(camera_path),
    )


def test_yume_camera_path_maps_to_text_action_prompt(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    sample = _make_sample(["push_in", "orbit_right"])

    prompt, payload = _aligned_prompt_for_sample(config.generation, sample, sample.prompt_target)

    assert prompt.startswith(
        "Person moves forward. Person moves right while turning left. The motion speed is moderate."
    )
    assert payload["camera_tokens"] == ["push_in", "orbit_right"]
    assert payload["control_source"] == "coarse_camera_path_text"
    assert payload["text_actions"] == [
        "Person moves forward.",
        "Person moves right while turning left.",
    ]


def test_yume_generate_records_action_alignment(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.repo_root.mkdir(parents=True, exist_ok=True)
    conditioning_image = tmp_path / "conditioning.png"
    conditioning_image.write_bytes(b"image")
    output_path = tmp_path / "prediction.mp4"
    sample = _make_sample(["pan_left"])
    recorded: dict[str, object] = {}

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["env"] = env
        output_path.write_bytes(b"video")

    monkeypatch.setattr("worldarena.models.adapters.yume.run_command", fake_run_command)

    adapter = YumeAdapter(config)
    result = adapter.generate(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_target,
    )

    command = recorded["command"]
    assert isinstance(command, list)
    prompt_index = command.index("--prompt")
    assert command[prompt_index + 1].startswith("Person turns left. The motion speed is moderate.")
    assert result["text_actions"] == ["Person turns left."]
    assert result["control_source"] == "camera_path_text"
    assert Path(result["action_spec_path"]).exists()
