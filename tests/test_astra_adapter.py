from __future__ import annotations

from pathlib import Path

from PIL import Image

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.astra import AstraAdapter, _astra_cam_type_for_sample, _generation_for_suite
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "models" / "astra.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")

    repo_root = tmp_path / "thirdparty" / "Astra"
    repo_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = tmp_path / "ckpt" / "EvanEternal--Astra"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wan_model_path = tmp_path / "ckpt" / "Wan-AI--Wan2.1-T2V-1.3B"
    wan_model_path.mkdir(parents=True, exist_ok=True)

    return ModelRuntimeConfig(
        config_path=config_path,
        name="astra",
        family="astra",
        python_bin="python",
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        entrypoint="scripts/infer_demo.py",
        output_ext=".mp4",
        supported_suites=["image_static", "image_dynamic", "video_static", "video_dynamic"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        control_signals=["text", "image", "camera_pose"],
        generation={
            "wan_model_path": str(wan_model_path),
            "cam_type": 1,
            "cam_type_from_sample": True,
            "start_frame": 0,
            "initial_condition_frames": 1,
            "frames_per_generation": 8,
            "total_frames_to_generate": 24,
            "max_history_frames": 100,
            "modality_type": "sekai",
            "camera_guidance_scale": 2.0,
            "text_guidance_scale": 1.0,
            "moe_num_experts": 3,
            "moe_top_k": 1,
            "cam_type_by_suite": {
                "video_static": 5,
            },
        },
    )


def _make_sample(*, suite: str = "image_static", camera_path: list[str] | None = None) -> BenchmarkSample:
    is_video = suite.startswith("video_")
    return BenchmarkSample(
        sample_id=f"{suite}-001",
        suite=suite,
        split="test",
        asset_level="video" if is_video else "image",
        modality="video" if is_video else "image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path=f"/dataset/{suite}.mp4" if is_video else f"/dataset/{suite}.png",
        relative_path=f"{suite}/sample",
        reference_path=f"/dataset/{suite}.mp4" if is_video else f"/dataset/{suite}.png",
        prediction_stem=f"{suite}_sample",
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
        camera_path=list(camera_path or []),
        generation_mode="static",
    )


def test_generation_for_suite_applies_by_suite_overrides() -> None:
    resolved = _generation_for_suite(
        {
            "cam_type": 1,
            "cam_type_by_suite": {"video_static": 5},
        },
        "video_static",
    )

    assert resolved["cam_type"] == 5


def test_astra_cam_type_uses_sample_camera_path() -> None:
    sample = _make_sample(camera_path=["push_in", "pan_left"])

    cam_type, source, camera_path = _astra_cam_type_for_sample(
        sample,
        {"cam_type": 1, "cam_type_from_sample": True},
    )

    assert cam_type == 4
    assert source == "sample_camera_path"
    assert camera_path == ["push_in", "pan_left"]


def test_build_model_adapter_returns_astra_adapter(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    adapter = build_model_adapter(config)

    assert isinstance(adapter, AstraAdapter)


def test_prompt_for_all_suites_uses_prompt_target(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    adapter = AstraAdapter(config)
    for suite in ("image_static", "image_dynamic", "video_static", "video_dynamic"):
        sample = _make_sample(suite=suite, camera_path=["fixed"])
        prompt = adapter.prompt_for(sample)
        assert prompt == sample.prompt_target


def test_generate_builds_runner_command_and_writes_output(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    adapter = AstraAdapter(config)
    sample = _make_sample(suite="video_static", camera_path=["push_in", "pan_left"])
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (832, 480), color="white").save(conditioning_image)
    output_path = tmp_path / "predictions" / "sample.mp4"
    observed: dict[str, object] = {}

    def _fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd
        del env
        observed["command"] = command
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")

    monkeypatch.setattr("worldarena.models.adapters.astra.run_command", _fake_run_command)

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
    assert "worldarena.models.adapters.astra_runner" in command
    assert "--checkpoint_dir" in command
    assert "--wan_model_path" in command
    assert "--cam_type" in command
    assert "4" in command
    assert payload["prediction_path"] == str(output_path)
    assert payload["cam_type"] == 4
    assert payload["cam_type_source"] == "sample_camera_path"
