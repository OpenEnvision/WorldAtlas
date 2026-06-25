from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.luciddreamer import (
    LucidDreamerAdapter,
    _generation_for_sample,
    _generation_for_suite,
)
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "models" / "luciddreamer.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")

    repo_root = tmp_path / "LucidDreamer"
    repo_root.mkdir(parents=True, exist_ok=True)

    return ModelRuntimeConfig(
        config_path=config_path,
        name="luciddreamer",
        family="luciddreamer",
        python_bin="python",
        repo_root=repo_root,
        checkpoint_dir=None,
        entrypoint="run.py",
        output_ext=".mp4",
        supported_suites=["image_static", "video_static"],
        prompt_mode="prompt_current",
        reference_mode="first_frame",
        control_signals=["text", "image"],
        generation={
            "campath_gen": "lookdown",
            "campath_render": "llff",
            "seed": 1,
            "diff_steps": 50,
            "negative_prompt": "",
            "campath_render_by_suite": {
                "video_static": "headbanging",
            },
        },
    )


def _make_sample(suite: str = "video_static", camera_path: list[str] | None = None) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-001",
        suite=suite,
        split="test",
        asset_level="video" if suite == "video_static" else "image",
        modality="video" if suite == "video_static" else "image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path=f"/dataset/{suite}.mp4" if suite == "video_static" else f"/dataset/{suite}.png",
        relative_path=f"{suite}/sample",
        reference_path=f"/dataset/{suite}.mp4" if suite == "video_static" else f"/dataset/{suite}.png",
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
    )


def test_generation_for_suite_applies_by_suite_overrides() -> None:
    resolved = _generation_for_suite(
        {
            "campath_render": "llff",
            "campath_render_by_suite": {"video_static": "headbanging"},
        },
        "video_static",
    )

    assert resolved["campath_render"] == "headbanging"


def test_generation_for_sample_maps_worldarena_camera_path_to_render_preset() -> None:
    sample = _make_sample("image_static", camera_path=["pull_out", "pan_left"])

    resolved = _generation_for_sample(
        {
            "campath_render": "llff",
            "use_camera_path_render": True,
            "campath_render_by_camera_token": {
                "pull_out": "back_and_forth",
                "pan_left": "headbanging",
            },
        },
        sample,
    )

    assert resolved["campath_render"] == "back_and_forth"
    assert resolved["_camera_path"] == ["pull_out", "pan_left"]
    assert resolved["_camera_path_primary"] == "pull_out"
    assert resolved["_campath_render_source"] == "sample_camera_path"


def test_build_model_adapter_returns_luciddreamer_adapter(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    adapter = build_model_adapter(config)

    assert isinstance(adapter, LucidDreamerAdapter)


def test_generate_builds_runner_command_and_records_sidecars(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    adapter = LucidDreamerAdapter(config)
    sample = _make_sample()
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (512, 512), color="white").save(conditioning_image)
    output_path = tmp_path / "predictions" / "sample.mp4"
    observed: dict[str, object] = {}

    def _fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd
        del env
        observed["command"] = command
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        output_path.with_suffix(".ply").write_bytes(b"ply")
        output_path.with_name(f"{output_path.stem}.depth{output_path.suffix}").write_bytes(b"depth")

    monkeypatch.setattr("worldarena.models.adapters.luciddreamer.run_command", _fake_run_command)

    payload = adapter.generate(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_current,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert output_path.exists()
    assert "-m" in command
    assert "worldarena.models.adapters.luciddreamer_runner" in command
    assert "--campath_render" in command
    assert "headbanging" in command
    assert payload["gaussian_splat_path"] == str(output_path.with_suffix(".ply"))
    assert payload["depth_video_path"] == str(
        output_path.with_name(f"{output_path.stem}.depth{output_path.suffix}")
    )


def test_generate_batch_builds_persistent_runner_command(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    config.python_bin = "/lucid/bin/python"
    config.generation["use_camera_path_render"] = True
    config.generation["campath_render_by_camera_token"] = {"pan_left": "headbanging"}
    adapter = LucidDreamerAdapter(config)
    sample = _make_sample("image_static", camera_path=["pan_left"])
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (512, 512), color="white").save(conditioning_image)
    output_path = tmp_path / "predictions" / "sample.mp4"
    observed: dict[str, object] = {}

    def _fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del env
        observed["command"] = command
        observed["cwd"] = cwd
        requests_path = Path(command[command.index("--requests_jsonl") + 1])
        results_path = Path(command[command.index("--results_jsonl") + 1])
        requests = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines()]
        assert len(requests) == 1
        assert requests[0]["sample_id"] == sample.sample_id
        assert requests[0]["min_seconds"] == 5.0
        assert requests[0]["input_size"] == 512
        assert requests[0]["camera_path"] == ["pan_left"]
        assert requests[0]["camera_path_primary"] == "pan_left"
        assert requests[0]["campath_render"] == "headbanging"
        assert requests[0]["campath_render_source"] == "sample_camera_path"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mp4")
        results_path.write_text(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "status": "generated",
                    "prediction_path": str(output_path),
                    "prompt": requests[0]["prompt"],
                    "video_duration_seconds": 6.667,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("worldarena.models.adapters.luciddreamer.run_command", _fake_run_command)

    payloads = adapter.generate_batch(
        [
            PreparedGenerationRequest(
                sample=sample,
                conditioning_image=conditioning_image,
                output_path=output_path,
                prompt=sample.prompt_current,
            )
        ]
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == "/lucid/bin/python"
    assert command[1] == "-u"
    assert command[2].endswith("luciddreamer_batch_runner.py")
    assert observed["cwd"] == config.repo_root
    assert payloads[sample.sample_id]["status"] == "generated"
    assert payloads[sample.sample_id]["video_duration_seconds"] == 6.667
