from __future__ import annotations

from pathlib import Path

from PIL import Image

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.luciddreamer import LucidDreamerAdapter, _generation_for_suite
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


def _make_sample(suite: str = "video_static") -> BenchmarkSample:
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
