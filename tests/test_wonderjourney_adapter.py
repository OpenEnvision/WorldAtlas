from __future__ import annotations

from pathlib import Path

from PIL import Image

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common import checkpoints
from worldarena.models.adapters.wonderjourney import WonderJourneyAdapter
from worldarena.models.adapters.wonderjourney_runner import _runtime_entrypoint
from worldarena.models.adapters.wonderworld import _generation_for_suite
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter


def _patch_workspace(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    workspace_root = tmp_path / "WorldArena"
    ckpt_root = workspace_root / "ckpt"
    monkeypatch.setattr(checkpoints, "PROJECT_ROOT", workspace_root)
    monkeypatch.setattr(checkpoints, "CANONICAL_CHECKPOINT_ROOT", ckpt_root)
    return workspace_root, ckpt_root


def _make_config(tmp_path: Path, monkeypatch) -> ModelRuntimeConfig:
    workspace_root, ckpt_root = _patch_workspace(monkeypatch, tmp_path)
    config_path = workspace_root / "config" / "models" / "wonderjourney.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")

    repo_root = workspace_root / "thirdparty" / "WonderJourney"
    (repo_root / "config").mkdir(parents=True, exist_ok=True)
    (repo_root / "run.py").write_text("print('stub')\n", encoding="utf-8")
    (repo_root / "config" / "base-config.yaml").write_text("device: cuda\n", encoding="utf-8")

    midas_checkpoint = ckpt_root / "WonderJourney" / "dpt_beit_large_512.pt"
    midas_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    midas_checkpoint.write_bytes(b"ckpt")

    return ModelRuntimeConfig(
        config_path=config_path,
        name="wonderjourney",
        family="wonderjourney",
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
            "midas_checkpoint": str(midas_checkpoint),
            "stable_diffusion_checkpoint": "ckpt/stabilityai--stable-diffusion-2-inpainting",
            "entity_limit": 2,
            "frames": 4,
            "num_scenes": 1,
            "num_keyframes": 2,
            "rotation_path": [0, 1],
            "save_fps": 8,
            "finetune_decoder_gen": False,
            "finetune_depth_model": False,
            "archive_sidecars": True,
            "frames_by_suite": {"video_static": 6},
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
            "frames": 4,
            "frames_by_suite": {"video_static": 6},
        },
        "video_static",
    )

    assert resolved["frames"] == 6


def test_registry_builds_wonderjourney_adapter(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)

    adapter = build_model_adapter(config)

    assert isinstance(adapter, WonderJourneyAdapter)


def test_runtime_entrypoint_removes_default_fp16_revision(tmp_path: Path) -> None:
    entrypoint = tmp_path / "run.py"
    entrypoint.write_text(
        'pipe = StableDiffusionInpaintPipeline.from_pretrained("repo", revision="fp16",)\n',
        encoding="utf-8",
    )

    patched = _runtime_entrypoint(entrypoint, tmp_path, None)

    assert patched != entrypoint
    assert 'revision="fp16"' not in patched.read_text(encoding="utf-8")


def test_generate_builds_runner_command(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, monkeypatch)
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (512, 512), color="white").save(conditioning_image)
    output_path = tmp_path / "prediction.mp4"
    sample = _make_sample()
    recorded: dict[str, object] = {}

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["env"] = env
        output_path.write_bytes(b"mp4")
        output_path.with_name(f"{output_path.stem}.reverse{output_path.suffix}").write_bytes(
            b"reverse"
        )
        output_path.with_name(f"{output_path.stem}.keyframes{output_path.suffix}").write_bytes(
            b"keyframes"
        )

    monkeypatch.setattr("worldarena.models.adapters.wonderjourney.run_command", fake_run_command)

    adapter = WonderJourneyAdapter(config)
    result = adapter.generate(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_current,
    )

    command = recorded["command"]
    assert isinstance(command, list)
    assert "worldarena.models.adapters.wonderjourney_runner" in command
    assert "--midas_checkpoint" in command
    assert "--stable_diffusion_checkpoint" in command
    assert "--rotation_path" in command
    assert "--frames" in command
    assert "6" in command
    assert result["prediction_path"] == str(output_path)
    assert result["scene_name"] == "dining space"
    assert result["reverse_video_path"] == str(
        output_path.with_name(f"{output_path.stem}.reverse{output_path.suffix}")
    )
    assert result["keyframes_video_path"] == str(
        output_path.with_name(f"{output_path.stem}.keyframes{output_path.suffix}")
    )
