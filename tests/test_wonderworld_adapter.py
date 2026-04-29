from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common import checkpoints
from worldarena.models.adapters.wonderworld import (
    WonderWorldAdapter,
    _generation_for_suite,
    _structured_prompts_for_sample,
)
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
    config_path = workspace_root / "config" / "models" / "wonderworld.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")

    repo_root = workspace_root / "thirdparty" / "WonderWorld"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "run.py").write_text("print('stub')\n", encoding="utf-8")
    repvit_checkpoint = ckpt_root / "WonderWorld" / "repvit_sam.pt"
    repvit_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    repvit_checkpoint.write_bytes(b"ckpt")

    return ModelRuntimeConfig(
        config_path=config_path,
        name="wonderworld",
        family="wonderworld",
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
            "repvit_checkpoint": str(repvit_checkpoint),
            "stable_diffusion_checkpoint": "sd2-community/stable-diffusion-2-inpainting",
            "depth_model_repo": "prs-eth/marigold-v1-0",
            "normal_model_repo": "prs-eth/marigold-normals-v0-1",
            "entity_limit": 3,
            "num_scenes": 1,
            "trajectory_frames": 8,
            "fps": 8,
            "gen_layer": False,
            "archive_sidecars": True,
        },
    )


def _make_sample(annotation_path: str | None = None, *, suite: str = "image_static") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-001",
        suite=suite,
        split="test",
        asset_level="image" if suite == "image_static" else "video",
        modality="image" if suite == "image_static" else "video",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/dining_spaces",
        path=f"/dataset/{suite}.png" if suite == "image_static" else f"/dataset/{suite}.mp4",
        relative_path=f"{suite}/sample",
        reference_path=f"/dataset/{suite}.png" if suite == "image_static" else f"/dataset/{suite}.mp4",
        prediction_stem=f"{suite}_sample",
        conditioning_strategy="reference_image" if suite == "image_static" else "first_frame",
        prompt_current="A photorealistic dining room scene.",
        prompt_target="Generate a photorealistic video of a static dining room scene.",
        style="photorealistic",
        environment="indoor",
        scene="dining space",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        annotation_path=annotation_path,
        has_annotation=annotation_path is not None,
        prompt_sequence=["Camera push in. A photorealistic dining room scene."],
        camera_path=["push_in"],
    )


def test_generation_for_suite_applies_by_suite_overrides() -> None:
    resolved = _generation_for_suite(
        {
            "gen_layer": False,
            "gen_layer_by_suite": {"video_static": True},
        },
        "video_static",
    )

    assert resolved["gen_layer"] is True


def test_structured_prompts_use_annotation_entities(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotations" / "sample"
    annotation_dir.mkdir(parents=True)
    (annotation_dir / "caption.json").write_text(
        json.dumps(
            {
                "objects": ["table", "chair", "window"],
            }
        ),
        encoding="utf-8",
    )
    sample = _make_sample(str(annotation_dir))

    structured = _structured_prompts_for_sample(
        sample,
        prompt_text=sample.prompt_current,
        generation={"entity_limit": 2},
    )

    assert structured["scene_name"] == "dining space"
    assert structured["entities"] == ["table", "chair"]
    assert structured["entity_source"] == "annotation_explicit"
    assert structured["content_prompt"] == "dining space, table, chair"


def test_registry_builds_wonderworld_adapter(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)

    adapter = build_model_adapter(config)

    assert isinstance(adapter, WonderWorldAdapter)


def test_generate_builds_runner_command(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, monkeypatch)
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (512, 512), color="white").save(conditioning_image)
    annotation_dir = tmp_path / "annotations" / "sample"
    annotation_dir.mkdir(parents=True)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], repeats=8, axis=0)
    np.save(annotation_dir / "poses.npy", poses)
    np.save(
        annotation_dir / "intrinsics.npy",
        np.repeat(np.asarray([[960.0, 960.0, 256.0, 256.0]], dtype=np.float32), repeats=8, axis=0),
    )
    output_path = tmp_path / "prediction.mp4"
    sample = _make_sample()
    sample.annotation_path = str(annotation_dir)
    sample.has_annotation = True
    recorded: dict[str, object] = {}

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["env"] = env
        output_path.write_bytes(b"mp4")
        output_path.with_name(f"{output_path.stem}.worldscore.zip").write_bytes(b"zip")

    monkeypatch.setattr("worldarena.models.adapters.wonderworld.run_command", fake_run_command)

    adapter = WonderWorldAdapter(config)
    result = adapter.generate(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_current,
    )

    command = recorded["command"]
    assert isinstance(command, list)
    assert "worldarena.models.adapters.wonderworld_worldscore_runner" in command
    assert "--annotation_path" in command
    assert "--prompt_sequence_json" in command
    assert "--repvit_checkpoint" in command
    assert "--stable_diffusion_checkpoint" in command
    assert "--depth_model_repo" in command
    assert "--normal_model_repo" in command
    assert "--trajectory_frames" in command
    assert result["prediction_path"] == str(output_path)
    assert result["archive_path"] == str(output_path.with_name("prediction.worldscore.zip"))
    assert result["scene_name"] == "dining space"
    assert result["entities"] == []
