from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common import checkpoints
from worldarena.models.adapters import pose_synthesis
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.lingbot_world import _camera_delta
from worldarena.models.adapters.pose_synthesis import (
    _synthetic_image_poses,
    resolve_pose_annotation_path,
)
from worldarena.models.adapters.wonderworld import (
    WonderWorldAdapter,
    _generation_for_suite,
    _structured_prompts_for_sample,
)
from worldarena.models.adapters.wonderworld_batch_runner import (
    BACKGROUND_RGB,
    _validate_rendered_frames,
)
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter


def _patch_workspace(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    workspace_root = tmp_path / "WorldAtlas Arena"
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
            "min_render_content_fraction": 0.02,
            "max_render_blank_frame_ratio": 0.2,
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
        output_path.with_name(f"{output_path.stem}.sidecar.zip").write_bytes(b"zip")

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
    assert "worldarena.models.adapters.wonderworld_batch_runner" in command
    assert "--annotation_path" in command
    assert "--prompt_sequence_json" in command
    assert "--repvit_checkpoint" in command
    assert "--stable_diffusion_checkpoint" in command
    assert "--depth_model_repo" in command
    assert "--normal_model_repo" in command
    assert "--trajectory_frames" in command
    assert "--min_render_content_fraction" in command
    assert "--max_render_blank_frame_ratio" in command
    assert result["prediction_path"] == str(output_path)
    assert result["archive_path"] == str(output_path.with_name("prediction.sidecar.zip"))
    assert result["scene_name"] == "dining space"
    assert result["entities"] == []


def test_generate_scales_synthetic_poses_to_wonderworld_units(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, monkeypatch)
    config.generation["camera_speed"] = 0.002
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (512, 512), color="white").save(conditioning_image)
    output_path = tmp_path / "prediction.mp4"
    sample = _make_sample()

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del command, cwd, env
        output_path.write_bytes(b"mp4")

    monkeypatch.setattr("worldarena.models.adapters.wonderworld.run_command", fake_run_command)

    result = WonderWorldAdapter(config).generate(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_current,
    )

    poses = np.load(Path(result["pose_annotation_path"]) / "poses.npy")
    np.testing.assert_allclose(poses[-1, 2, 3], 0.002, atol=1e-7)
    assert result["pose_source"] == "synthetic"
    assert result["synthetic_pose_translation_scale"] == 0.002


def test_generate_batch_scales_synthetic_poses_to_wonderworld_units(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, monkeypatch)
    config.generation["camera_speed"] = 0.002
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (512, 512), color="white").save(conditioning_image)
    output_path = tmp_path / "prediction.mp4"
    sample = _make_sample()

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        del cwd, env
        spec_path = Path(command[command.index("--batch_spec") + 1])
        results_path = Path(command[command.index("--batch_results") + 1])
        row = json.loads(spec_path.read_text(encoding="utf-8").strip())
        assert "--min_render_content_fraction" in row["args"]
        assert "--max_render_blank_frame_ratio" in row["args"]
        output_path.write_bytes(b"mp4")
        results_path.write_text(
            json.dumps(
                {
                    sample.sample_id: {
                        "status": "generated",
                        "prediction_path": str(output_path),
                    }
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("worldarena.models.adapters.wonderworld.run_command", fake_run_command)
    request = PreparedGenerationRequest(
        sample=sample,
        conditioning_image=conditioning_image,
        output_path=output_path,
        prompt=sample.prompt_current,
    )

    result = WonderWorldAdapter(config).generate_batch([request])[sample.sample_id]

    poses = np.load(Path(result["pose_annotation_path"]) / "poses.npy")
    np.testing.assert_allclose(poses[-1, 2, 3], 0.002, atol=1e-7)
    assert result["pose_source"] == "synthetic"
    assert result["synthetic_pose_translation_scale"] == 0.002


def test_render_validation_accepts_frames_with_scene_content() -> None:
    frames = [Image.new("RGB", (32, 32), color=(20 + index, 60, 120)) for index in range(10)]

    summary = _validate_rendered_frames(
        frames,
        min_content_fraction=0.02,
        max_blank_frame_ratio=0.2,
    )

    assert summary["blank_frame_count"] == 0
    assert summary["blank_frame_ratio"] == 0.0


def test_render_validation_rejects_gray_background_collapse() -> None:
    background = tuple(round(channel * 255) for channel in BACKGROUND_RGB)
    frames = [Image.new("RGB", (32, 32), color=(20, 60, 120))]
    frames.extend(Image.new("RGB", (32, 32), color=background) for _ in range(9))

    with pytest.raises(RuntimeError, match="9/10 frames"):
        _validate_rendered_frames(
            frames,
            min_content_fraction=0.02,
            max_blank_frame_ratio=0.2,
        )


@pytest.mark.parametrize(
    ("min_content_fraction", "max_blank_frame_ratio"),
    [(-0.1, 0.2), (1.1, 0.2), (0.02, -0.1), (0.02, 1.1)],
)
def test_render_validation_rejects_invalid_thresholds(
    min_content_fraction: float,
    max_blank_frame_ratio: float,
) -> None:
    with pytest.raises(ValueError):
        _validate_rendered_frames(
            [Image.new("RGB", (8, 8), color="red")],
            min_content_fraction=min_content_fraction,
            max_blank_frame_ratio=max_blank_frame_ratio,
        )


@pytest.mark.parametrize(
    "camera_token",
    ["push_in", "pull_out", "move_left", "move_right", "orbit_left", "orbit_right"],
)
def test_synthetic_pose_scales_translation_but_not_rotation(camera_token: str) -> None:
    scale = 0.001

    poses = _synthetic_image_poses([camera_token], frame_count=5, translation_scale=scale)
    unscaled_delta = _camera_delta(camera_token)

    np.testing.assert_allclose(poses[-1, :3, 3], unscaled_delta[:3, 3] * scale, atol=1e-7)
    np.testing.assert_allclose(poses[-1, :3, :3], unscaled_delta[:3, :3], atol=1e-6)


def test_synthetic_pose_keeps_pure_rotation_translation_free() -> None:
    poses = _synthetic_image_poses(["pan_right"], frame_count=5, translation_scale=0.001)
    expected = _camera_delta("pan_right")

    np.testing.assert_allclose(poses[-1, :3, 3], np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(poses[-1, :3, :3], expected[:3, :3], atol=1e-6)


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_synthetic_pose_rejects_invalid_translation_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="translation_scale"):
        _synthetic_image_poses(["push_in"], frame_count=5, translation_scale=scale)


def test_pose_cache_identity_includes_translation_scale(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pose_synthesis, "project_root", lambda: tmp_path)
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (640, 480), color="white").save(conditioning_image)
    sample = SimpleNamespace(
        annotation_path=None,
        modality="image",
        sample_id="image-static-001",
        relative_path="static/sample.png",
        camera_path=["push_in"],
        prediction_stem="image_static_sample",
    )
    config = SimpleNamespace(name="test-model")

    path_a, source_a = resolve_pose_annotation_path(
        config,
        sample,
        conditioning_image,
        frame_count=5,
        namespace="test",
        focal_scale=1.0,
        translation_scale=0.001,
    )
    path_b, source_b = resolve_pose_annotation_path(
        config,
        sample,
        conditioning_image,
        frame_count=5,
        namespace="test",
        focal_scale=1.0,
        translation_scale=0.002,
    )

    assert source_a == source_b == "synthetic"
    assert path_a != path_b
    np.testing.assert_allclose(np.load(path_a / "poses.npy")[-1, 2, 3], 0.001, atol=1e-7)
    np.testing.assert_allclose(np.load(path_b / "poses.npy")[-1, 2, 3], 0.002, atol=1e-7)
