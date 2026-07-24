from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices
from worldarena.common import checkpoints
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.lingbot_world import (
    LingBotWorldAdapter,
    _annotation_action_path,
    _generation_kwargs_for_runtime,
    _prepare_annotation_action_path,
    _prepare_lingbot_intrinsics,
)
from worldarena.models.config import ModelRuntimeConfig


def _patch_workspace(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    workspace_root = tmp_path / "WorldAtlas Arena"
    ckpt_root = workspace_root / "ckpt"
    monkeypatch.setattr(checkpoints, "PROJECT_ROOT", workspace_root)
    monkeypatch.setattr(checkpoints, "CANONICAL_CHECKPOINT_ROOT", ckpt_root)
    return workspace_root, ckpt_root


def _make_config(tmp_path: Path, monkeypatch) -> ModelRuntimeConfig:
    workspace_root, ckpt_root = _patch_workspace(monkeypatch, tmp_path)
    config_path = workspace_root / "config" / "models" / "lingbot_world.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    repo_root = workspace_root / "thirdparty" / "lingbot-world"
    repo_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ckpt_root / "lingbot-world-base-cam"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ModelRuntimeConfig(
        config_path=config_path,
        name="lingbot-world-fast",
        family="lingbot-world",
        python_bin="python",
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        entrypoint="generate_fast.py",
        output_ext=".mp4",
        supported_suites=["video_dynamic", "image_static", "image_dynamic"],
        prompt_mode="prompt_current",
        reference_mode="first_frame",
        control_signals=["trajectory"],
        generation={"size": "832*480", "frame_num": 81},
    )


def _make_sample(annotation_path: Path) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="sample-001",
        suite="video_dynamic",
        split="test",
        asset_level="video",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="dynamic/photorealistic/videos/articulated",
        path=str(annotation_path.parent / "sample.mp4"),
        relative_path="dynamic/sample.mp4",
        reference_path=str(annotation_path.parent / "sample.mp4"),
        prediction_stem="sample_pred",
        conditioning_strategy="first_frame",
        prompt_current="A dog runs through a meadow.",
        prompt_target="There is articulated motion in the dog.",
        style="photorealistic",
        environment="outdoor",
        scene="meadow",
        motion_category="articulated",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        annotation_path=str(annotation_path),
        has_annotation=True,
        has_pose=True,
    )


def _make_image_sample(
    *,
    suite: str,
    prediction_stem: str,
    camera_path: list[str],
    generation_mode: str | None,
) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{prediction_stem}-001",
        suite=suite,
        split="test",
        asset_level="image",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/indoor/living_spaces",
        path=f"/dataset/{prediction_stem}.png",
        relative_path=f"{suite}/{prediction_stem}.png",
        reference_path=f"/dataset/{prediction_stem}.png",
        prediction_stem=prediction_stem,
        conditioning_strategy="self",
        prompt_current="A reference image.",
        prompt_target="Camera pan right. Generate a video rollout from the reference image.",
        style="photorealistic",
        environment="indoor",
        scene="living space",
        motion_category=None if suite == "image_static" else "articulated",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=camera_path,
        generation_mode=generation_mode,
        annotation_path="/tmp/image_annotations.json::static/sample.png",
        has_annotation=True,
        has_pose=False,
    )


def test_prepare_lingbot_intrinsics_converts_normalized_values() -> None:
    intrinsics = np.asarray([[0.44, 0.7822222, 0.5, 0.5]], dtype=np.float32)
    converted, changed = _prepare_lingbot_intrinsics(intrinsics)

    assert changed is True
    np.testing.assert_allclose(
        converted,
        np.asarray([[366.08, 375.46667, 416.0, 240.0]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_prepare_annotation_action_path_materializes_pixel_intrinsics(tmp_path: Path, monkeypatch) -> None:
    annotation_path = tmp_path / "annotations" / "sample_ann"
    annotation_path.mkdir(parents=True)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], repeats=5, axis=0)
    intrinsics = np.repeat(
        np.asarray([[0.44, 0.7822222, 0.5, 0.5]], dtype=np.float32),
        repeats=5,
        axis=0,
    )
    np.save(annotation_path / "poses.npy", poses)
    np.save(annotation_path / "intrinsics.npy", intrinsics)
    (annotation_path / "indexes.txt").write_text("0\n1\n2\n3\n4\n", encoding="utf-8")

    config = _make_config(tmp_path, monkeypatch)
    sample = _make_sample(annotation_path)

    prepared = _prepare_annotation_action_path(config, sample, annotation_path)

    assert prepared != annotation_path
    assert str(prepared).startswith(str(tmp_path / "WorldAtlas Arena" / "cache"))
    exported = np.load(prepared / "intrinsics.npy")
    np.testing.assert_allclose(
        exported[0],
        np.asarray([366.08, 375.46667, 416.0, 240.0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    assert (prepared / "indexes.txt").is_symlink()


def test_prepare_annotation_action_path_keeps_pixel_intrinsics(tmp_path: Path, monkeypatch) -> None:
    annotation_path = tmp_path / "annotations" / "sample_ann"
    annotation_path.mkdir(parents=True)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], repeats=5, axis=0)
    intrinsics = np.repeat(
        np.asarray([[366.08, 375.46667, 416.0, 240.0]], dtype=np.float32),
        repeats=5,
        axis=0,
    )
    np.save(annotation_path / "poses.npy", poses)
    np.save(annotation_path / "intrinsics.npy", intrinsics)

    config = _make_config(tmp_path, monkeypatch)
    sample = _make_sample(annotation_path)

    prepared = _prepare_annotation_action_path(config, sample, annotation_path)

    assert prepared == annotation_path


def test_annotation_action_path_synthesizes_dynamic_image_controls(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)
    sample = _make_image_sample(
        suite="image_dynamic",
        prediction_stem="image_dynamic_sample",
        camera_path=["fixed"],
        generation_mode="dynamic",
    )

    prepared = _annotation_action_path(config, sample, action_string=None)

    assert prepared is not None
    assert str(prepared).startswith(str(tmp_path / "WorldAtlas Arena" / "cache"))
    poses = np.load(prepared / "poses.npy")
    intrinsics = np.load(prepared / "intrinsics.npy")
    assert poses.shape == (81, 4, 4)
    assert intrinsics.shape == (81, 4)
    np.testing.assert_allclose(poses[0], np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(poses[-1], np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(
        intrinsics[0],
        np.asarray([500.0, 500.0, 416.0, 240.0], dtype=np.float32),
    )


def test_annotation_action_path_synthesizes_static_multi_step_image_controls(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)
    sample = _make_image_sample(
        suite="image_static",
        prediction_stem="image_static_sample",
        camera_path=["pan_right", "move_right", "pull_out"],
        generation_mode="static",
    )

    prepared = _annotation_action_path(config, sample, action_string=None)

    assert prepared is not None
    poses = np.load(prepared / "poses.npy")
    intrinsics = np.load(prepared / "intrinsics.npy")
    assert poses.shape == (81, 4, 4)
    assert intrinsics.shape == (81, 4)
    assert not np.allclose(poses[0], poses[-1])
    expected = synthetic_camera_matrices(
        ["pan_right", "move_right", "pull_out"],
        target_frames=81,
    ).matrices
    np.testing.assert_allclose(poses, expected, atol=1e-6)
    # The move-right segment ends at frame 54. Pull-out then moves in the
    # rotated local frame, so the final world-X sign is not a valid direction test.
    assert poses[54, 0, 3] > 0.0


def test_generation_kwargs_for_runtime_uses_full_settings() -> None:
    runtime = SimpleNamespace(
        generation={
            "size": "832*480",
            "frame_num": 81,
            "sample_solver": "unipc",
            "sample_steps": 20,
            "sample_shift": 4.0,
            "sample_guide_scale": 6.0,
            "base_seed": 7,
        },
        max_area_configs={"832*480": 399360},
        cfg=SimpleNamespace(sample_shift=3.0, sample_steps=40, sample_guide_scale=5.0),
        offload_model=True,
        entrypoint_name="generate.py",
    )

    kwargs = _generation_kwargs_for_runtime(
        runtime,
        action_path="/tmp/action",
        action_string="w-5",
        allow_act2cam=True,
    )

    assert kwargs["sample_solver"] == "unipc"
    assert kwargs["sampling_steps"] == 20
    assert kwargs["guide_scale"] == 6.0
    assert kwargs["allow_act2cam"] is True
    assert kwargs["action_string"] == "w-5"
    assert "chunk_size" not in kwargs


def test_generation_kwargs_for_runtime_uses_fast_settings() -> None:
    runtime = SimpleNamespace(
        generation={
            "size": "832*480",
            "frame_num": 81,
            "sample_shift": 8.0,
            "base_seed": 11,
            "chunk_size": 5,
            "max_attention_size": 8192,
        },
        max_area_configs={"832*480": 399360},
        cfg=SimpleNamespace(sample_shift=3.0, sample_steps=40, sample_guide_scale=5.0),
        offload_model=False,
        entrypoint_name="generate_fast.py",
    )

    kwargs = _generation_kwargs_for_runtime(
        runtime,
        action_path="/tmp/action",
        action_string=None,
        allow_act2cam=False,
    )

    assert kwargs["chunk_size"] == 5
    assert kwargs["max_attention_size"] == 8192
    assert kwargs["shift"] == 8.0
    assert kwargs["offload_model"] is False
    assert "sample_solver" not in kwargs


def test_generation_kwargs_for_runtime_keeps_tuple_guide_scale() -> None:
    runtime = SimpleNamespace(
        generation={"size": "832*480", "frame_num": 81},
        max_area_configs={"832*480": 399360},
        cfg=SimpleNamespace(sample_shift=3.0, sample_steps=40, sample_guide_scale=(4.0, 6.0)),
        offload_model=True,
        entrypoint_name="generate.py",
    )

    kwargs = _generation_kwargs_for_runtime(
        runtime,
        action_path="/tmp/action",
        action_string=None,
        allow_act2cam=False,
    )

    assert kwargs["guide_scale"] == (4.0, 6.0)


def test_generation_kwargs_for_runtime_rejects_fast_action_string() -> None:
    runtime = SimpleNamespace(
        generation={"size": "832*480", "frame_num": 81},
        max_area_configs={"832*480": 399360},
        cfg=SimpleNamespace(sample_shift=3.0, sample_steps=40, sample_guide_scale=5.0),
        offload_model=True,
        entrypoint_name="generate_fast.py",
    )

    with pytest.raises(ValueError, match="generate_fast.py does not support action_string"):
        _generation_kwargs_for_runtime(
            runtime,
            action_path="/tmp/action",
            action_string="w-5",
            allow_act2cam=True,
        )


def test_lingbot_adapter_generate_batch_invokes_persistent_runner(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, monkeypatch)
    config.entrypoint = "generate.py"
    config.python_bin = "/share/project/yesliu/miniconda3/envs/lingbot-world/bin/python"
    config.generation = {"size": "832*480", "frame_num": 81, "device_id": 2}
    adapter = LingBotWorldAdapter(config)
    sample = _make_image_sample(
        suite="image_static",
        prediction_stem="batch_sample",
        camera_path=["fixed"],
        generation_mode="static",
    )
    conditioning_image = tmp_path / "conditioning.png"
    Image.new("RGB", (32, 32), color="white").save(conditioning_image)
    output_path = tmp_path / "predictions" / "batch_sample.mp4"

    captured: dict[str, object] = {}

    def fake_run_command(command, *, cwd, env):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        spec_path = Path(command[command.index("--batch-spec-path") + 1])
        results_path = Path(command[command.index("--results-path") + 1])
        rows = [json.loads(line) for line in spec_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0]["sample"]["sample_id"] == sample.sample_id
        output_path.write_bytes(b"video")
        results_path.write_text(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "status": "generated",
                    "prediction_path": str(output_path),
                    "prompt": sample.prompt_current,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("worldarena.models.adapters.lingbot_world.run_command", fake_run_command)

    results = adapter.generate_batch(
        [
            PreparedGenerationRequest(
                sample=sample,
                conditioning_image=conditioning_image,
                output_path=output_path,
                prompt=sample.prompt_current,
            )
        ]
    )

    command = captured["command"]
    assert command[0] == config.python_bin
    assert command[1:3] == ["-m", "worldarena.models.adapters.lingbot_world_batch_runner"]
    assert command[command.index("--device-id") + 1] == "2"
    assert str(config.config_path) in command
    assert results[sample.sample_id]["status"] == "generated"
    assert results[sample.sample_id]["prediction_path"] == str(output_path)
