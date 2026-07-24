from __future__ import annotations

from enum import Enum
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters import alayaworld_batch_runner as alayaworld_batch_runner_module
from worldarena.models.adapters.alayaworld import (
    AlayaWorldAdapter,
    _build_control_payload,
)
from worldarena.models.adapters.alayaworld_batch_runner import (
    RuntimePaths,
    _build_camera_metadata,
    _patch_inference_config,
    _release_rollout_only_state,
    _runtime_env,
)
from worldarena.models.adapters.alayaworld_upstream_runner import (
    _install_attention_backend_override,
)
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


def _make_sample(
    camera_path: list[str],
    *,
    suite: str = "image_static",
    prompt_sequence: list[str] | None = None,
) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-alaya-001",
        suite=suite,
        split="test",
        asset_level="image",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/test_scene",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/outdoor/test_scene/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem=f"{suite}_alaya_001",
        conditioning_strategy="reference_image",
        prompt_current="A road crosses a mountain valley.",
        prompt_target="Camera pushes in. A road crosses a mountain valley.",
        style="photorealistic",
        environment="outdoor",
        scene="test_scene",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        prompt_sequence=list(prompt_sequence or []),
        camera_path=camera_path,
    )


def test_alayaworld_config_builds_registered_adapter() -> None:
    config = load_model_config(Path("config/models/alayaworld.yaml"))
    adapter = build_model_adapter(config)

    assert isinstance(adapter, AlayaWorldAdapter)
    assert adapter.supports_batch_generation()
    assert config.supported_suites == [
        "image_static",
        "image_dynamic",
        "video_static",
        "video_dynamic",
    ]
    assert config.control_signals == ["text", "image", "camera_pose"]
    assert config.generation["rounds"] == 45
    assert config.generation["sampling_steps"] == 4
    assert config.generation["attention_type"] == "xformers"
    assert config.generation["persistent"] is True


def test_runtime_env_keeps_worldarena_importable_after_upstream_chdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/existing-pythonpath")

    runtime_env = _runtime_env({"offline": True})

    entries = runtime_env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(Path(__file__).resolve().parents[1])
    assert entries[1] == "/tmp/existing-pythonpath"


def test_persistent_decode_cleanup_retains_only_rollout_latents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(alayaworld_batch_runner_module, "_empty_cuda_cache", lambda: None)
    cache = SimpleNamespace(
        spatial_bank=object(),
        history=torch.ones(1),
        history_action_t_indices=torch.ones(1),
        explicit_nearby=torch.ones(1),
        context=torch.ones(1),
        negative_context=torch.ones(1),
        sink_latent=torch.ones(1),
        sink_indices=torch.ones(1),
        video_pixels=torch.ones(1),
    )

    _release_rollout_only_state(cache)

    assert cache.spatial_bank is None
    assert cache.history is None
    assert cache.context is None
    assert cache.sink_latent is None
    assert cache.video_pixels.numel() == 0


@pytest.mark.parametrize(
    "camera_token",
    [
        "fixed",
        "push_in",
        "pull_out",
        "move_left",
        "move_right",
        "pedestal_up",
        "pedestal_down",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "roll_cw",
        "roll_ccw",
        "orbit_left",
        "orbit_right",
    ],
)
def test_worldarena_camera_tokens_are_accepted(camera_token: str) -> None:
    payload = _build_control_payload(
        _make_sample([camera_token]),
        {"rounds": 2, "action_source": "worldarena_camera_path"},
    )

    assert payload["camera_tokens"] == [camera_token]
    assert payload["round_actions"] == [camera_token, camera_token]
    assert payload["control_source"] == "camera_path"


def test_multisegment_camera_path_is_balanced_over_rounds() -> None:
    payload = _build_control_payload(
        _make_sample(["push_in", "move_right", "pan_left"]),
        {
            "rounds": 5,
            "fps": 24,
            "temporal_stride": 8,
            "chunk_latent_frames": 4,
            "sink_latent_frames": 1,
            "history_latent_frames": 16,
        },
    )

    assert payload["round_actions"] == [
        "push_in",
        "push_in",
        "move_right",
        "move_right",
        "pan_left",
    ]
    assert payload["generated_frame_count"] == 160
    assert payload["trajectory_frame_count"] == 297
    assert payload["duration_seconds"] == pytest.approx(160 / 24)


def test_prompt_camera_prefix_is_used_when_camera_path_is_empty() -> None:
    payload = _build_control_payload(
        _make_sample([]),
        {"rounds": 1, "action_source": "worldarena_camera_path"},
    )

    assert payload["camera_tokens"] == ["push_in"]
    assert payload["control_source"] == "prompt_target"


def test_camera_metadata_extends_first_action_through_visual_prefix() -> None:
    generation = {
        "temporal_stride": 8,
        "chunk_latent_frames": 4,
        "sink_latent_frames": 1,
        "history_latent_frames": 16,
        "forward_distance_per_chunk": 2.54,
        "lateral_distance_per_chunk": 0.5348,
    }
    metadata = _build_camera_metadata(
        {"sample_id": "sample-1", "round_actions": ["push_in", "move_right"]},
        generation,
    )
    poses = metadata["cam_c2w"].numpy()

    assert poses.shape == (201, 4, 4)
    np.testing.assert_allclose(poses[0], np.eye(4, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(poses[8, :3, 3], [0.0, 0.0, 0.635], atol=1e-4)
    np.testing.assert_allclose(poses[136, :3, 3], [0.0, 0.0, 10.795], atol=1e-4)
    np.testing.assert_allclose(
        poses[-1, :3, 3],
        np.asarray([0.5348, 0.0, 13.335], dtype=np.float32),
        atol=1e-4,
    )
    assert metadata["frame_end"] == 201
    assert metadata["has_camera"] is True
    assert tuple(metadata["intrinsic"].shape) == (3, 3)


def test_batch_spec_preserves_prompt_controls_and_prompt_switch(tmp_path: Path) -> None:
    config = load_model_config(Path("config/models/alayaworld.yaml"))
    adapter = AlayaWorldAdapter(config)
    sample = _make_sample(
        ["orbit_left", "push_in"],
        prompt_sequence=[
            "A quiet mountain valley.",
            "A giant bird made of fire descends from the sky.",
        ],
    )
    request = PreparedGenerationRequest(
        sample=sample,
        conditioning_image=tmp_path / "reference.png",
        output_path=tmp_path / "predictions" / "sample.mp4",
        prompt=sample.prompt_target,
    )

    payload = adapter.batch_spec_payload(request)

    assert payload["prompt"] == sample.prompt_target
    assert payload["camera_tokens"] == ["orbit_left", "push_in"]
    assert len(payload["round_actions"]) == 45
    assert payload["round_actions"][:23] == ["orbit_left"] * 23
    assert payload["round_actions"][23:] == ["push_in"] * 22
    assert payload["skill_prompt"] == sample.prompt_sequence[-1]


def test_runtime_config_redirects_every_external_weight(tmp_path: Path) -> None:
    repo_root = tmp_path / "AlayaWorld"
    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "infer.yaml").write_text(
        "sample:\n  height: 544\n  width: 960\npaths: {}\n",
        encoding="utf-8",
    )
    paths = RuntimePaths(
        model_file=tmp_path / "weights" / "merged.safetensors",
        gemma_dir=tmp_path / "weights" / "gemma",
        da3_repo=tmp_path / "Depth-Anything-3",
        da3_cache=tmp_path / "hf-cache",
        taehv_file=None,
    )
    output_path = tmp_path / "work" / "infer.yaml"

    _patch_inference_config(
        repo_root=repo_root,
        paths=paths,
        generation={"height": 480, "width": 832, "attention_type": "xformers"},
        output_path=output_path,
    )
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert payload["paths"]["model"] == str(paths.model_file)
    assert payload["paths"]["gemma"] == str(paths.gemma_dir)
    assert payload["paths"]["da3_repo"] == str(paths.da3_repo)
    assert payload["paths"]["da3_cache"] == str(paths.da3_cache)
    assert payload["sample"]["height"] == 480
    assert payload["sample"]["width"] == 832
    assert payload["runtime"]["attention_type"] == "xformers"
    assert payload["validation"]["save_joystick"] is False


def test_attention_override_routes_upstream_hardcoded_fa3_to_xformers() -> None:
    class FakeAttentionFunction(Enum):
        PYTORCH = "pytorch"
        XFORMERS = "xformers"
        FLASH_ATTENTION_3 = "flash_attention_3"

        def __call__(
            self,
            q: object,
            k: object,
            v: object,
            heads: int,
            mask: object | None = None,
            window_size: tuple[int, int] | None = None,
        ) -> tuple[object, ...]:
            return ("original", self.value, q, k, v, heads, mask, window_size)

    class FakeXFormersAttention:
        def __call__(
            self,
            q: object,
            k: object,
            v: object,
            heads: int,
            mask: object | None = None,
        ) -> tuple[object, ...]:
            return ("xformers", q, k, v, heads, mask)

    class FakePytorchAttention(FakeXFormersAttention):
        pass

    attention_module = SimpleNamespace(
        AttentionFunction=FakeAttentionFunction,
        XFormersAttention=FakeXFormersAttention,
        PytorchAttention=FakePytorchAttention,
    )

    installed = _install_attention_backend_override("xformers", attention_module)

    assert installed is True
    assert FakeAttentionFunction.FLASH_ATTENTION_3("q", "k", "v", 8, "mask") == (
        "xformers",
        "q",
        "k",
        "v",
        8,
        "mask",
    )
    assert FakeAttentionFunction.PYTORCH("q", "k", "v", 8)[0] == "original"


def test_alayaworld_batch_runner_dry_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "AlayaWorld"
    for relative_path in (
        "configs/infer.yaml",
        "inference/run.py",
        "flash_alaya/utils/pipeline.py",
        "flash_alaya/alaya/memory/da3_depth.py",
    ):
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    checkpoint_dir = tmp_path / "ckpt" / "AlayaWorld"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "merged_infer.safetensors").write_bytes(b"checkpoint")
    gemma_dir = tmp_path / "ckpt" / "gemma"
    gemma_dir.mkdir()
    (gemma_dir / "config.json").write_text("{}\n", encoding="utf-8")
    da3_repo = tmp_path / "Depth-Anything-3"
    (da3_repo / "src" / "depth_anything_3").mkdir(parents=True)
    da3_cache = tmp_path / "hf-cache"
    da3_cache.mkdir()

    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"not decoded during dry-run")
    spec_path = tmp_path / "batch.jsonl"
    spec_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "suite": "image_static",
                "prediction_stem": "sample-1",
                "conditioning_image": str(input_path),
                "output_path": str(tmp_path / "output.mp4"),
                "prompt": "A quiet mountain valley.",
                "round_actions": ["push_in", "move_right"],
                "seed": 1234,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generation_path = tmp_path / "generation.json"
    generation_path.write_text(
        json.dumps(
            {
                "gemma_dir": str(gemma_dir),
                "da3_repo": str(da3_repo),
                "da3_cache": str(da3_cache),
                "compile": "none",
                "flex_attn": False,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldarena.models.adapters.alayaworld_batch_runner",
            "--repo_root",
            str(repo_root),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "WORLDARENA_ALAYAWORLD_DRY_RUN": "1",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["dry_run"] is True
    assert payload["request_count"] == 1
    assert payload["round_actions"] == ["push_in", "move_right"]
    assert payload["trajectory_frame_count"] == 201
    assert payload["persistent_engine"] is True
    assert payload["checkpoint_load_count"] == 1
    assert "--no-flex-attn" in payload["command"]
    assert "worldarena.models.adapters.alayaworld_upstream_runner" in payload["command"]
    metadata = torch.load(
        next((tmp_path / "_alayaworld_work").rglob("*_camera.pt")),
        map_location="cpu",
        weights_only=False,
    )
    assert tuple(metadata["cam_c2w"].shape) == (201, 4, 4)
