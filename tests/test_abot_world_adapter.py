from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.abot_world import (
    ABotWorldAdapter,
    KEY_ORDER,
    _action_for_value,
    _build_control_payload,
)
from worldarena.models.adapters.abot_world_batch_runner import _patch_checkpoint_paths
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


def _make_sample(camera_path: list[str], *, suite: str = "image_static") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"{suite}-001",
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
        prediction_stem=f"{suite}_sample_001",
        conditioning_strategy="reference_image",
        prompt_current="A road crosses a mountain valley.",
        prompt_target="Camera orbit left. A road crosses a mountain valley.",
        style="photorealistic",
        environment="outdoor",
        scene="test_scene",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=camera_path,
    )


def _active_keys(action: dict[str, bool]) -> set[str]:
    return {key for key, enabled in action.items() if enabled}


def test_abot_world_config_builds_registered_adapter() -> None:
    config = load_model_config(Path("config/models/abot_world.yaml"))
    adapter = build_model_adapter(config)

    assert isinstance(adapter, ABotWorldAdapter)
    assert config.supported_suites == ["image_static", "image_dynamic"]
    assert config.control_signals == ["text", "image", "actions"]
    assert config.reference_mode == "first_frame"
    assert config.prompt_mode == "prompt_target"
    assert config.generation["use_fp8_gemm"] is False
    assert config.generation["quant_type"] == "none"
    assert config.generation["encoder_device"] == "cuda"
    assert config.generation["require_persistent_batch"] is True
    assert adapter.supports_batch_generation() is True
    assert adapter.batch_checkpoint_load_policy() == "load_once"


@pytest.mark.parametrize(
    ("camera_token", "expected_keys"),
    [
        ("fixed", set()),
        ("push_in", {"W"}),
        ("pull_out", {"S"}),
        ("move_left", {"A"}),
        ("move_right", {"D"}),
        ("pan_left", {"S", "A", "J"}),
        ("pan_right", {"S", "D", "L"}),
        ("orbit_left", {"W", "A", "L"}),
        ("orbit_right", {"W", "D", "J"}),
        ("tilt_up", {"I"}),
        ("tilt_down", {"K"}),
    ],
)
def test_camera_path_maps_to_abot_native_keys(
    camera_token: str,
    expected_keys: set[str],
) -> None:
    payload = _build_control_payload(
        _make_sample([camera_token]),
        {"blocks": 5, "action_source": "worldarena_camera_path"},
    )

    assert payload["camera_tokens"] == [camera_token]
    assert payload["control_source"] == "camera_path"
    assert _active_keys(payload["actions"][0]) == expected_keys
    assert list(payload["actions"][0]) == list(KEY_ORDER)


def test_multi_segment_camera_path_spans_all_blocks() -> None:
    payload = _build_control_payload(
        _make_sample(["pan_left", "move_right", "orbit_right"]),
        {"blocks": 5, "action_source": "worldarena_camera_path"},
    )

    assert payload["block_count"] == 5
    assert len(payload["block_actions"]) == 5
    assert [_active_keys(action) for action in payload["block_actions"]] == [
        {"S", "A", "J"},
        {"S", "A", "J"},
        {"D"},
        {"D"},
        {"W", "D", "J"},
    ]


def test_config_action_plan_accepts_native_key_chords() -> None:
    payload = _build_control_payload(
        _make_sample([]),
        {
            "blocks": 3,
            "action_source": "config_action_plan",
            "action_plan": {"default": ["W+A", "L", "fixed"]},
        },
    )

    assert [_active_keys(action) for action in payload["block_actions"]] == [
        {"W", "A"},
        {"L"},
        set(),
    ]
    assert payload["control_source"] == "config_action_plan"


def test_unsupported_camera_degree_of_freedom_is_explicit() -> None:
    with pytest.raises(ValueError, match="unsupported ABot-World action"):
        _action_for_value("roll_cw")


def test_batch_spec_preserves_prompt_image_and_controls(tmp_path: Path) -> None:
    config = load_model_config(Path("config/models/abot_world.yaml"))
    adapter = ABotWorldAdapter(config)
    sample = _make_sample(["orbit_left"])
    request = PreparedGenerationRequest(
        sample=sample,
        conditioning_image=tmp_path / "reference.png",
        output_path=tmp_path / "predictions" / "sample.mp4",
        prompt=sample.prompt_target,
    )

    payload = adapter.batch_spec_payload(request)

    assert payload["sample_id"] == sample.sample_id
    assert payload["prompt"] == sample.prompt_target
    assert payload["conditioning_image"] == str((tmp_path / "reference.png").resolve())
    assert payload["camera_path"] == ["orbit_left"]
    assert _active_keys(payload["block_actions"][0]) == {"W", "A", "L"}


def test_checkpoint_paths_are_redirected_to_worldarena_root(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt" / "ABot-World-0-5B-LF"
    config = SimpleNamespace(
        model_kwargs=SimpleNamespace(model_name="old"),
        text_encoder_kwargs=SimpleNamespace(tokenizer_path="old", encoder_pth_path="old"),
        vae_kwargs=SimpleNamespace(pretrained_path="old"),
    )

    _patch_checkpoint_paths(config, checkpoint_dir)

    assert config.model_kwargs.model_name == str(checkpoint_dir)
    assert config.taew2_2_checkpoint == str(checkpoint_dir / "taew2_2.pth")
    assert config.text_encoder_kwargs.tokenizer_path == str(
        checkpoint_dir / "google" / "umt5-xxl"
    )
    assert config.vae_kwargs.pretrained_path == str(checkpoint_dir / "Wan2.2_VAE.pth")


def test_abot_world_batch_runner_dry_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "ABot-World"
    for relative_path in (
        "configs/default_config.yaml",
        "configs/long_forcing_dmd.yaml",
        "pipeline/causal_inference.py",
        "utils/wan_wrapper.py",
        "web_client/pipeline_loader.py",
    ):
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    checkpoint_dir = tmp_path / "ABot-World-0-5B-LF"
    checkpoint_dir.mkdir()
    for filename in (
        "Wan2.2_VAE.pth",
        "taew2_2.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors",
    ):
        (checkpoint_dir / filename).write_bytes(b"checkpoint")
    (checkpoint_dir / "google" / "umt5-xxl").mkdir(parents=True)

    spec_path = tmp_path / "batch.jsonl"
    spec_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "conditioning_image": str(tmp_path / "input.png"),
                "output_path": str(tmp_path / "output.mp4"),
                "prompt": "Camera moves forward.",
                "block_actions": [_action_for_value("W"), _action_for_value("L")],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generation_path = tmp_path / "generation.json"
    generation_path.write_text("{}\n", encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldarena.models.adapters.abot_world_batch_runner",
            "--repo_root",
            str(repo_root),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ],
        cwd=project_root,
        env={
            **os.environ,
            "PYTHONPATH": str(project_root),
            "WORLDARENA_ABOT_WORLD_DRY_RUN": "1",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["dry_run"] is True
    assert payload["persistent_engine"] is True
    assert payload["checkpoint_load_count"] == 1
    assert payload["request_count"] == 1
    assert payload["blocks"] == [2]
