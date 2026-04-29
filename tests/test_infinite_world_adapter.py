from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.infinite_world import (
    InfiniteWorldAdapter,
    _build_action_payload,
)
from worldarena.models.config import ModelRuntimeConfig, load_model_config


def _make_sample(camera_path: list[str], *, index: int = 1) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"image-static-{index:03d}",
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/test_scene",
        path="/tmp/reference.png",
        relative_path="static/photorealistic/outdoor/test_scene/reference.png",
        reference_path="/tmp/reference.png",
        prediction_stem=f"image_static_sample_{index:03d}",
        conditioning_strategy="reference_image",
        prompt_current="A static outdoor scene.",
        prompt_target="Camera orbit left. A static outdoor scene.",
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


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        config_path=tmp_path / "infinite_world.yaml",
        name="infinite-world",
        family="infinite_world",
        python_bin="python",
        repo_root=tmp_path / "repo",
        checkpoint_dir=tmp_path / "ckpt",
        entrypoint="scripts/infworld_inference.py",
        output_ext=".mp4",
        supported_suites=["image_static"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        generation={
            "validation_num_frames": 81,
            "chunk_stride": 20,
            "num_chunks": 1,
            "action_source": "worldarena_prompt",
        },
    )


@pytest.mark.parametrize(
    ("camera_token", "expected"),
    [
        ("push_in", {"move": "go forward", "view": "no-op"}),
        ("pull_out", {"move": "go back", "view": "no-op"}),
        ("move_left", {"move": "go left", "view": "no-op"}),
        ("move_right", {"move": "go right", "view": "no-op"}),
        ("pan_left", {"move": "go back and go left", "view": "turn left"}),
        ("pan_right", {"move": "go back and go right", "view": "turn right"}),
        ("orbit_left", {"move": "go forward and go left", "view": "turn right"}),
        ("orbit_right", {"move": "go forward and go right", "view": "turn left"}),
    ],
)
def test_infinite_world_camera_path_maps_to_user_action_definition(
    camera_token: str,
    expected: dict[str, str],
) -> None:
    payload = _build_action_payload(
        {"action_source": "worldarena_prompt"},
        "image_static",
        sample=_make_sample([camera_token]),
    )

    assert payload["actions"] == [expected]
    assert payload["camera_tokens"] == [camera_token]
    assert payload["control_source"] == "camera_path"
    assert payload["prompt_target"] == "Camera orbit left. A static outdoor scene."


def test_infinite_world_multi_segment_camera_path_spans_required_frames() -> None:
    payload = _build_action_payload(
        {
            "action_source": "worldarena_prompt",
            "validation_num_frames": 81,
            "chunk_stride": 20,
            "num_chunks": 1,
        },
        "image_static",
        sample=_make_sample(["pan_left", "move_right", "orbit_right"]),
    )

    assert payload["frames_per_action"] == 27
    assert len(payload["frame_actions"]) == 81
    assert payload["frame_actions"][0] == {"move": "go back and go left", "view": "turn left"}
    assert payload["frame_actions"][27] == {"move": "go right", "view": "no-op"}
    assert payload["frame_actions"][54] == {"move": "go forward and go right", "view": "turn left"}


def test_infinite_world_config_uses_prompt_target_and_image_suites() -> None:
    config = load_model_config(Path("config/models/infinite_world.yaml"))
    adapter = InfiniteWorldAdapter(config)

    assert "image_static" in config.supported_suites
    assert "image_dynamic" in config.supported_suites
    assert config.prompt_mode == "prompt_target"
    assert config.generation["action_source"] == "worldarena_prompt"
    assert adapter.prompt_for(_make_sample(["orbit_left"])) == "Camera orbit left. A static outdoor scene."


def test_infinite_world_batch_invokes_single_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    adapter = InfiniteWorldAdapter(config)
    monkeypatch.setattr(
        "worldarena.models.adapters.infinite_world.worldarena_project_root",
        lambda: tmp_path,
    )
    calls: list[list[str]] = []

    def fake_run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        del cwd, env
        calls.append(command)
        request_file = Path(command[command.index("--requests_json") + 1])
        results_file = Path(command[command.index("--results_json") + 1])
        requests = json.loads(request_file.read_text(encoding="utf-8"))["requests"]
        results = []
        for request in requests:
            Path(request["output_path"]).write_bytes(b"video")
            results.append(
                {
                    "status": "generated",
                    "sample_id": request["sample_id"],
                    "prediction_path": request["output_path"],
                    "prompt": request["prompt"],
                    "camera_path": request["camera_path"],
                    "actions": request["actions"],
                    "control_source": request["control_source"],
                }
            )
        results_file.write_text(json.dumps({"results": results}), encoding="utf-8")

    monkeypatch.setattr("worldarena.models.adapters.infinite_world.run_command", fake_run_command)
    requests = [
        PreparedGenerationRequest(
            sample=_make_sample(["orbit_left"], index=1),
            conditioning_image=tmp_path / "reference.png",
            output_path=tmp_path / "out" / "sample_1.mp4",
            prompt="Camera orbit left. A static outdoor scene.",
        ),
        PreparedGenerationRequest(
            sample=_make_sample(["pan_right"], index=2),
            conditioning_image=tmp_path / "reference.png",
            output_path=tmp_path / "out" / "sample_2.mp4",
            prompt="Camera pan right. A static outdoor scene.",
        ),
    ]

    results = adapter.generate_batch(requests)

    assert len(calls) == 1
    assert set(results) == {"image-static-001", "image-static-002"}
    assert results["image-static-001"]["status"] == "generated"
