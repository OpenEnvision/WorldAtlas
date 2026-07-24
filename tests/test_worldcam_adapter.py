from __future__ import annotations

from pathlib import Path

import numpy as np

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters import worldcam as worldcam_module
from worldarena.models.adapters.worldcam import WorldCamAdapter, _resolve_worldcam_generation
from worldarena.models.config import load_model_config


def _sample(annotation_path: Path) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="video-static-worldcam",
        suite="video_static",
        split="test",
        asset_level="video_static",
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="photorealistic/indoor/workspaces/0-5s",
        path="/tmp/reference.mp4",
        relative_path="indoor/workspaces/0-5s/reference.mp4",
        reference_path="/tmp/reference.mp4",
        prediction_stem="video_static_worldcam",
        conditioning_strategy="first_65_frames",
        prompt_current="A static room.",
        prompt_target="A static room.",
        style="photorealistic",
        environment="indoor",
        scene="workspaces",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        annotation_path=str(annotation_path),
        has_annotation=True,
        has_pose=True,
        fps=29.97002997002997,
        duration_seconds=149 / 29.97002997002997,
        conditioning_frame_count=65,
    )


def test_worldcam_auto_values_follow_gt_pose_count_and_fps(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation"
    annotation_path.mkdir()
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 149, axis=0)
    np.save(annotation_path / "poses.npy", poses)

    config = load_model_config(Path("config/models/worldcam.yaml"))
    resolved = _resolve_worldcam_generation(_sample(annotation_path), config.generation)

    assert resolved["camera_frames"] == 149
    assert resolved["output_frames"] == 149
    assert resolved["output_fps"] == 29.97002997002997
    assert resolved["num_ar_steps"] == 30


def test_worldcam_generate_passes_resolved_values(tmp_path: Path, monkeypatch) -> None:
    annotation_path = tmp_path / "annotation"
    annotation_path.mkdir()
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 149, axis=0)
    np.save(annotation_path / "poses.npy", poses)
    np.save(annotation_path / "intrinsics.npy", np.ones((149, 4), dtype=np.float32))

    captured: dict[str, list[str]] = {}
    output_path = tmp_path / "prediction.mp4"

    def _fake_run(command, **_kwargs) -> None:
        captured["command"] = command
        output_path.touch()

    monkeypatch.setattr(worldcam_module, "run_command", _fake_run)
    config = load_model_config(Path("config/models/worldcam.yaml"))
    result = WorldCamAdapter(config).generate(
        _sample(annotation_path),
        tmp_path / "conditioning.mp4",
        output_path,
        "A static room.",
    )

    command = captured["command"]
    assert command[command.index("--camera_frames") + 1] == "149"
    assert command[command.index("--output_frames") + 1] == "149"
    assert command[command.index("--output_fps") + 1] == "29.97002997002997"
    assert command[command.index("--num_ar_steps") + 1] == "30"
    assert result["resolved_generation"]["gt_frame_count"] == 149
