from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.open_dreamer import OpenDreamerAdapter
from worldarena.models.adapters.open_dreamer_actions import (
    build_vpt_action_sequence,
    segment_interval_counts,
)
from worldarena.models.adapters.open_dreamer_batch_runner import (
    plan_row,
    prepare_context_frames,
    resolve_checkpoint_layout,
    write_video,
)
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


CONFIG_PATH = Path("config/models/open_dreamer.yaml")


def _sample(*, suite: str = "video_static") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="open-dreamer-sample",
        suite=suite,
        split="test",
        asset_level=suite,
        modality="video",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/test_scene",
        path="/tmp/reference.mp4",
        relative_path="static/photorealistic/outdoor/test_scene/reference.mp4",
        reference_path="/tmp/reference.mp4",
        prediction_stem="open_dreamer_sample",
        conditioning_strategy="reference_video",
        prompt_current="A quiet outdoor scene.",
        prompt_target="Camera moves forward. A quiet outdoor scene.",
        style="photorealistic",
        environment="outdoor",
        scene="test_scene",
        motion_category=None,
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
        camera_path=["push_in", "pan_right"],
        generation_mode="dynamic" if suite.endswith("_dynamic") else "static",
        width=1920,
        height=1080,
        fps=30.0,
    )


def _request(tmp_path: Path, *, suite: str = "video_static") -> PreparedGenerationRequest:
    return PreparedGenerationRequest(
        sample=_sample(suite=suite),
        conditioning_image=tmp_path / "conditioning.mp4",
        output_path=tmp_path / "output.mp4",
        prompt="",
    )


def _orbax_checkpoint(root: Path, *, step: int = 250000, items: tuple[str, ...]) -> Path:
    step_dir = root / str(step)
    for item in items:
        (step_dir / item).mkdir(parents=True, exist_ok=True)
        (step_dir / item / "_METADATA").write_bytes(b"{}")
    (step_dir / "_CHECKPOINT_METADATA").write_bytes(b"{}")
    return step_dir


def test_open_dreamer_config_selects_the_adapter() -> None:
    config = load_model_config(CONFIG_PATH)
    adapter = build_model_adapter(config)

    assert isinstance(adapter, OpenDreamerAdapter)
    assert adapter.batch_checkpoint_load_policy() == "load_once"
    # The model has no text conditioning path at all.
    assert config.prompt_mode == "none"
    assert adapter.prompt_for(_sample()) == ""
    # The rollout needs real context frames, so the harness must hand over a clip
    # holding at least as many frames as the rollout encodes.
    assert config.reference_mode == "reference_video"
    assert config.generation["conditioning_video_frames"] >= config.generation["context_frames"]


def test_open_dreamer_payload_holds_the_camera_actions_for_the_suite(tmp_path: Path) -> None:
    adapter = OpenDreamerAdapter(load_model_config(CONFIG_PATH))

    static_payload = adapter.batch_spec_payload(_request(tmp_path))
    dynamic_payload = adapter.batch_spec_payload(_request(tmp_path, suite="video_dynamic"))

    assert static_payload["camera_path"] == ["push_in", "pan_right"]
    assert static_payload["camera_source"] == "worldarena_camera_path"
    assert dynamic_payload["camera_path"] == ["fixed"]
    assert dynamic_payload["camera_source"] == "dynamic_fixed"


def test_open_dreamer_camera_aliases_resolve_to_canonical_tokens(tmp_path: Path) -> None:
    adapter = OpenDreamerAdapter(load_model_config(CONFIG_PATH))
    request = _request(tmp_path)
    request.sample.camera_path = ["dolly_in", "truck_left", "stay"]

    payload = adapter.batch_spec_payload(request)

    assert payload["camera_path"] == ["push_in", "move_left", "fixed"]


def test_open_dreamer_action_segments_match_the_synthetic_camera_split() -> None:
    # The camera metrics interpolate the reference trajectory over frame_count - 1
    # transitions with the same base/remainder split, so segment boundaries must
    # line up with the actions the model is given.
    assert segment_interval_counts(2, 9) == [5, 4]
    assert segment_interval_counts(3, 8) == [3, 3, 2]
    assert segment_interval_counts(1, 0) == [0]


def test_open_dreamer_actions_are_noop_over_the_context_window() -> None:
    actions, metadata = build_vpt_action_sequence(
        ["push_in", "pan_right"],
        context_frames=4,
        horizon=8,
        mouse_pan_pixels=20.0,
        mouse_tilt_pixels=13.0,
    )

    assert len(actions) == 12
    assert metadata["total_frames"] == 12
    assert metadata["context_action_policy"] == "noop"
    # Frames 0..2 are pure context transitions; frame 3 already carries the motion
    # that produces the first generated frame.
    for action in actions[:3]:
        assert action == {
            "mouse": {"dx": 0.0, "dy": 0.0, "buttons": [], "dwheel": 0.0},
            "keyboard": {"keys": []},
        }
    assert actions[3]["keyboard"]["keys"] == ["key.keyboard.w"]
    # The trailing entry is dropped by upstream shift_actions.
    assert actions[-1]["keyboard"]["keys"] == []

    walk = [index for index, action in enumerate(actions) if action["keyboard"]["keys"]]
    yaw = [index for index, action in enumerate(actions) if action["mouse"]["dx"] > 0]
    assert walk == [3, 4, 5, 6]
    assert yaw == [7, 8, 9, 10]
    assert [segment["frames"] for segment in metadata["action_segments"]] == [4, 4]
    assert metadata["action_segments"][1]["mouse_dx_degrees_per_frame"] == pytest.approx(3.0)


def test_open_dreamer_rejects_camera_tokens_the_vpt_action_space_cannot_express() -> None:
    with pytest.raises(ValueError, match="roll_cw"):
        build_vpt_action_sequence(
            ["roll_cw"],
            context_frames=2,
            horizon=4,
            mouse_pan_pixels=20.0,
            mouse_tilt_pixels=13.0,
        )


def test_open_dreamer_rejects_rotation_without_a_mouse_delta() -> None:
    with pytest.raises(ValueError, match="mouse_pan_pixels"):
        build_vpt_action_sequence(
            ["pan_left"],
            context_frames=2,
            horizon=4,
            mouse_pan_pixels=0.0,
            mouse_tilt_pixels=0.0,
        )


def test_open_dreamer_context_clip_matches_the_trained_frame_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "conditioning.png"
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(image_path)

    clip, details = prepare_context_frames(
        image_path,
        context_frames=4,
        model_height=368,
        model_width=640,
        content_height=360,
        fit_mode="crop",
    )

    assert clip.shape == (4, 368, 640, 3)
    assert clip.dtype == np.uint8
    # The 4px training pad is zero-filled top and bottom, matching padding_H.
    assert not clip[:, :4].any()
    assert not clip[:, -4:].any()
    assert clip[:, 4:-4].any()
    assert details["pad_top"] == 4
    assert details["pad_bottom"] == 4
    assert details["conditioning_frames_repeated"] is True


def test_open_dreamer_pad_fit_mode_preserves_the_full_source_frame(tmp_path: Path) -> None:
    image_path = tmp_path / "square.png"
    Image.new("RGB", (512, 512), color=(255, 0, 0)).save(image_path)

    clip, _ = prepare_context_frames(
        image_path,
        context_frames=1,
        model_height=368,
        model_width=640,
        content_height=360,
        fit_mode="pad",
    )

    content = clip[0, 4:-4]
    # A square source is letterboxed left and right instead of being cropped.
    assert not content[:, :10].any()
    assert not content[:, -10:].any()
    assert content[:, 320].any()


def test_open_dreamer_rejects_odd_frame_sizes_instead_of_letting_ffmpeg_rescale(
    tmp_path: Path,
) -> None:
    # imageio's macro_block_size would silently stretch a cropped 360-row
    # prediction back to the padded geometry; the writer must reject bad sizes.
    with pytest.raises(ValueError, match="even dimensions"):
        write_video(tmp_path / "odd.mp4", np.zeros((2, 361, 640, 3), dtype=np.uint8), fps=20.0)


def test_open_dreamer_checkpoint_layout_resolves_the_latest_step(tmp_path: Path) -> None:
    root = tmp_path / "open-dreamer"
    _orbax_checkpoint(root, step=100000, items=("tokenizer", "dynamics_ema"))
    step_dir = _orbax_checkpoint(root, step=250000, items=("tokenizer", "dynamics_ema", "meta"))

    layout = resolve_checkpoint_layout(root, use_ema=True)

    assert layout["step"] == 250000
    assert layout["step_dir"] == step_dir
    assert layout["dynamics_item"] == "dynamics_ema"
    assert layout["checkpoint_path"] == root.resolve()


def test_open_dreamer_checkpoint_layout_accepts_a_run_directory(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _orbax_checkpoint(root / "checkpoints", step=1000, items=("tokenizer", "dynamics"))

    layout = resolve_checkpoint_layout(root, use_ema=False)

    assert layout["checkpoint_path"] == (root / "checkpoints").resolve()
    assert layout["dynamics_item"] == "dynamics"


def test_open_dreamer_checkpoint_layout_reports_a_missing_dynamics_item(tmp_path: Path) -> None:
    root = tmp_path / "open-dreamer"
    _orbax_checkpoint(root, step=250000, items=("tokenizer", "dynamics_ema"))

    # Released checkpoints ship dynamics_ema only; asking for the online weights
    # must fail before JAX touches the device.
    with pytest.raises(FileNotFoundError, match="dynamics"):
        resolve_checkpoint_layout(root, use_ema=False)


def test_open_dreamer_plan_row_uses_the_configured_rollout_length() -> None:
    generation = dict(load_model_config(CONFIG_PATH).generation)
    plan = plan_row({"camera_path": ["push_in"]}, generation)

    assert plan["context_frames"] == generation["context_frames"]
    assert plan["horizon"] == generation["horizon"]
    assert len(plan["actions"]) == generation["context_frames"] + generation["horizon"]
    assert plan["action_metadata"]["native_action_space"] == "minecraft_vpt"


def test_open_dreamer_batch_runner_dry_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "open-dreamer"
    (repo_root / "dreamer").mkdir(parents=True)
    spec_path = tmp_path / "batch.jsonl"
    spec_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "conditioning_image": str(tmp_path / "conditioning.mp4"),
                "output_path": str(tmp_path / "predictions" / "output.mp4"),
                "prompt": "",
                "camera_path": ["push_in", "pan_right"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generation_path = tmp_path / "generation.json"
    generation_path.write_text(
        json.dumps({"context_frames": 4, "horizon": 16, "fps": 20}),
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldarena.models.adapters.open_dreamer_batch_runner",
            "--repo_root",
            str(repo_root),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ],
        cwd=project_root,
        env={
            **os.environ,
            "PYTHONPATH": str(project_root),
            "WORLDARENA_OPEN_DREAMER_DRY_RUN": "1",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["status"] == "dry_run"
    assert payload["total_frames"] == 20
    assert payload["camera_path"] == ["push_in", "pan_right"]
    assert payload["native_action_space"] == "minecraft_vpt"

    # The VPT stream is written out so a run can be replayed with upstream tooling.
    actions = [
        json.loads(line)
        for line in Path(payload["vpt_actions_path"]).read_text().splitlines()
        if line.strip()
    ]
    assert len(actions) == 20
    assert Path(payload["action_sidecar"]).is_file()
