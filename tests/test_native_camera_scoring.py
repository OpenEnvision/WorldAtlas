"""Regression tests for the native_intended camera_error "small-motion floor" fix.

Before the fix, `score_native_intended_cameras` normalized trajectory error by
`ate / max(achieved_motion, floor)`. Whenever the model's real camera motion
was small (routine for compact, per-keypress discrete-action control), this
blew the relative error up toward the 1.0 cap even when the direction was
completely correct -- indistinguishable, from the score alone, from ignoring
the instruction. These tests pin down the fixed behavior: correct-but-modest
motion should score well, wrong-direction motion should still score poorly,
and "hold still" intents should still penalize drift.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from worldarena.benchmark.native_camera import score_native_intended_cameras
from worldarena.benchmark.schemas import BenchmarkSample


def _sample(*, camera_path: list[str], sample_id: str = "sample") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=sample_id,
        suite="image_static",
        split="test",
        asset_level="image_static",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="static/photorealistic/outdoor/aquatic_landscape",
        path="/tmp/input.png",
        relative_path="static/photorealistic/outdoor/aquatic_landscape/000.png",
        reference_path="/tmp/input.png",
        prediction_stem=sample_id,
        conditioning_strategy="reference_image",
        prompt_current="A coastal scene.",
        prompt_target="A coastal scene.",
        style="photorealistic",
        environment="outdoor",
        scene="aquatic_landscape",
        motion_category=None,
        source_name="unit",
        source_type="unit",
        source_group_id="unit",
        license_bucket="unit",
        camera_path=camera_path,
    )


def _poses_translating(direction: np.ndarray, total_distance: float, n_frames: int = 12) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=n_frames, axis=0)
    direction = direction / np.linalg.norm(direction)
    for index in range(n_frames):
        t = index / max(n_frames - 1, 1)
        poses[index, :3, 3] = (direction * total_distance * t).astype(np.float32)
    return poses


def _poses_static_with_jitter(n_frames: int = 12, jitter: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=n_frames, axis=0)
    if jitter:
        poses[:, :3, 3] += rng.normal(scale=jitter, size=(n_frames, 3)).astype(np.float32)
    return poses


def test_small_but_correct_pull_out_scores_well_above_legacy() -> None:
    """A camera that pulls out (moves -Z) by only 0.09 units -- well below the
    old hard-coded 1.0-unit reference -- should score decently once direction
    is judged, not be crushed by the small-motion floor."""
    sample = _sample(camera_path=["pull_out"])
    poses = _poses_translating(direction=np.asarray([0.0, 0.0, -1.0]), total_distance=0.09)

    result = score_native_intended_cameras(sample, "/tmp/does_not_exist.mp4", poses)

    legacy_accuracy = result["raw"]["legacy_accuracy"]
    fixed_accuracy = result["normalized"]
    assert legacy_accuracy < 0.6, f"expected the legacy formula to still be dragged down by tiny motion, got {legacy_accuracy}"
    assert fixed_accuracy > legacy_accuracy + 0.15, (
        f"fixed accuracy ({fixed_accuracy}) should clearly beat the legacy floor-bugged "
        f"accuracy ({legacy_accuracy}) for correct-direction-but-modest motion"
    )
    assert fixed_accuracy > 0.7


def test_large_correct_pull_out_scores_near_perfect() -> None:
    sample = _sample(camera_path=["pull_out"])
    poses = _poses_translating(direction=np.asarray([0.0, 0.0, -1.0]), total_distance=1.0)

    result = score_native_intended_cameras(sample, "/tmp/does_not_exist.mp4", poses)

    assert result["normalized"] > 0.9


def test_wrong_direction_with_ample_motion_still_scores_poorly() -> None:
    """Moving substantially in the *opposite* direction of the instruction
    must still be penalized -- the fix must not become a rubber stamp."""
    sample = _sample(camera_path=["pull_out"])
    # Instruction is pull_out (-Z); camera instead pushes in (+Z) by a lot.
    poses = _poses_translating(direction=np.asarray([0.0, 0.0, 1.0]), total_distance=0.5)

    result = score_native_intended_cameras(sample, "/tmp/does_not_exist.mp4", poses)

    assert result["normalized"] < 0.15


@pytest.mark.parametrize("small_distance", [0.09, 0.3])
def test_direction_score_monotonic_in_small_motion_regime(small_distance: float) -> None:
    """Regardless of how small the achieved motion is (as long as it is above
    pure noise), correct-direction motion should never score worse than
    wrong-direction motion of the same magnitude."""
    sample_correct = _sample(camera_path=["move_left"], sample_id="correct")
    sample_wrong = _sample(camera_path=["move_left"], sample_id="wrong")

    correct_poses = _poses_translating(direction=np.asarray([-1.0, 0.0, 0.0]), total_distance=small_distance)
    wrong_poses = _poses_translating(direction=np.asarray([1.0, 0.0, 0.0]), total_distance=small_distance)

    correct_result = score_native_intended_cameras(sample_correct, "/tmp/does_not_exist.mp4", correct_poses)
    wrong_result = score_native_intended_cameras(sample_wrong, "/tmp/does_not_exist.mp4", wrong_poses)

    assert correct_result["normalized"] > wrong_result["normalized"]


def test_fixed_intent_penalizes_large_drift_but_tolerates_noise() -> None:
    sample = _sample(camera_path=["fixed"])

    still = score_native_intended_cameras(sample, "/tmp/does_not_exist.mp4", _poses_static_with_jitter(jitter=0.0))
    drifting_poses = _poses_translating(direction=np.asarray([1.0, 0.0, 0.0]), total_distance=1.0)
    drifting = score_native_intended_cameras(sample, "/tmp/does_not_exist.mp4", drifting_poses)

    assert still["normalized"] > 0.95
    assert drifting["normalized"] < 0.1


def test_dreamx_style_three_step_action_seq_is_not_collapsed_into_one_action(tmp_path) -> None:
    """Regression test for a real bug found while validating the floor-effect
    fix against cached MegaSAM reconstructions: dreamx_world's sidecar
    payloads store a *sequential* list of per-segment actions under the
    "action_seq" key (each with its own `action_speed_list` entry). The
    scorer used to treat any 3-item "action_seq" list as a single
    simultaneous combined action (matrix-game's convention for a different
    key), silently discarding two of the three intended camera segments.
    """
    prediction_path = tmp_path / "clip.mp4"
    prediction_path.write_bytes(b"")
    sidecar = tmp_path / "clip_actions.json"
    sidecar.write_text(
        json.dumps(
            {
                "action_source": "worldarena_camera_path",
                "control_source": "camera_path_native",
                "camera_path": ["pan_left", "pull_out", "move_left"],
                "native_action_space": "dreamx_action_keys",
                "action_seq": ["j", "s", "a"],
                "action_speed_list": [6, 4, 4],
            }
        ),
        encoding="utf-8",
    )
    sample = _sample(camera_path=["pan_left", "pull_out", "move_left"], sample_id="clip")

    n_frames = 30
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], repeats=n_frames, axis=0)
    # Segment 1 (0..9): turn left in place. Segment 2 (10..19): pull out.
    # Segment 3 (20..29): move left. Large, unambiguous motions throughout.
    rotation = np.eye(3, dtype=np.float32)
    for index in range(1, 10):
        rotation = rotation @ _rot_y(-3.0)
        poses[index, :3, :3] = rotation
    for index in range(10, n_frames):
        poses[index, :3, :3] = rotation
    for index in range(10, 20):
        poses[index, :3, 3] = poses[index - 1, :3, 3] + rotation @ np.asarray([0.0, 0.0, -0.1], dtype=np.float32)
    for index in range(20, n_frames):
        poses[index, :3, 3] = poses[index - 1, :3, 3] + rotation @ np.asarray([-0.1, 0.0, 0.0], dtype=np.float32)

    result = score_native_intended_cameras(sample, str(prediction_path), poses)

    labels = [segment["label"] for segment in result["details"]["native_intent_segments"]]
    assert labels == ["j", "s", "a"], f"expected 3 sequential segments, got {labels}"
    assert result["normalized"] > 0.8


def _rot_y(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
