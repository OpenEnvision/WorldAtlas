"""Matrix-Game interactive world model adapter for WorldAtlas.

Encodes keyboard/mouse action sequences from ``camera_path`` tokens and routes
inference to Matrix-Game 1/2/3 runner modules depending on model variant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any
from uuid import uuid4

import numpy as np

from worldatlas.common.checkpoints import rehome_legacy_workspace_path
from worldatlas.common.progress import log_progress
from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.models.adapters.base import ModelAdapter, PreparedGenerationRequest, plan_rollout
from worldatlas.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
    run_sequential_generate_batch,
    worldatlas_project_root,
)
from worldatlas.models.adapters.external_batch import run_logged_command
from worldatlas.models.adapters.windowed_rollout import align_matrix_game_2_latent_frames

_LOAD_ONCE_RUNNERS = {
    "matrix_game_1": "worldatlas.models.adapters.matrix_game.matrix_game_1_batch_runner",
    "matrix_game_2": "worldatlas.models.adapters.matrix_game.matrix_game_2_batch_runner",
}


_DEFAULT_ACTIONS = ["forward", "camera_r", "forward_right", "camera_l", "forward"]
_MATRIX_GAME_1_AND_3_KEYBOARD_IDX = {
    "forward": 0,
    "back": 1,
    "left": 2,
    "right": 3,
    "jump": 4,
    "attack": 5,
}
_MATRIX_GAME_2_SPECS = {
    "universal": {
        "keyboard_idx": {
            "forward": 0,
            "back": 1,
            "left": 2,
            "right": 3,
        },
        "mouse_enabled": True,
        "mouse_step": 0.1,
    },
    "gta_drive": {
        "keyboard_idx": {
            "forward": 0,
            "back": 1,
        },
        "mouse_enabled": True,
        "mouse_step": 0.1,
    },
    "templerun": {
        "keyboard_idx": {
            "nomove": 0,
            "jump": 1,
            "slide": 2,
            "turnleft": 3,
            "turnright": 4,
            "leftside": 5,
            "rightside": 6,
        },
        "mouse_enabled": False,
        "mouse_step": 0.0,
    },
}
_COMBINATION_MAP = {
    "forward_left": ["forward", "left"],
    "forward_right": ["forward", "right"],
    "back_left": ["back", "left"],
    "back_right": ["back", "right"],
    "forward_jump": ["forward", "jump"],
    "back_jump": ["back", "jump"],
    "left_jump": ["left", "jump"],
    "right_jump": ["right", "jump"],
    "forward_attack": ["forward", "attack"],
    "back_attack": ["back", "attack"],
    "left_attack": ["left", "attack"],
    "right_attack": ["right", "attack"],
    "jump_attack": ["jump", "attack"],
    "idle": [],
}
_WORLDATLAS_CAMERA_ACTIONS = {
    "fixed": "idle",
    "push_in": "forward",
    "pull_out": "back",
    "pan_left": "back_left_camera_l",
    "pan_right": "back_right_camera_r",
    "move_left": "left",
    "move_right": "right",
    "orbit_left": "forward_left_camera_r",
    "orbit_right": "forward_right_camera_l",
}
_WORLDATLAS_CAMERA_PHRASES = {
    "camera fixed": "fixed",
    "fixed camera": "fixed",
    "camera push in": "push_in",
    "camera pushes in": "push_in",
    "camera move forward": "push_in",
    "camera moves forward": "push_in",
    "camera pull out": "pull_out",
    "camera pulls out": "pull_out",
    "camera move backward": "pull_out",
    "camera moves backward": "pull_out",
    "camera pan left": "pan_left",
    "camera pans left": "pan_left",
    "camera pan right": "pan_right",
    "camera pans right": "pan_right",
    "camera move left": "move_left",
    "camera moves left": "move_left",
    "camera truck left": "move_left",
    "camera move right": "move_right",
    "camera moves right": "move_right",
    "camera truck right": "move_right",
    "camera orbit left": "orbit_left",
    "camera orbits left": "orbit_left",
    "camera orbit right": "orbit_right",
    "camera orbits right": "orbit_right",
}
_MATRIX_GAME3_GT_CAMERA_ACTION_SOURCES = {
    "gt_camera",
    "gt_camera_path",
    "video_gt_camera",
    "vipe",
    "vipe_camera",
    "vipe_camera_path",
    "vipe_gt_camera",
    "vipe_gt_camera_path",
}


def _matrix_game_variant(family: str) -> str:
    normalized = family.lower().replace("-", "_")
    if normalized in {"matrix_game", "matrix_game_3"}:
        return "matrix_game_3"
    if normalized in {"matrix_game_1", "matrix_game_2"}:
        return normalized
    raise ValueError(f"Unsupported Matrix-Game family: {family}")


def matrix_game_batch_runner_module(family: str) -> str | None:
    """Persistent in-process runner for variants that can keep weights resident."""
    return _LOAD_ONCE_RUNNERS.get(_matrix_game_variant(family))


def _matrix_game_2_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("mode", "universal")).strip().lower()
    if mode not in _MATRIX_GAME_2_SPECS:
        supported = ", ".join(sorted(_MATRIX_GAME_2_SPECS))
        raise ValueError(f"Unsupported Matrix-Game-2 mode: {mode}. Supported values: {supported}")
    return mode


def _camera_value_map(step: float) -> dict[str, np.ndarray]:
    return {
        "camera_up": np.asarray([step, 0.0], dtype=np.float32),
        "camera_down": np.asarray([-step, 0.0], dtype=np.float32),
        "camera_l": np.asarray([0.0, -step], dtype=np.float32),
        "camera_r": np.asarray([0.0, step], dtype=np.float32),
        "camera_ul": np.asarray([step, -step], dtype=np.float32),
        "camera_ur": np.asarray([step, step], dtype=np.float32),
        "camera_dl": np.asarray([-step, -step], dtype=np.float32),
        "camera_dr": np.asarray([-step, step], dtype=np.float32),
    }


def _known_tokens(
    *,
    keyboard_idx: dict[str, int],
    camera_map: dict[str, np.ndarray],
    extra_tokens: set[str] | None = None,
) -> set[str]:
    return set(keyboard_idx) | set(camera_map) | set(_COMBINATION_MAP) | (extra_tokens or set())


def _tokenize_action(action: str, known_tokens: set[str]) -> list[str]:
    normalized = str(action).strip()
    if not normalized:
        return []
    if normalized in known_tokens:
        return [normalized]

    pieces = normalized.split("_")
    tokens: list[str] = []
    cursor = 0
    while cursor < len(pieces):
        matched: str | None = None
        matched_width = 0
        for end in range(len(pieces), cursor, -1):
            candidate = "_".join(pieces[cursor:end])
            if candidate in known_tokens:
                matched = candidate
                matched_width = end - cursor
                break
        if matched is None:
            raise ValueError(f"Unsupported Matrix-Game action token: {action}")
        tokens.append(matched)
        cursor += matched_width
    return tokens


def _expand_action_tokens(action: str, known_tokens: set[str]) -> list[str]:
    expanded: list[str] = []
    for token in _tokenize_action(action, known_tokens):
        expanded.extend(_COMBINATION_MAP.get(token, [token]))
    return expanded


def _action_plan(config: dict[str, Any], suite: str) -> list[str]:
    plan = dict(config.get("action_plan", {}))
    by_suite = plan.get("by_suite", {})
    actions = by_suite.get(suite) or plan.get("default") or _DEFAULT_ACTIONS
    return [str(action) for action in actions]


# action_source values that mean "take the itinerary from the benchmark sample".
_WORLDATLAS_CAMERA_ACTION_SOURCES = frozenset(
    {
        "worldatlas_prompt",
        "prompt",
        "prompt_target",
        "camera_path",
        "worldatlas",
        "worldatlas_camera",
        "worldatlas_camera_path",
    }
)


def _normalized_camera_token(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _worldatlas_camera_token(sample: BenchmarkSample) -> tuple[str, str]:
    camera_path = [
        _normalized_camera_token(token)
        for token in sample.camera_path
        if str(token).strip()
    ]
    if camera_path:
        return camera_path[0], "camera_path"

    prompt_prefix = str(sample.prompt_target or sample.prompt_current or "").split(".", 1)[0]
    normalized_prefix = " ".join(
        prompt_prefix.strip().lower().replace("-", " ").replace("_", " ").split()
    )
    if normalized_prefix in _WORLDATLAS_CAMERA_PHRASES:
        return _WORLDATLAS_CAMERA_PHRASES[normalized_prefix], "prompt_target"

    if sample.suite == "image_dynamic" or sample.generation_mode == "dynamic":
        return "fixed", "dynamic_default"
    return "fixed", "fallback"


def _worldatlas_camera_tokens(sample: BenchmarkSample) -> tuple[list[str], str]:
    camera_path = [
        _normalized_camera_token(token)
        for token in sample.camera_path
        if str(token).strip()
    ]
    if camera_path:
        return camera_path, "camera_path"

    token, source = _worldatlas_camera_token(sample)
    return [token], source


def _worldatlas_matrix_game_actions(sample: BenchmarkSample) -> tuple[list[str], list[str], str]:
    if sample.suite == "image_dynamic" or sample.generation_mode == "dynamic":
        return ["fixed"], ["idle"], "dynamic_idle"

    camera_tokens, source = _worldatlas_camera_tokens(sample)
    actions: list[str] = []
    for camera_token in camera_tokens:
        action = _WORLDATLAS_CAMERA_ACTIONS.get(camera_token)
        if action is None:
            raise ValueError(f"unsupported Matrix-Game camera token: {camera_token!r}")
        actions.append(action)
    return camera_tokens, actions or ["idle"], source


def _worldatlas_matrix_game_2_action(sample: BenchmarkSample) -> tuple[str, str, str]:
    camera_tokens, actions, source = _worldatlas_matrix_game_actions(sample)
    return camera_tokens[0], actions[0], source


def _segment_rows(
    *,
    tokens: list[str],
    frames: int,
    keyboard_idx: dict[str, int],
    mouse_enabled: bool,
    camera_map: dict[str, np.ndarray],
    pulse_period_frames: dict[str, int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    keyboard_dim = max(keyboard_idx.values(), default=-1) + 1
    base_keyboard = np.zeros(keyboard_dim, dtype=np.float32)
    base_mouse = np.zeros(2, dtype=np.float32)
    pulse_period_frames = pulse_period_frames or {}
    pulsed_tokens: list[str] = []

    for token in tokens:
        if token in keyboard_idx:
            if token in pulse_period_frames:
                pulsed_tokens.append(token)
            else:
                base_keyboard[keyboard_idx[token]] = 1.0
            continue
        if token in camera_map:
            base_mouse[:] = camera_map[token]
            continue
        raise ValueError(f"Unsupported Matrix-Game action token after expansion: {token}")

    keyboard_rows: list[np.ndarray] = []
    mouse_rows: list[np.ndarray] = [] if mouse_enabled else []
    for frame_index in range(frames):
        row = base_keyboard.copy()
        for token in pulsed_tokens:
            if frame_index % pulse_period_frames[token] == 0:
                row[keyboard_idx[token]] = 1.0
        keyboard_rows.append(row)
        if mouse_enabled:
            mouse_rows.append(base_mouse.copy())
    return keyboard_rows, mouse_rows if mouse_enabled else None


def _pad_or_trim_rows(
    rows: list[np.ndarray],
    *,
    total_frames: int,
    fallback: np.ndarray,
) -> list[np.ndarray]:
    if not rows:
        rows.append(fallback.copy())
    while len(rows) < total_frames:
        rows.append(rows[-1].copy())
    return rows[:total_frames]


def _build_fixed_length_action_payload(
    *,
    config: dict[str, Any],
    suite: str,
    variant: str,
    keyboard_idx: dict[str, int],
    total_frames: int,
    mouse_enabled: bool,
    mouse_step: float,
    default_frames_per_action: int,
    pulse_period_frames: dict[str, int] | None = None,
) -> dict[str, Any]:
    action_plan = dict(config.get("action_plan", {}))
    actions = _action_plan(config, suite)
    frames_per_action = int(action_plan.get("frames_per_action", default_frames_per_action))
    camera_map = _camera_value_map(mouse_step)
    known_tokens = _known_tokens(
        keyboard_idx=keyboard_idx,
        camera_map=camera_map,
        extra_tokens={"nomove"} if "nomove" not in keyboard_idx else None,
    )

    keyboard_rows: list[np.ndarray] = []
    mouse_rows: list[np.ndarray] = []
    for action in actions:
        tokens = _expand_action_tokens(action, known_tokens)
        segment_keyboard, segment_mouse = _segment_rows(
            tokens=tokens,
            frames=frames_per_action,
            keyboard_idx=keyboard_idx,
            mouse_enabled=mouse_enabled,
            camera_map=camera_map,
            pulse_period_frames=pulse_period_frames,
        )
        keyboard_rows.extend(segment_keyboard)
        if mouse_enabled and segment_mouse is not None:
            mouse_rows.extend(segment_mouse)

    keyboard_rows = _pad_or_trim_rows(
        keyboard_rows,
        total_frames=total_frames,
        fallback=np.zeros(max(keyboard_idx.values(), default=-1) + 1, dtype=np.float32),
    )

    payload = {
        "variant": variant,
        "actions": actions,
        "keyboard_condition": np.stack(keyboard_rows).tolist(),
        "total_frames": total_frames,
    }
    if mouse_enabled:
        mouse_rows = _pad_or_trim_rows(
            mouse_rows,
            total_frames=total_frames,
            fallback=np.zeros(2, dtype=np.float32),
        )
        payload["mouse_condition"] = np.stack(mouse_rows).tolist()
    return payload


def _matrix_game_1_payload(
    config: dict[str, Any],
    suite: str,
    *,
    sample: BenchmarkSample | None = None,
) -> dict[str, Any]:
    total_frames = int(config.get("video_length", 65))
    action_source = str(config.get("action_source", "config_action_plan")).strip().lower()
    camera_tokens: list[str] | None = None
    control_source: str | None = None
    payload_config = config

    if action_source in _WORLDATLAS_CAMERA_ACTION_SOURCES:
        if sample is None:
            raise ValueError(
                "Matrix-Game-1 worldatlas camera action_source requires a BenchmarkSample"
            )
        camera_tokens, matrix_actions, control_source = _worldatlas_matrix_game_actions(sample)
        # Every action is held for the same number of frames, so an inverse pair in a
        # closed itinerary stays balanced and the camera returns to its anchor.
        payload_config = dict(config)
        payload_config["action_plan"] = {
            "default": matrix_actions,
            "by_suite": {suite: matrix_actions},
            "frames_per_action": max(total_frames // max(len(matrix_actions), 1), 1),
        }

    payload = _build_fixed_length_action_payload(
        config=payload_config,
        suite=suite,
        variant="matrix_game_1",
        keyboard_idx=_MATRIX_GAME_1_AND_3_KEYBOARD_IDX,
        total_frames=total_frames,
        mouse_enabled=True,
        mouse_step=0.05,
        default_frames_per_action=max(
            total_frames // max(len(_action_plan(payload_config, suite)), 1), 1
        ),
        pulse_period_frames={"attack": int(config.get("attack_period_frames", 8))},
    )
    payload["video_length"] = total_frames
    payload["action_source"] = action_source
    if camera_tokens is not None:
        payload.update(
            {
                "control_source": control_source,
                "camera_token": camera_tokens[0],
                "camera_tokens": camera_tokens,
                "camera_path": list(sample.camera_path) if sample is not None else [],
            }
        )
    return payload


def _matrix_game_2_worldatlas_prompt_payload(
    *,
    config: dict[str, Any],
    suite: str,
    sample: BenchmarkSample | None,
    mode: str,
    spec: dict[str, Any],
    num_output_frames: int,
    total_frames: int,
    action_source: str = "worldatlas_prompt",
) -> dict[str, Any]:
    if sample is None:
        raise ValueError("Matrix-Game-2 worldatlas_prompt action_source requires a BenchmarkSample")

    camera_tokens, matrix_actions, control_source = _worldatlas_matrix_game_actions(sample)
    prompt_config = dict(config)
    frames_per_action = max(math.ceil(total_frames / max(len(matrix_actions), 1)), 1)
    prompt_config["action_plan"] = {
        "default": matrix_actions,
        "by_suite": {suite: matrix_actions},
        "frames_per_action": frames_per_action,
    }
    payload = _build_fixed_length_action_payload(
        config=prompt_config,
        suite=suite,
        variant="matrix_game_2",
        keyboard_idx=spec["keyboard_idx"],
        total_frames=total_frames,
        mouse_enabled=bool(spec["mouse_enabled"]),
        mouse_step=float(spec["mouse_step"]),
        default_frames_per_action=total_frames,
    )
    payload.update(
        {
            "mode": mode,
            "num_output_frames": num_output_frames,
            "action_source": action_source,
            "control_source": control_source,
            "camera_token": camera_tokens[0],
            "camera_tokens": camera_tokens,
            "camera_path": list(sample.camera_path),
            "prompt_current": sample.prompt_current,
            "prompt_target": sample.prompt_target,
            "prompt_control_note": (
                "image_dynamic uses idle/no-action controls so Matrix-Game-2 can freely evolve the scene"
                if sample.suite == "image_dynamic" or sample.generation_mode == "dynamic"
                else "image_static maps the full WorldAtlas camera_path to Matrix-Game-2 universal controls"
            ),
        }
    )
    return payload


def _matrix_game_2_payload(
    config: dict[str, Any],
    suite: str,
    sample: BenchmarkSample | None = None,
) -> dict[str, Any]:
    mode = _matrix_game_2_mode(config)
    spec = _MATRIX_GAME_2_SPECS[mode]
    target_seconds = config.get("target_duration_seconds")
    if target_seconds is None:
        num_output_frames = int(config.get("num_output_frames", 15))
        total_frames = int(config.get("num_frames", (num_output_frames - 1) * 4 + 1))
    else:
        # Pixel frames follow ``(num_output_frames - 1) * 4 + 1``, so the latent count
        # is the unit and the leading anchor frame is the base. Causal inference
        # further requires that latent count to be a multiple of num_frame_per_block
        # (3 in every official YAML).
        plan = plan_rollout(
            target_seconds=float(target_seconds),
            native_fps=float(config.get("output_fps", 12)),
            unit_frames=4,
            base_frames=1,
        )
        num_output_frames = align_matrix_game_2_latent_frames(plan.unit_count + 1)
        total_frames = (num_output_frames - 1) * 4 + 1
    action_source = str(config.get("action_source", "worldatlas_plan")).strip().lower()
    if action_source in {"official", "official_bench", "official_bench_actions"}:
        payload = _matrix_game_2_official_bench_payload(
            mode=mode,
            total_frames=total_frames,
            seed=int(config.get("seed", 42)),
        )
        payload["mode"] = mode
        payload["num_output_frames"] = num_output_frames
        payload["action_source"] = "official_bench"
        return payload

    if action_source in {
        "worldatlas_prompt",
        "prompt",
        "prompt_target",
        "camera_path",
        "worldatlas_camera_path",
    }:
        return _matrix_game_2_worldatlas_prompt_payload(
            config=config,
            suite=suite,
            sample=sample,
            mode=mode,
            spec=spec,
            num_output_frames=num_output_frames,
            total_frames=total_frames,
            action_source=action_source,
        )

    if sample is not None and action_source in {"worldatlas", "worldatlas_camera"}:
        return _matrix_game_2_worldatlas_prompt_payload(
            config=config,
            suite=suite,
            sample=sample,
            mode=mode,
            spec=spec,
            num_output_frames=num_output_frames,
            total_frames=total_frames,
            action_source=action_source,
        )

    payload = _build_fixed_length_action_payload(
        config=config,
        suite=suite,
        variant="matrix_game_2",
        keyboard_idx=spec["keyboard_idx"],
        total_frames=total_frames,
        mouse_enabled=bool(spec["mouse_enabled"]),
        mouse_step=float(spec["mouse_step"]),
        default_frames_per_action=int(dict(config.get("action_plan", {})).get("frames_per_action", 12)),
    )
    payload["mode"] = mode
    payload["num_output_frames"] = num_output_frames
    payload["action_source"] = action_source
    return payload


def _combine_official_matrix_game_2_actions(
    data: list[dict[str, np.ndarray]],
    *,
    total_frames: int,
    keyboard_dim: int,
    mouse_enabled: bool,
    rng: random.Random,
) -> dict[str, Any]:
    if total_frames % 4 != 1:
        raise ValueError(f"Matrix-Game-2 official action frame count must be 4n+1, got {total_frames}")

    keyboard_condition = np.zeros((total_frames, keyboard_dim), dtype=np.float32)
    mouse_condition = np.zeros((total_frames, 2), dtype=np.float32) if mouse_enabled else None
    current_frame = 0
    selections = [12]

    while current_frame < total_frames:
        segment_frames = selections[rng.randint(0, len(selections) - 1)]
        selected = data[rng.randint(0, len(data) - 1)]
        keyboard_segment = selected["keyboard_condition"]
        mouse_segment = selected.get("mouse_condition")

        if current_frame == 0:
            keyboard_condition[:1] = keyboard_segment[:1]
            if mouse_enabled and mouse_condition is not None and mouse_segment is not None:
                mouse_condition[:1] = mouse_segment[:1]
            current_frame = 1
            continue

        segment_frames = min(segment_frames, total_frames - current_frame)
        repeat_time = segment_frames // 4
        keyboard_condition[current_frame : current_frame + segment_frames] = np.tile(
            keyboard_segment,
            (repeat_time, 1),
        )
        if mouse_enabled and mouse_condition is not None and mouse_segment is not None:
            mouse_condition[current_frame : current_frame + segment_frames] = np.tile(
                mouse_segment,
                (repeat_time, 1),
            )
        current_frame += segment_frames

    payload: dict[str, Any] = {
        "variant": "matrix_game_2",
        "actions": ["official_bench"],
        "keyboard_condition": keyboard_condition.tolist(),
        "total_frames": total_frames,
    }
    if mouse_enabled and mouse_condition is not None:
        payload["mouse_condition"] = mouse_condition.tolist()
    return payload


def _matrix_game_2_official_bench_payload(
    *,
    mode: str,
    total_frames: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    if mode == "universal":
        single_actions = ["forward", "left", "right"]
        double_actions = ["forward_left", "forward_right"]
        camera_actions = ["camera_l", "camera_r"]
        actions_to_test = (
            double_actions * 5
            + camera_actions * 5
            + single_actions * 5
            + [f"{action}_{camera}" for action in single_actions + double_actions for camera in camera_actions]
        )
        keyboard_idx = {"forward": 0, "back": 1, "left": 2, "right": 3}
        camera_map = {
            "camera_up": [0.1, 0.0],
            "camera_down": [-0.1, 0.0],
            "camera_l": [0.0, -0.1],
            "camera_r": [0.0, 0.1],
            "camera_ur": [0.1, 0.1],
            "camera_ul": [0.1, -0.1],
            "camera_dr": [-0.1, 0.1],
            "camera_dl": [-0.1, -0.1],
        }
        base_actions = single_actions + camera_actions
        data: list[dict[str, np.ndarray]] = []
        for action_name in actions_to_test:
            keyboard_condition = np.zeros((4, 4), dtype=np.float32)
            mouse_condition = np.zeros((4, 2), dtype=np.float32)
            for sub_action in base_actions:
                if sub_action not in action_name:
                    continue
                if sub_action in camera_map:
                    mouse_condition[:] = np.asarray(camera_map[sub_action], dtype=np.float32)
                elif sub_action in keyboard_idx:
                    keyboard_condition[:, keyboard_idx[sub_action]] = 1.0
            data.append(
                {
                    "keyboard_condition": keyboard_condition,
                    "mouse_condition": mouse_condition,
                }
            )
        return _combine_official_matrix_game_2_actions(
            data,
            total_frames=total_frames,
            keyboard_dim=4,
            mouse_enabled=True,
            rng=rng,
        )

    if mode == "gta_drive":
        single_actions = ["forward", "back"]
        camera_actions = ["camera_l", "camera_r"]
        actions_to_test = (
            camera_actions * 2
            + single_actions * 2
            + [f"{action}_{camera}" for action in single_actions for camera in camera_actions]
        )
        keyboard_idx = {"forward": 0, "back": 1}
        camera_map = {"camera_l": [0.0, -0.1], "camera_r": [0.0, 0.1]}
        base_actions = single_actions + camera_actions
        data = []
        for action_name in actions_to_test:
            keyboard_condition = np.zeros((4, 2), dtype=np.float32)
            mouse_condition = np.zeros((4, 2), dtype=np.float32)
            for sub_action in base_actions:
                if sub_action not in action_name:
                    continue
                if sub_action in camera_map:
                    mouse_condition[:] = np.asarray(camera_map[sub_action], dtype=np.float32)
                elif sub_action in keyboard_idx:
                    keyboard_condition[:, keyboard_idx[sub_action]] = 1.0
            data.append(
                {
                    "keyboard_condition": keyboard_condition,
                    "mouse_condition": mouse_condition,
                }
            )
        return _combine_official_matrix_game_2_actions(
            data,
            total_frames=total_frames,
            keyboard_dim=2,
            mouse_enabled=True,
            rng=rng,
        )

    if mode == "templerun":
        actions_to_test = ["jump", "slide", "leftside", "rightside", "turnleft", "turnright", "nomove"]
        keyboard_idx = {
            "nomove": 0,
            "jump": 1,
            "slide": 2,
            "turnleft": 3,
            "turnright": 4,
            "leftside": 5,
            "rightside": 6,
        }
        data = []
        for action_name in actions_to_test:
            keyboard_condition = np.zeros((4, 7), dtype=np.float32)
            for sub_action, column in keyboard_idx.items():
                if sub_action in action_name:
                    keyboard_condition[:, column] = 1.0
            data.append({"keyboard_condition": keyboard_condition})
        return _combine_official_matrix_game_2_actions(
            data,
            total_frames=total_frames,
            keyboard_dim=7,
            mouse_enabled=False,
            rng=rng,
        )

    raise ValueError(f"Unsupported Matrix-Game-2 mode for official actions: {mode}")


def _annotation_dir_for_sample(sample: BenchmarkSample) -> Path | None:
    if not sample.annotation_path:
        return None
    annotation_path = str(sample.annotation_path)
    if "::" in annotation_path:
        return None
    remapped = rehome_legacy_workspace_path(annotation_path)
    if remapped is None:
        return None
    path = Path(remapped).expanduser()
    return path.resolve() if path.is_dir() else None


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_gt_c2w_poses(annotation_dir: Path) -> np.ndarray:
    for name in ("poses_c2w.npy", "poses.npy"):
        path = annotation_dir / name
        if not path.is_file():
            continue
        poses = np.asarray(np.load(path), dtype=np.float32)
        if poses.ndim == 3 and poses.shape[1:] == (4, 4):
            return poses
    raise FileNotFoundError(f"VIPE camera pose file not found under {annotation_dir}")


def _video_static_target_frames_and_fps(
    *,
    config: dict[str, Any],
    sample: BenchmarkSample,
    annotation_dir: Path | None,
) -> tuple[int, float, dict[str, Any]]:
    quality = _read_json_dict(annotation_dir / "quality_report.json") if annotation_dir is not None else {}
    frame_count = quality.get("source_video_frame_count") or quality.get("frame_count")
    if frame_count is None and annotation_dir is not None and (annotation_dir / "frame_indices.npy").is_file():
        frame_count = int(np.load(annotation_dir / "frame_indices.npy").shape[0])
    if frame_count is None and sample.duration_seconds and sample.fps:
        frame_count = int(round(float(sample.duration_seconds) * float(sample.fps)))
    if frame_count is None:
        frame_count = int(config.get("save_num_frames", 0) or config.get("video_length", 85))

    fps = quality.get("fps") or sample.fps or config.get("fps", 17)
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        fps = float(config.get("fps", 17))

    return max(int(frame_count), 1), fps, quality


def _matrix_game3_iteration_frame_count(first_clip_frames: int, followup_frames: int, num_iterations: int) -> int:
    return first_clip_frames + (max(num_iterations, 1) - 1) * followup_frames


def _matrix_game3_iterations_for_saved_frames(
    *,
    saved_frames: int,
    first_clip_frames: int,
    followup_frames: int,
) -> int:
    if saved_frames <= first_clip_frames:
        return 1
    return 1 + math.ceil((saved_frames - first_clip_frames) / followup_frames)


def _interp_columns(values: np.ndarray, target_count: int) -> np.ndarray:
    if values.shape[0] == target_count:
        return values.astype(np.float32)
    source_x = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=target_count, dtype=np.float32)
    columns = [np.interp(target_x, source_x, values[:, axis]) for axis in range(values.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def _camera_pitch_yaw_from_c2w(poses_c2w: np.ndarray) -> np.ndarray:
    forward = poses_c2w[:, :3, 2].astype(np.float64)
    norm = np.linalg.norm(forward, axis=1, keepdims=True)
    forward = forward / np.maximum(norm, 1e-8)
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    horizontal = np.maximum(np.linalg.norm(forward[:, [0, 2]], axis=1), 1e-8)
    pitch = np.unwrap(np.arctan2(-forward[:, 1], horizontal))
    return np.stack([pitch, yaw], axis=1).astype(np.float32)


def _nearest_rotations(poses_c2w: np.ndarray, target_count: int) -> np.ndarray:
    if poses_c2w.shape[0] == target_count:
        return poses_c2w[:, :3, :3].astype(np.float32)
    indices = np.rint(np.linspace(0, poses_c2w.shape[0] - 1, num=target_count)).astype(np.int64)
    return poses_c2w[indices, :3, :3].astype(np.float32)


def _action_label(keyboard_row: np.ndarray, mouse_row: np.ndarray) -> str:
    parts: list[str] = []
    if keyboard_row[0] > 0.5:
        parts.append("forward")
    if keyboard_row[1] > 0.5:
        parts.append("back")
    if keyboard_row[2] > 0.5:
        parts.append("left")
    if keyboard_row[3] > 0.5:
        parts.append("right")
    if mouse_row[0] > 0.0:
        parts.append("camera_up")
    elif mouse_row[0] < 0.0:
        parts.append("camera_down")
    if mouse_row[1] > 0.0:
        parts.append("camera_r")
    elif mouse_row[1] < 0.0:
        parts.append("camera_l")
    return "_".join(parts) if parts else "idle"


def _compress_action_labels(labels: list[str], *, max_items: int = 64) -> list[str]:
    compressed: list[str] = []
    for label in labels:
        if compressed and compressed[-1] == label:
            continue
        compressed.append(label)
    if len(compressed) <= max_items:
        return compressed
    head = compressed[: max_items // 2]
    tail = compressed[-(max_items - len(head)) :]
    return head + ["..."] + tail


def _scaled_positive_deadzone(
    values: np.ndarray,
    *,
    percentile: float,
    scale: float,
    minimum: float,
) -> float:
    nonzero = values[values > 0.0]
    if not nonzero.size:
        return minimum
    return max(float(np.percentile(nonzero, percentile)) * scale, minimum)


def _quantize_gt_camera_actions_dense(
    *,
    poses_c2w: np.ndarray,
    saved_frames: int,
    total_frames: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keyboard = np.zeros((total_frames, 6), dtype=np.float32)
    mouse = np.zeros((total_frames, 2), dtype=np.float32)
    if poses_c2w.shape[0] < 2 or saved_frames < 2:
        return keyboard, mouse, ["idle"]

    positions = _interp_columns(poses_c2w[:, :3, 3], saved_frames)
    pitch_yaw = _interp_columns(_camera_pitch_yaw_from_c2w(poses_c2w), saved_frames)
    rotations = _nearest_rotations(poses_c2w, saved_frames)

    deltas_world = positions[1:] - positions[:-1]
    local_deltas = np.einsum("nij,nj->ni", np.swapaxes(rotations[:-1], 1, 2), deltas_world)
    planar_steps = np.linalg.norm(local_deltas[:, [0, 2]], axis=1)
    translation_deadzone = _scaled_positive_deadzone(
        planar_steps,
        percentile=float(config.get("gt_camera_translation_deadzone_percentile", 35)),
        scale=float(config.get("gt_camera_translation_deadzone_scale", 0.35)),
        minimum=float(config.get("gt_camera_translation_deadzone", 1e-6)),
    )

    angle_deltas = np.diff(pitch_yaw, axis=0)
    angle_abs = np.abs(angle_deltas)
    angle_deadzone = _scaled_positive_deadzone(
        angle_abs,
        percentile=float(config.get("gt_camera_angle_deadzone_percentile", 40)),
        scale=float(config.get("gt_camera_angle_deadzone_scale", 0.35)),
        minimum=math.radians(float(config.get("gt_camera_angle_deadzone_degrees", 0.03))),
    )

    mouse_step = float(config.get("gt_camera_mouse_step", 0.1))
    labels: list[str] = []
    usable_rows = min(saved_frames - 1, total_frames)
    for index in range(usable_rows):
        local_x = float(local_deltas[index, 0])
        local_z = float(local_deltas[index, 2])
        if local_z > translation_deadzone:
            keyboard[index, _MATRIX_GAME_1_AND_3_KEYBOARD_IDX["forward"]] = 1.0
        elif local_z < -translation_deadzone:
            keyboard[index, _MATRIX_GAME_1_AND_3_KEYBOARD_IDX["back"]] = 1.0
        if local_x > translation_deadzone:
            keyboard[index, _MATRIX_GAME_1_AND_3_KEYBOARD_IDX["right"]] = 1.0
        elif local_x < -translation_deadzone:
            keyboard[index, _MATRIX_GAME_1_AND_3_KEYBOARD_IDX["left"]] = 1.0

        delta_pitch = float(angle_deltas[index, 0])
        delta_yaw = float(angle_deltas[index, 1])
        if delta_pitch > angle_deadzone:
            mouse[index, 0] = mouse_step
        elif delta_pitch < -angle_deadzone:
            mouse[index, 0] = -mouse_step
        if delta_yaw > angle_deadzone:
            mouse[index, 1] = mouse_step
        elif delta_yaw < -angle_deadzone:
            mouse[index, 1] = -mouse_step
        labels.append(_action_label(keyboard[index], mouse[index]))

    if usable_rows > 0 and usable_rows < total_frames:
        keyboard[usable_rows:] = keyboard[usable_rows - 1]
        mouse[usable_rows:] = mouse[usable_rows - 1]
    return keyboard, mouse, _compress_action_labels(labels)


def _dominant_translation_token(local_delta: np.ndarray, *, deadzone: float) -> str | None:
    local_x = float(local_delta[0])
    local_z = float(local_delta[2])
    if max(abs(local_x), abs(local_z)) <= deadzone:
        return None
    if abs(local_z) >= abs(local_x):
        return "forward" if local_z > 0.0 else "back"
    return "right" if local_x > 0.0 else "left"


def _dominant_rotation_vector(angle_delta: np.ndarray, *, deadzone: float, mouse_step: float) -> np.ndarray:
    mouse = np.zeros(2, dtype=np.float32)
    delta_pitch = float(angle_delta[0])
    delta_yaw = float(angle_delta[1])
    if max(abs(delta_pitch), abs(delta_yaw)) <= deadzone:
        return mouse
    if abs(delta_yaw) >= abs(delta_pitch):
        mouse[1] = mouse_step if delta_yaw > 0.0 else -mouse_step
    else:
        mouse[0] = mouse_step if delta_pitch > 0.0 else -mouse_step
    return mouse


def _coarse_segment_bounds(active_frames: int, config: dict[str, Any]) -> list[tuple[int, int]]:
    segment_frames = max(int(config.get("gt_camera_segment_frames", 24)), 2)
    max_segments = max(int(config.get("gt_camera_max_segments", 24)), 1)
    segment_count = max(math.ceil(active_frames / segment_frames), 1)
    segment_count = min(segment_count, max_segments, active_frames)
    edges = np.rint(np.linspace(0, active_frames, num=segment_count + 1)).astype(np.int64)
    bounds: list[tuple[int, int]] = []
    for start, end in zip(edges[:-1], edges[1:]):
        start_i = int(start)
        end_i = int(end)
        if end_i > start_i:
            bounds.append((start_i, end_i))
    return bounds


def _coarse_action_label(keyboard_token: str | None, mouse_row: np.ndarray) -> str:
    keyboard = np.zeros(6, dtype=np.float32)
    if keyboard_token is not None:
        keyboard[_MATRIX_GAME_1_AND_3_KEYBOARD_IDX[keyboard_token]] = 1.0
    return _action_label(keyboard, mouse_row)


def _quantize_gt_camera_actions_coarse(
    *,
    poses_c2w: np.ndarray,
    saved_frames: int,
    total_frames: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keyboard = np.zeros((total_frames, 6), dtype=np.float32)
    mouse = np.zeros((total_frames, 2), dtype=np.float32)
    active_frames = min(saved_frames, total_frames)
    if poses_c2w.shape[0] < 2 or active_frames < 2:
        return keyboard, mouse, ["idle"]

    positions = _interp_columns(poses_c2w[:, :3, 3], saved_frames)
    pitch_yaw = _interp_columns(_camera_pitch_yaw_from_c2w(poses_c2w), saved_frames)
    rotations = _nearest_rotations(poses_c2w, saved_frames)
    bounds = _coarse_segment_bounds(active_frames, config)
    if not bounds:
        return keyboard, mouse, ["idle"]

    segment_local_deltas: list[np.ndarray] = []
    segment_angle_deltas: list[np.ndarray] = []
    for start, end in bounds:
        pose_end = max(end - 1, start)
        world_delta = positions[pose_end] - positions[start]
        local_delta = rotations[start].T @ world_delta
        segment_local_deltas.append(local_delta.astype(np.float32))
        segment_angle_deltas.append((pitch_yaw[pose_end] - pitch_yaw[start]).astype(np.float32))

    local_delta_array = np.stack(segment_local_deltas)
    angle_delta_array = np.stack(segment_angle_deltas)
    planar_steps = np.linalg.norm(local_delta_array[:, [0, 2]], axis=1)
    translation_deadzone = _scaled_positive_deadzone(
        planar_steps,
        percentile=float(config.get("gt_camera_translation_deadzone_percentile", 50)),
        scale=float(config.get("gt_camera_translation_deadzone_scale", 0.6)),
        minimum=float(config.get("gt_camera_translation_deadzone", 1e-6)),
    )
    angle_deadzone = _scaled_positive_deadzone(
        np.abs(angle_delta_array),
        percentile=float(config.get("gt_camera_angle_deadzone_percentile", 50)),
        scale=float(config.get("gt_camera_angle_deadzone_scale", 0.6)),
        minimum=math.radians(float(config.get("gt_camera_angle_deadzone_degrees", 0.25))),
    )

    mouse_step = float(config.get("gt_camera_mouse_step", 0.05))
    pulse_period = max(int(config.get("gt_camera_coarse_pulse_period_frames", 6)), 1)
    pulse_on = max(int(config.get("gt_camera_coarse_pulse_on_frames", 2)), 1)
    pulse_on = min(pulse_on, pulse_period)
    labels: list[str] = []
    for (start, end), local_delta, angle_delta in zip(bounds, segment_local_deltas, segment_angle_deltas):
        keyboard_token = _dominant_translation_token(local_delta, deadzone=translation_deadzone)
        mouse_row = _dominant_rotation_vector(angle_delta, deadzone=angle_deadzone, mouse_step=mouse_step)
        labels.append(_coarse_action_label(keyboard_token, mouse_row))
        for frame_index in range(start, end):
            if (frame_index - start) % pulse_period >= pulse_on:
                continue
            if keyboard_token is not None:
                keyboard[frame_index, _MATRIX_GAME_1_AND_3_KEYBOARD_IDX[keyboard_token]] = 1.0
            mouse[frame_index] = mouse_row

    return keyboard, mouse, _compress_action_labels(labels)


def _quantize_gt_camera_actions(
    *,
    poses_c2w: np.ndarray,
    saved_frames: int,
    total_frames: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mode = str(config.get("gt_camera_control_mode", "coarse")).strip().lower()
    if mode in {"dense", "frame", "framewise", "per_frame"}:
        return _quantize_gt_camera_actions_dense(
            poses_c2w=poses_c2w,
            saved_frames=saved_frames,
            total_frames=total_frames,
            config=config,
        )
    if mode not in {"coarse", "segment", "segmented", "rough"}:
        raise ValueError(f"Unsupported gt_camera_control_mode: {mode}")
    return _quantize_gt_camera_actions_coarse(
        poses_c2w=poses_c2w,
        saved_frames=saved_frames,
        total_frames=total_frames,
        config=config,
    )


def _matrix_game_3_video_static_gt_payload(
    config: dict[str, Any],
    suite: str,
    sample: BenchmarkSample,
) -> dict[str, Any]:
    if suite != "video_static":
        raise ValueError(f"VIPE GT camera action source is only valid for video_static, got {suite}")
    annotation_dir = _annotation_dir_for_sample(sample)
    if annotation_dir is None:
        raise FileNotFoundError(f"video_static sample has no annotation directory: {sample.sample_id}")

    action_plan = dict(config.get("action_plan", {}))
    first_clip_frames = int(action_plan.get("first_clip_frames", 57))
    clip_overlap_frames = int(action_plan.get("clip_overlap_frames", 16))
    followup_frames = 56 - clip_overlap_frames
    saved_frames, output_fps, quality = _video_static_target_frames_and_fps(
        config=config,
        sample=sample,
        annotation_dir=annotation_dir,
    )
    num_iterations = _matrix_game3_iterations_for_saved_frames(
        saved_frames=saved_frames,
        first_clip_frames=first_clip_frames,
        followup_frames=followup_frames,
    )
    total_frames = _matrix_game3_iteration_frame_count(
        first_clip_frames,
        followup_frames,
        num_iterations,
    )
    poses_c2w = _load_gt_c2w_poses(annotation_dir)
    keyboard_condition, mouse_condition, actions = _quantize_gt_camera_actions(
        poses_c2w=poses_c2w,
        saved_frames=saved_frames,
        total_frames=total_frames,
        config=config,
    )
    return {
        "variant": "matrix_game_3",
        "actions": actions,
        "keyboard_condition": keyboard_condition.tolist(),
        "mouse_condition": mouse_condition.tolist(),
        "total_frames": total_frames,
        "num_iterations": num_iterations,
        "save_num_frames": saved_frames,
        "output_fps": output_fps,
        "action_source": str(config.get("action_source", "vipe_gt_camera")).strip().lower(),
        "control_source": "vipe_gt_camera",
        "camera_token": None,
        "camera_tokens": None,
        "camera_path": list(sample.camera_path),
        "prompt_current": sample.prompt_current,
        "prompt_target": sample.prompt_target,
        "gt_camera_annotation_path": str(annotation_dir),
        "gt_camera_pose_count": int(poses_c2w.shape[0]),
        "gt_video_frame_count": saved_frames,
        "gt_video_fps": output_fps,
        "gt_camera_quality_status": quality.get("status"),
        "gt_camera_control_mode": str(config.get("gt_camera_control_mode", "coarse")).strip().lower(),
        "prompt_control_note": (
            "video_static uses the VIPE/GT camera trajectory quantized into Matrix-Game-3 "
            "keyboard and mouse controls; output fps/frame count are matched to the GT video."
        ),
    }


def _matrix_game_3_payload(
    config: dict[str, Any],
    suite: str,
    sample: BenchmarkSample | None = None,
) -> dict[str, Any]:
    action_plan = dict(config.get("action_plan", {}))
    action_source = str(config.get("action_source", "worldatlas_plan")).strip().lower()
    if sample is not None and sample.suite == "video_static" and action_source in _MATRIX_GAME3_GT_CAMERA_ACTION_SOURCES:
        return _matrix_game_3_video_static_gt_payload(config, suite, sample)

    camera_tokens: list[str] | None = None
    control_source = None
    if action_source in {
        "worldatlas_prompt",
        "prompt",
        "prompt_target",
        "camera_path",
        "worldatlas",
        "worldatlas_camera",
        "worldatlas_camera_path",
    }:
        if sample is None:
            raise ValueError("Matrix-Game-3 worldatlas_prompt action_source requires a BenchmarkSample")
        camera_tokens, matrix_actions, control_source = _worldatlas_matrix_game_actions(sample)
        actions = matrix_actions
        payload_config = None
    else:
        actions = _action_plan(config, suite)
        payload_config = config
    frames_per_action = int(action_plan.get("frames_per_action", 12))
    first_clip_frames = int(action_plan.get("first_clip_frames", 57))
    clip_overlap_frames = int(action_plan.get("clip_overlap_frames", 16))
    followup_frames = 56 - clip_overlap_frames
    configured_iterations = action_plan.get("num_iterations") or config.get("num_iterations")
    target_seconds = config.get("target_duration_seconds")

    if target_seconds is not None:
        # Iterations after the first each add ``followup_frames``, so folding the first
        # clip's surplus into the base makes ``unit_count`` read directly as the
        # iteration count.
        num_iterations = plan_rollout(
            target_seconds=float(target_seconds),
            native_fps=float(config.get("fps", 17)),
            unit_frames=followup_frames,
            base_frames=first_clip_frames - followup_frames,
        ).unit_count
    elif configured_iterations is None:
        target_frames = max(first_clip_frames, len(actions) * frames_per_action)
        if target_frames <= first_clip_frames:
            num_iterations = 1
        else:
            num_iterations = 1 + math.ceil((target_frames - first_clip_frames) / followup_frames)
    else:
        num_iterations = max(int(configured_iterations), 1)

    total_frames = first_clip_frames + (num_iterations - 1) * followup_frames
    if payload_config is None:
        prompt_config = dict(config)
        saved_frames = int(config.get("save_num_frames") or total_frames)
        action_frames = min(saved_frames, total_frames)
        frames_per_action = max(action_frames // max(len(matrix_actions), 1), 1)
        prompt_config["action_plan"] = {
            "default": matrix_actions,
            "by_suite": {suite: matrix_actions},
            "frames_per_action": frames_per_action,
        }
        payload_config = prompt_config
    payload = _build_fixed_length_action_payload(
        config=payload_config,
        suite=suite,
        variant="matrix_game_3",
        keyboard_idx=_MATRIX_GAME_1_AND_3_KEYBOARD_IDX,
        total_frames=total_frames,
        mouse_enabled=True,
        mouse_step=0.1,
        default_frames_per_action=frames_per_action,
    )
    payload["num_iterations"] = num_iterations
    payload["action_source"] = action_source
    if sample is not None:
        payload.update(
            {
                "control_source": control_source,
                "camera_token": camera_tokens[0] if camera_tokens else None,
                "camera_tokens": camera_tokens if payload_config is not config else None,
                "camera_path": list(sample.camera_path),
                "prompt_current": sample.prompt_current,
                "prompt_target": sample.prompt_target,
                "prompt_control_note": (
                    "dynamic suites use idle/no-action controls so Matrix-Game-3 can freely evolve the scene"
                    if sample.suite == "image_dynamic" or sample.generation_mode == "dynamic"
                    else "static suites map the full WorldAtlas camera_path to Matrix-Game-3 controls"
                ),
            }
        )
    return payload


def _build_action_payload(
    config: dict[str, Any],
    suite: str,
    *,
    family: str = "matrix_game_3",
    sample: BenchmarkSample | None = None,
) -> dict[str, Any]:
    variant = _matrix_game_variant(family)
    if variant == "matrix_game_1":
        return _matrix_game_1_payload(config, suite, sample=sample)
    if variant == "matrix_game_2":
        return _matrix_game_2_payload(config, suite, sample=sample)
    return _matrix_game_3_payload(config, suite, sample=sample)


def _matrix_game_3_command(
    *,
    python_bin: str,
    repo_root: Path,
    checkpoint_dir: Path,
    action_spec_path: Path,
    conditioning_image: Path,
    output_path: Path,
    prompt: str,
    generation: dict[str, Any],
    action_payload: dict[str, Any],
) -> list[str]:
    command = [
        python_bin,
        "-m",
        "worldatlas.models.adapters.matrix_game.matrix_game_runner",
        "--repo_root",
        str(repo_root),
        "--ckpt_dir",
        str(checkpoint_dir),
        "--actions_json",
        str(action_spec_path),
        "--image_path",
        str(conditioning_image),
        "--prompt",
        prompt,
        "--output_dir",
        str(output_path.parent),
        "--save_name",
        output_path.stem,
        "--size",
        str(generation.get("size", "704*1280")),
        "--fps",
        str(float(action_payload.get("output_fps", generation.get("fps", 17)))),
        "--seed",
        str(int(generation.get("seed", 42))),
        "--num_iterations",
        str(int(action_payload["num_iterations"])),
        "--num_inference_steps",
        str(int(generation.get("num_inference_steps", 3))),
        "--sample_guide_scale",
        str(float(generation.get("sample_guide_scale", 5.0))),
        "--save_num_frames",
        str(int(action_payload.get("save_num_frames", generation.get("save_num_frames", 0) or 0))),
        "--ulysses_size",
        str(int(generation.get("ulysses_size", 1))),
        "--vae_type",
        str(generation.get("vae_type", "mg_lightvae_v2")),
    ]
    extend_command_with_options(
        command,
        payload=generation,
        value_options={
            "sample_shift": float,
            "lightvae_pruning_rate": float,
            "fa_version": str,
            "async_vae_warmup_iters": int,
        },
        flag_options=(
            "visualize_ops",
            "use_base_model",
            "use_int8",
            "verify_quant",
            "use_async_vae",
            "compile_vae",
            "t5_fsdp",
            "t5_cpu",
            "dit_fsdp",
            "convert_model_dtype",
        ),
    )
    return command


def _matrix_game_1_command(
    *,
    python_bin: str,
    repo_root: Path,
    checkpoint_dir: Path,
    action_spec_path: Path,
    conditioning_image: Path,
    output_path: Path,
    prompt: str,
    generation: dict[str, Any],
) -> list[str]:
    command = [
        python_bin,
        "-m",
        "worldatlas.models.adapters.matrix_game.matrix_game_1_runner",
        "--repo_root",
        str(repo_root),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--actions_json",
        str(action_spec_path),
        "--image_path",
        str(conditioning_image),
        "--output_path",
        str(output_path),
        "--prompt",
        prompt,
    ]
    extend_command_with_options(
        command,
        payload=generation,
        value_options={
            "video_length": int,
            "guidance_scale": float,
            "inference_steps": int,
            "shift": float,
            "num_pre_frames": int,
            "num_steps": int,
            "rel_l1_thresh": float,
            "fps": int,
            "seed": int,
            "chunk_video_length": int,
        },
        flag_options=("bfloat16",),
    )
    if "prefer_flash_attn3" in generation:
        command.append(
            "--prefer_flash_attn3"
            if bool(generation.get("prefer_flash_attn3"))
            else "--no-prefer_flash_attn3"
        )
    resolution = generation.get("resolution")
    if resolution is not None:
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("Matrix-Game-1 generation.resolution must be a 2-item list like [1280, 720]")
        command.extend(["--resolution", str(int(resolution[0])), str(int(resolution[1]))])
    return command


def _matrix_game_2_cli_generation(
    generation: dict[str, Any],
    action_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prefer the planned latent count on the action payload over the YAML default.

    Memory-track configs keep ``num_output_frames: 150`` (the short-clip default)
    and grow the actual rollout through ``target_duration_seconds`` → 183 latents
    / 729 pixel actions (181 from plan_rollout, snapped up to a multiple of the
    3-latent inference block). The runner sizes noise and checks action length
    against ``--num_output_frames``, so the CLI must follow the payload.
    """
    cli = dict(generation)
    if action_payload and action_payload.get("num_output_frames") is not None:
        cli["num_output_frames"] = int(action_payload["num_output_frames"])
    if action_payload and action_payload.get("save_num_frames"):
        cli["save_num_frames"] = int(action_payload["save_num_frames"])
    return cli


def _matrix_game_2_command(
    *,
    python_bin: str,
    repo_root: Path,
    checkpoint_dir: Path,
    action_spec_path: Path,
    conditioning_image: Path,
    output_path: Path,
    prompt: str,
    generation: dict[str, Any],
    action_payload: dict[str, Any] | None = None,
) -> list[str]:
    del prompt
    command = [
        python_bin,
        "-m",
        "worldatlas.models.adapters.matrix_game.matrix_game_2_runner",
        "--repo_root",
        str(repo_root),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--actions_json",
        str(action_spec_path),
        "--image_path",
        str(conditioning_image),
        "--output_path",
        str(output_path),
        "--mode",
        _matrix_game_2_mode(generation),
    ]
    extend_command_with_options(
        command,
        payload=_matrix_game_2_cli_generation(generation, action_payload),
        value_options={
            "num_output_frames": int,
            "save_num_frames": int,
            "seed": int,
            "output_fps": int,
            "config_path": str,
            "checkpoint_filename": str,
        },
        flag_options=("compile_vae",),
    )
    return command


class MatrixGameAdapter(ModelAdapter):
    """Generate interactive game-world videos via Matrix-Game runner modules."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_checkpoint_load_policy(self) -> str:
        if matrix_game_batch_runner_module(self.config.family):
            return "load_once"
        return "reload_per_sample"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        runner_module = matrix_game_batch_runner_module(self.config.family)
        if runner_module is None:
            return run_sequential_generate_batch(self, requests)
        if not requests:
            return {}
        if self.config.repo_root is None:
            raise ValueError("Matrix-Game adapters require repo_root")

        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError("Matrix-Game batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)
        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"{self.config.family.lower().replace('.', '_')}_{uuid4().hex}.jsonl"
        generation_path = spec_path.with_suffix(".generation.json")
        generation_path.write_text(
            json.dumps(self.config.generation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with spec_path.open("w", encoding="utf-8") as file:
            for request in requests:
                action_payload = _build_action_payload(
                    self.config.generation,
                    request.sample.suite,
                    family=self.config.family,
                    sample=request.sample,
                )
                file.write(
                    json.dumps(
                        {
                            "sample_id": request.sample.sample_id,
                            "suite": request.sample.suite,
                            "prediction_stem": request.sample.prediction_stem,
                            "conditioning_image": str(
                                request.conditioning_image.expanduser().resolve()
                            ),
                            "output_path": str(request.output_path.expanduser().resolve()),
                            "prompt": request.prompt,
                            "action_payload": action_payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        project_root = worldatlas_project_root()
        env = build_env(
            extra_pythonpaths=[project_root, self.config.repo_root],
            overrides=self.config.env,
        )
        env.setdefault("PYTHONUNBUFFERED", "1")
        command = [
            self.config.python_bin,
            "-m",
            runner_module,
            "--repo_root",
            str(self.config.repo_root),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ]
        if self.config.checkpoint_dir is not None:
            command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])
        log_path = spec_path.with_suffix(".log")
        log_progress(
            "external_runner_start",
            label="Matrix-Game",
            samples=len(requests),
            runner=runner_module,
            log=str(log_path),
        )
        run_logged_command(
            command,
            cwd=project_root,
            env=env,
            log_path=log_path,
            label="Matrix-Game",
        )
        log_progress(
            "external_runner_done",
            label="Matrix-Game",
            samples=len(requests),
            log=str(log_path),
        )

        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            if request.output_path.is_file() and request.output_path.stat().st_size > 0:
                results[request.sample.sample_id] = {
                    "status": "generated",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                    "batch_spec_path": str(spec_path),
                    "generation_log": str(log_path),
                }
            else:
                results[request.sample.sample_id] = {
                    "status": "failed",
                    "error": f"Matrix-Game output was not written: {request.output_path}",
                    "prompt": request.prompt,
                    "batch_spec_path": str(spec_path),
                    "generation_log": str(log_path),
                }
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        generation = self.config.generation
        variant = _matrix_game_variant(self.config.family)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        action_payload = _build_action_payload(
            generation,
            sample.suite,
            family=self.config.family,
            sample=sample,
        )
        action_spec_path = output_path.parent / f"{output_path.stem}_actions.json"
        action_spec_path.write_text(
            json.dumps(action_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        project_root = worldatlas_project_root()
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("Matrix-Game adapters require repo_root")
        checkpoint_dir = self.config.checkpoint_dir
        if checkpoint_dir is None:
            raise ValueError("Matrix-Game adapters require checkpoint_dir")

        if variant == "matrix_game_1":
            command = _matrix_game_1_command(
                python_bin=self.config.python_bin,
                repo_root=repo_root,
                checkpoint_dir=checkpoint_dir,
                action_spec_path=action_spec_path,
                conditioning_image=conditioning_image,
                output_path=output_path,
                prompt=prompt,
                generation=generation,
            )
        elif variant == "matrix_game_2":
            command = _matrix_game_2_command(
                python_bin=self.config.python_bin,
                repo_root=repo_root,
                checkpoint_dir=checkpoint_dir,
                action_spec_path=action_spec_path,
                conditioning_image=conditioning_image,
                output_path=output_path,
                prompt=prompt,
                generation=generation,
                action_payload=action_payload,
            )
        else:
            command = _matrix_game_3_command(
                python_bin=self.config.python_bin,
                repo_root=repo_root,
                checkpoint_dir=checkpoint_dir,
                action_spec_path=action_spec_path,
                conditioning_image=conditioning_image,
                output_path=output_path,
                prompt=prompt,
                generation=generation,
                action_payload=action_payload,
            )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"Matrix-Game output was not written: {output_path}")
        return {
            "actions": action_payload["actions"],
            "action_source": action_payload.get("action_source"),
            "camera_token": action_payload.get("camera_token"),
            "camera_path": action_payload.get("camera_path"),
            "control_source": action_payload.get("control_source"),
            "prompt_control_note": action_payload.get("prompt_control_note"),
            "output_fps": action_payload.get("output_fps"),
            "saved_frames": action_payload.get("save_num_frames"),
            "gt_camera_annotation_path": action_payload.get("gt_camera_annotation_path"),
            "gt_video_frame_count": action_payload.get("gt_video_frame_count"),
            "command": command,
            "action_spec_path": str(action_spec_path),
            "prediction_path": str(output_path),
            "prompt": prompt,
            "variant": variant,
        }
