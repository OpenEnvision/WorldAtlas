from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter
from worldarena.models.adapters.common import build_env, extend_command_with_options, run_command


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


def _matrix_game_variant(family: str) -> str:
    normalized = family.lower().replace("-", "_")
    if normalized in {"matrix_game", "matrix_game_3"}:
        return "matrix_game_3"
    if normalized in {"matrix_game_1", "matrix_game_2"}:
        return normalized
    raise ValueError(f"Unsupported Matrix-Game family: {family}")


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


def _matrix_game_1_payload(config: dict[str, Any], suite: str) -> dict[str, Any]:
    total_frames = int(config.get("video_length", 65))
    payload = _build_fixed_length_action_payload(
        config=config,
        suite=suite,
        variant="matrix_game_1",
        keyboard_idx=_MATRIX_GAME_1_AND_3_KEYBOARD_IDX,
        total_frames=total_frames,
        mouse_enabled=True,
        mouse_step=0.05,
        default_frames_per_action=max(total_frames // max(len(_action_plan(config, suite)), 1), 1),
        pulse_period_frames={"attack": int(config.get("attack_period_frames", 8))},
    )
    payload["video_length"] = total_frames
    return payload


def _matrix_game_2_payload(config: dict[str, Any], suite: str) -> dict[str, Any]:
    mode = _matrix_game_2_mode(config)
    spec = _MATRIX_GAME_2_SPECS[mode]
    num_output_frames = int(config.get("num_output_frames", 15))
    total_frames = int(config.get("num_frames", (num_output_frames - 1) * 4 + 1))
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
    return payload


def _matrix_game_3_payload(config: dict[str, Any], suite: str) -> dict[str, Any]:
    action_plan = dict(config.get("action_plan", {}))
    actions = _action_plan(config, suite)
    frames_per_action = int(action_plan.get("frames_per_action", 12))
    first_clip_frames = int(action_plan.get("first_clip_frames", 57))
    clip_overlap_frames = int(action_plan.get("clip_overlap_frames", 16))
    followup_frames = 56 - clip_overlap_frames
    configured_iterations = action_plan.get("num_iterations") or config.get("num_iterations")

    if configured_iterations is None:
        target_frames = max(first_clip_frames, len(actions) * frames_per_action)
        if target_frames <= first_clip_frames:
            num_iterations = 1
        else:
            num_iterations = 1 + math.ceil((target_frames - first_clip_frames) / followup_frames)
    else:
        num_iterations = max(int(configured_iterations), 1)

    total_frames = first_clip_frames + (num_iterations - 1) * followup_frames
    payload = _build_fixed_length_action_payload(
        config=config,
        suite=suite,
        variant="matrix_game_3",
        keyboard_idx=_MATRIX_GAME_1_AND_3_KEYBOARD_IDX,
        total_frames=total_frames,
        mouse_enabled=True,
        mouse_step=0.1,
        default_frames_per_action=frames_per_action,
    )
    payload["num_iterations"] = num_iterations
    return payload


def _build_action_payload(config: dict[str, Any], suite: str, *, family: str = "matrix_game_3") -> dict[str, Any]:
    variant = _matrix_game_variant(family)
    if variant == "matrix_game_1":
        return _matrix_game_1_payload(config, suite)
    if variant == "matrix_game_2":
        return _matrix_game_2_payload(config, suite)
    return _matrix_game_3_payload(config, suite)


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
        "worldarena.models.adapters.matrix_game_runner",
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
        str(int(generation.get("fps", 17))),
        "--seed",
        str(int(generation.get("seed", 42))),
        "--num_iterations",
        str(int(action_payload["num_iterations"])),
        "--num_inference_steps",
        str(int(generation.get("num_inference_steps", 3))),
        "--sample_guide_scale",
        str(float(generation.get("sample_guide_scale", 5.0))),
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
        "worldarena.models.adapters.matrix_game_1_runner",
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
        },
        flag_options=("bfloat16",),
    )
    resolution = generation.get("resolution")
    if resolution is not None:
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("Matrix-Game-1 generation.resolution must be a 2-item list like [1280, 720]")
        command.extend(["--resolution", str(int(resolution[0])), str(int(resolution[1]))])
    return command


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
) -> list[str]:
    del prompt
    command = [
        python_bin,
        "-m",
        "worldarena.models.adapters.matrix_game_2_runner",
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
        payload=generation,
        value_options={
            "num_output_frames": int,
            "seed": int,
            "output_fps": int,
            "config_path": str,
            "checkpoint_filename": str,
        },
        flag_options=("compile_vae",),
    )
    return command


class MatrixGameAdapter(ModelAdapter):
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
        action_payload = _build_action_payload(generation, sample.suite, family=self.config.family)
        action_spec_path = output_path.parent / f"{output_path.stem}_actions.json"
        action_spec_path.write_text(
            json.dumps(action_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        project_root = Path(__file__).resolve().parents[3]
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("Matrix-Game adapters require repo_root")
        checkpoint_dir = self.config.checkpoint_dir or repo_root

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
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
            "variant": variant,
        }
