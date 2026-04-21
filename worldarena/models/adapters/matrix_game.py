from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.models.adapters.base import ModelAdapter
from worldarena.models.adapters.common import build_env, run_command
from worldarena.benchmark.schemas import BenchmarkSample


KEYBOARD_DIM = 6
KEYBOARD_IDX = {
    "forward": 0,
    "back": 1,
    "left": 2,
    "right": 3,
    "jump": 4,
    "attack": 5,
}
MOUSE_STEP = 0.1
CAMERA_VALUE_MAP = {
    "camera_up": [MOUSE_STEP, 0.0],
    "camera_down": [-MOUSE_STEP, 0.0],
    "camera_l": [0.0, -MOUSE_STEP],
    "camera_r": [0.0, MOUSE_STEP],
    "camera_ul": [MOUSE_STEP, -MOUSE_STEP],
    "camera_ur": [MOUSE_STEP, MOUSE_STEP],
    "camera_dl": [-MOUSE_STEP, -MOUSE_STEP],
    "camera_dr": [-MOUSE_STEP, MOUSE_STEP],
}
COMBINATION_MAP = {
    "forward_left": ["forward", "left"],
    "forward_right": ["forward", "right"],
    "back_left": ["back", "left"],
    "back_right": ["back", "right"],
    "idle": [],
    "nomove": [],
}


def _encode_action(action: str) -> tuple[np.ndarray, np.ndarray]:
    keyboard = np.zeros(KEYBOARD_DIM, dtype=np.float32)
    mouse = np.zeros(2, dtype=np.float32)
    for item in COMBINATION_MAP.get(action, [action]):
        if item in KEYBOARD_IDX:
            keyboard[KEYBOARD_IDX[item]] = 1.0
        elif item in CAMERA_VALUE_MAP:
            mouse[:] = np.asarray(CAMERA_VALUE_MAP[item], dtype=np.float32)
    return keyboard, mouse


def _action_plan(config: dict[str, Any], suite: str) -> list[str]:
    plan = dict(config.get("action_plan", {}))
    by_suite = plan.get("by_suite", {})
    actions = by_suite.get(suite) or plan.get("default") or ["forward", "camera_r", "forward_right", "camera_l"]
    return [str(action) for action in actions]


def _build_action_payload(config: dict[str, Any], suite: str) -> dict[str, Any]:
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
    keyboard_rows: list[np.ndarray] = []
    mouse_rows: list[np.ndarray] = []
    for action in actions:
        keyboard, mouse = _encode_action(action)
        keyboard_rows.extend([keyboard.copy()] * frames_per_action)
        mouse_rows.extend([mouse.copy()] * frames_per_action)
    if not keyboard_rows:
        keyboard_rows.append(np.zeros(KEYBOARD_DIM, dtype=np.float32))
        mouse_rows.append(np.zeros(2, dtype=np.float32))
    while len(keyboard_rows) < total_frames:
        keyboard_rows.append(keyboard_rows[-1].copy())
        mouse_rows.append(mouse_rows[-1].copy())

    return {
        "actions": actions,
        "keyboard_condition": np.stack(keyboard_rows[:total_frames]).tolist(),
        "mouse_condition": np.stack(mouse_rows[:total_frames]).tolist(),
        "num_iterations": num_iterations,
    }


class MatrixGameAdapter(ModelAdapter):
    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        action_payload = _build_action_payload(generation, sample.suite)
        action_spec_path = output_path.parent / f"{output_path.stem}_actions.json"
        action_spec_path.write_text(
            json.dumps(action_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        project_root = Path(__file__).resolve().parents[3]

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.matrix_game_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--ckpt_dir",
            str(self.config.checkpoint_dir or self.config.repo_root),
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
        if "sample_shift" in generation:
            command.extend(["--sample_shift", str(float(generation["sample_shift"]))])
        if "lightvae_pruning_rate" in generation:
            command.extend(["--lightvae_pruning_rate", str(float(generation["lightvae_pruning_rate"]))])
        if generation.get("fa_version") is not None:
            command.extend(["--fa_version", str(generation["fa_version"])])
        for flag in (
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
        ):
            if generation.get(flag, False):
                command.append(f"--{flag}")
        if generation.get("async_vae_warmup_iters", 0):
            command.extend(["--async_vae_warmup_iters", str(int(generation["async_vae_warmup_iters"]))])

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
        }
