from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter
from worldarena.models.adapters.common import build_env, extend_command_with_options, run_command


MOVE_ACTION_MAP = {
    "no-op": 0,
    "go forward": 1,
    "go back": 2,
    "go left": 3,
    "go right": 4,
    "go forward and go left": 5,
    "go forward and go right": 6,
    "go back and go left": 7,
    "go back and go right": 8,
    "uncertain": 9,
}
VIEW_ACTION_MAP = {
    "no-op": 0,
    "turn up": 1,
    "turn down": 2,
    "turn left": 3,
    "turn right": 4,
    "turn up and turn left": 5,
    "turn up and turn right": 6,
    "turn down and turn left": 7,
    "turn down and turn right": 8,
    "uncertain": 9,
}
STRING_ACTION_ALIASES = {
    "noop": {"move": "no-op", "view": "no-op"},
    "no_op": {"move": "no-op", "view": "no-op"},
    "no-op": {"move": "no-op", "view": "no-op"},
    "none": {"move": "no-op", "view": "no-op"},
    "forward": {"move": "go forward", "view": "no-op"},
    "go_forward": {"move": "go forward", "view": "no-op"},
    "back": {"move": "go back", "view": "no-op"},
    "backward": {"move": "go back", "view": "no-op"},
    "go_back": {"move": "go back", "view": "no-op"},
    "left": {"move": "go left", "view": "no-op"},
    "go_left": {"move": "go left", "view": "no-op"},
    "right": {"move": "go right", "view": "no-op"},
    "go_right": {"move": "go right", "view": "no-op"},
    "forward_left": {"move": "go forward and go left", "view": "no-op"},
    "forward_right": {"move": "go forward and go right", "view": "no-op"},
    "back_left": {"move": "go back and go left", "view": "no-op"},
    "back_right": {"move": "go back and go right", "view": "no-op"},
    "camera_up": {"move": "no-op", "view": "turn up"},
    "look_up": {"move": "no-op", "view": "turn up"},
    "up": {"move": "no-op", "view": "turn up"},
    "camera_down": {"move": "no-op", "view": "turn down"},
    "look_down": {"move": "no-op", "view": "turn down"},
    "down": {"move": "no-op", "view": "turn down"},
    "camera_left": {"move": "no-op", "view": "turn left"},
    "camera_l": {"move": "no-op", "view": "turn left"},
    "turn_left": {"move": "no-op", "view": "turn left"},
    "camera_right": {"move": "no-op", "view": "turn right"},
    "camera_r": {"move": "no-op", "view": "turn right"},
    "turn_right": {"move": "no-op", "view": "turn right"},
    "camera_up_left": {"move": "no-op", "view": "turn up and turn left"},
    "camera_up_right": {"move": "no-op", "view": "turn up and turn right"},
    "camera_down_left": {"move": "no-op", "view": "turn down and turn left"},
    "camera_down_right": {"move": "no-op", "view": "turn down and turn right"},
    "uncertain": {"move": "uncertain", "view": "uncertain"},
}
MOVE_PART_ALIASES = {
    "forward": "go forward",
    "go_forward": "go forward",
    "back": "go back",
    "backward": "go back",
    "go_back": "go back",
    "left": "go left",
    "go_left": "go left",
    "right": "go right",
    "go_right": "go right",
}
VIEW_PART_ALIASES = {
    "camera_up": "turn up",
    "up": "turn up",
    "look_up": "turn up",
    "camera_down": "turn down",
    "down": "turn down",
    "look_down": "turn down",
    "camera_left": "turn left",
    "camera_l": "turn left",
    "turn_left": "turn left",
    "camera_right": "turn right",
    "camera_r": "turn right",
    "turn_right": "turn right",
}


def _sanitize_action_token(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_interaction(action: str | dict[str, str]) -> dict[str, str]:
    if isinstance(action, dict):
        move = str(action.get("move", "no-op"))
        view = str(action.get("view", "no-op"))
        if move not in MOVE_ACTION_MAP:
            raise ValueError(f"Unsupported Infinite-World move action: {move}")
        if view not in VIEW_ACTION_MAP:
            raise ValueError(f"Unsupported Infinite-World view action: {view}")
        return {"move": move, "view": view}

    normalized = _sanitize_action_token(str(action))
    if normalized in STRING_ACTION_ALIASES:
        return dict(STRING_ACTION_ALIASES[normalized])

    if any(separator in normalized for separator in ("+", "|", ",")):
        move = "no-op"
        view = "no-op"
        for part in normalized.replace("|", "+").replace(",", "+").split("+"):
            token = part.strip()
            if not token:
                continue
            if token in MOVE_PART_ALIASES:
                move = MOVE_PART_ALIASES[token]
                continue
            if token in VIEW_PART_ALIASES:
                view = VIEW_PART_ALIASES[token]
                continue
            raise ValueError(f"Unsupported Infinite-World interaction part: {action}")
        return {"move": move, "view": view}

    raise ValueError(f"Unsupported Infinite-World interaction: {action}")


def _interaction_plan(config: dict[str, Any], suite: str) -> list[str | dict[str, str]]:
    plan = dict(config.get("action_plan", {}))
    by_suite = plan.get("by_suite", {})
    actions = by_suite.get(suite) or plan.get("default") or [
        "forward",
        "camera_right",
        "forward",
        "camera_left",
        "forward",
    ]
    return list(actions)


def _build_action_payload(
    config: dict[str, Any],
    suite: str,
    *,
    validation_num_frames: int = 81,
) -> dict[str, Any]:
    action_plan = dict(config.get("action_plan", {}))
    actions = _interaction_plan(config, suite)
    frames_per_action = int(action_plan.get("frames_per_action", 16))
    chunk_stride = int(config.get("chunk_stride", 20))
    num_chunks = max(int(config.get("num_chunks", 1)), 1)
    required_frames = validation_num_frames + (num_chunks - 1) * chunk_stride

    frame_actions: list[dict[str, str]] = []
    for action in actions:
        frame_action = _normalize_interaction(action)
        frame_actions.extend([dict(frame_action)] * frames_per_action)

    if not frame_actions:
        frame_actions.append({"move": "no-op", "view": "no-op"})
    while len(frame_actions) < required_frames:
        frame_actions.append(dict(frame_actions[-1]))

    return {
        "actions": actions,
        "frame_actions": frame_actions[:required_frames],
        "frames_per_action": frames_per_action,
        "required_frames": required_frames,
        "validation_num_frames": validation_num_frames,
        "chunk_stride": chunk_stride,
        "num_chunks": num_chunks,
    }


class InfiniteWorldAdapter(ModelAdapter):
    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if self.config.repo_root is None:
            raise ValueError("Infinite-World adapter requires repo_root")

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        action_payload = _build_action_payload(
            generation,
            sample.suite,
            validation_num_frames=int(generation.get("validation_num_frames", 81)),
        )
        action_spec_path = output_path.parent / f"{output_path.stem}_actions.json"
        action_spec_path.write_text(
            json.dumps(action_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        project_root = Path(__file__).resolve().parents[3]

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.infinite_world_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--ckpt_dir",
            str(self.config.checkpoint_dir or ""),
            "--actions_json",
            str(action_spec_path),
            "--image_path",
            str(conditioning_image),
            "--prompt",
            prompt,
            "--output_path",
            str(output_path),
        ]
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "config_path": str,
                "bucket_config_name": str,
                "fps": int,
                "seed": int,
                "num_chunks": int,
                "num_sampling_steps": int,
                "sample_shift": float,
                "sample_guide_scale": float,
                "quality": int,
                "negative_prompt": str,
                "device": str,
            },
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"Infinite-World output was not written: {output_path}")
        return {
            "actions": action_payload["actions"],
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
        }


__all__ = [
    "InfiniteWorldAdapter",
    "_build_action_payload",
]
