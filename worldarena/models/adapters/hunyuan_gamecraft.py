from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
    run_sequential_generate_batch,
)
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample


_DEFAULT_ACTIONS = ["w", "d", "w", "a", "w"]
_SUPPORTED_ACTIONS = {"w", "a", "s", "d", "left_rot", "right_rot", "up_rot", "down_rot"}
_WORLD_ARENA_ACTION_SOURCE_VALUES = {
    "worldarena",
    "worldarena_camera",
    "worldarena_camera_path",
    "camera_path",
    "prompt_target",
}
_WORLD_ARENA_GAMECRAFT_ACTIONS = {
    "fixed": ("w", 0.0, "exact_no_motion"),
    "push_in": ("w", 1.0, "exact_translation"),
    "pull_out": ("s", 1.0, "exact_translation"),
    "move_left": ("a", 1.0, "exact_translation"),
    "move_right": ("d", 1.0, "exact_translation"),
    "pan_left": ("left_rot", 1.0, "exact_rotation"),
    "pan_right": ("right_rot", 1.0, "exact_rotation"),
    "tilt_up": ("up_rot", 1.0, "exact_rotation"),
    "tilt_down": ("down_rot", 1.0, "exact_rotation"),
    "pedestal_up": ("up_rot", 1.0, "coarse_vertical_to_tilt"),
    "pedestal_down": ("down_rot", 1.0, "coarse_vertical_to_tilt"),
    "orbit_left": ("right_rot", 1.0, "coarse_orbit_to_counter_rotation"),
    "orbit_right": ("left_rot", 1.0, "coarse_orbit_to_counter_rotation"),
}


def _normalize_camera_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_action_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    else:
        raw_items = [str(item).strip() for item in value]
    return [item for item in raw_items if item]


def _normalize_speed_list(
    value: object,
    *,
    action_count: int,
    default_speed: float,
) -> list[float]:
    if action_count <= 0:
        return []
    if value is None:
        return [default_speed] * action_count
    if isinstance(value, (int, float, str)):
        return [float(value)] * action_count

    speeds = [float(item) for item in value]
    if len(speeds) == 1:
        return speeds * action_count
    if len(speeds) != action_count:
        raise ValueError(
            "Hunyuan-GameCraft action speed list length mismatch: "
            f"expected {action_count}, got {len(speeds)}"
        )
    return speeds


def _action_plan_for_suite(generation: dict[str, Any], suite: str) -> tuple[list[str], list[float]]:
    plan = dict(generation.get("action_plan", {}))
    actions_by_suite = dict(plan.get("by_suite", {}))
    speeds_by_suite = dict(plan.get("speed_by_suite", {}))

    actions = _normalize_action_list(actions_by_suite.get(suite) or plan.get("default") or _DEFAULT_ACTIONS)
    if not actions:
        raise ValueError("Hunyuan-GameCraft action plan is empty")

    invalid_actions = sorted({action for action in actions if action not in _SUPPORTED_ACTIONS})
    if invalid_actions:
        raise ValueError(
            "Unsupported Hunyuan-GameCraft action token(s): " + ", ".join(invalid_actions)
        )

    default_speed = float(generation.get("action_speed", 0.2))
    speeds = _normalize_speed_list(
        speeds_by_suite.get(suite, plan.get("speed_default")),
        action_count=len(actions),
        default_speed=default_speed,
    )
    return actions, speeds


def _worldarena_action_plan_for_sample(
    generation: dict[str, Any],
    sample: BenchmarkSample,
) -> dict[str, Any]:
    camera_tokens = [
        _normalize_camera_token(token)
        for token in _camera_path_for_sample(sample)
        if str(token).strip()
    ] or ["fixed"]
    base_speed = float(generation.get("action_speed", 0.2))
    speed_overrides = dict(generation.get("worldarena_action_speeds", {}))
    unsupported_action = str(generation.get("unsupported_action", "w"))
    if unsupported_action not in _SUPPORTED_ACTIONS:
        unsupported_action = "w"

    actions: list[str] = []
    speeds: list[float] = []
    approximations: list[dict[str, str]] = []
    unsupported_tokens: list[str] = []
    for camera_token in camera_tokens:
        native = _WORLD_ARENA_GAMECRAFT_ACTIONS.get(camera_token)
        if native is None:
            actions.append(unsupported_action)
            speeds.append(0.0)
            unsupported_tokens.append(camera_token)
            approximations.append(
                {
                    "camera_token": camera_token,
                    "native_action": unsupported_action,
                    "type": "unsupported_no_motion",
                }
            )
            continue

        action, speed_scale, approximation = native
        actions.append(action)
        token_speed = speed_overrides.get(camera_token)
        speeds.append(float(token_speed) if token_speed is not None else base_speed * float(speed_scale))
        if not approximation.startswith("exact"):
            approximations.append(
                {
                    "camera_token": camera_token,
                    "native_action": action,
                    "type": approximation,
                }
            )

    control_source = "camera_path"
    if unsupported_tokens:
        control_source = "unsupported_camera_path"
    elif approximations:
        control_source = "coarse_camera_path"

    return {
        "action_source": "worldarena_camera_path",
        "control_source": control_source,
        "camera_tokens": camera_tokens,
        "camera_path": list(sample.camera_path),
        "native_action_space": "hunyuan_gamecraft",
        "actions": actions,
        "action_speeds": speeds,
        "approximations": approximations,
        "unsupported_camera_tokens": unsupported_tokens,
    }


def _action_payload_for_sample(generation: dict[str, Any], sample: BenchmarkSample) -> dict[str, Any]:
    action_source = str(generation.get("action_source", "worldarena_plan")).strip().lower()
    if action_source in _WORLD_ARENA_ACTION_SOURCE_VALUES:
        return _worldarena_action_plan_for_sample(generation, sample)

    actions, action_speeds = _action_plan_for_suite(generation, sample.suite)
    return {
        "action_source": action_source,
        "control_source": "config_action_plan",
        "camera_tokens": [],
        "camera_path": list(sample.camera_path),
        "native_action_space": "hunyuan_gamecraft",
        "actions": actions,
        "action_speeds": action_speeds,
        "approximations": [],
        "unsupported_camera_tokens": [],
    }


class HunyuanGameCraftAdapter(ModelAdapter):
    def supports_batch_generation(self) -> bool:
        return True

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        return run_sequential_generate_batch(self, requests)

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        repo_root = self.config.repo_root
        if repo_root is None:
            raise ValueError("Hunyuan-GameCraft adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Hunyuan-GameCraft adapter requires checkpoint_dir")

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]
        action_payload = _action_payload_for_sample(generation, sample)
        actions = list(action_payload["actions"])
        action_speeds = list(action_payload["action_speeds"])
        action_spec_path = output_path.parent / f"{output_path.stem}_actions.json"
        action_spec_path.write_text(
            json.dumps(action_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        prompt_prefix = str(generation.get("prompt_prefix", ""))
        prompt_suffix = str(generation.get("prompt_suffix", ""))
        prompt_text = f"{prompt_prefix}{prompt}{prompt_suffix}".strip()

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.hunyuan_gamecraft_runner",
            "--repo_root",
            str(repo_root),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--sample_name",
            output_path.stem,
            "--prompt",
            prompt_text,
            "--action_list",
            *actions,
            "--action_speed_list",
            *(f"{speed:g}" for speed in action_speeds),
        ]
        command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "model_variant": str,
                "checkpoint_filename": str,
                "num_gpus": int,
                "master_port": int,
                "height": int,
                "width": int,
                "cfg_scale": float,
                "sample_n_frames": int,
                "infer_steps": int,
                "flow_shift_eval_video": float,
                "seed": int,
                "add_pos_prompt": str,
                "add_neg_prompt": str,
                "use_deepcache": int,
            },
            bool_value_options=(
                "image_start",
                "cpu_offload",
                "disable_sp",
                "use_fp8",
                "use_sage",
                "keep_work_dir",
            ),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"Hunyuan-GameCraft output was not written: {output_path}")
        return {
            "actions": actions,
            "action_speeds": action_speeds,
            "action_source": action_payload["action_source"],
            "control_source": action_payload["control_source"],
            "camera_tokens": action_payload["camera_tokens"],
            "camera_path": action_payload["camera_path"],
            "action_spec_path": str(action_spec_path),
            "action_alignment": action_payload,
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
        }


__all__ = ["HunyuanGameCraftAdapter", "_action_payload_for_sample", "_action_plan_for_suite"]
