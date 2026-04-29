from __future__ import annotations

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


_DEFAULT_ACTIONS = ["w", "d", "w", "a", "w"]
_SUPPORTED_ACTIONS = {"w", "a", "s", "d", "left_rot", "right_rot", "up_rot", "down_rot"}


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
        actions, action_speeds = _action_plan_for_suite(generation, sample.suite)

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
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
        }


__all__ = ["HunyuanGameCraftAdapter", "_action_plan_for_suite"]
