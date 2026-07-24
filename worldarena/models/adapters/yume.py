"""Yume model adapter for WorldAtlas Arena.

Yume generates camera-controlled videos from a conditioning image. Camera motion
is derived from ``camera_path`` tokens or explicit movement config fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import build_env, extend_command_with_options, run_command
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample


_YUME_OPTION_NAMES = (
    "mode",
    "caption_model_dir",
    "fps",
    "sample_steps",
    "sample_num",
    "frame_zero",
    "shift",
    "seed",
    "resolution",
    "gpu_index",
    "refine_from_image",
    "load_caption_model",
    "memory_optimization",
    "vae_memory_optimization",
    "camera_movement1",
    "camera_movement2",
)
_WORLD_ARENA_ACTION_SOURCE_VALUES = {
    "worldarena",
    "worldarena_camera",
    "worldarena_camera_path",
    "camera_path",
    "prompt_target",
}
_YUME_TEXT_ACTIONS = {
    "fixed": ("Camera remains fixed.", "exact_no_motion"),
    "push_in": ("Person moves forward.", "exact_translation"),
    "pull_out": ("Person moves backward.", "exact_translation"),
    "move_left": ("Person moves left.", "exact_translation"),
    "move_right": ("Person moves right.", "exact_translation"),
    "pan_left": ("Person turns left.", "exact_rotation"),
    "pan_right": ("Person turns right.", "exact_rotation"),
    "orbit_left": ("Person moves left while turning right.", "coarse_orbit_to_text"),
    "orbit_right": ("Person moves right while turning left.", "coarse_orbit_to_text"),
    "tilt_up": ("Person looks up.", "coarse_tilt_to_text"),
    "tilt_down": ("Person looks down.", "coarse_tilt_to_text"),
    "pedestal_up": ("Camera moves upward.", "coarse_vertical_to_text"),
    "pedestal_down": ("Camera moves downward.", "coarse_vertical_to_text"),
}


def _normalize_camera_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _suite_value(payload: dict[str, Any], name: str, suite: str) -> Any:
    by_suite = payload.get(f"{name}_by_suite")
    if isinstance(by_suite, dict) and suite in by_suite:
        return by_suite[suite]
    return payload.get(name)


def _generation_payload_for_sample(generation: dict[str, Any], sample: BenchmarkSample) -> dict[str, Any]:
    payload = dict(generation)
    for option_name in _YUME_OPTION_NAMES:
        resolved = _suite_value(generation, option_name, sample.suite)
        if resolved is not None:
            payload[option_name] = resolved
    return payload


def _action_payload_for_sample(generation: dict[str, Any], sample: BenchmarkSample) -> dict[str, Any]:
    action_source = str(generation.get("action_source", "prompt_target")).strip().lower()
    if action_source not in _WORLD_ARENA_ACTION_SOURCE_VALUES:
        return {
            "action_source": action_source,
            "control_source": "config_or_prompt",
            "camera_tokens": [],
            "camera_path": list(sample.camera_path),
            "native_action_space": "yume_text_prompt",
            "text_actions": [],
            "approximations": [],
            "unsupported_camera_tokens": [],
        }

    camera_tokens = [
        _normalize_camera_token(token)
        for token in _camera_path_for_sample(sample)
        if str(token).strip()
    ] or ["fixed"]
    text_actions: list[str] = []
    approximations: list[dict[str, str]] = []
    unsupported_tokens: list[str] = []
    for camera_token in camera_tokens:
        mapped = _YUME_TEXT_ACTIONS.get(camera_token)
        if mapped is None:
            raise ValueError(f"unsupported YUME camera token: {camera_token!r}")
        text_action, approximation = mapped
        text_actions.append(text_action)
        if not approximation.startswith("exact"):
            approximations.append(
                {
                    "camera_token": camera_token,
                    "text_action": text_action,
                    "type": approximation,
                }
            )

    control_source = "camera_path_text"
    if approximations:
        control_source = "coarse_camera_path_text"

    return {
        "action_source": "worldarena_camera_path",
        "control_source": control_source,
        "camera_tokens": camera_tokens,
        "camera_path": list(sample.camera_path),
        "native_action_space": "yume_text_prompt",
        "text_actions": text_actions,
        "approximations": approximations,
        "unsupported_camera_tokens": unsupported_tokens,
    }


def _aligned_prompt_for_sample(
    generation: dict[str, Any],
    sample: BenchmarkSample,
    prompt: str,
) -> tuple[str, dict[str, Any]]:
    action_payload = _action_payload_for_sample(generation, sample)
    text_actions = [str(item).strip() for item in action_payload["text_actions"] if str(item).strip()]
    if not text_actions:
        return prompt, action_payload
    action_text = " ".join(text_actions)
    speed_cue = str(generation.get("worldarena_action_speed_cue", "") or "").strip()
    prefix = f"{action_text} {speed_cue}".strip()
    return f"{prefix} {prompt}".strip(), action_payload


class YumeAdapter(ModelAdapter):
    """Generate camera-controlled videos via the Yume upstream repo."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if self.config.repo_root is None:
            raise ValueError("YUME adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("YUME adapter requires checkpoint_dir")
        if not requests:
            return {}

        generation = self.config.generation
        project_root = Path(__file__).resolve().parents[3]
        output_dirs = {request.output_path.parent.resolve() for request in requests}
        if len(output_dirs) != 1:
            raise ValueError("YUME batch generation requires a single output directory")
        output_dir = next(iter(output_dirs))
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_dir = output_dir / "_batch_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"yume_{uuid4().hex}.jsonl"
        with spec_path.open("w", encoding="utf-8") as file:
            for request in requests:
                generation_payload = _generation_payload_for_sample(generation, request.sample)
                prompt_text, action_payload = _aligned_prompt_for_sample(
                    generation_payload,
                    request.sample,
                    request.prompt,
                )
                action_spec_path = request.output_path.parent / f"{request.output_path.stem}_actions.json"
                action_spec_path.write_text(
                    json.dumps(action_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                file.write(
                    json.dumps(
                        {
                            "sample_id": request.sample.sample_id,
                            "prediction_stem": request.sample.prediction_stem,
                            "suite": request.sample.suite,
                            "conditioning_image": str(request.conditioning_image.expanduser().resolve()),
                            "output_path": str(request.output_path),
                            "prompt": prompt_text,
                            "source_prompt": request.prompt,
                            "generation": generation_payload,
                            "action_alignment": action_payload,
                            "action_spec_path": str(action_spec_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.yume_batch_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--checkpoint_dir",
            str(self.config.checkpoint_dir),
            "--entrypoint",
            str(self.config.entrypoint or "webapp_single_gpu.py"),
            "--batch_spec_path",
            str(spec_path),
        ]
        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)

        results: dict[str, dict[str, Any]] = {}
        for request in requests:
            generation_payload = _generation_payload_for_sample(generation, request.sample)
            prompt_text, action_payload = _aligned_prompt_for_sample(
                generation_payload,
                request.sample,
                request.prompt,
            )
            action_spec_path = request.output_path.parent / f"{request.output_path.stem}_actions.json"
            if request.output_path.exists():
                results[request.sample.sample_id] = {
                    "status": "generated",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "prompt": prompt_text,
                    "source_prompt": request.prompt,
                    "action_source": action_payload["action_source"],
                    "control_source": action_payload["control_source"],
                    "camera_tokens": action_payload["camera_tokens"],
                    "camera_path": action_payload["camera_path"],
                    "text_actions": action_payload["text_actions"],
                    "action_spec_path": str(action_spec_path),
                    "action_alignment": action_payload,
                }
            else:
                results[request.sample.sample_id] = {
                    "status": "failed",
                    "error": f"YUME batch output was not written: {request.output_path}",
                    "command": command,
                    "prediction_path": str(request.output_path),
                    "prompt": prompt_text,
                    "source_prompt": request.prompt,
                    "action_source": action_payload["action_source"],
                    "control_source": action_payload["control_source"],
                    "camera_tokens": action_payload["camera_tokens"],
                    "camera_path": action_payload["camera_path"],
                    "text_actions": action_payload["text_actions"],
                    "action_spec_path": str(action_spec_path),
                    "action_alignment": action_payload,
                }
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if self.config.repo_root is None:
            raise ValueError("YUME adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("YUME adapter requires checkpoint_dir")

        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parents[3]

        runner_payload = _generation_payload_for_sample(generation, sample)
        prompt_text, action_payload = _aligned_prompt_for_sample(runner_payload, sample, prompt)
        action_spec_path = output_path.parent / f"{output_path.stem}_actions.json"
        action_spec_path.write_text(
            json.dumps(action_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        command = [
            self.config.python_bin,
            "-m",
            "worldarena.models.adapters.yume_runner",
            "--repo_root",
            str(self.config.repo_root),
            "--entrypoint",
            str(self.config.entrypoint or "webapp_single_gpu.py"),
            "--conditioning_image",
            str(conditioning_image.expanduser().resolve()),
            "--output_path",
            str(output_path),
            "--prompt",
            prompt_text,
        ]
        command.extend(["--checkpoint_dir", str(self.config.checkpoint_dir)])

        extend_command_with_options(
            command,
            payload=runner_payload,
            value_options={
                "mode": str,
                "caption_model_dir": str,
                "fps": int,
                "sample_steps": int,
                "sample_num": int,
                "frame_zero": int,
                "shift": float,
                "seed": int,
                "resolution": str,
                "gpu_index": int,
                "camera_movement1": str,
                "camera_movement2": str,
            },
            bool_value_options=(
                "refine_from_image",
                "load_caption_model",
                "memory_optimization",
                "vae_memory_optimization",
            ),
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=project_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"YUME output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt_text,
            "source_prompt": prompt,
            "action_source": action_payload["action_source"],
            "control_source": action_payload["control_source"],
            "camera_tokens": action_payload["camera_tokens"],
            "camera_path": action_payload["camera_path"],
            "text_actions": action_payload["text_actions"],
            "action_spec_path": str(action_spec_path),
            "action_alignment": action_payload,
        }


__all__ = ["YumeAdapter", "_action_payload_for_sample", "_aligned_prompt_for_sample"]
