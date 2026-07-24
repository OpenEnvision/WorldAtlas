"""InfiniteWorld interactive world model adapter for WorldAtlas Arena.

Maps ``camera_path`` tokens to discrete move/view action indices consumed by the
InfiniteWorld upstream repo and launches ``infinite_world_runner``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.checkpoints import project_root as worldarena_project_root
from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest
from worldarena.models.adapters.common import (
    build_env,
    extend_command_with_options,
    run_command,
)
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample


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
_WORLD_ARENA_CAMERA_ACTIONS = {
    "fixed": {"move": "no-op", "view": "no-op"},
    "push_in": {"move": "go forward", "view": "no-op"},
    "pull_out": {"move": "go back", "view": "no-op"},
    "move_left": {"move": "go left", "view": "no-op"},
    "move_right": {"move": "go right", "view": "no-op"},
    # WorldScore diagram convention: pan moves backward diagonally while turning
    # in the same direction; orbit moves forward diagonally toward the subject.
    "pan_left": {"move": "go back and go left", "view": "turn left"},
    "pan_right": {"move": "go back and go right", "view": "turn right"},
    "orbit_left": {"move": "go forward and go left", "view": "turn right"},
    "orbit_right": {"move": "go forward and go right", "view": "turn left"},
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


def _worldarena_actions_for_sample(sample: BenchmarkSample) -> tuple[list[str], list[dict[str, str]], str]:
    raw_camera_path = _camera_path_for_sample(sample) or ["fixed"]
    camera_tokens = [_sanitize_action_token(token) for token in raw_camera_path if str(token).strip()] or ["fixed"]
    control_source = "camera_path" if sample.camera_path else "annotation_or_default_camera_path"
    actions: list[dict[str, str]] = []
    for camera_token in camera_tokens:
        action = _WORLD_ARENA_CAMERA_ACTIONS.get(camera_token)
        if action is None:
            raise ValueError(f"unsupported Infinite-World camera token: {camera_token!r}")
        actions.append(dict(action))
    return camera_tokens, actions or [dict(_WORLD_ARENA_CAMERA_ACTIONS["fixed"])], control_source


def _build_action_payload(
    config: dict[str, Any],
    suite: str,
    *,
    validation_num_frames: int = 81,
    sample: BenchmarkSample | None = None,
) -> dict[str, Any]:
    action_plan = dict(config.get("action_plan", {}))
    num_chunks = max(int(config.get("num_chunks", 1)), 1)
    chunk_stride = int(config.get("chunk_stride", validation_num_frames - 1))
    required_frames = 1 + num_chunks * max(validation_num_frames - 1, 1)
    action_source = str(config.get("action_source", "worldarena_plan")).strip().lower()
    camera_tokens: list[str] = []
    control_source = action_source

    if sample is not None and action_source in {"worldarena_prompt", "camera_path", "prompt_target"}:
        camera_tokens, normalized_actions, control_source = _worldarena_actions_for_sample(sample)
        actions: list[str | dict[str, str]] = normalized_actions
        frames_per_action = max(math.ceil(required_frames / max(len(actions), 1)), 1)
    else:
        actions = _interaction_plan(config, suite)
        frames_per_action = int(action_plan.get("frames_per_action", 16))

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
        "action_source": action_source,
        "control_source": control_source,
        "camera_tokens": camera_tokens,
        "camera_path": list(sample.camera_path) if sample is not None else [],
        "prompt_current": sample.prompt_current if sample is not None else None,
        "prompt_target": sample.prompt_target if sample is not None else None,
    }


def _request_payload_for(
    request: PreparedGenerationRequest,
    *,
    generation: dict[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    action_payload = _build_action_payload(
        generation,
        request.sample.suite,
        validation_num_frames=int(generation.get("validation_num_frames", 81)),
        sample=request.sample,
    )
    action_spec_path = runtime_root / "actions" / f"{request.output_path.stem}_actions.json"
    action_spec_path.parent.mkdir(parents=True, exist_ok=True)
    action_spec_path.write_text(
        json.dumps(action_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "sample_id": request.sample.sample_id,
        "suite": request.sample.suite,
        "prompt": request.prompt,
        "prompt_source": "prompt_target",
        "prompt_current": request.sample.prompt_current,
        "prompt_target": request.sample.prompt_target,
        "image_path": str(request.conditioning_image),
        "actions_json": str(action_spec_path),
        "output_path": str(request.output_path),
        "camera_path": action_payload["camera_path"],
        "camera_tokens": action_payload["camera_tokens"],
        "actions": action_payload["actions"],
        "action_spec_path": str(action_spec_path),
        "action_source": action_payload["action_source"],
        "control_source": action_payload["control_source"],
        "seed": int(generation.get("seed", 42)),
    }


class InfiniteWorldAdapter(ModelAdapter):
    """Generate action-controlled world videos via the InfiniteWorld upstream repo."""

    def supports_batch_generation(self) -> bool:
        """Supports batch generation -> bool."""
        return True

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        if self.config.repo_root is None:
            raise ValueError("Infinite-World adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Infinite-World adapter requires checkpoint_dir")

        generation = dict(self.config.generation)
        project_root = worldarena_project_root()
        runtime_root = project_root / "cache" / "model_runtime" / "infinite_world"
        runtime_root.mkdir(parents=True, exist_ok=True)
        for request in requests:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)

        request_payloads = [
            _request_payload_for(request, generation=generation, runtime_root=runtime_root)
            for request in requests
        ]
        with TemporaryDirectory(prefix="worldarena_infinite_world_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            requests_json = temp_dir / "requests.json"
            results_json = temp_dir / "results.json"
            requests_json.write_text(
                json.dumps({"requests": request_payloads}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            command = [
                self.config.python_bin,
                "-m",
                "worldarena.models.adapters.infinite_world_runner",
                "--repo_root",
                str(self.config.repo_root),
                "--ckpt_dir",
                str(self.config.checkpoint_dir),
                "--requests_json",
                str(requests_json),
                "--results_json",
                str(results_json),
            ]
            extend_command_with_options(
                command,
                payload=generation,
                value_options={
                    "config_path": str,
                    "bucket_config_name": str,
                    "fps": int,
                    "seed": int,
                    "batch_size": int,
                    "num_chunks": int,
                    "num_sampling_steps": int,
                    "sample_shift": float,
                    "sample_guide_scale": float,
                    "quality": int,
                    "target_duration_sec": float,
                    "negative_prompt": str,
                    "device": str,
                },
            )

            env = build_env(
                extra_pythonpaths=[project_root, self.config.repo_root],
                overrides=self.config.env,
            )
            run_command(command, cwd=project_root, env=env)
            if not results_json.exists():
                raise FileNotFoundError(f"Infinite-World runner did not write results: {results_json}")
            raw_results = json.loads(results_json.read_text(encoding="utf-8"))

        results: dict[str, dict[str, Any]] = {}
        for item in raw_results.get("results", []):
            sample_id = str(item.get("sample_id", ""))
            if sample_id:
                results[sample_id] = dict(item)
        return results

    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        if self.config.repo_root is None:
            raise ValueError("Infinite-World adapter requires repo_root")
        if self.config.checkpoint_dir is None:
            raise ValueError("Infinite-World adapter requires checkpoint_dir")

        request = PreparedGenerationRequest(
            sample=sample,
            conditioning_image=conditioning_image,
            output_path=output_path,
            prompt=prompt,
        )
        payload = self.generate_batch([request]).get(sample.sample_id, {})
        if not payload:
            raise RuntimeError(f"Infinite-World runner returned no result for sample {sample.sample_id}")
        if payload.get("status") == "failed":
            raise RuntimeError(str(payload.get("error", "Infinite-World generation failed")))
        return payload


__all__ = [
    "InfiniteWorldAdapter",
    "_build_action_payload",
    "_request_payload_for",
]
