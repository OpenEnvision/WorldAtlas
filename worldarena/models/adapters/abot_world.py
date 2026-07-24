"""ABot-World action-conditioned rollout adapter.

The upstream project exposes an interactive streaming pipeline.  This adapter
turns WorldArena camera paths into its native ``W/A/S/D/I/J/K/L`` controls and
delegates a whole benchmark shard to one long-lived runner process so the model
is loaded only once.
"""

from __future__ import annotations

import math
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter


KEY_ORDER = ("W", "A", "S", "D", "I", "J", "K", "L")

_CAMERA_TOKEN_ALIASES = {
    "dolly_in": "push_in",
    "dolly_out": "pull_out",
    "zoom_in": "push_in",
    "zoom_out": "pull_out",
    "truck_left": "move_left",
    "truck_right": "move_right",
    "tiltup": "tilt_up",
    "tiltdown": "tilt_down",
    "stay": "fixed",
    "idle": "fixed",
    "none": "fixed",
}

# Keep the compound pan/orbit convention aligned with the other interactive
# WorldArena adapters: pan backs away while turning toward the same side;
# orbit moves forward on an arc while looking toward the subject.
_WORLD_ARENA_CAMERA_KEYS = {
    "fixed": (),
    "push_in": ("W",),
    "pull_out": ("S",),
    "move_left": ("A",),
    "move_right": ("D",),
    "pan_left": ("S", "A", "J"),
    "pan_right": ("S", "D", "L"),
    "orbit_left": ("W", "A", "L"),
    "orbit_right": ("W", "D", "J"),
    "tilt_up": ("I",),
    "tilt_down": ("K",),
}

_ACTION_ALIASES = {
    "forward": ("W",),
    "back": ("S",),
    "backward": ("S",),
    "left": ("A",),
    "right": ("D",),
    "look_up": ("I",),
    "look_down": ("K",),
    "look_left": ("J",),
    "look_right": ("L",),
    "turn_left": ("J",),
    "turn_right": ("L",),
}

_CAMERA_PROMPT_PREFIXES = {
    "camera fixed": "fixed",
    "fixed camera": "fixed",
    "camera push in": "push_in",
    "camera pushes in": "push_in",
    "camera pull out": "pull_out",
    "camera pulls out": "pull_out",
    "camera move left": "move_left",
    "camera moves left": "move_left",
    "camera move right": "move_right",
    "camera moves right": "move_right",
    "camera pan left": "pan_left",
    "camera pans left": "pan_left",
    "camera pan right": "pan_right",
    "camera pans right": "pan_right",
    "camera orbit left": "orbit_left",
    "camera orbits left": "orbit_left",
    "camera orbit right": "orbit_right",
    "camera orbits right": "orbit_right",
    "camera tilt up": "tilt_up",
    "camera tilts up": "tilt_up",
    "camera tilt down": "tilt_down",
    "camera tilts down": "tilt_down",
}


def _normalize_token(value: str) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _CAMERA_TOKEN_ALIASES.get(token, token)


def _keys_dict(keys: tuple[str, ...] | list[str] | set[str]) -> dict[str, bool]:
    active = {str(key).strip().upper() for key in keys}
    unknown = active.difference(KEY_ORDER)
    if unknown:
        raise ValueError(f"unsupported ABot-World keys: {sorted(unknown)}")
    return {key: key in active for key in KEY_ORDER}


def _action_for_value(value: Any) -> dict[str, bool]:
    """Normalize a camera token, key chord, mapping, or eight-value action row."""
    if isinstance(value, dict):
        return _keys_dict([key for key, enabled in value.items() if bool(enabled)])
    if isinstance(value, (list, tuple)):
        if len(value) == len(KEY_ORDER) and all(
            isinstance(item, (bool, int, float)) for item in value
        ):
            return _keys_dict(
                [key for key, enabled in zip(KEY_ORDER, value) if bool(enabled)]
            )
        return _keys_dict([str(item) for item in value])

    raw = str(value or "").strip()
    token = _normalize_token(raw)
    if token in _WORLD_ARENA_CAMERA_KEYS:
        return _keys_dict(_WORLD_ARENA_CAMERA_KEYS[token])
    if token in _ACTION_ALIASES:
        return _keys_dict(_ACTION_ALIASES[token])
    if any(separator in raw for separator in ("+", "|", ",")):
        normalized = raw.replace("|", "+").replace(",", "+")
        return _keys_dict([part for part in normalized.split("+") if part.strip()])
    if raw.upper() in KEY_ORDER:
        return _keys_dict([raw])
    raise ValueError(
        f"unsupported ABot-World action {value!r}; native controls are "
        "W/A/S/D/I/J/K/L and WorldArena camera tokens"
    )


def _camera_tokens_for_sample(sample: BenchmarkSample) -> tuple[list[str], str]:
    tokens = [_normalize_token(value) for value in sample.camera_path if str(value).strip()]
    if tokens:
        return tokens, "camera_path"

    prompt_prefix = str(sample.prompt_target or sample.prompt_current or "").split(".", 1)[0]
    normalized_prefix = " ".join(
        prompt_prefix.strip().lower().replace("-", " ").replace("_", " ").split()
    )
    prompt_token = _CAMERA_PROMPT_PREFIXES.get(normalized_prefix)
    if prompt_token is not None:
        return [prompt_token], "prompt_target"
    return ["fixed"], "dynamic_default" if sample.suite == "image_dynamic" else "fallback"


def _configured_actions(generation: dict[str, Any], suite: str) -> list[Any]:
    plan = dict(generation.get("action_plan", {}))
    by_suite = dict(plan.get("by_suite", {}))
    actions = by_suite.get(suite) or plan.get("default")
    return list(actions or ["fixed"])


def _build_control_payload(
    sample: BenchmarkSample,
    generation: dict[str, Any],
) -> dict[str, Any]:
    action_source = str(generation.get("action_source", "worldarena_camera_path")).strip().lower()
    if action_source in {
        "worldarena",
        "worldarena_camera",
        "worldarena_camera_path",
        "camera_path",
        "prompt_target",
    }:
        camera_tokens, control_source = _camera_tokens_for_sample(sample)
        actions: list[Any] = camera_tokens
    elif action_source in {"config", "config_action_plan", "action_plan"}:
        actions = _configured_actions(generation, sample.suite)
        camera_tokens = []
        control_source = "config_action_plan"
    else:
        raise ValueError(f"unsupported ABot-World action_source: {action_source!r}")

    normalized_actions = [_action_for_value(action) for action in actions]
    requested_blocks = max(int(generation.get("blocks", 5)), 1)
    block_count = max(requested_blocks, len(normalized_actions))
    frames_per_action = max(math.ceil(block_count / len(normalized_actions)), 1)
    block_actions: list[dict[str, bool]] = []
    for action in normalized_actions:
        block_actions.extend([dict(action) for _ in range(frames_per_action)])
    block_actions = block_actions[:block_count]
    while len(block_actions) < block_count:
        block_actions.append(dict(block_actions[-1]))

    return {
        "action_source": action_source,
        "control_source": control_source,
        "camera_path": list(sample.camera_path),
        "camera_tokens": camera_tokens,
        "key_order": list(KEY_ORDER),
        "actions": normalized_actions,
        "block_actions": block_actions,
        "block_count": block_count,
        "frames_per_action": frames_per_action,
    }


class ABotWorldAdapter(ExternalBatchAdapter):
    """Generate first-frame, text, and keyboard-conditioned ABot-World videos."""

    runner_module = "worldarena.models.adapters.abot_world_batch_runner"
    label = "ABot-World"

    def batch_checkpoint_load_policy(self) -> str:
        """The batch runner constructs one pipeline before its request loop."""
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        payload.update(_build_control_payload(request.sample, dict(self.config.generation)))
        payload["seed"] = int(self.config.generation.get("seed", 42))
        return payload

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        controls_by_sample = {
            request.sample.sample_id: _build_control_payload(
                request.sample,
                dict(self.config.generation),
            )
            for request in requests
        }
        results = super().generate_batch(requests)
        for sample_id, result in results.items():
            controls = controls_by_sample.get(sample_id)
            if controls is not None:
                result.update(controls)
        return results


__all__ = [
    "ABotWorldAdapter",
    "KEY_ORDER",
    "_action_for_value",
    "_build_control_payload",
]
