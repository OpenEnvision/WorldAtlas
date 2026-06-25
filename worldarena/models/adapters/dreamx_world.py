from __future__ import annotations

from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter
from worldarena.models.adapters.lingbot_world import _camera_path_for_sample


DREAMX_ACTION_BY_CAMERA_TOKEN = {
    "fixed": "w",
    "push_in": "w",
    "pull_out": "s",
    "move_left": "a",
    "move_right": "d",
    "pan_left": "j",
    "pan_right": "l",
    "orbit_left": "wj",
    "orbit_right": "wl",
}
DREAMX_ACTION_SPEED_BY_ACTION = {
    "w": 4,
    "s": 4,
    "a": 4,
    "d": 4,
    "j": 6,
    "l": 6,
    "i": 6,
    "k": 6,
    "wj": 6,
    "wl": 6,
    "sj": 6,
    "sl": 6,
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _suite_override(generation: dict[str, Any], key: str, suite: str) -> Any:
    by_suite = generation.get(f"{key}_by_suite")
    if isinstance(by_suite, dict) and suite in by_suite:
        return by_suite[suite]
    return generation.get(key)


def _dreamx_actions_for_request(
    request: PreparedGenerationRequest,
    generation: dict[str, Any],
) -> tuple[list[str], list[int]]:
    explicit_actions = _suite_override(generation, "action_seq", request.sample.suite)
    explicit_speeds = _suite_override(generation, "action_speed_list", request.sample.suite)
    if explicit_actions:
        actions = [str(item).strip().lower() for item in _as_list(explicit_actions) if str(item).strip()]
        speeds = [int(item) for item in _as_list(explicit_speeds)]
        if not speeds:
            speeds = [int(generation.get("action_speed", 4))] * len(actions)
        while len(speeds) < len(actions):
            speeds.append(speeds[-1])
        return actions, speeds[: len(actions)]

    configured_map = generation.get("action_map") if isinstance(generation.get("action_map"), dict) else {}
    camera_path = _camera_path_for_sample(request.sample) or ["push_in"]
    max_segments = int(generation.get("max_action_segments", 3))
    actions: list[str] = []
    speeds: list[int] = []
    for camera_token in camera_path[:max(max_segments, 1)]:
        token = str(camera_token).strip().lower()
        action = str(configured_map.get(token, DREAMX_ACTION_BY_CAMERA_TOKEN.get(token, "w"))).strip().lower()
        if not action:
            continue
        actions.append(action)
        speeds.append(
            int(
                generation.get(
                    "rotation_speed" if any(key in action for key in ("j", "l", "i", "k")) else "action_speed",
                    DREAMX_ACTION_SPEED_BY_ACTION.get(action, 4),
                )
            )
        )
    if not actions:
        actions = [str(generation.get("default_action", "w"))]
        speeds = [int(generation.get("action_speed", 4))]
    return actions, speeds


class DreamXWorldAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.dreamx_world_batch_runner"
    label = "DreamX World"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        generation = dict(self.config.generation)
        action_seq, action_speed_list = _dreamx_actions_for_request(request, generation)
        payload.update(
            {
                "camera_path": list(request.sample.camera_path),
                "dreamx_action_seq": action_seq,
                "dreamx_action_speed_list": action_speed_list,
            }
        )
        return payload


__all__ = ["DreamXWorldAdapter"]
