"""Echo-WM Base adapter for WorldAtlas."""

from __future__ import annotations

import math
import re
from typing import Any

from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter
from worldatlas.models.adapters.lingbot_world.common import _camera_path_for_sample


ECHO_WM_DEFAULT_FPS = 24.0
ECHO_WM_DEFAULT_FRAMES = 241
_PROMPT_FIELDS = ("Environment", "Character", "Style", "Perspective", "Sounds", "Speech")
_ACTION_KEYS = {
    "fixed": "none",
    "push_in": "w",
    "pull_out": "s",
    "move_left": "a",
    "move_right": "d",
    "pan_left": "j",
    "pan_right": "l",
    "tilt_up": "i",
    "tilt_down": "k",
    "pedestal_up": "i",
    "pedestal_down": "k",
    # Approximate an orbit by strafing while yawing back toward the subject.
    "orbit_left": "al",
    "orbit_right": "dj",
    "roll_cw": "none",
    "roll_ccw": "none",
}


def resolve_echo_frame_count(generation: dict[str, Any]) -> int:
    """Resolve an Echo-WM length on the LTX ``1 + 8m`` frame lattice."""
    fps = float(generation.get("fps", ECHO_WM_DEFAULT_FPS))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Echo-WM fps must be positive and finite, got {fps!r}")
    seconds = generation.get("target_duration_seconds")
    requested = (
        int(generation.get("num_frames", ECHO_WM_DEFAULT_FRAMES))
        if seconds is None
        else int(math.ceil(float(seconds) * fps))
    )
    if requested < 9:
        raise ValueError(f"Echo-WM requires at least 9 frames, got {requested}")
    return 1 + 8 * int(math.ceil((requested - 1) / 8))


def echo_action_string(camera_path: list[str], *, num_frames: int) -> tuple[str, list[str]]:
    """Map WorldAtlas camera tokens to Echo-WM's WASD/IJKL action DSL."""
    tokens = [str(value).strip().lower() for value in camera_path if str(value).strip()]
    tokens = tokens or ["fixed"]
    unknown = [token for token in tokens if token not in _ACTION_KEYS]
    if unknown:
        raise ValueError(f"unsupported Echo-WM camera action(s): {unknown}")
    intervals = int(num_frames) - 1
    base, remainder = divmod(intervals, len(tokens))
    if base <= 0:
        raise ValueError("Echo-WM output is too short for the requested camera path")
    segments = [
        f"{_ACTION_KEYS[token]}-{base + (1 if index < remainder else 0)}"
        for index, token in enumerate(tokens)
    ]
    approximations = [
        token
        for token in tokens
        if token in {"pedestal_up", "pedestal_down", "orbit_left", "orbit_right", "roll_cw", "roll_ccw"}
    ]
    return ",".join(segments), approximations


def echo_prompt(prompt: str, *, style: str | None, scene: str | None) -> str:
    """Wrap a benchmark prompt in Echo-WM Base's documented six-field schema."""
    text = " ".join(str(prompt).split())
    if not text:
        raise ValueError("Echo-WM requires a non-empty prompt")
    if all(re.search(rf"(?im)^\s*{re.escape(field)}\s*:", prompt) for field in _PROMPT_FIELDS):
        return prompt.strip()
    return "\n\n".join(
        (
            f"Environment: {text}",
            f"Character: {scene or 'No specific character is required.'}",
            f"Style: {style or 'Photorealistic cinematic world simulation.'}",
            "Perspective: Preserve the reference-image viewpoint and composition; "
            "camera motion follows the requested actions.",
            "Sounds: Natural environmental audio matching the visible scene and motion.",
            "Speech: None.",
        )
    )


class EchoWMAdapter(ExternalBatchAdapter):
    """Generate audio-video rollouts with Echo-WM Base."""

    runner_module = "worldatlas.models.adapters.echo_wm.echo_wm_batch_runner"
    label = "Echo-WM"

    def batch_checkpoint_load_policy(self) -> str:
        """The batch runner caches checkpoint state dicts once per resident worker."""
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        frame_count = resolve_echo_frame_count(self.config.generation)
        camera_path = _camera_path_for_sample(request.sample) or ["fixed"]
        action, approximations = echo_action_string(camera_path, num_frames=frame_count)
        payload.update(
            {
                "prompt": echo_prompt(
                    request.prompt,
                    style=request.sample.style,
                    scene=request.sample.scene,
                ),
                "action_str": action,
                "camera_path": camera_path,
                "action_approximations": approximations,
                "num_frames": frame_count,
            }
        )
        return payload


__all__ = [
    "EchoWMAdapter",
    "echo_action_string",
    "echo_prompt",
    "resolve_echo_frame_count",
]
