"""WorldArena to Minecraft/VPT action translation for the Open Dreamer adapter.

Open Dreamer is action-conditioned: instead of a text prompt or a camera-pose
trajectory it consumes one VPT action dictionary per frame, holding keyboard keys
and applying raw mouse deltas.  WorldArena stores camera intent as symbolic path
segments, so converting those segments into VPT events belongs at this adapter
boundary -- the same place ``camera_i2v_common`` converts them into RealEstate10K
matrices.

The module deliberately avoids ``jax`` and the upstream ``dreamer`` package so
the mapping stays testable from the core evaluation environment; only
``open_dreamer_batch_runner`` loads the JAX runtime.
"""

from __future__ import annotations

from typing import Any, Sequence

from worldarena.benchmark.synthetic_camera import (
    SUPPORTED_SYNTHETIC_CAMERA_TOKENS,
    _TOKEN_ALIASES,
)
from worldarena.models.adapters.camera_i2v_common import normalize_camera_path


# Upstream ``dreamer/actions.py`` scales raw VPT pixel deltas into degrees with
# ``CAMERA_SCALER = 360 / 2400`` and clips to ``CAMERA_MAXVAL`` degrees before the
# mu-law binning.  Mirrored here so sidecars can report the requested rotation in
# degrees without importing the upstream package.
VPT_DEGREES_PER_PIXEL = 360.0 / 2400.0
VPT_MAX_CAMERA_DEGREES = 30.0

NATIVE_ACTION_SPACE = "minecraft_vpt"

# Minecraft/VPT exposes no roll axis, so ``roll_cw`` / ``roll_ccw`` are absent
# here and rejected loudly instead of being degraded into a silent no-op that
# would still be scored as an attempted roll.
_VPT_SEGMENT_ACTIONS: dict[str, tuple[tuple[str, ...], float, float, str]] = {
    "fixed": ((), 0.0, 0.0, "exact_no_motion"),
    "push_in": (("key.keyboard.w",), 0.0, 0.0, "exact_translation"),
    "pull_out": (("key.keyboard.s",), 0.0, 0.0, "exact_translation"),
    "move_left": (("key.keyboard.a",), 0.0, 0.0, "exact_translation"),
    "move_right": (("key.keyboard.d",), 0.0, 0.0, "exact_translation"),
    "pan_left": ((), -1.0, 0.0, "exact_rotation"),
    "pan_right": ((), 1.0, 0.0, "exact_rotation"),
    # VPT mouse dy is screen-space: positive dy looks down.
    "tilt_up": ((), 0.0, -1.0, "exact_rotation"),
    "tilt_down": ((), 0.0, 1.0, "exact_rotation"),
    # Minecraft has no free vertical translation; jump rises and sneak lowers the
    # eye height, which is directionally right but much smaller in magnitude.
    "pedestal_up": (("key.keyboard.space",), 0.0, 0.0, "coarse_vertical_to_jump"),
    "pedestal_down": (("key.keyboard.left.shift",), 0.0, 0.0, "coarse_vertical_to_sneak"),
    # Orbiting is approximated by strafing while yawing the opposite way, which
    # keeps the subject roughly centred but does not hold a constant radius.
    "orbit_left": (("key.keyboard.a",), 1.0, 0.0, "coarse_orbit_to_strafe_and_yaw"),
    "orbit_right": (("key.keyboard.d",), -1.0, 0.0, "coarse_orbit_to_strafe_and_yaw"),
}

_ROTATION_TOKENS = frozenset(
    token
    for token, (_, dx_scale, dy_scale, _) in _VPT_SEGMENT_ACTIONS.items()
    if dx_scale or dy_scale
)
# Tokens WorldArena can express but the VPT action space cannot.
INEXPRESSIBLE_CAMERA_TOKENS = frozenset(
    SUPPORTED_SYNTHETIC_CAMERA_TOKENS - set(_VPT_SEGMENT_ACTIONS)
)


def canonical_camera_path(value: Any) -> list[str]:
    """Normalize path tokens and resolve the synthetic-camera aliases."""
    return [_TOKEN_ALIASES.get(token, token) for token in normalize_camera_path(value)]


def segment_interval_counts(segment_count: int, intervals: int) -> list[int]:
    """Split ``intervals`` frame transitions across ``segment_count`` segments.

    Mirrors the split in ``synthetic_camera.synthetic_camera_matrices`` so the
    actions handed to the model change segment on the same frames as the
    reference trajectory that the camera metrics score against.
    """
    segment_count = max(int(segment_count), 1)
    intervals = max(int(intervals), 0)
    base = intervals // segment_count
    remainder = intervals % segment_count
    return [base + (1 if index < remainder else 0) for index in range(segment_count)]


def vpt_action(
    keys: Sequence[str] = (),
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> dict[str, Any]:
    """Build one VPT action dictionary in the layout ``parse_action_dicts`` reads."""
    return {
        "mouse": {"dx": float(dx), "dy": float(dy), "buttons": [], "dwheel": 0.0},
        "keyboard": {"keys": [str(key) for key in keys]},
    }


def noop_vpt_action() -> dict[str, Any]:
    """Return the held-nothing, no-mouse-motion action."""
    return vpt_action()


def _clipped_degrees(pixels: float) -> tuple[float, bool]:
    degrees = pixels * VPT_DEGREES_PER_PIXEL
    clipped = abs(degrees) > VPT_MAX_CAMERA_DEGREES
    if clipped:
        degrees = VPT_MAX_CAMERA_DEGREES if degrees > 0 else -VPT_MAX_CAMERA_DEGREES
    return degrees, clipped


def build_vpt_action_sequence(
    camera_path: Any,
    *,
    context_frames: int,
    horizon: int,
    mouse_pan_pixels: float,
    mouse_tilt_pixels: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Translate a WorldArena camera path into a per-frame VPT action sequence.

    The returned list has ``context_frames + horizon`` entries, where entry ``t``
    is the action executed at frame ``t`` (the motion that produces frame
    ``t + 1``).  Upstream ``shift_actions`` re-aligns that to "the action that
    produced this frame" before the rollout splits context from future, so the
    camera path is laid out over the ``horizon`` generated transitions only:
    context frames come from real footage whose true actions are unknown, and the
    trailing entry is dropped by the shift.
    """
    context_frames = int(context_frames)
    horizon = int(horizon)
    if context_frames < 1:
        raise ValueError(f"context_frames must be >= 1, got {context_frames}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    tokens = canonical_camera_path(camera_path)
    unsupported = [token for token in tokens if token not in _VPT_SEGMENT_ACTIONS]
    if unsupported:
        detail = ", ".join(sorted(set(unsupported)))
        inexpressible = sorted(set(unsupported) & INEXPRESSIBLE_CAMERA_TOKENS)
        hint = (
            f" ({', '.join(inexpressible)} has no Minecraft/VPT equivalent)"
            if inexpressible
            else ""
        )
        raise ValueError(
            f"unsupported Open Dreamer camera token(s): {detail}{hint}; "
            f"supported tokens: {', '.join(sorted(_VPT_SEGMENT_ACTIONS))}"
        )

    rotation_tokens = [token for token in tokens if token in _ROTATION_TOKENS]
    if rotation_tokens and mouse_pan_pixels <= 0.0 and mouse_tilt_pixels <= 0.0:
        raise ValueError(
            "camera path requests rotation "
            f"({', '.join(sorted(set(rotation_tokens)))}) but both "
            "mouse_pan_pixels and mouse_tilt_pixels are zero"
        )

    total_frames = context_frames + horizon
    actions = [noop_vpt_action() for _ in range(total_frames)]
    counts = segment_interval_counts(len(tokens), horizon)

    segments: list[dict[str, Any]] = []
    approximations: list[dict[str, str]] = []
    clipped = False
    # Frame ``context_frames`` is the first generated frame, so its incoming
    # transition is stored at index ``context_frames - 1``.
    cursor = context_frames - 1
    for token, frame_count in zip(tokens, counts):
        keys, dx_scale, dy_scale, approximation = _VPT_SEGMENT_ACTIONS[token]
        dx = dx_scale * float(mouse_pan_pixels)
        dy = dy_scale * float(mouse_tilt_pixels)
        dx_degrees, dx_clipped = _clipped_degrees(dx)
        dy_degrees, dy_clipped = _clipped_degrees(dy)
        clipped = clipped or dx_clipped or dy_clipped
        for _ in range(frame_count):
            actions[cursor] = vpt_action(keys, dx=dx, dy=dy)
            cursor += 1
        segments.append(
            {
                "camera_token": token,
                "keys": list(keys),
                "mouse_dx": dx,
                "mouse_dy": dy,
                "mouse_dx_degrees_per_frame": dx_degrees,
                "mouse_dy_degrees_per_frame": dy_degrees,
                "frames": int(frame_count),
                "approximation": approximation,
            }
        )
        if not approximation.startswith("exact"):
            approximations.append(
                {
                    "camera_token": token,
                    "native_action": "+".join([*keys, *_mouse_labels(dx, dy)]) or "noop",
                    "approximation": approximation,
                }
            )

    metadata = {
        "camera_path": tokens,
        "native_action_space": NATIVE_ACTION_SPACE,
        "action_segments": segments,
        "approximations": approximations,
        "context_frames": context_frames,
        "horizon": horizon,
        "total_frames": total_frames,
        "context_action_policy": "noop",
        "mouse_pan_pixels": float(mouse_pan_pixels),
        "mouse_tilt_pixels": float(mouse_tilt_pixels),
        "mouse_degrees_per_pixel": VPT_DEGREES_PER_PIXEL,
        "mouse_delta_clipped": clipped,
    }
    return actions, metadata


def _mouse_labels(dx: float, dy: float) -> list[str]:
    labels: list[str] = []
    if dx:
        labels.append("yaw_right" if dx > 0 else "yaw_left")
    if dy:
        labels.append("pitch_down" if dy > 0 else "pitch_up")
    return labels


__all__ = [
    "INEXPRESSIBLE_CAMERA_TOKENS",
    "NATIVE_ACTION_SPACE",
    "VPT_DEGREES_PER_PIXEL",
    "VPT_MAX_CAMERA_DEGREES",
    "build_vpt_action_sequence",
    "canonical_camera_path",
    "noop_vpt_action",
    "segment_interval_counts",
    "vpt_action",
]
