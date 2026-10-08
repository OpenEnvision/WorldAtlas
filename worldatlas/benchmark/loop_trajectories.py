"""Closed-loop camera trajectories that revisit earlier viewpoints for memory evaluation.

Ordinary ``camera_path`` synthesis accumulates rigid deltas without ever returning to
the anchor pose, so a revisit event never occurs and memory metrics have nothing to
score. The loops defined here close by construction: ``retrace`` loops return along the
algebraic inverse of their outbound leg, and ``circuit`` loops close through rotational
symmetry without ever backtracking.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from worldatlas.benchmark.synthetic_camera import (
    camera_delta,
    segment_frame_bounds,
    synthetic_camera_matrices,
)


TOKEN_INVERSES: dict[str, str] = {
    "fixed": "fixed",
    "push_in": "pull_out",
    "pull_out": "push_in",
    "move_left": "move_right",
    "move_right": "move_left",
    "pedestal_up": "pedestal_down",
    "pedestal_down": "pedestal_up",
    "pan_left": "pan_right",
    "pan_right": "pan_left",
    "tilt_up": "tilt_down",
    "tilt_down": "tilt_up",
    "roll_cw": "roll_ccw",
    "roll_ccw": "roll_cw",
    "orbit_left": "orbit_right",
    "orbit_right": "orbit_left",
}

# Rigid deltas are float32 like the rest of the pose pipeline, and the geodesic angle
# reads through an arccos near 1, which amplifies float32 epsilon to roughly 0.03 deg
# once a couple dozen tokens have accumulated. A step angle that fails to divide a full
# turn misses closure by tens of degrees, so this floor still separates the two by three
# orders of magnitude.
DEFAULT_CLOSURE_TRANSLATION_RATIO = 1e-4
DEFAULT_CLOSURE_ROTATION_DEGREES = 0.1


@dataclass(frozen=True, slots=True)
class LoopLeg:
    """One outbound movement ending at a named waypoint."""

    tokens: tuple[str, ...]
    waypoint: str


@dataclass(frozen=True, slots=True)
class LoopDefinition:
    """A closed camera itinerary expressed as legs between named waypoints."""

    loop_id: str
    family: str
    description: str
    legs: tuple[LoopLeg, ...]
    start_waypoint: str = "A"


@dataclass(frozen=True, slots=True)
class LoopWaypointVisit:
    """One arrival at a waypoint, located in the resampled frame timeline."""

    label: str
    frame_index: int
    visit_ordinal: int


@dataclass(frozen=True, slots=True)
class LoopRevisitPair:
    """A nominal (first-visit, revisit) frame pair for one waypoint."""

    label: str
    first_frame: int
    revisit_frame: int

    @property
    def frame_gap(self) -> int:
        """Number of frames elapsed between the two visits."""
        return self.revisit_frame - self.first_frame


@dataclass(frozen=True, slots=True)
class LoopCameraTrajectory:
    """A synthesized closed-loop trajectory with its revisit structure."""

    loop_id: str
    family: str
    matrices: np.ndarray
    camera_path: tuple[str, ...]
    visits: tuple[LoopWaypointVisit, ...]
    revisit_pairs: tuple[LoopRevisitPair, ...]
    closure_translation: float
    closure_rotation_degrees: float
    path_extent: float

    @property
    def closure_translation_ratio(self) -> float:
        """Closure translation error relative to the trajectory's spatial extent."""
        return self.closure_translation / max(self.path_extent, 1e-8)


def invert_tokens(tokens: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return the token sequence that exactly undoes ``tokens``.

    Each supported token has a rigid-delta inverse, so reversing the order and
    substituting inverses yields an identity product regardless of step sizes.
    """
    inverted: list[str] = []
    for token in reversed(list(tokens)):
        if token not in TOKEN_INVERSES:
            raise ValueError(f"token has no defined inverse: {token!r}")
        inverted.append(TOKEN_INVERSES[token])
    return tuple(inverted)


def repeat_tokens_to_integer_laps(
    tokens: tuple[str, ...] | list[str],
    minimum_count: int,
) -> tuple[str, ...]:
    """Grow a token sequence to at least ``minimum_count`` entries, whole laps only.

    Adapters that stretch a short clip to a long rollout must not spread the extra
    chunks across individual tokens. Handing out a remainder (``divmod``) gives the
    first few tokens one extra copy, which unbalances a retrace loop's forward and
    reverse legs and leaves a circuit loop on a fractional turn -- the camera never
    comes back and the return gate reads it as a model that forgot the scene.

    Repeating the whole sequence keeps both invariants: inverse tokens stay paired
    and circuits stay on integer turns. The count rounds up so the rollout still
    covers the requested duration.
    """
    ordered = [str(token).strip() for token in tokens if str(token).strip()]
    if not ordered:
        ordered = ["fixed"]
    lap_length = len(ordered)
    requested = max(int(minimum_count), 1)
    laps = max(1, -(-requested // lap_length))
    return tuple(ordered) * laps


def _repeat(token: str, count: int) -> tuple[str, ...]:
    return tuple(token for _ in range(max(int(count), 0)))


def _retrace_legs(outbound: tuple[LoopLeg, ...], *, start_waypoint: str) -> tuple[LoopLeg, ...]:
    """Append inverse legs so the itinerary walks back through every waypoint."""
    labels = [start_waypoint, *(leg.waypoint for leg in outbound)]
    return (
        *outbound,
        *(
            LoopLeg(tokens=invert_tokens(leg.tokens), waypoint=labels[index])
            for index, leg in reversed(list(enumerate(outbound)))
        ),
    )


def _yaw_loop_360(runtime: dict[str, Any]) -> tuple[LoopLeg, ...]:
    steps = int(runtime.get("loop_yaw_full_turn_steps", 12))
    quarter = max(steps // 4, 1)
    return (
        LoopLeg(tokens=_repeat("pan_right", quarter), waypoint="B"),
        LoopLeg(tokens=_repeat("pan_right", quarter), waypoint="C"),
        LoopLeg(tokens=_repeat("pan_right", quarter), waypoint="D"),
        LoopLeg(tokens=_repeat("pan_right", steps - 3 * quarter), waypoint="A"),
    )


def _yaw_loop_180(runtime: dict[str, Any]) -> tuple[LoopLeg, ...]:
    steps = int(runtime.get("loop_yaw_half_turn_steps", 6))
    return _retrace_legs(
        (LoopLeg(tokens=_repeat("pan_right", steps), waypoint="B"),),
        start_waypoint="A",
    )


def _dolly_loop(runtime: dict[str, Any]) -> tuple[LoopLeg, ...]:
    steps = int(runtime.get("loop_dolly_steps", 4))
    return _retrace_legs(
        (LoopLeg(tokens=_repeat("push_in", steps), waypoint="B"),),
        start_waypoint="A",
    )


def _square_loop(runtime: dict[str, Any]) -> tuple[LoopLeg, ...]:
    edge = int(runtime.get("loop_square_edge_steps", 3))
    turn = int(runtime.get("loop_square_turn_steps", 3))
    corner = (*_repeat("push_in", edge), *_repeat("pan_right", turn))
    return (
        LoopLeg(tokens=corner, waypoint="B"),
        LoopLeg(tokens=corner, waypoint="C"),
        LoopLeg(tokens=corner, waypoint="D"),
        LoopLeg(tokens=corner, waypoint="A"),
    )


def _double_yaw_loop(runtime: dict[str, Any]) -> tuple[LoopLeg, ...]:
    steps = int(runtime.get("loop_yaw_half_turn_steps", 6))
    out = _repeat("pan_right", steps)
    back = invert_tokens(out)
    return (
        LoopLeg(tokens=out, waypoint="B"),
        LoopLeg(tokens=back, waypoint="A"),
        LoopLeg(tokens=out, waypoint="B"),
        LoopLeg(tokens=back, waypoint="A"),
    )


def _palindrome_loop(runtime: dict[str, Any]) -> tuple[LoopLeg, ...]:
    edge = int(runtime.get("loop_palindrome_edge_steps", 3))
    turn = int(runtime.get("loop_palindrome_turn_steps", 2))
    return _retrace_legs(
        (
            LoopLeg(tokens=_repeat("push_in", edge), waypoint="B"),
            LoopLeg(
                tokens=(*_repeat("pan_right", turn), *_repeat("push_in", edge)),
                waypoint="C",
            ),
        ),
        start_waypoint="A",
    )


_LOOP_BUILDERS = {
    "yaw_loop_360": (
        "circuit",
        "Full 360-degree in-place sweep returning to the anchor heading",
        _yaw_loop_360,
    ),
    "yaw_loop_180": (
        "retrace",
        "Rotate away by half a turn and retrace back to the anchor heading",
        _yaw_loop_180,
    ),
    "dolly_loop": (
        "retrace",
        "Translate forward and retrace back to the anchor position",
        _dolly_loop,
    ),
    "square_loop": (
        "circuit",
        "Closed square circuit that never backtracks along its own path",
        _square_loop,
    ),
    "double_yaw_loop": (
        "retrace",
        "Two consecutive rotational round trips, revisiting both waypoints twice",
        _double_yaw_loop,
    ),
    "palindrome_loop": (
        "retrace",
        "Three-waypoint out-and-back, revisiting the middle waypoint",
        _palindrome_loop,
    ),
}

SUPPORTED_LOOP_IDS = tuple(sorted(_LOOP_BUILDERS))


def loop_definition(loop_id: str, runtime: dict[str, Any] | None = None) -> LoopDefinition:
    """Build the leg itinerary for one loop identifier."""
    key = str(loop_id or "").strip().lower()
    entry = _LOOP_BUILDERS.get(key)
    if entry is None:
        raise ValueError(
            f"unsupported loop trajectory {loop_id!r}; expected one of {SUPPORTED_LOOP_IDS}"
        )
    family, description, builder = entry
    legs = builder(dict(runtime or {}))
    if not legs:
        raise ValueError(f"loop trajectory {key!r} produced no legs")
    return LoopDefinition(loop_id=key, family=family, description=description, legs=legs)


def loop_camera_path(loop_id: str, runtime: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Expand a loop identifier into its flat ``camera_path`` token sequence."""
    definition = loop_definition(loop_id, runtime)
    return tuple(token for leg in definition.legs for token in leg.tokens)


def _rotation_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Geodesic SO(3) angle between two rotation matrices, in degrees."""
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _waypoint_visits(
    definition: LoopDefinition,
    frame_bounds: tuple[int, ...],
) -> tuple[LoopWaypointVisit, ...]:
    """Locate each waypoint arrival in the resampled frame timeline."""
    token_boundaries = [0]
    for leg in definition.legs:
        token_boundaries.append(token_boundaries[-1] + len(leg.tokens))

    labels = [definition.start_waypoint, *(leg.waypoint for leg in definition.legs)]
    seen: dict[str, int] = {}
    visits: list[LoopWaypointVisit] = []
    for label, token_index in zip(labels, token_boundaries):
        ordinal = seen.get(label, 0)
        seen[label] = ordinal + 1
        visits.append(
            LoopWaypointVisit(
                label=label,
                frame_index=int(frame_bounds[token_index]),
                visit_ordinal=ordinal,
            )
        )
    return tuple(visits)


def _revisit_pairs(visits: tuple[LoopWaypointVisit, ...]) -> tuple[LoopRevisitPair, ...]:
    """Enumerate every ordered pair of visits that share a waypoint label."""
    by_label: dict[str, list[LoopWaypointVisit]] = {}
    for visit in visits:
        by_label.setdefault(visit.label, []).append(visit)

    pairs: list[LoopRevisitPair] = []
    for label, label_visits in by_label.items():
        ordered = sorted(label_visits, key=lambda item: item.frame_index)
        for outer in range(len(ordered)):
            for inner in range(outer + 1, len(ordered)):
                if ordered[inner].frame_index <= ordered[outer].frame_index:
                    continue
                pairs.append(
                    LoopRevisitPair(
                        label=label,
                        first_frame=ordered[outer].frame_index,
                        revisit_frame=ordered[inner].frame_index,
                    )
                )
    return tuple(sorted(pairs, key=lambda item: (item.first_frame, item.revisit_frame)))


def loop_camera_trajectory(
    loop_id: str,
    *,
    target_frames: int,
    runtime: dict[str, Any] | None = None,
) -> LoopCameraTrajectory:
    """Synthesize a closed-loop pose trajectory together with its revisit pairs."""
    runtime = dict(runtime or {})
    definition = loop_definition(loop_id, runtime)
    camera_path = tuple(token for leg in definition.legs for token in leg.tokens)
    frame_count = max(int(target_frames), len(camera_path) + 1)

    trajectory = synthetic_camera_matrices(
        list(camera_path),
        target_frames=frame_count,
        runtime=runtime,
    )
    matrices = trajectory.matrices
    frame_bounds = segment_frame_bounds(len(camera_path), frame_count)
    visits = _waypoint_visits(definition, frame_bounds)

    closure = np.eye(4, dtype=np.float64)
    for token in camera_path:
        closure = closure @ camera_delta(token, runtime).astype(np.float64)
    closure_translation = float(np.linalg.norm(closure[:3, 3]))
    closure_rotation = _rotation_angle_degrees(np.eye(3), closure[:3, :3])
    positions = matrices[:, :3, 3]
    path_extent = float(np.linalg.norm(np.ptp(positions, axis=0)))

    return LoopCameraTrajectory(
        loop_id=definition.loop_id,
        family=definition.family,
        matrices=matrices,
        camera_path=camera_path,
        visits=visits,
        revisit_pairs=_revisit_pairs(visits),
        closure_translation=closure_translation,
        closure_rotation_degrees=closure_rotation,
        path_extent=path_extent,
    )


def verify_loop_closure(
    trajectory: LoopCameraTrajectory,
    *,
    max_translation_ratio: float = DEFAULT_CLOSURE_TRANSLATION_RATIO,
    max_rotation_degrees: float = DEFAULT_CLOSURE_ROTATION_DEGREES,
) -> None:
    """Raise when a synthesized loop fails to return to its anchor pose.

    Circuit loops close through rotational symmetry, so a runtime override such as a
    step angle that does not divide a full turn silently breaks closure. Failing loudly
    here keeps a broken itinerary from reaching generation.
    """
    if trajectory.closure_rotation_degrees > float(max_rotation_degrees):
        raise ValueError(
            f"loop {trajectory.loop_id!r} does not close in rotation: "
            f"{trajectory.closure_rotation_degrees:.6f} deg exceeds {max_rotation_degrees} deg"
        )
    ratio = trajectory.closure_translation_ratio
    if ratio > float(max_translation_ratio):
        raise ValueError(
            f"loop {trajectory.loop_id!r} does not close in translation: "
            f"residual ratio {ratio:.3e} exceeds {max_translation_ratio:.3e}"
        )
    if not trajectory.revisit_pairs:
        raise ValueError(f"loop {trajectory.loop_id!r} defines no revisit pairs")


def loop_trajectory_manifest(
    loop_id: str,
    *,
    target_frames: int,
    fps: float,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe a loop trajectory as a JSON-serializable generation sidecar."""
    trajectory = loop_camera_trajectory(loop_id, target_frames=target_frames, runtime=runtime)
    verify_loop_closure(trajectory)
    frame_rate = max(float(fps), 1e-6)
    return {
        "loop_id": trajectory.loop_id,
        "loop_family": trajectory.family,
        "camera_path": list(trajectory.camera_path),
        "target_frames": int(len(trajectory.matrices)),
        "fps": float(fps),
        "duration_seconds": round(len(trajectory.matrices) / frame_rate, 6),
        "closure_translation_ratio": round(trajectory.closure_translation_ratio, 9),
        "closure_rotation_degrees": round(trajectory.closure_rotation_degrees, 9),
        "waypoint_visits": [
            {
                "label": visit.label,
                "frame_index": visit.frame_index,
                "visit_ordinal": visit.visit_ordinal,
            }
            for visit in trajectory.visits
        ],
        "nominal_revisit_pairs": [
            {
                "label": pair.label,
                "first_frame": pair.first_frame,
                "revisit_frame": pair.revisit_frame,
                "frame_gap": pair.frame_gap,
                "gap_seconds": round(pair.frame_gap / frame_rate, 6),
            }
            for pair in trajectory.revisit_pairs
        ],
    }


__all__ = [
    "DEFAULT_CLOSURE_ROTATION_DEGREES",
    "DEFAULT_CLOSURE_TRANSLATION_RATIO",
    "SUPPORTED_LOOP_IDS",
    "TOKEN_INVERSES",
    "LoopCameraTrajectory",
    "LoopDefinition",
    "LoopLeg",
    "LoopRevisitPair",
    "LoopWaypointVisit",
    "invert_tokens",
    "loop_camera_path",
    "loop_camera_trajectory",
    "loop_definition",
    "loop_trajectory_manifest",
    "repeat_tokens_to_integer_laps",
    "verify_loop_closure",
]
