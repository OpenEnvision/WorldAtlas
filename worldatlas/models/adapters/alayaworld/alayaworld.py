"""AlayaWorld continuous-camera adapter for WorldAtlas.

AlayaWorld consumes a first-frame image, a pixel-rate ``cam_c2w`` trajectory,
and a text prompt.  The batch runner materializes those official case files;
this adapter is responsible for turning WorldAtlas's symbolic camera path into
chunk-aligned controls without importing the heavyweight upstream runtime.
"""

from __future__ import annotations

from typing import Any

from worldatlas.benchmark.loop_trajectories import repeat_tokens_to_integer_laps
from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.benchmark.taxonomy import MEMORY_SUITE
from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


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

ALAYAWORLD_CAMERA_TOKENS = frozenset(
    {
        "fixed",
        "push_in",
        "pull_out",
        "move_left",
        "move_right",
        "pedestal_up",
        "pedestal_down",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "roll_cw",
        "roll_ccw",
        "orbit_left",
        "orbit_right",
    }
)

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
    "camera pedestal up": "pedestal_up",
    "camera pedestal down": "pedestal_down",
}


def _normalize_camera_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    token = _CAMERA_TOKEN_ALIASES.get(token, token)
    if token not in ALAYAWORLD_CAMERA_TOKENS:
        raise ValueError(
            f"unsupported AlayaWorld camera action {value!r}; supported tokens: "
            f"{sorted(ALAYAWORLD_CAMERA_TOKENS)}"
        )
    return token


def _camera_tokens_for_sample(sample: BenchmarkSample) -> tuple[list[str], str]:
    values = [value for value in sample.camera_path if str(value).strip()]
    if values:
        return [_normalize_camera_token(value) for value in values], "camera_path"

    prompt_prefix = str(sample.prompt_target or sample.prompt_current or "").split(".", 1)[0]
    normalized_prefix = " ".join(
        prompt_prefix.strip().lower().replace("-", " ").replace("_", " ").split()
    )
    prompt_token = _CAMERA_PROMPT_PREFIXES.get(normalized_prefix)
    if prompt_token is not None:
        return [prompt_token], "prompt_target"
    return ["fixed"], "fallback"


def _configured_camera_tokens(generation: dict[str, Any], suite: str) -> list[str]:
    plan = dict(generation.get("action_plan", {}))
    by_suite = dict(plan.get("by_suite", {}))
    values = by_suite.get(suite) or plan.get("default") or ["fixed"]
    return [_normalize_camera_token(value) for value in values]


def _expand_round_actions(
    tokens: list[str],
    rounds: int,
    *,
    closed_loop: bool = False,
) -> list[str]:
    """Stretch the itinerary to cover ``rounds`` rounds.

    Closed-loop memory samples repeat whole laps. Handing out a ``divmod``
    remainder gives the leading tokens one extra copy, which unbalances the
    forward and reverse legs of a retrace loop and leaves a circuit on a
    fractional turn; the return gate then reads a broken itinerary as a model
    that forgot the scene. The other suites keep the per-token spread their
    published predictions were generated with.
    """
    if not tokens:
        tokens = ["fixed"]
    round_count = max(int(rounds), len(tokens), 1)
    if closed_loop:
        return list(repeat_tokens_to_integer_laps(tokens, round_count))
    base, remainder = divmod(round_count, len(tokens))
    expanded: list[str] = []
    for index, token in enumerate(tokens):
        expanded.extend([token] * (base + (1 if index < remainder else 0)))
    return expanded


def _build_control_payload(
    sample: BenchmarkSample,
    generation: dict[str, Any],
) -> dict[str, Any]:
    action_source = str(
        generation.get("action_source", "worldatlas_camera_path")
    ).strip().lower()
    if action_source in {
        "worldatlas",
        "worldatlas_camera",
        "worldatlas_camera_path",
        "camera_path",
        "prompt_target",
    }:
        camera_tokens, control_source = _camera_tokens_for_sample(sample)
    elif action_source in {"config", "config_action_plan", "action_plan"}:
        camera_tokens = _configured_camera_tokens(generation, sample.suite)
        control_source = "config_action_plan"
    else:
        raise ValueError(f"unsupported AlayaWorld action_source: {action_source!r}")

    requested_rounds = max(int(generation.get("rounds", 45)), 1)
    round_actions = _expand_round_actions(
        camera_tokens,
        requested_rounds,
        closed_loop=sample.suite == MEMORY_SUITE,
    )
    trajectory_actions = None
    memory_policy = str(generation.get("memory_action_policy", "legacy_repeat"))
    if memory_policy not in {"legacy_repeat", "single_sequence"}:
        raise ValueError(f"unsupported memory_action_policy: {memory_policy!r}")
    if sample.suite == MEMORY_SUITE and memory_policy == "single_sequence":
        from worldatlas.benchmark.memory_duration import sample_action_timeline
        # The native chunk budget controls time, not how often the task repeats.
        # Pose runners interpolate this original itinerary over the full budget.
        trajectory_actions = list(camera_tokens)
        round_actions = sample_action_timeline(trajectory_actions, requested_rounds)
    elif sample.suite == MEMORY_SUITE and generation.get("memory_reference_rounds") is not None:
        from worldatlas.benchmark.memory_duration import sample_action_timeline
        trajectory_actions = _expand_round_actions(
            camera_tokens, int(generation["memory_reference_rounds"]), closed_loop=True,
        )
        # The pose runner samples the full 60s path over the shorter frame budget;
        # do not round 23 requested rounds back up to 32/48 whole action laps.
        round_actions = sample_action_timeline(trajectory_actions, requested_rounds)
    temporal_stride = max(int(generation.get("temporal_stride", 8)), 1)
    chunk_latent_frames = max(int(generation.get("chunk_latent_frames", 4)), 1)
    sink_latent_frames = max(int(generation.get("sink_latent_frames", 1)), 0)
    history_latent_frames = max(int(generation.get("history_latent_frames", 16)), 0)
    fps = max(float(generation.get("fps", 24.0)), 1e-6)
    frames_per_chunk = temporal_stride * chunk_latent_frames
    generated_frames = len(round_actions) * frames_per_chunk
    prefix_latent_frames = sink_latent_frames + history_latent_frames
    trajectory_frames = (prefix_latent_frames + len(round_actions) * chunk_latent_frames) * temporal_stride + 1

    return {
        "action_source": action_source,
        "control_source": control_source,
        "camera_path": list(sample.camera_path),
        "camera_tokens": camera_tokens,
        "round_actions": round_actions,
        **({"trajectory_actions": trajectory_actions} if trajectory_actions is not None else {}),
        "round_count": len(round_actions),
        "frames_per_chunk": frames_per_chunk,
        "generated_frame_count": generated_frames,
        "trajectory_frame_count": trajectory_frames,
        "duration_seconds": generated_frames / fps,
    }


class AlayaWorldAdapter(ExternalBatchAdapter):
    """Generate camera-controlled autoregressive videos with AlayaWorld."""

    runner_module = "worldatlas.models.adapters.alayaworld.alayaworld_batch_runner"
    label = "AlayaWorld"

    def batch_checkpoint_load_policy(self) -> str:
        """The batch runner keeps one Alaya engine resident per shard."""
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        payload.update(_build_control_payload(request.sample, dict(self.config.generation)))
        prompt_sequence = [
            str(value).strip()
            for value in request.sample.prompt_sequence
            if str(value).strip()
        ]
        payload.update(
            {
                "prompt_sequence": prompt_sequence,
                "seed": int(self.config.generation.get("seed", 1234)),
            }
        )
        if prompt_sequence and prompt_sequence[-1] != request.prompt:
            payload["skill_prompt"] = prompt_sequence[-1]
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
    "ALAYAWORLD_CAMERA_TOKENS",
    "AlayaWorldAdapter",
    "_build_control_payload",
    "_expand_round_actions",
    "_normalize_camera_token",
]
