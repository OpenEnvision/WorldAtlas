"""Alaya-EVOKE camera-controlled adapter for WorldAtlas.

EVOKE consumes a first-frame image (i2v), an optional reference video (v2v),
a vipe ``cam_c2w`` trajectory, and a text prompt.  This adapter turns
WorldAtlas's symbolic camera path into chunk-aligned controls without
importing the upstream runtime.
"""

from __future__ import annotations

from typing import Any

from worldatlas.benchmark.loop_trajectories import repeat_tokens_to_integer_laps
from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.benchmark.taxonomy import MEMORY_SUITE
from worldatlas.models.adapters.base import PreparedGenerationRequest, plan_rollout
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


PIXEL_FRAMES_PER_CHUNK = 36
LATENT_FRAMES_PER_CHUNK = 33
OUTPUT_FPS = 24.0
DEFAULT_NUM_CHUNKS = 7

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

EVOKE_CAMERA_TOKENS = frozenset(
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

_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}


def _normalize_camera_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    token = _CAMERA_TOKEN_ALIASES.get(token, token)
    if token not in EVOKE_CAMERA_TOKENS:
        raise ValueError(
            f"unsupported EVOKE camera action {value!r}; supported tokens: "
            f"{sorted(EVOKE_CAMERA_TOKENS)}"
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


def _expand_chunk_actions(
    tokens: list[str],
    chunks: int,
    *,
    closed_loop: bool = False,
) -> list[str]:
    """Stretch the itinerary to cover ``chunks`` chunks.

    Closed-loop memory samples repeat whole laps. Handing out a ``divmod``
    remainder gives the leading tokens one extra copy, which unbalances the
    forward and reverse legs of a retrace loop and leaves a circuit on a
    fractional turn; the return gate then reads a broken itinerary as a model
    that forgot the scene. The other suites keep the per-token spread their
    published predictions were generated with.
    """
    if not tokens:
        tokens = ["fixed"]
    chunk_count = max(int(chunks), len(tokens), 1)
    if closed_loop:
        return list(repeat_tokens_to_integer_laps(tokens, chunk_count))
    base, remainder = divmod(chunk_count, len(tokens))
    expanded: list[str] = []
    for index, token in enumerate(tokens):
        expanded.extend([token] * (base + (1 if index < remainder else 0)))
    return expanded


def resolve_num_chunks(generation: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``num_chunks`` so persistent decode covers the target duration.

    Official persistent decode emits ``36 * chunks - 3`` pixel frames.  The
    ``NUM_FRAMES`` knob only picks the chunk count: ``33 * chunks``.
    """
    resolved = dict(generation)
    fps = max(float(resolved.get("fps", OUTPUT_FPS)), 1e-6)
    configured = max(int(resolved.get("num_chunks", DEFAULT_NUM_CHUNKS)), 1)
    target = resolved.get("target_duration_seconds")
    if target is None:
        chunks = configured
    else:
        needed = max(int(-(-(float(target) * fps + 3.0) // PIXEL_FRAMES_PER_CHUNK)), 1)
        chunks = max(needed, configured)
        plan = plan_rollout(
            target_seconds=float(target),
            native_fps=fps,
            unit_frames=PIXEL_FRAMES_PER_CHUNK,
            base_frames=0,
            min_units=chunks,
        )
        details = plan.as_details()
        details["rollout_total_frames"] = PIXEL_FRAMES_PER_CHUNK * chunks - 3
        details["rollout_achieved_seconds"] = details["rollout_total_frames"] / fps
        resolved["rollout_plan"] = details
    resolved["num_chunks"] = chunks
    resolved["num_frames"] = LATENT_FRAMES_PER_CHUNK * chunks
    return resolved


def resolve_sample_type(generation: dict[str, Any], sample: BenchmarkSample) -> str:
    explicit = str(generation.get("sample_type") or generation.get("mode") or "").strip().lower()
    if explicit in {"i2v", "v2v", "t2v"}:
        if explicit == "t2v":
            raise ValueError("EVOKE t2v cannot carry camera warp; WorldAtlas uses i2v/v2v")
        return explicit
    return "i2v"


def resolve_warp_enabled(generation: dict[str, Any], camera_tokens: list[str]) -> bool:
    raw = str(generation.get("warp", "auto")).strip().lower()
    if raw in {"on", "1", "true", "yes"}:
        return True
    if raw in {"off", "0", "false", "no"}:
        return False
    return any(token != "fixed" for token in camera_tokens)


def _prompt_schedule(
    *,
    prompt: str,
    prompt_sequence: list[str],
    num_chunks: int,
    enabled: bool,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    sequence = [str(value).strip() for value in prompt_sequence if str(value).strip()]
    skill = sequence[-1] if sequence and sequence[-1] != prompt else ""
    if not skill:
        return []
    switch_chunk = max(num_chunks // 2, 1)
    return [
        {"start_chunk": 0, "prompt": prompt},
        {"start_chunk": switch_chunk, "prompt": skill},
    ]


def _build_control_payload(
    sample: BenchmarkSample,
    generation: dict[str, Any],
    *,
    prompt: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_num_chunks(generation)
    action_source = str(resolved.get("action_source", "worldatlas_camera_path")).strip().lower()
    if action_source in {
        "worldatlas",
        "worldatlas_camera",
        "worldatlas_camera_path",
        "camera_path",
        "prompt_target",
    }:
        camera_tokens, control_source = _camera_tokens_for_sample(sample)
    elif action_source in {"config", "config_action_plan", "action_plan"}:
        camera_tokens = _configured_camera_tokens(resolved, sample.suite)
        control_source = "config_action_plan"
    else:
        raise ValueError(f"unsupported EVOKE action_source: {action_source!r}")

    chunk_actions = _expand_chunk_actions(
        camera_tokens,
        int(resolved["num_chunks"]),
        closed_loop=sample.suite == MEMORY_SUITE,
    )
    trajectory_actions = None
    memory_policy = str(resolved.get("memory_action_policy", "legacy_repeat"))
    if memory_policy not in {"legacy_repeat", "single_sequence"}:
        raise ValueError(f"unsupported memory_action_policy: {memory_policy!r}")
    if sample.suite == MEMORY_SUITE and memory_policy == "single_sequence":
        from worldatlas.benchmark.memory_duration import sample_action_timeline
        trajectory_actions = list(camera_tokens)
        chunk_actions = sample_action_timeline(trajectory_actions, int(resolved["num_chunks"]))
    elif sample.suite == MEMORY_SUITE and resolved.get("memory_reference_num_chunks") is not None:
        from worldatlas.benchmark.memory_duration import sample_action_timeline
        trajectory_actions = _expand_chunk_actions(
            camera_tokens, int(resolved["memory_reference_num_chunks"]), closed_loop=True,
        )
        chunk_actions = sample_action_timeline(trajectory_actions, int(resolved["num_chunks"]))
    fps = max(float(resolved.get("fps", OUTPUT_FPS)), 1e-6)
    generated_frames = PIXEL_FRAMES_PER_CHUNK * len(chunk_actions) - 3
    sample_type = resolve_sample_type(resolved, sample)
    warp_enabled = resolve_warp_enabled(resolved, camera_tokens)
    prompt_text = str(prompt if prompt is not None else sample.prompt_target or "").strip()
    prompt_sequence = [str(value).strip() for value in sample.prompt_sequence if str(value).strip()]
    schedule = _prompt_schedule(
        prompt=prompt_text,
        prompt_sequence=prompt_sequence,
        num_chunks=len(chunk_actions),
        enabled=str(resolved.get("enable_prompt_switching", True)).lower()
        not in {"0", "false", "no", "off"},
    )
    reference_video = ""
    if sample_type == "v2v":
        for candidate in (sample.path, sample.reference_path, sample.conditioning_path):
            if candidate and str(candidate).lower().endswith(tuple(_VIDEO_SUFFIXES)):
                reference_video = str(candidate)
                break
        if not reference_video:
            raise ValueError(
                f"EVOKE v2v sample {sample.sample_id} needs a video path; "
                "use sample_type=i2v or pass a video asset"
            )

    payload = {
        "action_source": action_source,
        "control_source": control_source,
        "camera_path": list(sample.camera_path),
        "camera_tokens": camera_tokens,
        "chunk_actions": chunk_actions,
        **({"trajectory_actions": trajectory_actions} if trajectory_actions is not None else {}),
        "num_chunks": len(chunk_actions),
        "num_frames": LATENT_FRAMES_PER_CHUNK * len(chunk_actions),
        "frames_per_chunk": PIXEL_FRAMES_PER_CHUNK,
        "generated_frame_count": generated_frames,
        "trajectory_frame_count": PIXEL_FRAMES_PER_CHUNK * len(chunk_actions),
        "duration_seconds": generated_frames / fps,
        "sample_type": sample_type,
        "warp_enabled": warp_enabled,
        "prompt_sequence": prompt_sequence,
        "segment_prompts": schedule,
        "reference_video": reference_video,
        "modality": sample.modality,
    }
    if resolved.get("rollout_plan") is not None:
        payload["rollout_plan"] = resolved["rollout_plan"]
    return payload


class EvokeAdapter(ExternalBatchAdapter):
    """Generate camera-controlled few-step videos with Alaya-EVOKE."""

    runner_module = "worldatlas.models.adapters.evoke.evoke_batch_runner"
    label = "Alaya-EVOKE"

    def batch_checkpoint_load_policy(self) -> str:
        """The official infer_batch driver keeps one pipeline resident per shard."""
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        generation = resolve_num_chunks(dict(self.config.generation))
        controls = _build_control_payload(
            request.sample,
            generation,
            prompt=request.prompt,
        )
        payload.update(controls)
        payload["seed"] = int(generation.get("seed", 44))
        if controls["segment_prompts"]:
            payload["skill_prompt"] = controls["segment_prompts"][-1]["prompt"]
        return payload

    def generate_batch(
        self,
        requests: list[PreparedGenerationRequest],
    ) -> dict[str, dict[str, Any]]:
        generation = resolve_num_chunks(dict(self.config.generation))
        controls_by_sample = {
            request.sample.sample_id: _build_control_payload(
                request.sample,
                generation,
                prompt=request.prompt,
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
    "DEFAULT_NUM_CHUNKS",
    "EVOKE_CAMERA_TOKENS",
    "LATENT_FRAMES_PER_CHUNK",
    "OUTPUT_FPS",
    "PIXEL_FRAMES_PER_CHUNK",
    "EvokeAdapter",
    "_build_control_payload",
    "_expand_chunk_actions",
    "_normalize_camera_token",
    "resolve_num_chunks",
    "resolve_sample_type",
    "resolve_warp_enabled",
]
