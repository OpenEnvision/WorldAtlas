"""Reproduce action timing transformations actually applied during generation."""

from __future__ import annotations

import math
from typing import Any


def realize_matrix_game_2_actions(payload: dict[str, Any]) -> dict[str, Any]:
    """Recover the runner's 3-latent-block alignment, never arbitrary padding.

    Older sidecars describe the pre-alignment program (e.g. 181 latents / 721
    frames), although the runner executed 183 latents / 729 frames, repeating
    the last keyboard/mouse row. Only accept this identifiable, complete input
    format. The caller's object and any existing sidecar remain untouched.
    """
    if payload.get("variant") != "matrix_game_2":
        return payload
    latent_count = int(payload["num_output_frames"])
    if latent_count < 1:
        raise ValueError("Matrix-Game-2 latent frame count must be positive")
    original_frames = (latent_count - 1) * 4 + 1
    aligned_latents = math.ceil(latent_count / 3) * 3
    executed_frames = (aligned_latents - 1) * 4 + 1
    declared_frames = int(payload.get("total_frames", original_frames))
    if declared_frames not in {original_frames, executed_frames}:
        raise ValueError("Matrix-Game-2 sidecar length is not its declared latent timeline")
    result = dict(payload)
    lengths = {}
    for field in ("keyboard_condition", "mouse_condition"):
        rows = payload.get(field)
        if rows is None and field == "mouse_condition" and payload.get("mode") == "templerun":
            continue
        if not isinstance(rows, list) or len(rows) != declared_frames:
            raise ValueError(f"Matrix-Game-2 {field} is incomplete; cannot reconstruct executed controls")
        lengths[field] = len(rows)
        result[field] = list(rows) + [list(rows[-1]) for _ in range(executed_frames - len(rows))]
    result["total_frames"] = executed_frames
    result["num_output_frames"] = aligned_latents
    if executed_frames != declared_frames:
        result["action_timeline_alignment"] = {
            "method": "matrix_game_2_runner_repeat_last_to_3_latent_block",
            "recorded_frames": declared_frames,
            "executed_frames": executed_frames,
            "recorded_latents": latent_count,
            "executed_latents": aligned_latents,
            "appended_frames": executed_frames - declared_frames,
            "recorded_field_lengths": lengths,
        }
    return result
