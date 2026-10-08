"""Derive 30-second memory runs from the existing 60-second protocol.

FPS, checkpoints, quality settings, camera magnitudes and scoring stay inherited.
Only temporal budgets and their action/pose sampling change.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Sequence


def sample_action_timeline(actions: Sequence, count: int) -> list:
    """Sample a complete action timeline at uniformly spaced interval midpoints."""
    if not actions or count < 1:
        raise ValueError("a nonempty action timeline and positive count are required")
    return [actions[min((2 * i + 1) * len(actions) // (2 * count), len(actions) - 1)]
            for i in range(count)]


def retime_action_runs(actions: Sequence[str], speeds: Sequence[float], count: int) -> tuple[list[str], list[float]]:
    """Fit fixed-length native commands while preserving each ordered motion leg.

Each action has one speed. Merging identical adjacent commands and
redistributing their integrated speed preserves every leg, including inverse legs.
"""
    if not actions or len(actions) != len(speeds):
        raise ValueError("actions and speeds must be nonempty and aligned")
    runs: list[list] = []
    for action, speed in zip(actions, speeds):
        if runs and runs[-1][0] == action and runs[-1][1] == speed:
            runs[-1][2] += 1
        else:
            runs.append([action, speed, 1])
    if count < len(runs):
        raise ValueError(f"{count} native commands cannot preserve {len(runs)} motion legs")
    remaining = count - len(runs)
    quotas = [remaining * run[2] / len(actions) for run in runs]
    sizes = [1 + math.floor(q) for q in quotas]
    for i in sorted(range(len(runs)), key=lambda i: (-(quotas[i] % 1), i))[:count - sum(sizes)]:
        sizes[i] += 1
    result_actions, result_speeds = [], []
    for (action, speed, old_count), size in zip(runs, sizes):
        result_actions.extend([action] * size)
        result_speeds.extend([float(speed) * old_count / size] * size)
    return result_actions, result_speeds


def derive_30s_model(payload: dict[str, Any], key: str, source_config: str) -> dict[str, Any]:
    """Clone a published 60s model recipe, changing only its temporal controls."""
    result = deepcopy(payload)
    model = result["model"]
    generation = model["generation"]
    reference = deepcopy(generation)
    fps = float(generation.get("output_fps", generation.get("fps", 24)))
    if "target_duration_seconds" in generation:
        generation["target_duration_seconds"] = 30
    if key in {"alayaworld", "alayaworld_v11"}:
        stride = int(generation.get("temporal_stride", 8)) * int(generation.get("chunk_latent_frames", 4))
        generation["rounds"] = math.ceil(30 * fps / stride)
        generation["memory_reference_rounds"] = int(reference["rounds"])
    elif key == "evoke":
        from worldatlas.models.adapters.evoke.evoke import resolve_num_chunks
        generation["memory_reference_num_chunks"] = resolve_num_chunks(reference)["num_chunks"]
    elif key == "abot_world":
        generation["blocks"] = math.ceil(30 * fps / 12)
        generation["memory_reference_blocks"] = int(reference["blocks"])
    elif key == "infinite_world":
        generation["num_chunks"] = math.ceil((30 * fps - 1) / (int(generation["validation_num_frames"]) - 1))
        generation["target_duration_sec"] = 30
        generation["memory_action_duration_seconds"] = 30
    elif key == "dreamx_world":
        # Three-latent chunks; pixel frames = 4 * (latents - 1) + 1.
        generation["num_output_frames"] = 3 * math.ceil((math.ceil((30 * fps - 1) / 4) + 1) / 3)
    elif key == "matrix_game_1":
        generation["video_length"] = 1 + 4 * math.ceil((30 * fps - 1) / 4)
    elif key == "sana_wm":
        generation["num_frames"] = 1 + 8 * math.ceil((30 * fps - 1) / 8)
    elif key == "solarwm_h3":
        generation["memory_duration_seconds"] = 30
    elif "target_duration_seconds" not in generation:
        raise ValueError(f"No 30-second duration rule for {key}")
    model["name"] = model["name"] + "-30s"
    model.setdefault("metadata", {}).update(
        memory_duration_seconds=30, memory_reference_duration_seconds=60,
        memory_reference_config=source_config,
    )
    return result
