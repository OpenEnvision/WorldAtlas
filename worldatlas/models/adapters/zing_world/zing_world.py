"""WorldAtlas image/action requests for the official Zing-0.5 runtime."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter
from worldatlas.models.adapters.lingbot_world.common import _camera_path_for_sample

UPSTREAM_REVISION = "11212da06f63290f4d354cdc0324241ba6eac356"
CHECKPOINT_REVISION = "b9ca1ed1de750bf19a0e6562c6cd71fa7ef35648"
NATIVE_FPS = 24
ACTION_ORDER = "wasdijkl"
_ACTION_KEYS = {
    "fixed": "",
    "push_in": "w",
    "pull_out": "s",
    "move_left": "a",
    "move_right": "d",
    "tilt_up": "i",
    "pan_left": "j",
    "tilt_down": "k",
    "pan_right": "l",
    "orbit_left": "al",
    "orbit_right": "dj",
    "pedestal_up": "i",
    "pedestal_down": "k",
}
_APPROXIMATIONS = {"orbit_left", "orbit_right", "pedestal_up", "pedestal_down"}


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if (isinstance(value, bool) or not math.isfinite(float(value))
            or float(value) != int(value) or int(value) < minimum):
        raise ValueError(f"Zing {name} must be an integer >= {minimum}")
    return int(value)


def runtime_options(generation: dict[str, Any]) -> dict[str, Any]:
    """Resolve the native frame lattice without changing playback speed."""
    allowed = {
        "num_frames", "target_duration_seconds", "fps", "width", "height", "seed",
        "local_attn_size", "sink_size", "nproc_per_node", "require_persistent_batch",
    }
    unknown = set(generation) - allowed
    if unknown:
        raise ValueError(f"Unsupported Zing generation options: {sorted(unknown)}")
    if _integer(generation.get("nproc_per_node", 1), "nproc_per_node") != 1:
        raise ValueError("Zing uses one GPU per worker; use independent manifest shards")
    fps = _integer(generation.get("fps", NATIVE_FPS), "fps")
    if fps != NATIVE_FPS:
        raise ValueError("Zing uses native 24 FPS; changing FPS would change action duration")
    seconds = generation.get("target_duration_seconds")
    if seconds is None:
        requested = _integer(generation.get("num_frames", 121), "num_frames", 5)
    else:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Zing target_duration_seconds must be positive and finite")
        requested = max(5, math.ceil(seconds * fps))
    # The recorded video includes the reference; output.frames in Zing does not.
    num_frames = 1 + 4 * math.ceil((requested - 1) / 4)
    width = _integer(generation.get("width", 832), "width")
    height = _integer(generation.get("height", 480), "height")
    if width % 32 or height % 32:
        raise ValueError("Zing width and height must be divisible by 32")
    local = _integer(generation.get("local_attn_size", 97), "local_attn_size", -1)
    sink = _integer(generation.get("sink_size", 9), "sink_size", 0)
    # Match upstream _validate_cache_window with frames_per_block=4.
    if ((local == -1 and sink != 0)
            or (local != -1 and (local < 1 or math.ceil(local / 4) - math.ceil(sink / 4) < 2))):
        raise ValueError("Zing cache window is invalid; use 97/9, 33/5, or -1/0")
    seed = _integer(generation.get("seed", 42), "seed", 0)
    if seed >= 2**32:
        raise ValueError("Zing seed must be in [0, 2**32)")
    return {
        "num_frames": num_frames, "requested_frames": requested, "fps": fps,
        "width": width, "height": height, "seed": seed,
        "local_attn_size": local, "sink_size": sink,
    }


def sample_seed(base_seed: int, sample_id: str) -> int:
    """Keep each rollout reproducible across shard sizes, skips, and retries."""
    digest = hashlib.sha256(f"zing:{base_seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def keyboard_actions(camera_path: list[str], num_frames: int) -> tuple[list[list[int]], list[str]]:
    """Expand the entire camera itinerary into Zing's W/A/S/D/I/J/K/L rows."""
    tokens = [str(token).strip().lower() for token in camera_path if str(token).strip()] or ["fixed"]
    unknown = [token for token in tokens if token not in _ACTION_KEYS]
    if unknown:
        raise ValueError(f"Unsupported Zing camera actions (no native keyboard mapping): {unknown}")
    intervals = _integer(num_frames, "num_frames", 5) - 1
    base, remainder = divmod(intervals, len(tokens))
    if base == 0:
        raise ValueError("Zing rollout is too short to preserve every camera-path token")
    actions = []
    for index, token in enumerate(tokens):
        row = [int(key in _ACTION_KEYS[token]) for key in ACTION_ORDER]
        actions.extend([row.copy() for _ in range(base + (index < remainder))])
    return actions, sorted(set(tokens) & _APPROXIMATIONS)


def build_message(row: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Build one official JSONL record; all lengths include exactly one reference."""
    prompt = row["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Zing requires a non-empty prompt")
    prompt = prompt.strip()
    image = Path(row["conditioning_image"]).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Zing conditioning image not found: {image}")
    actions, _ = keyboard_actions(row.get("camera_path", []), options["num_frames"])
    return {
        "sample_id": row["sample_id"],
        "messages": [
            {"role": "user", "type": "text", "content": prompt},
            {
                "role": "target", "type": "video", "reference_frame_count": 1,
                "uri": str(image),
                "output": {
                    "frames": options["num_frames"] - 1,
                    "height": options["height"], "width": options["width"],
                },
                "controls": [{"type": "keyboard_direction_frame_interval", "actions": actions}],
            },
        ],
    }


class ZingWorldAdapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.zing_world.zing_world_batch_runner"
    label = "Zing-0.5"

    def __init__(self, config):
        super().__init__(config)
        runtime_options(config.generation)

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        options = runtime_options(self.config.generation)
        camera_path = _camera_path_for_sample(request.sample) or ["fixed"]
        _, approximations = keyboard_actions(camera_path, options["num_frames"])
        payload.update(camera_path=camera_path, action_approximations=approximations,
                       num_frames=options["num_frames"])
        return payload

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        if self.config.checkpoint_dir is None:
            raise ValueError("Zing requires a local checkpoint_dir")
        return super()._build_command(spec_path, generation_path)
