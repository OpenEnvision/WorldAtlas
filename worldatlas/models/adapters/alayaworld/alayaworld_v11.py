"""AlayaWorld v1.1 AR/DMD controls (ViGeo, separate from the v1 DA3 demo)."""

from __future__ import annotations

import hashlib
from typing import Any

from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.external_batch import ExternalBatchAdapter
from .alayaworld import _build_control_payload

UPSTREAM_REVISION = "ea03cfbb2e4c4e9102ed8ea8562e0b5370ca9b79"


def runtime_options(generation: dict[str, Any]) -> dict[str, Any]:
    options = {
        "variant": "dmd", "rounds": 4, "height": 544, "width": 960,
        "fps": 24, "temporal_stride": 8, "chunk_latent_frames": 4,
        "history_latent_frames": 4, "sink_latent_frames": 1,
        "nproc_per_node": 1, "attention_type": "flash_attention_2",
        "seed": 1234, "increment_seed": True, "offline": True,
        "continue_on_error": True,
    }
    options.update(generation)
    if options["variant"] not in {"ar", "dmd"}:
        raise ValueError("AlayaWorld v1.1 variant must be ar or dmd")
    for key, expected in (("fps", 24), ("temporal_stride", 8),
                          ("chunk_latent_frames", 4), ("history_latent_frames", 4),
                          ("sink_latent_frames", 1), ("nproc_per_node", 1)):
        if float(options[key]) != expected:
            raise ValueError(f"AlayaWorld v1.1 requires {key}={expected}")
        options[key] = expected
    for key in ("height", "width", "rounds"):
        value = int(options[key])
        if float(options[key]) != value or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        options[key] = value
    if options["height"] % 32 or options["width"] % 32:
        raise ValueError("AlayaWorld v1.1 height and width must be multiples of 32")
    if options["attention_type"] not in {"flash_attention_2", "pytorch", "xformers"}:
        raise ValueError("ViGeo requires FA3 disabled; use flash_attention_2, pytorch or xformers")
    for key in ("increment_seed", "offline", "continue_on_error"):
        if not isinstance(options[key], bool):
            raise ValueError(f"{key} must be a YAML boolean")
    return options


def sample_seed(sample_id: str, options: dict[str, Any]) -> int:
    offset = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:4], "big")
    return (int(options["seed"]) + (offset if options["increment_seed"] else 0)) % (2**31)


class AlayaWorldV11Adapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.alayaworld.alayaworld_v11_batch_runner"
    label = "AlayaWorld v1.1"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        options = runtime_options(dict(self.config.generation))
        payload = super().batch_spec_payload(request)
        controls = _build_control_payload(request.sample, options)
        # ViGeo seeds 1 + (N - 1) * stride = 25 pixels; output keeps the
        # final seed frame and appends 32 generated frames per round.
        generated = controls["generated_frame_count"]
        controls.update(
            generated_frame_count=generated + 1,
            trajectory_frame_count=25 + generated,
            duration_seconds=(generated + 1) / options["fps"],
        )
        payload.update(controls)
        payload["seed"] = sample_seed(request.sample.sample_id, options)
        sequence = [str(p).strip() for p in request.sample.prompt_sequence if str(p).strip()]
        rounds = controls["round_count"]
        if len(sequence) > rounds:
            raise ValueError("prompt_sequence has more segments than rollout rounds")
        if sequence:
            base, extra = divmod(rounds, len(sequence))
            payload["prompt_schedule"] = [
                prompt for i, prompt in enumerate(sequence)
                for _ in range(base + (i < extra))
            ]
        else:
            payload["prompt_schedule"] = []
        return payload
