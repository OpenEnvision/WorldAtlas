"""Open Dreamer model adapter for WorldAtlas Arena.

Open Dreamer is an action-conditioned world model (a JAX/Flax Dreamer 4
implementation trained on Minecraft/VPT gameplay).  It takes no text prompt: the
only steering signal is one VPT action per frame, so the adapter forwards the
sample's symbolic camera path and lets
``worldarena.models.adapters.open_dreamer_actions`` translate it into keyboard and
mouse events.
"""

from __future__ import annotations

from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter
from worldarena.models.adapters.open_dreamer_actions import canonical_camera_path


class OpenDreamerAdapter(ExternalBatchAdapter):
    """Roll out Open Dreamer through a checkpoint-resident batch runner."""

    runner_module = "worldarena.models.adapters.open_dreamer_batch_runner"
    label = "Open Dreamer"

    def batch_checkpoint_load_policy(self) -> str:
        """The runner restores one Orbax bundle and reuses it for every request."""
        return "load_once"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        sample = request.sample
        is_dynamic = sample.generation_mode == "dynamic" or sample.suite.endswith("_dynamic")
        # Dynamic suites score how the world evolves in front of a held camera, so
        # the action stream stays a no-op instead of driving the player around.
        camera_path = ["fixed"] if is_dynamic else canonical_camera_path(sample.camera_path)
        payload.update(
            {
                "camera_path": camera_path,
                "camera_source": "dynamic_fixed" if is_dynamic else "worldarena_camera_path",
                "generation_mode": sample.generation_mode,
                "modality": sample.modality,
                "source_width": sample.width,
                "source_height": sample.height,
                "source_fps": sample.fps,
            }
        )
        return payload


__all__ = ["OpenDreamerAdapter"]
