"""WorldArena adapter for Riemann Dynamics Matrix-Game 3.5."""

from __future__ import annotations

from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.camera_i2v_common import normalize_camera_path
from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class MatrixGame35Adapter(ExternalBatchAdapter):
    """Run anchor-image samples through the official camera-trajectory API."""

    runner_module = "worldarena.models.adapters.matrix_game_35_batch_runner"
    label = "Matrix-Game 3.5"

    def batch_checkpoint_load_policy(self) -> str:
        return "reload_per_sample"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        is_dynamic = (
            request.sample.generation_mode == "dynamic"
            or request.sample.suite.endswith("_dynamic")
        )
        camera_path = ["fixed"] if is_dynamic else normalize_camera_path(request.sample.camera_path)
        payload.update(
            {
                "camera_path": camera_path,
                "camera_source": "dynamic_fixed" if is_dynamic else "worldarena_camera_path",
                "generation_mode": request.sample.generation_mode,
                "source_width": request.sample.width,
                "source_height": request.sample.height,
            }
        )
        return payload


__all__ = ["MatrixGame35Adapter"]
