from __future__ import annotations

from typing import Any

from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.external_batch import ExternalBatchAdapter
from worldarena.models.adapters.gen3c import _camera_path_for_sample


def _sample_key_candidates(request: PreparedGenerationRequest) -> tuple[str, ...]:
    sample = request.sample
    return (
        sample.sample_id,
        sample.prediction_stem,
        sample.relative_path,
        sample.reference_path,
    )


def _camera_index_for_request(
    request: PreparedGenerationRequest,
    generation: dict[str, Any],
) -> int:
    mapping = generation.get("camera_index_by_sample")
    if isinstance(mapping, dict):
        for key in _sample_key_candidates(request):
            if key in mapping:
                return int(mapping[key])
    return int(generation.get("camera_index", generation.get("start_camera_idx", 0)))


class AC3DAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.ac3d_batch_runner"
    label = "AC3D"

    def batch_spec_payload(self, request: PreparedGenerationRequest) -> dict[str, Any]:
        payload = super().batch_spec_payload(request)
        generation = dict(self.config.generation)
        camera_index = _camera_index_for_request(request, generation)
        end_camera_idx = int(generation.get("end_camera_idx", camera_index + 1))
        if end_camera_idx <= camera_index:
            end_camera_idx = camera_index + 1
        payload.update(
            {
                "camera_index": camera_index,
                "start_camera_idx": camera_index,
                "end_camera_idx": end_camera_idx,
                "camera_path": _camera_path_for_sample(request.sample),
            }
        )
        return payload


__all__ = ["AC3DAdapter"]
