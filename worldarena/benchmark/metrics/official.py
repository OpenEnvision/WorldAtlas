from __future__ import annotations

from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, resolve_metric_backend


class OfficialMetric(Metric):
    supported_backends: set[str]
    auto_backend: str

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend=self.auto_backend,
            supported_backends=self.supported_backends,
        )

    def _base_details(self, prediction: LoadedMedia) -> dict[str, Any]:
        return {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }


__all__ = ["OfficialMetric"]
