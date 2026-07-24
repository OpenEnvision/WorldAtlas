"""Wrappers that delegate to WorldArena's in-tree official metric adapters."""

from __future__ import annotations

from typing import Any

from worldarena.benchmark.media import LoadedMedia, protocol_frame_details
from worldarena.benchmark.metrics.base import Metric, resolve_metric_backend


class OfficialMetric(Metric):
    """Base class for metrics delegated to the official runtime adapters."""

    supported_backends: set[str]
    auto_backend: str

    def _resolve_backend(self) -> str:
        """Validate configured backend against this metric's supported official runtimes."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend=self.auto_backend,
            supported_backends=self.supported_backends,
        )

    def _base_details(self, prediction: LoadedMedia) -> dict[str, Any]:
        """Seed metric details with protocol frame indices from the prediction."""
        return protocol_frame_details(prediction)


__all__ = ["OfficialMetric"]
