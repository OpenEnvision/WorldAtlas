"""MeT3R reconstruction-consistency metric implementations."""

from __future__ import annotations

from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.met3r_runtime import compute_met3r_consistency
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


MET3R_BACKENDS = {"met3r"}


def _default_normalization(distance: str) -> dict[str, Any]:
    """Pick sensible [0, 1] scaling defaults for each MeT3R distance name."""
    distance_name = distance.lower()
    if distance_name == "psnr":
        return {"lower": 0.0, "upper": 100.0, "higher_is_better": True}
    if distance_name == "ssim":
        return {"lower": 0.0, "upper": 1.0, "higher_is_better": True}
    if distance_name == "mse":
        return {"lower": 0.0, "upper": 4.0, "higher_is_better": False}
    if distance_name in {"rmse", "lpips"}:
        return {"lower": 0.0, "upper": 2.0, "higher_is_better": False}
    return {"lower": 0.0, "upper": 2.0, "higher_is_better": False}


class Met3RConsistencyMetric(Metric):
    """Score anchor-frame reconstruction consistency between reference and prediction."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Merge distance-specific normalization defaults with config overrides."""
        runtime_payload = dict(runtime or {})
        distance = str(runtime_payload.get("distance") or "cosine")
        super().__init__(
            "met3r_consistency",
            backend,
            {**_default_normalization(distance), **dict(normalization or {})},
        )
        self.runtime = runtime_payload
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the MeT3R reconstruction-consistency backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="met3r",
            supported_backends=MET3R_BACKENDS,
        )

    def _details(self, prediction: LoadedMedia) -> dict[str, Any]:
        """Collect protocol frame indices used during MeT3R scoring."""
        return {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Compare MeT3R feature consistency on paired anchor frames."""
        del sample

        backend = self._resolve_backend()
        details = self._details(prediction)
        try:
            result = compute_met3r_consistency(
                reference.anchor_frames,
                prediction.anchor_frames,
                runtime=self.runtime,
            )
            raw = result["raw"]
            details.update(result["details"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=str(result["backend"]),
                details=details,
                eligibility_status="official",
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="official",
                error=str(exc),
            )


__all__ = ["Met3RConsistencyMetric"]
