"""MeT3R reconstruction-consistency metric implementations."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.met3r_runtime import compute_met3r_video_consistency, _metric_options
from worldatlas.benchmark.met3r_protocol import PROTOCOL
from worldatlas.benchmark.metrics.base import (
    Metric,
    apply_display_error_ref,
    normalize_score,
    resolve_metric_backend,
)
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput


MET3R_BACKENDS = {"met3r"}


def _default_normalization(distance: str) -> dict[str, Any]:
    """WorldAtlas display transform only; retain the original paper raw score."""
    if distance != "cosine":
        raise ValueError("The published MEt3R metric requires cosine distance")
    return {"lower": 0.0, "upper": 2.0, "higher_is_better": False}


class Met3RConsistencyMetric(Metric):
    """Paper MEt3R on native consecutive generated views, independent of GT."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Merge distance-specific normalization defaults with config overrides."""
        runtime_payload = dict(runtime or {})
        distance = str(runtime_payload.get("distance") or "cosine")
        merged = {**_default_normalization(distance), **dict(normalization or {})}
        paper = {
            key: merged[key]
            for key in ("lower", "upper", "higher_is_better")
            if key in merged
        }
        extra = set(merged) - {"lower", "upper", "higher_is_better", "error_ref", "error_from"}
        if paper != _default_normalization("cosine") or extra:
            raise ValueError("MEt3R keeps the paper raw score and only the declared WorldAtlas 1-raw/2 display transform")
        if merged.get("error_ref") is not None and merged.get("error_from") != "raw":
            raise ValueError("MEt3R display error_ref must use error_from: raw")
        super().__init__("met3r_consistency", backend, merged)
        self.runtime = runtime_payload
        # Reject old MASt3R / bilinear / alternative-distance configurations
        # before constructing expensive models or emitting a misleading score.
        _metric_options(dict(runtime_payload, device="cpu"))
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
        """Describe the metric-specific protocol, not the shared anchor sampler."""
        return {
            "pairing_protocol": PROTOCOL,
            "implementation_revision": PROTOCOL,
            "input_image_used": False,
            "normalization_note": (
                "Paper raw is lower-is-better. Identity display is 1 - raw/2; "
                "Reports retain 1/(1+raw/error_ref) when error_ref is set; "
                "leaderboard publication averages per-video 1-raw/2 directly."
            ),
        }

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Score the generated video's consecutive native frames."""
        del sample, reference

        backend = self._resolve_backend()
        details = self._details(prediction)
        try:
            if prediction.modality != "video":
                raise ValueError("MEt3R sequence metric requires a generated video with at least two frames")
            result = compute_met3r_video_consistency(Path(prediction.path), runtime=self.runtime)
            raw = result["raw"]
            details.update(result["details"])
            if details.get("pair_count") != details.get("frame_count", 0) - 1:
                raise ValueError("MEt3R did not score every native consecutive frame pair")
            if self.normalization.get("error_ref") is not None:
                normalized = apply_display_error_ref(raw, self.normalization)
            else:
                normalized = normalize_score(raw, _default_normalization("cosine"))
            return MetricOutput(
                raw=raw,
                normalized=normalized,
                backend=str(result["backend"]),
                details=details,
                eligibility_status="official",
                protocol_name=PROTOCOL,
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="official",
                error=str(exc),
                protocol_name=PROTOCOL,
            )


__all__ = ["Met3RConsistencyMetric"]
