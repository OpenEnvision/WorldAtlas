from __future__ import annotations

from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.reconstruction_consistency_backend import (
    DEFAULT_RECONSTRUCTION_BACKEND,
    compute_reconstruction_consistency,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.common.video_io import read_video_frames


RECONSTRUCTION_CONSISTENCY_BACKENDS = {DEFAULT_RECONSTRUCTION_BACKEND}


class ReconstructionConsistencyMetric(Metric):
    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("reconstruction_consistency", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend=DEFAULT_RECONSTRUCTION_BACKEND,
            supported_backends=RECONSTRUCTION_CONSISTENCY_BACKENDS,
        )

    def _frames(self, prediction: LoadedMedia) -> tuple[list[Any], dict[str, Any]]:
        frame_source = str(self.runtime.get("frame_source") or "sampled")
        if frame_source == "full_video":
            frames = read_video_frames(prediction.path)
            return frames, {
                "frame_source": frame_source,
                "frame_indices": list(range(len(frames))),
                "sampled_frame_indices": list(prediction.sampled_frame_indices),
            }
        if frame_source != "sampled":
            raise ValueError(
                "reconstruction_consistency frame_source must be 'sampled' or 'full_video'"
            )
        return list(prediction.frames), {
            "frame_source": frame_source,
            "frame_indices": list(prediction.sampled_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del reference

        backend = self._resolve_backend()
        details: dict[str, Any] = {
            "prediction_path": prediction.path,
            "native_frame_count": prediction.native_frame_count,
        }
        if prediction.modality != "video":
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error="reconstruction_consistency requires a video prediction",
            )

        try:
            frames, frame_details = self._frames(prediction)
            details.update(frame_details)
            prompt = sample.prompt_target or sample.prompt_current
            payload = compute_reconstruction_consistency(
                frames,
                prompt=prompt,
                runtime=self.runtime,
            )
            details.update(payload.get("details", {}))
            raw = payload.get("raw")
            raw_float = float(raw) if isinstance(raw, (int, float)) else None
            return MetricOutput(
                raw=raw_float,
                normalized=normalize_score(raw_float, self.normalization),
                backend=str(payload.get("backend") or backend),
                details=details,
                eligibility_status="diagnostic",
                error=payload.get("error"),
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


__all__ = [
    "RECONSTRUCTION_CONSISTENCY_BACKENDS",
    "ReconstructionConsistencyMetric",
]
