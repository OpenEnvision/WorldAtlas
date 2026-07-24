"""3D reconstruction and geometry metric implementations."""

from __future__ import annotations

from typing import Any

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.reconstruction_consistency_backend import (
    DEFAULT_RECONSTRUCTION_BACKEND,
    compute_reconstruction_consistency,
    extract_wbench_frames,
)
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.common.video_io import read_video_frames


RECONSTRUCTION_CONSISTENCY_BACKENDS = {
    DEFAULT_RECONSTRUCTION_BACKEND,
    "da3_reprojection",
    "gaussian_lpips",  # legacy alias; resolves to da3_reprojection
}
RECONSTRUCTION_CONSISTENCY_METRICS = {
    "reconstruction_consistency",
    "geometric_consistency",
    "photometric_consistency",
}


class ReconstructionConsistencyEvaluator:
    """Evaluate the WBench reconstruction compute unit once per prediction."""

    def __init__(self, runtime: dict[str, Any] | None = None) -> None:
        self.runtime = dict(runtime or {})
        self._cache_key: tuple[str, str] | None = None
        self._cache_value: tuple[dict[str, Any], dict[str, Any]] | None = None

    def _frames(self, prediction: LoadedMedia) -> tuple[list[Any], dict[str, Any]]:
        """Decode frames using WBench's 3 FPS rule or an explicit legacy mode."""
        frame_source = str(self.runtime.get("frame_source") or "wbench_fps")
        if frame_source == "wbench_fps":
            frames, _, details = extract_wbench_frames(
                prediction.path,
                fps=float(self.runtime.get("fps", 3.0)),
                decode_threads=(
                    int(self.runtime["decode_threads"])
                    if self.runtime.get("decode_threads") is not None
                    else None
                ),
                decode_mode=str(self.runtime.get("decode_mode") or "random_seek"),
            )
            details["sampled_frame_indices"] = list(prediction.sampled_frame_indices)
            return frames, details
        if frame_source == "full_video":
            frames = read_video_frames(prediction.path)
            return frames, {
                "frame_source": frame_source,
                "frame_indices": list(range(len(frames))),
                "sampled_frame_indices": list(prediction.sampled_frame_indices),
            }
        if frame_source != "sampled":
            raise ValueError(
                "reconstruction consistency frame_source must be 'wbench_fps', "
                "'sampled', or 'full_video'"
            )
        return list(prediction.frames), {
            "frame_source": frame_source,
            "frame_indices": list(prediction.sampled_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
        }

    def evaluate(
        self,
        *,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the shared geometric/photometric payload for one prediction."""
        prompt = sample.prompt_target or sample.prompt_current
        cache_key = (str(prediction.path), str(prompt or ""))
        if self._cache_key == cache_key and self._cache_value is not None:
            return self._cache_value

        frames, frame_details = self._frames(prediction)
        payload = compute_reconstruction_consistency(
            frames,
            prompt=prompt,
            runtime=self.runtime,
        )
        value = (payload, frame_details)
        self._cache_key = cache_key
        self._cache_value = value
        return value


class ReconstructionConsistencyMetric(Metric):
    """Expose either WBench reconstruction sub-metric from one shared compute unit."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
        *,
        metric_name: str = "reconstruction_consistency",
        evaluator: ReconstructionConsistencyEvaluator | None = None,
    ) -> None:
        """Accept runtime controls such as frame_source and reconstruction model options."""
        if metric_name not in RECONSTRUCTION_CONSISTENCY_METRICS:
            raise ValueError(f"unsupported reconstruction consistency metric: {metric_name}")
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        self.evaluator = evaluator or ReconstructionConsistencyEvaluator(self.runtime)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the default reconstruction-consistency backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend=DEFAULT_RECONSTRUCTION_BACKEND,
            supported_backends=RECONSTRUCTION_CONSISTENCY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Run reconstruction-consistency scoring on selected prediction frames."""
        del reference

        backend = self._resolve_backend()
        eligibility_status = (
            "official" if self.name in {"geometric_consistency", "photometric_consistency"}
            else "diagnostic"
        )
        details: dict[str, Any] = {
            "prediction_path": prediction.path,
            "native_frame_count": prediction.native_frame_count,
            "compute_unit": "reconstruction_consistency",
        }
        if prediction.modality != "video":
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status=eligibility_status,
                error=f"{self.name} requires a video prediction",
            )

        try:
            payload, frame_details = self.evaluator.evaluate(
                sample=sample,
                prediction=prediction,
            )
            details.update(frame_details)
            details.update(payload.get("details", {}))
            if self.name == "photometric_consistency":
                raw = details.get("photometric_consistency")
            elif self.name == "geometric_consistency":
                raw = details.get("geometric_consistency")
            else:
                raw = payload.get("raw")
            raw_float = float(raw) if isinstance(raw, (int, float)) else None
            return MetricOutput(
                raw=raw_float,
                normalized=normalize_score(raw_float, self.normalization),
                backend=str(payload.get("backend") or backend),
                details=details,
                eligibility_status=eligibility_status,
                error=payload.get("error"),
            )
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status=eligibility_status,
                error=str(exc),
            )


__all__ = [
    "RECONSTRUCTION_CONSISTENCY_BACKENDS",
    "RECONSTRUCTION_CONSISTENCY_METRICS",
    "ReconstructionConsistencyEvaluator",
    "ReconstructionConsistencyMetric",
]
