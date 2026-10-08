"""3D reconstruction and geometry metric implementations."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import hashlib

import numpy as np

from worldatlas.benchmark.media import LoadedMedia
from worldatlas.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldatlas.benchmark.reconstruction_consistency_backend import (
    DEFAULT_RECONSTRUCTION_BACKEND,
    compute_reconstruction_consistency,
    extract_wbench_frames,
)
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput
from worldatlas.common.video_io import read_video_frames


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
        self._cache_key: str | None = None
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
        from worldatlas.benchmark.integrity import digest_json, path_content
        cache_key = digest_json({
            'prediction': path_content(Path(prediction.path)), 'prompt': prompt, 'runtime': self.runtime,
            'frame_source': prediction.evaluation_protocol,
            'indices': prediction.sampled_frame_indices, 'protocol_indices': prediction.protocol_frame_indices,
            'frames': [(frame.shape, str(frame.dtype), hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest())
                       for frame in [*prediction.frames, *prediction.anchor_frames]],
        })
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
                valid_frames = details.get("photo_valid_frames")
                side_error = details.get("photo_error")
            elif self.name == "geometric_consistency":
                raw = details.get("geometric_consistency")
                valid_frames = details.get("geo_valid_frames")
                side_error = details.get("geo_error")
            else:
                raw = payload.get("raw")
                valid_frames = details.get("num_frames")
                side_error = None
            # Shared DA3 compute can succeed on one side while the other has
            # zero valid frames. That used to become official 0.0 and enter
            # the mean. Missing valid_frames (legacy payloads) is left alone.
            empty_side = valid_frames is not None and int(valid_frames) <= 0
            if self.name in {"geometric_consistency", "photometric_consistency"} and (
                empty_side or side_error
            ):
                return MetricOutput(
                    raw=None,
                    normalized=None,
                    backend=str(payload.get("backend") or backend),
                    details=details,
                    eligibility_status=eligibility_status,
                    error=str(
                        side_error
                        or payload.get("error")
                        or f"{self.name} has no valid frames"
                    ),
                )
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
