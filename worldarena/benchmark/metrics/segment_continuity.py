"""Prediction-only segment continuity via TransNetV2 scene-cut detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from worldarena.benchmark.media import LoadedMedia, protocol_frame_details
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.benchmark.transnet_backend import (
    predict_transnet_single_frame_probabilities,
    predict_transnet_video_probabilities,
)

SEGMENT_CONTINUITY_BACKENDS = {"transnet_v2"}
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SCENE_LEN = 10


def _filter_cuts_by_min_scene_length(cuts: list[int], frame_count: int, min_scene_len: int) -> list[int]:
    """Remove cut candidates that would leave a scene shorter than ``min_scene_len``."""
    if not cuts or min_scene_len <= 1:
        return sorted(set(cuts))

    cuts = sorted(set(cuts))
    changed = True
    while changed:
        changed = False
        boundaries = [0, *cuts, frame_count]
        kept: list[int] = []
        for index, cut in enumerate(cuts):
            left_len = cut - boundaries[index]
            right_len = boundaries[index + 2] - cut
            if left_len >= min_scene_len and right_len >= min_scene_len:
                kept.append(cut)
            else:
                changed = True
        cuts = kept
    return cuts


def detect_scene_cuts(
    predictions: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
) -> dict[str, Any]:
    """Detect scene cuts from TransNetV2 single-frame probabilities."""
    probabilities = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if probabilities.size == 0:
        raise ValueError("empty TransNetV2 prediction sequence")

    binary = (probabilities >= float(threshold)).astype(np.uint8)
    candidate_cuts: list[int] = []
    previous = int(binary[0])
    for index in range(1, len(binary)):
        current = int(binary[index])
        if previous == 0 and current == 1 and index != 0:
            candidate_cuts.append(index)
        previous = current

    filtered_cuts = _filter_cuts_by_min_scene_length(
        candidate_cuts,
        len(probabilities),
        int(min_scene_len),
    )
    return {
        "candidate_cut_count": len(candidate_cuts),
        "cut_count": len(filtered_cuts),
        "cut_frame_indices": filtered_cuts,
        "candidate_cut_frame_indices": candidate_cuts,
        "threshold": float(threshold),
        "min_scene_len": int(min_scene_len),
    }


def compute_segment_continuity_score(
    frames: list[np.ndarray],
    *,
    runtime: dict[str, Any] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
    prediction_path: str | Path | None = None,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> dict[str, Any]:
    """Return a binary cut-free score for one prediction video."""
    runtime = dict(runtime or {})
    video_path = Path(prediction_path) if prediction_path is not None else None
    if video_path is not None and video_path.is_file():
        inference = predict_transnet_video_probabilities(
            video_path,
            runtime=runtime,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
        )
    else:
        inference = predict_transnet_single_frame_probabilities(frames, runtime=runtime)
    configured_min_scene_len = int(runtime.get("min_scene_len", min_scene_len))
    effective_min_scene_len = min(
        configured_min_scene_len,
        max(1, int(inference["frame_count"]) // 2),
    )
    cut_details = detect_scene_cuts(
        inference["probabilities"],
        threshold=float(runtime.get("threshold", threshold)),
        min_scene_len=effective_min_scene_len,
    )
    cut_details.update(
        {
            "configured_min_scene_len": configured_min_scene_len,
            "adaptive_min_scene_len": effective_min_scene_len != configured_min_scene_len,
        }
    )
    raw = 0.0 if cut_details["cut_count"] > 0 else 1.0
    return {
        "raw": raw,
        "backend": "transnet_v2",
        "details": {
            **cut_details,
            "frame_count": inference["frame_count"],
            "repo_root": inference["repo_root"],
            "weights_dir": inference["weights_dir"],
            "weights_file": inference.get("weights_file"),
            "inference_framework": inference.get("inference_framework", "unknown"),
            "inference_device": inference.get("inference_device", "unknown"),
            "torch_num_threads": inference.get("torch_num_threads"),
            "torch_num_interop_threads": inference.get("torch_num_interop_threads"),
            "analysis_frame_policy": inference.get("analysis_frame_policy", "loaded_sampled_frames"),
            "native_frame_count": inference.get("native_frame_count"),
            "native_window_start_index": inference.get("native_window_start_index"),
            "native_window_end_index": inference.get("native_window_end_index"),
            "cut_free": raw == 1.0,
        },
    }


class SegmentContinuityMetric(Metric):
    """Score whether a generated video contains detected scene cuts."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Store TransNetV2 runtime options for segment continuity."""
        super().__init__("segment_continuity", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the TransNetV2 segment-continuity backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="transnet_v2",
            supported_backends=SEGMENT_CONTINUITY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Return 1 when no scene cut is detected, else 0."""
        del sample
        del reference

        backend = self._resolve_backend()
        frames = list(prediction.frames or prediction.anchor_frames)
        details = protocol_frame_details(prediction)
        try:
            result = compute_segment_continuity_score(
                frames,
                runtime=self.runtime,
                prediction_path=prediction.path if prediction.modality == "video" else None,
                start_ratio=prediction.window_start_ratio,
                end_ratio=prediction.window_end_ratio,
            )
            details.update(result["details"])
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
            )
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


__all__ = [
    "SegmentContinuityMetric",
    "compute_segment_continuity_score",
    "detect_scene_cuts",
]
