"""PMF (perceptual motion fidelity) metric implementations."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


PMF_BACKENDS = {"numpy_fft"}
_PMF_EPSILON = 1e-12


def _uniform_indices(length: int, count: int) -> list[int]:
    """Pick evenly spaced frame indices when resampling to a shorter clip."""
    if length <= 0 or count <= 0:
        return []
    if length == 1 or count == 1:
        return [0]
    return [int(round(value)) for value in np.linspace(0, length - 1, num=count)]


def _resample_frames(frames: list[np.ndarray], count: int) -> list[np.ndarray]:
    """Downsample or upsample a frame list to a target length via uniform indexing."""
    if len(frames) == count:
        return frames
    return [frames[index] for index in _uniform_indices(len(frames), count)]


def _resize_frames(frames: list[np.ndarray], *, width: int, height: int) -> tuple[list[np.ndarray], bool]:
    """Resize frames to a common resolution, reporting whether any resize occurred."""
    resized = False
    output: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[1] == width and frame.shape[0] == height:
            output.append(frame)
            continue
        output.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC))
        resized = True
    return output, resized


def _normalized_spectral_energy(video: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Normalize spatiotemporal FFT power so total energy sums to one."""
    # `video` is RGB with shape (T, H, W, C); PMF applies a 3D DFT over H, W, T per channel.
    channel_first = np.transpose(video, (3, 1, 2, 0))
    spectrum = np.fft.fftn(channel_first, axes=(1, 2, 3))
    power = np.sum(np.abs(spectrum) ** 2, axis=0, dtype=np.float64)
    total_energy = float(np.sum(power, dtype=np.float64))
    if total_energy <= epsilon:
        degenerate = np.zeros_like(power, dtype=np.float64)
        degenerate[0, 0, 0] = 1.0
        return degenerate
    return power / total_energy


def compute_physical_motion_fidelity(
    prediction_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    *,
    epsilon: float = _PMF_EPSILON,
) -> dict[str, Any]:
    """Compare normalized 3D spectral energy between aligned prediction and reference clips."""
    if len(prediction_frames) < 2 or len(reference_frames) < 2:
        raise ValueError("pmf requires at least two frames in both prediction and reference videos")

    target_frame_count = min(len(prediction_frames), len(reference_frames))
    prediction_clip = _resample_frames(prediction_frames, target_frame_count)
    reference_clip = _resample_frames(reference_frames, target_frame_count)

    reference_height, reference_width = reference_clip[0].shape[:2]
    reference_clip, reference_resized = _resize_frames(
        reference_clip,
        width=reference_width,
        height=reference_height,
    )
    prediction_clip, prediction_resized = _resize_frames(
        prediction_clip,
        width=reference_width,
        height=reference_height,
    )

    prediction_video = np.stack(prediction_clip, axis=0).astype(np.float64, copy=False)
    reference_video = np.stack(reference_clip, axis=0).astype(np.float64, copy=False)

    prediction_energy = _normalized_spectral_energy(prediction_video, epsilon=epsilon)
    reference_energy = _normalized_spectral_energy(reference_video, epsilon=epsilon)
    total_variation = 0.5 * float(np.sum(np.abs(prediction_energy - reference_energy), dtype=np.float64))
    pmf = float(-np.log(max(total_variation, epsilon)))
    return {
        "raw": pmf,
        "details": {
            "tv_distance": total_variation,
            "epsilon_floor": float(epsilon),
            "aligned_frame_count": target_frame_count,
            "reference_resolution": [reference_width, reference_height],
            "prediction_frame_count": len(prediction_frames),
            "reference_frame_count": len(reference_frames),
            "temporal_alignment": "uniform_resample_to_min_length",
            "prediction_resized": prediction_resized,
            "reference_resized": reference_resized,
        },
    }


class PhysicalMotionFidelityMetric(Metric):
    """Score how closely prediction motion spectra match a reference video."""

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        """Register PMF with the numpy-FFT spectral backend."""
        super().__init__("pmf", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the numpy FFT PMF implementation."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="numpy_fft",
            supported_backends=PMF_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Compute PMF between paired prediction and reference video frames."""
        del sample

        backend = self._resolve_backend()
        details = {
            "frame_indices": list(prediction.protocol_frame_indices),
            "sampled_frame_indices": list(prediction.sampled_frame_indices),
            "reference_sampled_frame_indices": list(reference.sampled_frame_indices),
        }
        if prediction.modality != "video" or reference.modality != "video":
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="not_applicable",
                error="pmf requires video prediction and video reference",
            )
        try:
            payload = compute_physical_motion_fidelity(prediction.frames, reference.frames)
            raw = float(payload["raw"])
            details.update(payload["details"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=backend,
                details=details,
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


__all__ = ["PhysicalMotionFidelityMetric", "compute_physical_motion_fidelity"]
