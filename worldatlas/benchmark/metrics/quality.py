"""Image and video quality metric implementations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from worldatlas.benchmark.media import LoadedMedia, protocol_frame_details
from worldatlas.benchmark.metrics.base import Metric, normalize_score, resolve_metric_backend
from worldatlas.benchmark.quality_backends import (
    HPSV3_DEFAULT_P1,
    HPSV3_DEFAULT_P99,
    compute_hpsv3_norm_scores,
    compute_musiq_scores,
    compute_quality_component_scores,
)
from worldatlas.benchmark.reference_runtime import compute_reference_perceptual_quality
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput
from worldatlas.common.progress import log_exception
from worldatlas.common.video_io import probe_video_frame_count, read_video_frames, read_video_frames_at


QUALITY_BACKENDS = {"reference", "pyiqa"}
IMAGE_QUALITY_BACKENDS = {"musiq"}
HPSV3_NORM_BACKENDS = {"hpsv3"}
DISABLED_QUALITY_BACKENDS = {"heuristic"}


def _even_sample_indices(total: int, count: int) -> list[int]:
    """Return evenly spaced frame indices in ``[0, total)``."""
    if total <= 0:
        return []
    if count <= 0 or total <= count:
        return list(range(total))
    raw = [int(index) for index in np.linspace(0, total - 1, num=count, dtype=int)]
    unique: list[int] = []
    seen: set[int] = set()
    for index in raw:
        if index not in seen:
            seen.add(index)
            unique.append(index)
    return unique


def _even_sample_frames(frames: list[np.ndarray], count: int) -> list[np.ndarray]:
    """Uniformly subsample frames when HPSv3 evaluation budgets are capped."""
    if count <= 0 or len(frames) <= count:
        return frames
    return [frames[index] for index in _even_sample_indices(len(frames), count)]


def resolve_hpsv3_batch_size(runtime: dict[str, Any] | None = None) -> int:
    """Prefer WORLDATLAS_HPSV3_BATCH_SIZE, then metric runtime, then 8."""
    raw = os.environ.get("WORLDATLAS_HPSV3_BATCH_SIZE")
    if raw not in (None, ""):
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    configured = (runtime or {}).get("batch_size", 8)
    try:
        return max(1, int(configured))
    except (TypeError, ValueError):
        return 8


def _prediction_frames_for_hpsv3(
    prediction: LoadedMedia,
    *,
    sample_count: int,
    sample_from_path: bool,
) -> tuple[list[np.ndarray], str]:
    """Load evenly sampled frames from disk or in-memory media for HPSv3 scoring."""
    if sample_from_path and prediction.modality == "video":
        path = Path(prediction.path)
        try:
            total = probe_video_frame_count(path)
        except Exception:
            total = 0
        indices = _even_sample_indices(total, sample_count)
        if indices:
            frames = read_video_frames_at(path, indices)
            if len(frames) != len(indices):
                raise ValueError("HPSv3 decoder returned incomplete selected frames")
            return frames, "video_path_even"
        frames = read_video_frames(path)
        return _even_sample_frames(frames, sample_count), "video_path_even"
    return _even_sample_frames(prediction.frames or prediction.anchor_frames, sample_count), "loaded_media_even"


class PerceptualQualityMetric(Metric):
    """Score no-reference perceptual quality via reference runtime or CLIP-IQA."""

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        """Validate perceptual-quality backend selection at construction."""
        super().__init__("perceptual_quality", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Allow reference or pyiqa backends while blocking deprecated heuristics."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="reference",
            supported_backends=QUALITY_BACKENDS,
            disabled_backends=DISABLED_QUALITY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Average perceptual quality over prediction anchor frames."""
        backend = self._resolve_backend()
        details = protocol_frame_details(prediction)
        try:
            if backend == "reference":
                reference_result = compute_reference_perceptual_quality(prediction.anchor_frames)
                raw = reference_result["raw"]
                details.update(reference_result["details"])
                return MetricOutput(
                    raw=raw,
                    normalized=normalize_score(raw, self.normalization),
                    backend=str(reference_result["backend"]),
                    details=details,
                )
            if backend == "pyiqa":
                quality_result = compute_quality_component_scores(
                    prediction.anchor_frames,
                    component_name="clip_iqa",
                )
                raw = quality_result["raw"]
                details.update(
                    {
                        "frame_scores": list(quality_result["frame_scores"]),
                        "native_frame_scores": list(quality_result["native_frame_scores"]),
                        "native_score_range": list(quality_result["native_score_range"]),
                    }
                )
                return MetricOutput(
                    raw=raw,
                    normalized=normalize_score(raw, self.normalization),
                    backend=str(quality_result["backend"]),
                    details=details,
                )
            raise RuntimeError(f"unsupported quality backend resolved at runtime: {backend}")
        except Exception as exc:  # pragma: no cover - backend-specific runtime failures
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                error=str(exc),
            )


class SubjectiveQualityMetric(PerceptualQualityMetric):
    """Alias exposing perceptual quality under the subjective_quality metric name."""

    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        """Reuse perceptual-quality wiring under the subjective_quality name."""
        super().__init__(backend=backend, normalization=normalization)
        self.name = "subjective_quality"


class ImageQualityMetric(Metric):
    """Official imaging-quality score using MUSIQ on prediction frames."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Store MUSIQ model path and preprocessing overrides."""
        super().__init__("image_quality", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the MUSIQ image-quality backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="musiq",
            supported_backends=IMAGE_QUALITY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Average MUSIQ scores across prediction frames."""
        del sample
        del reference
        backend = self._resolve_backend()
        details = protocol_frame_details(prediction)
        try:
            quality_result = compute_musiq_scores(
                prediction.frames or prediction.anchor_frames,
                model_path=self.runtime.get("model_path"),
                preprocess_mode=str(self.runtime.get("preprocess_mode", "longer")),
            )
            raw = quality_result["raw"]
            details.update(
                {
                    "frame_scores": list(quality_result["frame_scores"]),
                    "native_frame_scores": list(quality_result["native_frame_scores"]),
                    "native_score_range": list(quality_result["native_score_range"]),
                    "preprocess_mode": quality_result["preprocess_mode"],
                    "weights_path": quality_result["weights_path"],
                }
            )
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=str(quality_result["backend"]),
                details=details,
                eligibility_status="official",
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="official",
                error=str(exc),
            )


class HPSv3NormMetric(Metric):
    """Official human-preference quality using percentile-normalized HPSv3 rewards."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Store HPSv3 sampling, checkpoint, and prompt-mode runtime options."""
        super().__init__("hpsv3_norm", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the HPSv3 normalized reward backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="hpsv3",
            supported_backends=HPSV3_NORM_BACKENDS,
        )

    def _prompt_for_sample(self, sample: BenchmarkSample) -> str:
        """Select optional conditioning text for HPSv3 based on runtime prompt_mode."""
        prompt_mode = str(self.runtime.get("prompt_mode", "empty")).strip().lower()
        if prompt_mode == "target":
            return sample.prompt_target or ""
        if prompt_mode == "current":
            return sample.prompt_current or ""
        if prompt_mode != "empty":
            raise ValueError(f"Unknown HPSv3 prompt_mode: {prompt_mode}")
        return ""

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Evaluate evenly sampled frames with HPSv3 and map rewards to [0, 1]."""
        del reference
        backend = self._resolve_backend()
        sample_count = int(self.runtime.get("sample_count", 20))
        batch_size = resolve_hpsv3_batch_size(self.runtime)
        sample_from_path = bool(self.runtime.get("sample_from_path", True))
        details = {
            **protocol_frame_details(prediction),
            "hpsv3_sample_count": sample_count,
            "hpsv3_batch_size": batch_size,
        }
        try:
            frames, sampling_policy = _prediction_frames_for_hpsv3(
                prediction,
                sample_count=sample_count,
                sample_from_path=sample_from_path,
            )
            result = compute_hpsv3_norm_scores(
                frames,
                prompt=self._prompt_for_sample(sample),
                p1=float(self.runtime.get("p1", HPSV3_DEFAULT_P1)),
                p99=float(self.runtime.get("p99", HPSV3_DEFAULT_P99)),
                batch_size=batch_size,
                root_path=self.runtime.get("root_path", self.runtime.get("hpsv3_root")),
                checkpoint=self.runtime.get("checkpoint", self.runtime.get("checkpoint_path")),
                config_path=self.runtime.get("config_path"),
                qwen_model_path=self.runtime.get("qwen_model_path"),
                device=self.runtime.get("device"),
            )
            raw = result["raw"]
            details.update(
                {
                    "hpsv3_sampling_policy": sampling_policy,
                    "implementation_revision": "complete_hpsv3_rewards_v2",
                    "hpsv3_evaluated_frame_count": len(frames),
                    "hpsv3_raw_reward_mean": result["raw_reward_mean"],
                    "hpsv3_raw_rewards": list(result["raw_rewards"]),
                    "hpsv3_normalization_percentiles": list(result["normalization_percentiles"]),
                    "score_0_100": result["score_0_100"],
                    "checkpoint_path": result["checkpoint_path"],
                    "config_path": result["config_path"],
                    "device": result["device"],
                }
            )
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization),
                backend=str(result["backend"]),
                details=details,
                eligibility_status="official",
            )
        except ModuleNotFoundError as exc:
            log_exception(
                "metric",
                exc,
                sample_id=sample.sample_id,
                metric=self.name,
                status="skip",
            )
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="not_applicable",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            log_exception(
                "metric",
                exc,
                sample_id=sample.sample_id,
                metric=self.name,
                status="fail",
            )
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="official",
                error=f"{type(exc).__name__}: {exc}",
            )
