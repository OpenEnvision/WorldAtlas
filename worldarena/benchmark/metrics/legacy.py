from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image

from worldarena.benchmark.annotations import annotation_text, flatten_instruction_labels
from worldarena.benchmark.clip_backend import clip_image_features, clip_text_features
from worldarena.benchmark.consistency_backends import (
    dinov2_image_features as _shared_dinov2_image_features,
    transition_scores as _shared_transition_scores,
)
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, clamp01, cosine_similarity, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput
from worldarena.common.checkpoints import hf_local_dir, resolve_checkpoint_path


SEMANTIC_ALIGNMENT_BACKENDS = {"clip"}
SUBJECT_CONSISTENCY_BACKENDS = {"dinov2"}
BACKGROUND_CONSISTENCY_BACKENDS = {"clip"}
DEPTH_ACCURACY_BACKENDS = {"depth_anything"}
BASIC_FRAME_METRIC_BACKENDS = {"framewise"}


def _protocol_details(prediction: LoadedMedia) -> dict[str, Any]:
    return {
        "frame_indices": list(prediction.protocol_frame_indices),
        "sampled_frame_indices": list(prediction.sampled_frame_indices),
    }


def _normalize_or_identity(value: float | None, spec: dict[str, Any]) -> float | None:
    if value is None:
        return None
    if not spec:
        return float(value)
    return normalize_score(float(value), spec)


def _paired_frames(
    prediction: LoadedMedia,
    reference: LoadedMedia,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(zip(prediction.frames, reference.frames))


def _resize_like(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == target.shape[:2]:
        return frame
    target_size = (target.shape[1], target.shape[0])
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)


def _dinov2_image_features(frames: list[np.ndarray]) -> np.ndarray:
    return _shared_dinov2_image_features(frames)


@lru_cache(maxsize=1)
def _load_depth_backend(
    model_name: str,
) -> tuple[Any, Any, str, str] | None:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except (ImportError, ModuleNotFoundError):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if device == "cuda" else "float32"
    torch_dtype = getattr(torch, dtype)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    return model, processor, device, dtype


def _predict_depth_maps(frames: list[np.ndarray], *, model_name: str) -> np.ndarray:
    backend = _load_depth_backend(model_name)
    if backend is None:
        raise ModuleNotFoundError("transformers/torch not available")

    import torch

    model, processor, device, dtype_name = backend
    inputs = processor(images=[Image.fromarray(frame) for frame in frames], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    target_dtype = getattr(torch, dtype_name)
    for key, value in list(inputs.items()):
        if torch.is_floating_point(value):
            inputs[key] = value.to(target_dtype)

    with torch.inference_mode():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.unsqueeze(1)
        resized = torch.nn.functional.interpolate(
            depth,
            size=frames[0].shape[:2],
            mode="bicubic",
            align_corners=False,
        )
    return resized.squeeze(1).detach().cpu().float().numpy()


def _default_depth_model_name() -> str:
    """Return the canonical local depth-anything checkpoint directory."""
    return str(hf_local_dir("depth-anything/Depth-Anything-V2-Small-hf", required=True))


def _resolved_depth_model_name(value: object | None) -> str:
    """Resolve a configured model reference into a local checkpoint directory."""
    if value is None:
        return _default_depth_model_name()
    token = str(value).strip()
    if not token:
        return _default_depth_model_name()
    if token.startswith(("ckpt/", "./", "../", "/", "~")):
        resolved = resolve_checkpoint_path(token, kind="dir", required=True)
        if resolved is None:
            raise FileNotFoundError(f"depth model path is missing: {token}")
        return str(resolved)
    if "/" in token:
        return str(hf_local_dir(token, required=True))
    return token


def _transition_scores(features: np.ndarray) -> list[float]:
    return _shared_transition_scores(features)


def _reference_semantic_text(sample: BenchmarkSample) -> tuple[str, str]:
    caption_text = annotation_text(sample.annotation_path).strip()
    instruction_labels = flatten_instruction_labels(sample.annotation_path)
    label_text = ". ".join(instruction_labels).strip()
    if caption_text and label_text:
        return f"{caption_text} Actions: {label_text}.", "annotation_caption+instructions"
    if caption_text:
        return caption_text, "annotation_caption"
    if label_text:
        return label_text, "annotation_instructions"
    prompt_fallback = sample.prompt_current.strip() or sample.prompt_target.strip()
    return prompt_fallback, "prompt_fallback"


def _psnr(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    mse = float(np.mean((left - right) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def _ssim_channel(left: np.ndarray, right: np.ndarray) -> float:
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    left = left.astype(np.float32)
    right = right.astype(np.float32)
    mu_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right, (11, 11), 1.5)

    mu_left_sq = mu_left * mu_left
    mu_right_sq = mu_right * mu_right
    mu_cross = mu_left * mu_right

    sigma_left_sq = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu_left_sq
    sigma_right_sq = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu_right_sq
    sigma_cross = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu_cross

    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (sigma_left_sq + sigma_right_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(ssim_map))


def _ssim_rgb(left: np.ndarray, right: np.ndarray) -> float:
    scores = [
        _ssim_channel(left[:, :, channel], right[:, :, channel])
        for channel in range(left.shape[2])
    ]
    return clamp01(float(np.mean(scores)))


def _depth_absrel(prediction_depth: np.ndarray, reference_depth: np.ndarray) -> float:
    scale = float(np.median(reference_depth) / max(np.median(prediction_depth), 1e-6))
    aligned_prediction = prediction_depth * scale
    valid = reference_depth > 1e-3
    if not np.any(valid):
        valid = np.ones_like(reference_depth, dtype=bool)
    error = np.abs(aligned_prediction - reference_depth) / np.maximum(reference_depth, 1e-6)
    return float(np.mean(error[valid]))


class SemanticAlignmentMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("semantic_alignment", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="clip",
            supported_backends=SEMANTIC_ALIGNMENT_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del reference

        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        try:
            reference_text, text_source = _reference_semantic_text(sample)
            if not reference_text:
                raise ValueError("reference semantic text is unavailable for this sample")
            features, loader_name = clip_image_features(prediction.frames or prediction.anchor_frames)
            text_features, _ = clip_text_features([reference_text])
            text_feature = text_features[0]
            frame_scores = [
                clamp01((cosine_similarity(feature, text_feature) + 1.0) / 2.0)
                for feature in features
            ]
            raw = float(np.mean(frame_scores)) if frame_scores else None
            details.update(
                {
                    "clip_loader": loader_name,
                    "reference_text_source": text_source,
                    "reference_text": reference_text,
                    "frame_scores": [round(value, 4) for value in frame_scores],
                }
            )
            return MetricOutput(
                raw=raw,
                normalized=_normalize_or_identity(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
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


class SubjectConsistencyMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("subject_consistency", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="dinov2",
            supported_backends=SUBJECT_CONSISTENCY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample
        del reference

        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        if len(prediction.frames) < 2:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error="not enough frames for subject consistency",
            )
        try:
            features = _dinov2_image_features(prediction.frames)
            transition_scores = _transition_scores(features)
            raw = float(np.mean(transition_scores)) if transition_scores else None
            details["transition_scores"] = [round(value, 4) for value in transition_scores]
            return MetricOutput(
                raw=raw,
                normalized=_normalize_or_identity(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
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


class BackgroundConsistencyMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("background_consistency", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="clip",
            supported_backends=BACKGROUND_CONSISTENCY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample
        del reference

        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        if len(prediction.frames) < 2:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error="not enough frames for background consistency",
            )
        try:
            features, loader_name = clip_image_features(prediction.frames)
            transition_scores = _transition_scores(features)
            raw = float(np.mean(transition_scores)) if transition_scores else None
            details["transition_scores"] = [round(value, 4) for value in transition_scores]
            details["clip_loader"] = loader_name
            return MetricOutput(
                raw=raw,
                normalized=_normalize_or_identity(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
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


class DepthAccuracyMetric(Metric):
    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("depth_accuracy", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="depth_anything",
            supported_backends=DEPTH_ACCURACY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample

        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        pairs = _paired_frames(prediction, reference)
        if not pairs:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error="no aligned frames available for depth accuracy",
            )
        try:
            model_name = _resolved_depth_model_name(self.runtime.get("model_name"))
            prediction_frames = [_resize_like(left, right) for left, right in pairs]
            reference_frames = [right for _, right in pairs]
            prediction_depth = _predict_depth_maps(prediction_frames, model_name=model_name)
            reference_depth = _predict_depth_maps(reference_frames, model_name=model_name)
            frame_errors = [
                _depth_absrel(prediction_depth[index], reference_depth[index])
                for index in range(prediction_depth.shape[0])
            ]
            raw = float(np.mean(frame_errors)) if frame_errors else None
            details.update(
                {
                    "model_name": model_name,
                    "frame_absrel": [round(value, 4) for value in frame_errors],
                }
            )
            return MetricOutput(
                raw=raw,
                normalized=_normalize_or_identity(raw, self.normalization),
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
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


class PsnrMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("psnr", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="framewise",
            supported_backends=BASIC_FRAME_METRIC_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample

        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        pairs = _paired_frames(prediction, reference)
        if not pairs:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error="no aligned frames available for PSNR",
            )
        frame_scores = [_psnr(_resize_like(left, right), right) for left, right in pairs]
        raw = float(np.mean(frame_scores)) if frame_scores else None
        details["frame_scores"] = [round(value, 4) for value in frame_scores]
        return MetricOutput(
            raw=raw,
            normalized=_normalize_or_identity(raw, self.normalization),
            backend=backend,
            details=details,
            eligibility_status="diagnostic",
        )


class SsimMetric(Metric):
    def __init__(self, backend: str, normalization: dict[str, Any]) -> None:
        super().__init__("ssim", backend, normalization)
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="framewise",
            supported_backends=BASIC_FRAME_METRIC_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample

        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        pairs = _paired_frames(prediction, reference)
        if not pairs:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error="no aligned frames available for SSIM",
            )
        frame_scores = [_ssim_rgb(_resize_like(left, right), right) for left, right in pairs]
        raw = float(np.mean(frame_scores)) if frame_scores else None
        details["frame_scores"] = [round(value, 4) for value in frame_scores]
        return MetricOutput(
            raw=raw,
            normalized=_normalize_or_identity(raw, self.normalization),
            backend=backend,
            details=details,
            eligibility_status="diagnostic",
        )


__all__ = [
    "BackgroundConsistencyMetric",
    "DepthAccuracyMetric",
    "PsnrMetric",
    "SemanticAlignmentMetric",
    "SsimMetric",
    "SubjectConsistencyMetric",
]
