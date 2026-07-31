"""Metric implementations for object permanence and scene memory."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from worldarena.benchmark.consistency_backends import dinov2_image_features
from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, clamp01, cosine_similarity, normalize_score, resolve_metric_backend
from worldarena.benchmark.metrics.legacy import _resize_like, _ssim_rgb
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


MEMORY_REVISIT_BACKENDS = {"dinov2"}
MEMORY_SYMMETRY_BACKENDS = {"mse_symmetry"}
REVISIT_RETURN_BACKENDS = {"pose_gate"}
ENTITY_REAPPEARANCE_BACKENDS = {"masked_dinov2"}


def _protocol_details(prediction: LoadedMedia) -> dict[str, Any]:
    """Collect protocol and sampled frame indices for memory metrics."""
    return {
        "frame_indices": list(prediction.protocol_frame_indices),
        "sampled_frame_indices": list(prediction.sampled_frame_indices),
    }


def _feature_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Map DINO cosine similarity to a [0, 1] feature match score."""
    return clamp01((cosine_similarity(left, right) + 1.0) / 2.0)


def _mse_similarity(left: np.ndarray, right: np.ndarray, *, tau: float) -> tuple[float, float]:
    """Return raw MSE and a tau-scaled similarity for revisit closure scoring."""
    left_float = left.astype(np.float32) / 255.0
    right_float = right.astype(np.float32) / 255.0
    mse = float(np.mean((left_float - right_float) ** 2))
    return mse, 1.0 - clamp01(mse / max(float(tau), 1e-6))


def _rotation_error_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Return the geodesic SO(3) angle between two rotation matrices."""
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def compute_revisit_return_gate(
    poses_c2w: np.ndarray,
    *,
    tail_fraction: float = 0.15,
    min_departure_translation: float = 0.10,
    min_departure_rotation_degrees: float = 45.0,
    max_return_translation_ratio: float = 0.12,
    max_return_rotation_degrees: float = 8.0,
    max_return_translation: float | None = None,
) -> dict[str, Any]:
    """Verify that a trajectory actually departs and returns to its anchor pose.

    Translation is normalized by the observed departure magnitude so monocular
    pose tracks with unknown metric scale remain usable. Pure rotational loops
    are accepted through the rotation departure branch.
    """
    poses = np.asarray(poses_c2w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3:
        raise ValueError(f"expected at least three [4,4] camera poses, got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("camera poses contain non-finite values")

    anchor = poses[0]
    translation = np.linalg.norm(poses[:, :3, 3] - anchor[:3, 3], axis=1)
    rotation = np.asarray(
        [_rotation_error_degrees(anchor[:3, :3], pose[:3, :3]) for pose in poses],
        dtype=np.float64,
    )
    max_translation = float(np.max(translation))
    max_rotation = float(np.max(rotation))
    translation_departed = max_translation >= float(min_departure_translation)
    rotation_departed = max_rotation >= float(min_departure_rotation_degrees)
    departure_passed = bool(translation_departed or rotation_departed)

    tail_count = max(1, int(math.ceil(len(poses) * clamp01(float(tail_fraction)))))
    tail_start = max(1, len(poses) - tail_count)
    translation_ratio = translation / max(max_translation, 1e-8)
    combined_error = (
        translation_ratio / max(float(max_return_translation_ratio), 1e-8)
        + rotation / max(float(max_return_rotation_degrees), 1e-8)
    )
    revisit_index = int(tail_start + np.argmin(combined_error[tail_start:]))
    revisit_translation = float(translation[revisit_index])
    revisit_translation_ratio = float(translation_ratio[revisit_index])
    revisit_rotation = float(rotation[revisit_index])

    translation_returned = revisit_translation_ratio <= float(max_return_translation_ratio)
    if max_return_translation is not None:
        translation_returned = translation_returned and revisit_translation <= float(
            max_return_translation
        )
    rotation_returned = revisit_rotation <= float(max_return_rotation_degrees)
    return_passed = bool(translation_returned and rotation_returned)
    passed = bool(departure_passed and return_passed)

    departure_score = max(
        clamp01(max_translation / max(float(min_departure_translation), 1e-8)),
        clamp01(max_rotation / max(float(min_departure_rotation_degrees), 1e-8)),
    )
    translation_return_score = 1.0 - clamp01(
        revisit_translation_ratio / max(float(max_return_translation_ratio), 1e-8)
    )
    rotation_return_score = 1.0 - clamp01(
        revisit_rotation / max(float(max_return_rotation_degrees), 1e-8)
    )
    return {
        "passed": passed,
        "gate_score": clamp01(
            departure_score * min(translation_return_score, rotation_return_score)
        ),
        "departure_passed": departure_passed,
        "return_passed": return_passed,
        "translation_departed": translation_departed,
        "rotation_departed": rotation_departed,
        "max_departure_translation": max_translation,
        "max_departure_rotation_degrees": max_rotation,
        "revisit_index": revisit_index,
        "revisit_translation": revisit_translation,
        "revisit_translation_ratio": revisit_translation_ratio,
        "revisit_rotation_degrees": revisit_rotation,
        "tail_start_index": tail_start,
    }


def _load_pose_array(path: Path) -> np.ndarray:
    """Load a [N,4,4] pose sidecar from a directory, NPY, or NPZ file."""
    if path.is_dir():
        for name in ("poses_c2w.npy", "poses.npy", "camera_poses.npy"):
            candidate = path / name
            if candidate.exists():
                return _load_pose_array(candidate)
        raise FileNotFoundError(f"pose directory has no supported pose file: {path}")
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        for key in ("poses_c2w", "poses", "data"):
            if key in payload:
                poses = payload[key]
                break
        else:
            raise KeyError(f"pose archive has no supported array: {path}")
    else:
        poses = np.load(path)
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected pose array [N,4,4], got {poses.shape}")
    return poses


def resolve_revisit_pose_sidecar(
    prediction_path: Path,
    runtime: dict[str, Any],
) -> tuple[np.ndarray, str]:
    """Resolve the estimated camera trajectory used by the return gate."""
    base = prediction_path.resolve().parent
    candidates: list[Path] = []
    explicit = runtime.get("pose_path") or runtime.get("prediction_pose_path")
    if explicit:
        candidate = Path(str(explicit)).expanduser()
        candidates.append(candidate if candidate.is_absolute() else base / candidate)
    stem = prediction_path.stem
    candidates.extend(
        [
            base / "annotations_vipe" / f"{stem}_vipe_ann",
            base / f"{stem}_vipe_ann",
            base / f"{stem}_poses_c2w.npy",
            base / f"{stem}_poses.npy",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return _load_pose_array(candidate), str(candidate)
    raise FileNotFoundError("revisit return gate requires a camera-pose sidecar")


def compute_entity_reappearance_consistency(
    anchor_descriptor: np.ndarray,
    revisit_descriptor: np.ndarray,
    *,
    anchor_area: float,
    revisit_area: float,
    return_gate_passed: bool = True,
    appearance_weight: float = 0.80,
) -> dict[str, float | bool]:
    """Score masked entity identity and scale without language supervision."""
    left = np.asarray(anchor_descriptor, dtype=np.float64).reshape(-1)
    right = np.asarray(revisit_descriptor, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("anchor and revisit descriptors must be non-empty and shape-matched")
    if anchor_area <= 0.0 or revisit_area <= 0.0:
        raise ValueError("anchor and revisit mask areas must be positive")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    appearance = 0.0 if denominator <= 1e-12 else clamp01(float(np.dot(left, right) / denominator))
    area_consistency = float(math.exp(-abs(math.log(float(revisit_area) / float(anchor_area)))))
    weight = clamp01(float(appearance_weight))
    conditional = weight * appearance + (1.0 - weight) * area_consistency
    return {
        "entity_reappearance_consistency": conditional if return_gate_passed else 0.0,
        "conditional_consistency": conditional,
        "appearance_consistency": appearance,
        "area_consistency": area_consistency,
        "return_gate_passed": bool(return_gate_passed),
    }


def _frame_mse(left: np.ndarray, right: np.ndarray) -> float:
    """Mean squared RGB error between two frames, resizing when shapes differ."""
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_CUBIC)
    left_float = left.astype(np.float32) / 255.0
    right_float = right.astype(np.float32) / 255.0
    return float(np.mean((left_float - right_float) ** 2))


def compute_memory_symmetry(
    frames: list[np.ndarray],
    *,
    mse_offset: float = 0.0,
    value_decay: float = 12.0,
    exp_power: float = 1.0,
    temporal_decay: float = 0.20,
) -> dict[str, Any]:
    """Score mirror symmetry by matching early frames to their late-frame counterparts."""
    if len(frames) < 2:
        raise ValueError("memory_symmetry requires at least two frames")

    pair_count = len(frames) // 2
    pair_mse = [_frame_mse(frames[index], frames[-index - 1]) for index in range(pair_count)]
    pair_similarities = [
        float(np.exp(-float(value_decay) * max(0.0, mse - float(mse_offset)) ** float(exp_power)))
        for mse in pair_mse
    ]
    distances = np.arange(pair_count, 0, -1, dtype=np.float32)
    weights = np.exp(-float(temporal_decay) * distances)
    total = float(np.sum(weights))
    weights = weights / total if total > 0.0 else np.full((pair_count,), 1.0 / pair_count)
    raw = float(np.sum(weights * np.asarray(pair_similarities, dtype=np.float32)))
    return {
        "raw": clamp01(raw),
        "details": {
            "pair_mse": [round(value, 8) for value in pair_mse],
            "pair_similarities": [round(value, 6) for value in pair_similarities],
            "pair_weights": [round(float(value), 6) for value in weights],
            "mse_offset": float(mse_offset),
            "value_decay": float(value_decay),
            "exp_power": float(exp_power),
            "temporal_decay": float(temporal_decay),
        },
    }


class MemoryRevisitConsistencyMetric(Metric):
    """Score whether a 360-degree revisit returns to the same generated scene."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Store closure weights and progress-gate hyperparameters for revisit scoring."""
        super().__init__("memory_revisit_consistency", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the DINO-based memory revisit backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="dinov2",
            supported_backends=MEMORY_REVISIT_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Blend DINO, SSIM, and MSE closure with a mid-trajectory progress gate."""
        del sample
        del reference

        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        frames = prediction.frames
        if len(frames) < 2:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error="not enough frames for memory revisit consistency",
            )

        try:
            start = frames[0]
            end = _resize_like(frames[-1], start)
            features = dinov2_image_features(frames)

            dino_revisit = _feature_similarity(features[0], features[-1])
            ssim_revisit = _ssim_rgb(start, end)
            mse, mse_revisit = _mse_similarity(
                start,
                end,
                tau=float(self.runtime.get("mse_tau", 0.08)),
            )

            closure_weights = dict(self.runtime.get("closure_weights", {}))
            dino_weight = float(closure_weights.get("dino", 0.55))
            ssim_weight = float(closure_weights.get("ssim", 0.25))
            mse_weight = float(closure_weights.get("mse", 0.20))
            weight_sum = max(dino_weight + ssim_weight + mse_weight, 1e-6)
            closure_score = (
                dino_weight * dino_revisit
                + ssim_weight * ssim_revisit
                + mse_weight * mse_revisit
            ) / weight_sum

            quarter_indices = sorted(
                {
                    int(round((len(frames) - 1) * fraction))
                    for fraction in (0.25, 0.5, 0.75)
                }
            )
            quarter_similarities = [
                _feature_similarity(features[0], features[index])
                for index in quarter_indices
                if 0 < index < len(frames) - 1
            ]
            mean_quarter_displacement = (
                float(np.mean([1.0 - value for value in quarter_similarities]))
                if quarter_similarities
                else 0.0
            )
            lower = float(self.runtime.get("progress_lower", 0.03))
            upper = float(self.runtime.get("progress_upper", 0.18))
            progress_score = clamp01((mean_quarter_displacement - lower) / max(upper - lower, 1e-6))
            progress_gate_weight = clamp01(float(self.runtime.get("progress_gate_weight", 0.35)))
            raw = closure_score * ((1.0 - progress_gate_weight) + progress_gate_weight * progress_score)

            return_gate: dict[str, Any] | None = None
            if bool(self.runtime.get("require_return_gate", False)):
                poses, pose_source = resolve_revisit_pose_sidecar(Path(prediction.path), self.runtime)
                return_gate = compute_revisit_return_gate(
                    poses,
                    tail_fraction=float(self.runtime.get("return_tail_fraction", 0.15)),
                    min_departure_translation=float(
                        self.runtime.get("min_departure_translation", 0.10)
                    ),
                    min_departure_rotation_degrees=float(
                        self.runtime.get("min_departure_rotation_degrees", 45.0)
                    ),
                    max_return_translation_ratio=float(
                        self.runtime.get("max_return_translation_ratio", 0.12)
                    ),
                    max_return_rotation_degrees=float(
                        self.runtime.get("max_return_rotation_degrees", 8.0)
                    ),
                    max_return_translation=self.runtime.get("max_return_translation"),
                )
                details["pose_source"] = pose_source
                details["conditional_visual_consistency"] = round(float(raw), 6)
                raw = float(raw) if return_gate["passed"] else 0.0

            details.update(
                {
                    "dino_revisit": round(dino_revisit, 4),
                    "ssim_revisit": round(ssim_revisit, 4),
                    "mse": round(mse, 6),
                    "mse_revisit": round(mse_revisit, 4),
                    "closure_score": round(float(closure_score), 4),
                    "quarter_indices": quarter_indices,
                    "quarter_dino_similarities": [round(value, 4) for value in quarter_similarities],
                    "mean_quarter_displacement": round(mean_quarter_displacement, 4),
                    "progress_score": round(progress_score, 4),
                    "progress_gate_weight": progress_gate_weight,
                    "return_gate": return_gate,
                    "metric_uses_vlm": False,
                    "metric_uses_llm_as_judge": False,
                }
            )
            return MetricOutput(
                raw=float(raw),
                normalized=normalize_score(float(raw), self.normalization) if self.normalization else float(raw),
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


class RevisitReturnGateMetric(Metric):
    """Expose the actual depart-and-return pose gate as a leaderboard metric."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("revisit_return_gate", backend, normalization)
        self.runtime = dict(runtime or {})
        resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="pose_gate",
            supported_backends=REVISIT_RETURN_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample, reference
        backend = "pose_gate" if self.backend == "auto" else self.backend
        details = _protocol_details(prediction)
        try:
            poses, pose_source = resolve_revisit_pose_sidecar(Path(prediction.path), self.runtime)
            gate = compute_revisit_return_gate(
                poses,
                tail_fraction=float(self.runtime.get("return_tail_fraction", 0.15)),
                min_departure_translation=float(
                    self.runtime.get("min_departure_translation", 0.10)
                ),
                min_departure_rotation_degrees=float(
                    self.runtime.get("min_departure_rotation_degrees", 45.0)
                ),
                max_return_translation_ratio=float(
                    self.runtime.get("max_return_translation_ratio", 0.12)
                ),
                max_return_rotation_degrees=float(
                    self.runtime.get("max_return_rotation_degrees", 8.0)
                ),
                max_return_translation=self.runtime.get("max_return_translation"),
            )
            raw = 1.0 if gate["passed"] else 0.0
            details.update(
                {
                    "pose_source": pose_source,
                    "return_gate": gate,
                    "metric_uses_vlm": False,
                    "metric_uses_llm_as_judge": False,
                }
            )
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization) if self.normalization else raw,
                backend=backend,
                details=details,
            )
        except FileNotFoundError as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="not_applicable",
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


def _load_entity_revisit_sidecar(
    prediction_path: Path,
    runtime: dict[str, Any],
) -> tuple[dict[str, np.ndarray], str]:
    """Load precomputed masked DINO descriptors and mask areas."""
    base = prediction_path.resolve().parent
    explicit = runtime.get("entity_revisit_sidecar_path")
    candidates: list[Path] = []
    if explicit:
        path = Path(str(explicit)).expanduser()
        candidates.append(path if path.is_absolute() else base / path)
    candidates.append(base / f"{prediction_path.stem}_entity_revisit.npz")
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = np.load(candidate)
        required = {
            "anchor_descriptor",
            "revisit_descriptor",
            "anchor_mask_area",
            "revisit_mask_area",
        }
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"entity revisit sidecar is missing: {sorted(missing)}")
        return {key: np.asarray(payload[key]) for key in required}, str(candidate)
    raise FileNotFoundError("entity reappearance metric requires an entity-revisit NPZ sidecar")


class EntityReappearanceConsistencyMetric(Metric):
    """Score a returning entity with masked DINO features and mask geometry."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("entity_reappearance_consistency", backend, normalization)
        self.runtime = dict(runtime or {})
        resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="masked_dinov2",
            supported_backends=ENTITY_REAPPEARANCE_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        del sample, reference
        backend = "masked_dinov2" if self.backend == "auto" else self.backend
        details = _protocol_details(prediction)
        try:
            payload, source = _load_entity_revisit_sidecar(
                Path(prediction.path), self.runtime
            )
            poses, pose_source = resolve_revisit_pose_sidecar(Path(prediction.path), self.runtime)
            gate = compute_revisit_return_gate(
                poses,
                tail_fraction=float(self.runtime.get("return_tail_fraction", 0.15)),
                min_departure_translation=float(
                    self.runtime.get("min_departure_translation", 0.10)
                ),
                min_departure_rotation_degrees=float(
                    self.runtime.get("min_departure_rotation_degrees", 45.0)
                ),
                max_return_translation_ratio=float(
                    self.runtime.get("max_return_translation_ratio", 0.12)
                ),
                max_return_rotation_degrees=float(
                    self.runtime.get("max_return_rotation_degrees", 8.0)
                ),
            )
            result = compute_entity_reappearance_consistency(
                payload["anchor_descriptor"],
                payload["revisit_descriptor"],
                anchor_area=float(payload["anchor_mask_area"].reshape(-1)[0]),
                revisit_area=float(payload["revisit_mask_area"].reshape(-1)[0]),
                return_gate_passed=bool(gate["passed"]),
                appearance_weight=float(self.runtime.get("appearance_weight", 0.80)),
            )
            raw = float(result["entity_reappearance_consistency"])
            details.update(
                {
                    "entity_sidecar_source": source,
                    "pose_source": pose_source,
                    "return_gate": gate,
                    **result,
                    "metric_uses_vlm": False,
                    "metric_uses_llm_as_judge": False,
                }
            )
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization) if self.normalization else raw,
                backend=backend,
                details=details,
            )
        except FileNotFoundError as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="not_applicable",
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


class MemorySymmetryMetric(Metric):
    """Score temporal mirror symmetry between paired early and late frames."""

    def __init__(
        self,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Store exponential decay knobs for symmetric frame-pair weighting."""
        super().__init__("memory_symmetry", backend, normalization)
        self.runtime = dict(runtime or {})
        self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Require the MSE-symmetry memory backend."""
        return resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="mse_symmetry",
            supported_backends=MEMORY_SYMMETRY_BACKENDS,
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Evaluate mirror-frame similarity across the prediction timeline."""
        del sample
        del reference
        backend = self._resolve_backend()
        details = _protocol_details(prediction)
        try:
            result = compute_memory_symmetry(
                prediction.frames,
                mse_offset=float(self.runtime.get("mse_offset", 0.0)),
                value_decay=float(self.runtime.get("value_decay", 12.0)),
                exp_power=float(self.runtime.get("exp_power", 1.0)),
                temporal_decay=float(self.runtime.get("temporal_decay", 0.20)),
            )
            details.update(result["details"])
            raw = float(result["raw"])
            return MetricOutput(
                raw=raw,
                normalized=normalize_score(raw, self.normalization) if self.normalization else raw,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
            )
        except Exception as exc:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details=details,
                eligibility_status="diagnostic",
                error=str(exc),
            )


__all__ = [
    "EntityReappearanceConsistencyMetric",
    "MemoryRevisitConsistencyMetric",
    "MemorySymmetryMetric",
    "RevisitReturnGateMetric",
    "compute_entity_reappearance_consistency",
    "compute_memory_symmetry",
    "compute_revisit_return_gate",
    "resolve_revisit_pose_sidecar",
]
