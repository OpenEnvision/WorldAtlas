"""JEDI (Joint Embedding Distance for Images) backend for video quality."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from worldarena.benchmark.metrics.base import normalize_score
from worldarena.common.checkpoints import checkpoint_env, filter_checkpoint_env_overrides


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_JEDI_REPO_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "thirdparty" / "JEDi",
)


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _as_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_jedi_repo_root(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("repo_root")
    if configured:
        repo_root = Path(str(configured)).expanduser()
        if not repo_root.exists():
            raise FileNotFoundError(f"JEDi repo_root does not exist: {repo_root}")
        if not (repo_root / "videojedi").is_dir():
            raise FileNotFoundError(f"JEDi repo_root does not contain a videojedi package: {repo_root}")
        return repo_root.resolve()

    for candidate in DEFAULT_JEDI_REPO_CANDIDATES:
        if (candidate / "videojedi").is_dir():
            return candidate.resolve()
    return None


def _load_jedi_metric_class(runtime: dict[str, Any]) -> tuple[Any, Path | None]:
    repo_root = _resolve_jedi_repo_root(runtime)
    if repo_root is not None:
        repo_path = str(repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    try:
        module = importlib.import_module("videojedi")
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise ModuleNotFoundError(
            "JEDi package is unavailable; install videojedi or set "
            "benchmark.metric_runtime.jedi.repo_root to the cloned repository. "
            f"Import error: {exc.__class__.__name__}: {exc}"
        ) from exc

    metric_class = getattr(module, "JEDiMetric", None)
    if metric_class is None:
        raise AttributeError("videojedi package does not expose JEDiMetric")
    return metric_class, repo_root


def _prepare_runtime_environment(runtime: dict[str, Any]) -> None:
    os.environ.update(checkpoint_env())
    os.environ.update(filter_checkpoint_env_overrides(dict(runtime.get("env", {}))))


def _resolve_runtime_path(value: Any, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resize_frame(frame: np.ndarray, analysis_size: int) -> np.ndarray:
    if frame.shape[0] == analysis_size and frame.shape[1] == analysis_size:
        return frame
    return cv2.resize(frame, (analysis_size, analysis_size), interpolation=cv2.INTER_AREA)


def _decode_video_frames(
    path: Path,
    *,
    analysis_size: int,
    max_source_frames: int | None,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    native_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices: set[int] | None = None
    if max_source_frames is not None and native_count > max_source_frames:
        target_indices = {
            int(round(value))
            for value in np.linspace(0, native_count - 1, num=max_source_frames)
        }

    frames: list[np.ndarray] = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if target_indices is not None and frame_index not in target_indices:
            frame_index += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(_resize_frame(rgb, analysis_size))
        frame_index += 1
        if target_indices is None and max_source_frames is not None and len(frames) >= max_source_frames:
            break
    capture.release()

    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {path}")
    return frames


def _clip_start_indices(
    frame_count: int,
    *,
    clip_frame_count: int,
    max_clips_per_video: int,
    clip_stride: int,
) -> list[int]:
    last_start = frame_count - clip_frame_count
    if last_start < 0:
        return []
    if max_clips_per_video <= 0:
        return list(range(0, last_start + 1, max(clip_stride, 1)))
    clip_count = min(max_clips_per_video, last_start + 1)
    return sorted(
        {
            int(round(value))
            for value in np.linspace(0, last_start, num=clip_count)
        }
    )


def _clips_for_video(path: Path, runtime: dict[str, Any]) -> list[np.ndarray]:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"JEDi requires video inputs, got: {path}")

    clip_frame_count = _as_positive_int(runtime.get("clip_frame_count"), 16)
    analysis_size = _as_positive_int(runtime.get("analysis_size"), 224)
    max_clips_per_video = int(runtime.get("max_clips_per_video", 1))
    clip_stride = _as_positive_int(runtime.get("clip_stride"), clip_frame_count)
    max_source_frames = _as_optional_positive_int(runtime.get("max_source_frames"))

    frames = _decode_video_frames(
        path,
        analysis_size=analysis_size,
        max_source_frames=max_source_frames,
    )
    starts = _clip_start_indices(
        len(frames),
        clip_frame_count=clip_frame_count,
        max_clips_per_video=max_clips_per_video,
        clip_stride=clip_stride,
    )
    if not starts:
        raise ValueError(
            f"video has {len(frames)} decoded frames, fewer than JEDi clip_frame_count={clip_frame_count}: {path}"
        )
    return [
        np.stack(frames[start : start + clip_frame_count], axis=0).astype(np.uint8, copy=False)
        for start in starts
    ]


def _build_clip_arrays(
    pairs: Sequence[tuple[Path, Path, str]],
    runtime: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    max_pairs = _as_optional_positive_int(runtime.get("max_pairs"))
    selected_pairs = list(pairs[:max_pairs]) if max_pairs is not None else list(pairs)
    prediction_clips: list[np.ndarray] = []
    reference_clips: list[np.ndarray] = []
    skipped: list[dict[str, str]] = []
    used_sample_ids: list[str] = []

    for prediction_path, reference_path, sample_id in selected_pairs:
        try:
            pred_clips = _clips_for_video(prediction_path, runtime)
            ref_clips = _clips_for_video(reference_path, runtime)
        except Exception as exc:
            if len(skipped) < 20:
                skipped.append(
                    {
                        "sample_id": sample_id,
                        "prediction_path": str(prediction_path),
                        "reference_path": str(reference_path),
                        "reason": str(exc),
                    }
                )
            continue
        prediction_clips.extend(pred_clips)
        reference_clips.extend(ref_clips)
        used_sample_ids.append(sample_id)

    min_clip_count = _as_positive_int(runtime.get("min_clip_count"), 2)
    if len(prediction_clips) < min_clip_count or len(reference_clips) < min_clip_count:
        raise ValueError(
            "JEDi needs at least "
            f"{min_clip_count} valid clips per distribution; got "
            f"{len(prediction_clips)} prediction clips and {len(reference_clips)} reference clips"
        )

    clip_frame_count = _as_positive_int(runtime.get("clip_frame_count"), 16)
    details = {
        "candidate_pair_count": len(selected_pairs),
        "used_pair_count": len(used_sample_ids),
        "used_sample_ids_preview": used_sample_ids[:20],
        "skipped_pair_count": len(selected_pairs) - len(used_sample_ids),
        "skipped_pairs_preview": skipped,
        "prediction_clip_count": len(prediction_clips),
        "reference_clip_count": len(reference_clips),
        "clip_frame_count": clip_frame_count,
        "analysis_size": _as_positive_int(runtime.get("analysis_size"), 224),
        "max_clips_per_video": int(runtime.get("max_clips_per_video", 1)),
        "clip_stride": _as_positive_int(runtime.get("clip_stride"), clip_frame_count),
        "max_source_frames": _as_optional_positive_int(runtime.get("max_source_frames")),
    }
    return np.stack(prediction_clips, axis=0), np.stack(reference_clips, axis=0), details


def _make_loader(clips: np.ndarray, batch_size: int) -> Any:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tensor = torch.from_numpy(clips).permute(0, 1, 4, 2, 3).float() / 255.0
    labels = torch.zeros((tensor.shape[0],), dtype=torch.long)
    return DataLoader(TensorDataset(tensor, labels), batch_size=batch_size, shuffle=False)


def compute_jedi_distance(
    *,
    prediction_reference_pairs: Sequence[tuple[Path, Path, str]],
    normalization: dict[str, Any],
    runtime: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_array, reference_array, details = _build_clip_arrays(
        prediction_reference_pairs,
        runtime,
    )

    prediction_npy = output_dir / "prediction_clips.npy"
    reference_npy = output_dir / "reference_clips.npy"
    np.save(prediction_npy, prediction_array)
    np.save(reference_npy, reference_array)

    _prepare_runtime_environment(runtime)
    metric_class, repo_root = _load_jedi_metric_class(runtime)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    batch_size = _as_positive_int(runtime.get("batch_size"), 4)
    num_samples = _as_positive_int(
        runtime.get("num_samples"),
        min(len(reference_array), len(prediction_array)),
    )
    model_base = repo_root or Path.cwd()
    model_dir = _resolve_runtime_path(runtime.get("model_dir"), model_base) or str(model_base)
    config_path = _resolve_runtime_path(runtime.get("config_path"), model_base)
    metric = metric_class(
        feature_path=str(feature_dir),
        model_dir=model_dir,
        config_path=config_path,
    )
    metric.load_features(
        train_loader=_make_loader(reference_array, batch_size),
        test_loader=_make_loader(prediction_array, batch_size),
        num_samples=num_samples,
    )
    raw = float(metric.compute_metric())
    if not np.isfinite(raw):
        raise ValueError(f"JEDi returned a non-finite score: {raw}")

    details.update(
        {
            "prediction_npy": str(prediction_npy),
            "reference_npy": str(reference_npy),
            "feature_dir": str(feature_dir),
            "repo_root": str(repo_root) if repo_root is not None else None,
            "model_dir": model_dir,
            "config_path": config_path,
            "batch_size": batch_size,
            "num_samples": num_samples,
            "torch_home": os.environ.get("TORCH_HOME"),
            "raw_direction": "lower_is_better",
        }
    )
    return {
        "raw": raw,
        "normalized": normalize_score(raw, normalization),
        "backend": "jedi",
        "details": details,
    }


__all__ = ["compute_jedi_distance"]
