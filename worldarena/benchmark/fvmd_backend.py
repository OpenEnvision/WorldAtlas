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
DEFAULT_FVMD_REPO_CANDIDATES = (
    Path("/share/project/bench/model/FVMD-frechet-video-motion-distance"),
    Path(__file__).resolve().parents[2] / "thirdparty" / "FVMD-frechet-video-motion-distance",
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


def _resolve_fvmd_repo_root(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("repo_root")
    if configured:
        repo_root = Path(str(configured)).expanduser()
        if not repo_root.exists():
            raise FileNotFoundError(f"FVMD repo_root does not exist: {repo_root}")
        if not (repo_root / "fvmd").is_dir():
            raise FileNotFoundError(f"FVMD repo_root does not contain an fvmd package: {repo_root}")
        return repo_root.resolve()

    for candidate in DEFAULT_FVMD_REPO_CANDIDATES:
        if (candidate / "fvmd").is_dir():
            return candidate.resolve()
    return None


def _load_fvmd_callable(runtime: dict[str, Any]) -> tuple[Any, Path | None]:
    repo_root = _resolve_fvmd_repo_root(runtime)
    if repo_root is not None:
        repo_path = str(repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    try:
        module = importlib.import_module("fvmd.fvmd")
    except Exception as exc:  # pragma: no cover - depends on optional FVMD runtime
        raise ModuleNotFoundError(
            "FVMD package is unavailable; install fvmd or set "
            "benchmark.metric_runtime.fvmd.repo_root to the cloned repository. "
            f"Import error: {exc.__class__.__name__}: {exc}"
        ) from exc

    fvmd_fn = getattr(module, "fvmd", None)
    if fvmd_fn is None:
        raise AttributeError("FVMD package does not expose fvmd.fvmd")
    return fvmd_fn, repo_root


def _prepare_runtime_environment(runtime: dict[str, Any]) -> None:
    os.environ.update(checkpoint_env())
    os.environ.update(filter_checkpoint_env_overrides(dict(runtime.get("env", {}))))


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
        raise ValueError(f"FVMD requires video inputs, got: {path}")

    clip_frame_count = _as_positive_int(runtime.get("clip_frame_count"), 16)
    analysis_size = _as_positive_int(runtime.get("analysis_size"), 256)
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
            f"video has {len(frames)} decoded frames, fewer than FVMD clip_frame_count={clip_frame_count}: {path}"
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
            "FVMD needs at least "
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
        "analysis_size": _as_positive_int(runtime.get("analysis_size"), 256),
        "max_clips_per_video": int(runtime.get("max_clips_per_video", 1)),
        "clip_stride": _as_positive_int(runtime.get("clip_stride"), clip_frame_count),
        "max_source_frames": _as_optional_positive_int(runtime.get("max_source_frames")),
    }
    return np.stack(prediction_clips, axis=0), np.stack(reference_clips, axis=0), details


def compute_frechet_video_motion_distance(
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
    fvmd_fn, repo_root = _load_fvmd_callable(runtime)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    raw = float(
        fvmd_fn(
            log_dir=str(log_dir),
            gen_path=str(prediction_npy),
            gt_path=str(reference_npy),
        )
    )
    if not np.isfinite(raw):
        raise ValueError(f"FVMD returned a non-finite score: {raw}")

    details.update(
        {
            "prediction_npy": str(prediction_npy),
            "reference_npy": str(reference_npy),
            "log_dir": str(log_dir),
            "repo_root": str(repo_root) if repo_root is not None else None,
            "torch_home": os.environ.get("TORCH_HOME"),
            "raw_direction": "lower_is_better",
        }
    )
    return {
        "raw": raw,
        "normalized": normalize_score(raw, normalization),
        "backend": "fvmd",
        "details": details,
    }


__all__ = ["compute_frechet_video_motion_distance"]
