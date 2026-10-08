"""Shared clip sampling for suite-level distribution metrics (JEDi, FVMD).

Distribution metrics compare two *sets* of clips and never need prediction and
reference to be paired, so both sides are decoded through the same spec here.

The spec samples a fixed wall-clock window rather than a fixed frame count.
FVMD derives its features from per-frame pixel displacement, so feeding it a
30 fps reference clip against a 16 fps prediction clip would report the frame
rate gap as a motion difference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from worldatlas.common.video_io import declared_frame_count_off_by_one


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class ClipDecodeError(RuntimeError):
    """Raised when a single video cannot contribute clips."""


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


def _as_optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


@dataclass(frozen=True, slots=True)
class ClipSamplingSpec:
    """How videos are turned into fixed-length clips for a distribution metric."""

    clip_frame_count: int = 16
    analysis_size: int = 224
    sample_fps: float | None = 8.0
    max_clips_per_video: int = 1
    clip_stride: int | None = None
    preserve_aspect_ratio: bool = False
    max_source_frames: int | None = None
    min_clip_count: int = 2

    @classmethod
    def from_runtime(
        cls,
        runtime: dict[str, Any],
        *,
        default_analysis_size: int,
    ) -> "ClipSamplingSpec":
        clip_frame_count = _as_positive_int(runtime.get("clip_frame_count"), 16)
        return cls(
            clip_frame_count=clip_frame_count,
            analysis_size=_as_positive_int(runtime.get("analysis_size"), default_analysis_size),
            sample_fps=_as_optional_positive_float(runtime.get("sample_fps")),
            max_clips_per_video=int(runtime.get("max_clips_per_video", 1)),
            clip_stride=_as_optional_positive_int(runtime.get("clip_stride")),
            preserve_aspect_ratio=bool(runtime.get("preserve_aspect_ratio", False)),
            max_source_frames=_as_optional_positive_int(runtime.get("max_source_frames")),
            min_clip_count=_as_positive_int(runtime.get("min_clip_count"), 2),
        )

    @property
    def effective_stride(self) -> int:
        return self.clip_stride or self.clip_frame_count

    @property
    def window_seconds(self) -> float | None:
        if self.sample_fps is None:
            return None
        return self.clip_frame_count / self.sample_fps

    def describe(self) -> dict[str, Any]:
        return {
            "clip_frame_count": self.clip_frame_count,
            "analysis_size": self.analysis_size,
            "sample_fps": self.sample_fps,
            "window_seconds": self.window_seconds,
            "max_clips_per_video": self.max_clips_per_video,
            "clip_stride": self.effective_stride,
            "preserve_aspect_ratio": self.preserve_aspect_ratio,
            "max_source_frames": self.max_source_frames,
            "min_clip_count": self.min_clip_count,
        }

    def signature(self) -> str:
        digest = hashlib.sha256()
        for key, value in sorted(self.describe().items()):
            digest.update(f"{key}={value}".encode("utf-8"))
        return digest.hexdigest()[:16]


@dataclass(slots=True)
class ClipBuildReport:
    """Bookkeeping for one side of a distribution comparison."""

    candidate_video_count: int = 0
    used_video_count: int = 0
    clip_count: int = 0
    skipped_video_count: int = 0
    skipped_videos_preview: list[dict[str, str]] = field(default_factory=list)
    used_ids_preview: list[str] = field(default_factory=list)
    source_fps_observed: list[float] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_video_count": self.candidate_video_count,
            "used_video_count": self.used_video_count,
            "clip_count": self.clip_count,
            "skipped_video_count": self.skipped_video_count,
            "skipped_videos_preview": self.skipped_videos_preview,
            "used_ids_preview": self.used_ids_preview,
        }
        if self.source_fps_observed:
            observed = np.asarray(self.source_fps_observed, dtype=np.float64)
            payload["source_fps_min"] = round(float(observed.min()), 3)
            payload["source_fps_max"] = round(float(observed.max()), 3)
            payload["source_fps_median"] = round(float(np.median(observed)), 3)
        return payload


def _resize_frame(frame: np.ndarray, spec: ClipSamplingSpec) -> np.ndarray:
    size = spec.analysis_size
    height, width = frame.shape[:2]
    if height == size and width == size:
        return frame
    if not spec.preserve_aspect_ratio:
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

    scale = size / min(height, width)
    scaled = cv2.resize(
        frame,
        (max(size, int(round(width * scale))), max(size, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    top = (scaled.shape[0] - size) // 2
    left = (scaled.shape[1] - size) // 2
    return scaled[top : top + size, left : left + size]


def _resample_indices(
    *,
    native_count: int,
    source_fps: float,
    sample_fps: float,
) -> list[int]:
    """Source frame indices approximating a constant ``sample_fps`` stream."""
    step = source_fps / sample_fps
    count = int(math.floor(native_count / step))
    if count <= 0:
        return []
    return [min(native_count - 1, int(round(index * step))) for index in range(count)]


def _decode_frames(path: Path, spec: ClipSamplingSpec, *, verified_frame_count: int | None = None) -> tuple[list[np.ndarray], float | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ClipDecodeError(f"failed to open video: {path}")

    try:
        native_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # A frozen reference roster may pin an ffprobe-verified count. Predictions
        # never get that roster, but they still use the same OpenCV off-by-one
        # recovery as per-sample decoding: the video stays in the population.
        if verified_frame_count is not None:
            if verified_frame_count <= 0:
                raise ClipDecodeError(f"Invalid verified frame count: {verified_frame_count}")
            native_count = verified_frame_count
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None

        keep: set[int] | None = None
        if spec.sample_fps is not None:
            if source_fps is None:
                raise ClipDecodeError(
                    f"cannot align to sample_fps={spec.sample_fps} because the source frame "
                    f"rate is unavailable: {path}"
                )
            # Upsampling would duplicate frames and fabricate zero motion, which FVMD
            # reads as a real signal, so refuse instead of silently corrupting the clip.
            if source_fps < spec.sample_fps - 1e-6:
                raise ClipDecodeError(
                    f"source frame rate {source_fps:.3f} is below sample_fps="
                    f"{spec.sample_fps}: {path}"
                )
            if native_count > 0:
                keep = set(
                    _resample_indices(
                        native_count=native_count,
                        source_fps=source_fps,
                        sample_fps=spec.sample_fps,
                    )
                )
                if not keep:
                    raise ClipDecodeError(
                        f"video is shorter than one resampled frame at sample_fps="
                        f"{spec.sample_fps}: {path}"
                    )

        frames: list[np.ndarray] = []
        frame_index = 0
        budget_reached = False
        while True:
            success, frame = capture.read()
            if not success:
                break
            if keep is None or frame_index in keep:
                frames.append(_resize_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), spec))
            frame_index += 1
            if spec.max_source_frames is not None and len(frames) >= spec.max_source_frames:
                budget_reached = True
                break
    finally:
        capture.release()

    if not budget_reached and native_count > 0 and frame_index != native_count:
        if (
            verified_frame_count is None
            and declared_frame_count_off_by_one(native_count, frame_index)
        ):
            # CAP_PROP_FRAME_COUNT is often one past the last decodable frame.
            # Resample from the observed length instead of dropping the video.
            return _decode_frames(path, spec, verified_frame_count=frame_index)
        raise ClipDecodeError(f"Incomplete video decode: expected {native_count}, got {frame_index}: {path}")
    if not frames:
        raise ClipDecodeError(f"video contains no decodable frames: {path}")
    return frames, source_fps


def _clip_start_indices(frame_count: int, spec: ClipSamplingSpec) -> list[int]:
    """Stride-aligned start offsets, spread over the video and never overrunning it."""
    stride = spec.effective_stride
    last_start = frame_count - spec.clip_frame_count
    if last_start < 0:
        return []
    if spec.max_clips_per_video <= 0:
        return list(range(0, last_start + 1, stride))
    # Spread the picks over the stride-aligned slots rather than over the raw
    # frame range: rounding a frame offset to a stride can land past
    # ``last_start`` and yield a short final clip.
    slots = last_start // stride + 1
    clip_count = min(spec.max_clips_per_video, slots)
    if clip_count <= 1:
        return [0]
    return sorted(
        {
            int(round(value)) * stride
            for value in np.linspace(0, slots - 1, num=clip_count)
        }
    )


def clips_for_video(path: Path, spec: ClipSamplingSpec, *, verified_frame_count: int | None = None) -> tuple[list[np.ndarray], float | None]:
    """Decode ``path`` into ``uint8`` clips shaped ``(clip_frame_count, size, size, 3)``."""
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ClipDecodeError(f"distribution metrics require video inputs, got: {path}")

    frames, source_fps = (_decode_frames(path, spec) if verified_frame_count is None else
                          _decode_frames(path, spec, verified_frame_count=verified_frame_count))
    starts = _clip_start_indices(len(frames), spec)
    if not starts:
        window = (
            f"{spec.clip_frame_count} frames at {spec.sample_fps} fps"
            if spec.sample_fps is not None
            else f"{spec.clip_frame_count} frames"
        )
        raise ClipDecodeError(
            f"video yields {len(frames)} sampled frames, fewer than one clip of {window}: {path}"
        )
    clips = [
        np.stack(frames[start : start + spec.clip_frame_count], axis=0).astype(np.uint8, copy=False)
        for start in starts
    ]
    # Fail this one video rather than letting a ragged clip reach the caller's
    # np.stack, which would abort the whole distribution comparison.
    expected = (spec.clip_frame_count, spec.analysis_size, spec.analysis_size, 3)
    for clip in clips:
        if clip.shape != expected:
            raise ClipDecodeError(
                f"clip shape {clip.shape} does not match expected {expected}: {path}"
            )
    return clips, source_fps


def build_clip_array(
    entries: Sequence[tuple[Path, str]],
    spec: ClipSamplingSpec,
    *,
    label: str,
    verified_counts: dict[str, int] | None = None,
) -> tuple[np.ndarray, ClipBuildReport]:
    """Decode every entry into one stacked clip array.

    ``entries`` are ``(path, identifier)`` pairs; the identifier only feeds
    diagnostics so callers can pass a sample id or a corpus-relative path.
    ``verified_counts`` is reference-only: ffprobe-confirmed frame counts from
    the frozen roster. Predictions omit it; a one-frame container lie is still
    recovered in ``_decode_frames`` so the prediction set is not silently cut.
    """
    report = ClipBuildReport(candidate_video_count=len(entries))
    clips: list[np.ndarray] = []

    def decode(entry):
        path, identifier = entry
        try:
            verified = None if verified_counts is None else verified_counts.get(str(path))
            video_clips, source_fps = clips_for_video(
                path, spec, verified_frame_count=verified
            )
            return path, identifier, video_clips, source_fps, None
        except Exception as exc:
            return path, identifier, None, None, str(exc)

    workers = int(os.environ.get('WORLDATLAS_DISTRIBUTION_DECODE_WORKERS', '1'))
    if not 1 <= workers <= 8:
        raise ValueError('Distribution decode workers must be between 1 and 8')
    context = ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()
    with context as pool:
        # map preserves input order even when a later video finishes first.
        # Pixel sampling, clip boundaries and feature batch boundaries stay fixed.
        decoded = pool.map(decode, entries) if pool else map(decode, entries)
        for path, identifier, video_clips, source_fps, error in decoded:
            if error is not None:
                report.skipped_video_count += 1
                if len(report.skipped_videos_preview) < 20:
                    report.skipped_videos_preview.append(
                        {"id": identifier, "path": str(path), "reason": error}
                    )
                continue
            clips.extend(video_clips)
            report.used_video_count += 1
            if source_fps is not None:
                report.source_fps_observed.append(source_fps)
            if len(report.used_ids_preview) < 20:
                report.used_ids_preview.append(identifier)

    if len(clips) < spec.min_clip_count:
        raise ValueError(
            f"{label} side produced {len(clips)} clips, fewer than min_clip_count="
            f"{spec.min_clip_count}"
        )

    report.clip_count = len(clips)
    return np.stack(clips, axis=0), report


# The reference side may draw more clips per video to lift its sample count, but
# the sampling window itself must stay identical or the two distributions are no
# longer measured on the same footing.
REFERENCE_OVERRIDE_KEYS = frozenset(
    {"max_clips_per_video", "clip_stride", "max_source_frames", "min_clip_count"}
)


def reference_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    """Apply ``runtime['reference']`` overrides on top of the shared runtime."""
    overrides = dict(runtime.get("reference") or {})
    unknown = sorted(set(overrides) - REFERENCE_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(
            "reference-side overrides may only change "
            f"{sorted(REFERENCE_OVERRIDE_KEYS)}; got {unknown}"
        )
    merged = {key: value for key, value in runtime.items() if key != "reference"}
    merged.update(overrides)
    return merged


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Save ``array`` so a concurrent reader never observes a partial file.

    The reference cache is shared across runs, and a half-written ``.npy`` would
    be silently loaded as a valid cache hit later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(staging, array)
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staging.write_text(text, encoding="utf-8")
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)


def clip_set_signature(
    entries: Iterable[tuple[Path, str]],
    spec: ClipSamplingSpec,
    *,
    extra: Sequence[str] = (),
) -> str:
    """Bind compressed video bytes, population identities and clip protocol."""
    from worldatlas.benchmark.integrity import path_content
    digest = hashlib.sha256()
    digest.update(b"content_and_sample_identity_v2")
    digest.update(json.dumps(path_content(Path(__file__)), sort_keys=True).encode('utf-8'))
    digest.update(cv2.getBuildInformation().encode('utf-8'))
    digest.update(spec.signature().encode("ascii"))
    for token in extra:
        digest.update(b"\x00extra\x00")
        digest.update(str(token).encode("utf-8"))
    for path, sample_id in sorted(entries, key=lambda item: (str(item[0]), item[1])):
        digest.update(b"\x00path\x00")
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\x00sample_id\x00")
        digest.update(sample_id.encode("utf-8"))
        digest.update(json.dumps(path_content(Path(path)), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:20]


def require_complete_clip_report(report: dict[str, Any], expected_videos: int, *, label: str) -> None:
    if (report.get("candidate_video_count") != expected_videos
            or report.get("used_video_count") != expected_videos
            or report.get("skipped_video_count") != 0 or report.get("clip_count", 0) <= 0):
        raise ValueError(f"{label} distribution is incomplete; skipped/missing videos cannot be scored: {report}")


def validate_distribution_features(features: Any, count: int, *, label: str) -> np.ndarray:
    array = np.asarray(features)
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} features must be finite and cover all {count} clips; got {array.shape}")
    return array


def distribution_cache_ready(features: Path) -> bool:
    from worldatlas.benchmark.sample_identity import file_sha256
    metadata = features.with_suffix('.json')
    if not features.is_file() or not metadata.is_file():
        return False
    report = json.loads(metadata.read_text(encoding='utf-8'))
    if report.get('features_sha256') != file_sha256(features):
        raise ValueError(f"Distribution feature cache has missing/mismatched content hash: {features}")
    validate_distribution_features(np.load(features, allow_pickle=False), int(report.get('clip_count', 0)), label='cached reference')
    return True


def distribution_cache_metadata(features: Path, report: dict[str, Any]) -> str:
    from worldatlas.benchmark.sample_identity import file_sha256
    return json.dumps({**report, 'features_sha256': file_sha256(features)}, indent=2, sort_keys=True) + '\n'


__all__ = [
    "REFERENCE_OVERRIDE_KEYS",
    "VIDEO_EXTENSIONS",
    "ClipBuildReport",
    "ClipDecodeError",
    "ClipSamplingSpec",
    "atomic_save_npy",
    "atomic_write_text",
    "build_clip_array",
    "clip_set_signature",
    "clips_for_video",
    "reference_runtime",
]
