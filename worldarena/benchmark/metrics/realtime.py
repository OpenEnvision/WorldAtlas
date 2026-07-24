"""Real-time interaction metric implementations (latency, action response)."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from worldarena.benchmark.media import LoadedMedia
from worldarena.benchmark.metrics.base import Metric, clamp01, normalize_score, resolve_metric_backend
from worldarena.benchmark.schemas import BenchmarkSample, MetricOutput


REALTIME_THROUGHPUT_METRICS = {
    "effective_generation_fps",
    "realtime_factor",
}
INTERACTION_LATENCY_METRICS = {
    "action_response_latency",
    "latency_budget_pass_rate",
}
REALTIME_METRICS = REALTIME_THROUGHPUT_METRICS | INTERACTION_LATENCY_METRICS
SUPPORTED_REALTIME_BACKENDS = {"generation_metadata", "action_flow"}


def _not_applicable(metric_name: str, backend: str, reason: str, **details: Any) -> MetricOutput:
    """Build a skipped MetricOutput when a realtime metric cannot run."""
    return MetricOutput(
        raw=None,
        normalized=None,
        backend=backend,
        details=details,
        eligibility_status="not_applicable",
        error=reason,
    )


def _coerce_float(value: Any) -> float | None:
    """Parse a value to a finite float or return None when invalid."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _coerce_int(value: Any) -> int | None:
    """Round a coerced float to the nearest integer frame or count."""
    number = _coerce_float(value)
    if number is None:
        return None
    return int(round(number))


@lru_cache(maxsize=128)
def _load_generation_record_file(path_text: str) -> tuple[dict[str, Any], ...]:
    """Parse JSONL generation records from a cached file path."""
    path = Path(path_text)
    records: list[dict[str, Any]] = []
    if not path.exists():
        return ()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return tuple(records)


def _candidate_record_files(prediction_path: Path) -> list[Path]:
    """Search parent directories for generation_records JSONL sidecars."""
    directories: list[Path] = []
    current = prediction_path.resolve().parent
    for _ in range(3):
        directories.append(current)
        if current.parent == current:
            break
        current = current.parent

    files: list[Path] = []
    for directory in dict.fromkeys(directories):
        files.extend(sorted(directory.glob("generation_records*.jsonl")))
    return files


def _resolve_record_path(value: Any, *, base_dir: Path) -> Path | None:
    """Resolve a possibly relative generation-record path against a base dir."""
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        return path.resolve()
    except OSError:
        return path


def _find_generation_record(sample: BenchmarkSample, prediction_path: Path) -> dict[str, Any] | None:
    """Match a prediction video to its generation timing/action record."""
    resolved_prediction = prediction_path.resolve()
    fallback_by_sample: dict[str, Any] | None = None
    for records_file in _candidate_record_files(prediction_path):
        base_dir = records_file.parent
        for record in _load_generation_record_file(str(records_file.resolve())):
            for key in ("prediction_path", "output_path"):
                record_path = _resolve_record_path(record.get(key), base_dir=base_dir)
                if record_path is not None and record_path == resolved_prediction:
                    return dict(record)
            record_sample_id = str(record.get("sample_id") or record.get("case_id") or "")
            if record_sample_id == sample.sample_id:
                record_suite = record.get("suite")
                if record_suite in (None, sample.suite):
                    fallback_by_sample = dict(record)
    return fallback_by_sample


def _video_probe(path: Path) -> dict[str, float | int | None]:
    """Read frame count, FPS, and duration from a video with OpenCV."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    duration = (frame_count / fps) if frame_count > 0 and fps > 0 else None
    return {
        "frame_count": max(frame_count, 0),
        "fps": fps if fps > 0 else None,
        "duration_seconds": duration,
    }


def _target_fps(
    *,
    runtime: dict[str, Any],
    record: dict[str, Any] | None,
    sample: BenchmarkSample,
    probe_fps: float | None,
) -> float | None:
    """Resolve target FPS from runtime, records, sample metadata, or probe."""
    for value in [
        runtime.get("target_fps"),
        (record or {}).get("target_fps"),
        (record or {}).get("output_fps"),
        (record or {}).get("fps"),
        sample.fps,
        probe_fps,
    ]:
        number = _coerce_float(value)
        if number is not None and number > 0:
            return number
    return None


def _generation_wall_time(record: dict[str, Any] | None) -> float | None:
    """Extract wall-clock generation time from a generation record."""
    if not record:
        return None
    for key in [
        "generation_wall_time_seconds",
        "wall_time_seconds",
        "elapsed_seconds",
        "runtime_seconds",
    ]:
        value = _coerce_float(record.get(key))
        if value is not None and value > 0:
            return value
    return None


class RealtimeThroughputMetric(Metric):
    """Score effective FPS or realtime factor from generation metadata."""
    def __init__(
        self,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Validate generation-metadata backend and store runtime options."""
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        resolve_metric_backend(
            metric_name=metric_name,
            configured_backend=backend,
            auto_backend="generation_metadata",
            supported_backends={"generation_metadata"},
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Derive throughput from wall time, frame count, and video duration."""
        del reference

        backend = resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="generation_metadata",
            supported_backends={"generation_metadata"},
        )
        prediction_path = Path(prediction.path)
        record = _find_generation_record(sample, prediction_path)
        wall_time = _generation_wall_time(record)
        if wall_time is None:
            return _not_applicable(
                self.name,
                backend,
                "generation timing metadata unavailable",
                prediction_path=str(prediction_path),
            )

        probe = _video_probe(prediction_path)
        frame_count = int(probe["frame_count"] or 0)
        fps = _coerce_float(probe.get("fps"))
        if frame_count <= 0:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details={"prediction_path": str(prediction_path), "generation_wall_time_seconds": wall_time},
                error="prediction video frame count unavailable",
            )

        target_fps = _target_fps(runtime=self.runtime, record=record, sample=sample, probe_fps=fps)
        duration_seconds = _coerce_float(probe.get("duration_seconds"))
        if duration_seconds is None:
            fallback_fps = target_fps or fps
            if fallback_fps is not None and fallback_fps > 0:
                duration_seconds = frame_count / fallback_fps

        if self.name == "effective_generation_fps":
            raw = frame_count / wall_time
            normalized = clamp01(raw / target_fps) if target_fps else normalize_score(raw, self.normalization)
        else:
            if duration_seconds is None:
                return MetricOutput(
                    raw=None,
                    normalized=None,
                    backend=backend,
                    details={
                        "prediction_path": str(prediction_path),
                        "frame_count": frame_count,
                        "generation_wall_time_seconds": wall_time,
                    },
                    error="prediction video duration unavailable",
                )
            raw = duration_seconds / wall_time
            normalized = normalize_score(raw, self.normalization or {"lower": 0.0, "upper": 1.0})

        return MetricOutput(
            raw=round(float(raw), 6),
            normalized=normalized,
            backend=backend,
            details={
                "prediction_path": str(prediction_path),
                "generation_record_found": record is not None,
                "generation_wall_time_seconds": round(float(wall_time), 6),
                "frame_count": frame_count,
                "output_fps": fps,
                "target_fps": target_fps,
                "output_duration_seconds": duration_seconds,
            },
        )


def _load_action_payload(record: dict[str, Any] | None, prediction_path: Path) -> dict[str, Any] | None:
    """Load inline or sidecar action JSON referenced by a generation record."""
    if not record:
        return None
    inline = record.get("action_payload")
    if isinstance(inline, dict):
        return dict(inline)
    action_spec_path = record.get("action_spec_path")
    if action_spec_path:
        resolved = _resolve_record_path(action_spec_path, base_dir=prediction_path.resolve().parent)
        if resolved is not None and resolved.exists():
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    return None


def _row_tuple(keyboard_row: Any, mouse_row: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Normalize keyboard and mouse rows into hashable tuples for segmentation."""
    keyboard = tuple(float(value) for value in (keyboard_row or []))
    mouse = tuple(float(value) for value in (mouse_row or []))
    return keyboard, mouse


def _action_segments(payload: dict[str, Any], frame_count: int) -> list[dict[str, Any]]:
    """Split keyboard/mouse timelines into constant-action frame segments."""
    keyboard_rows = list(payload.get("keyboard_condition") or [])
    mouse_rows = list(payload.get("mouse_condition") or [])
    total_rows = max(len(keyboard_rows), len(mouse_rows), int(payload.get("total_frames") or 0), 0)
    if total_rows <= 0 or frame_count <= 1:
        return []
    usable_rows = min(total_rows, frame_count)

    starts = [0]
    previous = _row_tuple(
        keyboard_rows[0] if keyboard_rows else None,
        mouse_rows[0] if mouse_rows else None,
    )
    for index in range(1, usable_rows):
        current = _row_tuple(
            keyboard_rows[index] if index < len(keyboard_rows) else None,
            mouse_rows[index] if index < len(mouse_rows) else None,
        )
        if current != previous:
            starts.append(index)
            previous = current
    starts.append(usable_rows)

    segments: list[dict[str, Any]] = []
    for left, right in zip(starts[:-1], starts[1:]):
        segments.append(
            {
                "start_frame": left,
                "end_frame": right,
                "keyboard": keyboard_rows[left] if left < len(keyboard_rows) else [],
                "mouse": mouse_rows[left] if left < len(mouse_rows) else [],
            }
        )
    return segments


def _response_spec(segment: dict[str, Any]) -> tuple[str, float] | None:
    """Infer expected horizontal or radial motion direction from an action segment."""
    keyboard = np.asarray(segment.get("keyboard") or [], dtype=np.float32)
    mouse = np.asarray(segment.get("mouse") or [], dtype=np.float32)

    if mouse.shape[0] >= 2 and abs(float(mouse[1])) > 1e-6:
        return "horizontal", -float(np.sign(mouse[1]))
    if keyboard.shape[0] >= 4:
        if keyboard[2] > 0.5 and keyboard[3] <= 0.5:
            return "horizontal", 1.0
        if keyboard[3] > 0.5 and keyboard[2] <= 0.5:
            return "horizontal", -1.0
    if keyboard.shape[0] >= 2:
        if keyboard[0] > 0.5 and keyboard[1] <= 0.5:
            return "radial", 1.0
        if keyboard[1] > 0.5 and keyboard[0] <= 0.5:
            return "radial", -1.0
    return None


def _decode_video_frames(path: Path, *, max_width: int) -> list[np.ndarray]:
    """Decode grayscale frames downscaled for optical-flow latency analysis."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if width > max_width:
            scale = max_width / float(width)
            gray = cv2.resize(gray, (max_width, max(int(round(height * scale)), 1)))
        frames.append(gray)
    capture.release()
    return frames


def _flow_signal(left: np.ndarray, right: np.ndarray, spec: tuple[str, float]) -> float:
    """Measure signed optical-flow response along the expected action axis."""
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        left,
        right,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    kind, sign = spec
    if kind == "horizontal":
        return float(sign * np.mean(flow[:, :, 0]))

    height, width = left.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx -= (width - 1) / 2.0
    yy -= (height - 1) / 2.0
    norm = np.sqrt(xx**2 + yy**2)
    norm[norm < 1.0] = 1.0
    radial = flow[:, :, 0] * (xx / norm) + flow[:, :, 1] * (yy / norm)
    return float(sign * np.mean(radial))


def _first_response_latency_frames(
    *,
    frames: list[np.ndarray],
    segment: dict[str, Any],
    next_start: int,
    spec: tuple[str, float],
    min_response_signal: float,
    baseline_window_frames: int,
) -> int | None:
    """Detect the first frame where flow exceeds a baseline threshold."""
    start = int(segment["start_frame"])
    search_start = max(start - 1, 0)
    search_end = min(max(next_start, start + 1), len(frames) - 1)
    if search_start >= search_end:
        return None

    baseline_start = max(0, search_start - baseline_window_frames)
    baseline_values = [
        _flow_signal(frames[index], frames[index + 1], spec)
        for index in range(baseline_start, search_start)
    ]
    if baseline_values:
        threshold = max(
            min_response_signal,
            float(np.mean(baseline_values) + 2.0 * np.std(baseline_values)),
        )
    else:
        threshold = min_response_signal

    for index in range(search_start, search_end):
        if _flow_signal(frames[index], frames[index + 1], spec) >= threshold:
            response_frame = index + 1
            return max(response_frame - start, 0)
    return None


def _measure_action_latencies(
    *,
    prediction_path: Path,
    payload: dict[str, Any],
    fps: float,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Estimate per-action response latencies from flow in the prediction video."""
    probe = _video_probe(prediction_path)
    frame_count = int(probe["frame_count"] or 0)
    if frame_count <= 1:
        return {"events": [], "latencies_seconds": [], "missing_response_count": 0}
    segments = _action_segments(payload, frame_count)
    if len(segments) <= 1:
        return {"events": [], "latencies_seconds": [], "missing_response_count": 0}

    frames = _decode_video_frames(
        prediction_path,
        max_width=int(runtime.get("max_analysis_width", 320)),
    )
    frame_count = min(frame_count, len(frames))
    min_response_signal = float(runtime.get("min_response_signal", 0.04))
    baseline_window_frames = int(runtime.get("baseline_window_frames", 4))
    timeout_seconds = float(runtime.get("latency_timeout_seconds", 1.0))
    timeout_frames = max(int(round(timeout_seconds * fps)), 1)

    events: list[dict[str, Any]] = []
    latencies_seconds: list[float] = []
    missing_response_count = 0
    for index, segment in enumerate(segments[1:], start=1):
        start = int(segment["start_frame"])
        if start >= frame_count - 1:
            continue
        spec = _response_spec(segment)
        if spec is None:
            continue
        next_start = min(int(segments[index + 1]["start_frame"]) if index + 1 < len(segments) else frame_count, frame_count)
        latency_frames = _first_response_latency_frames(
            frames=frames,
            segment=segment,
            next_start=next_start,
            spec=spec,
            min_response_signal=min_response_signal,
            baseline_window_frames=baseline_window_frames,
        )
        detected = latency_frames is not None
        if latency_frames is None:
            latency_frames = timeout_frames
            missing_response_count += 1
        latency_seconds = float(latency_frames) / float(fps)
        latencies_seconds.append(latency_seconds)
        events.append(
            {
                "start_frame": start,
                "end_frame": int(segment["end_frame"]),
                "response_kind": spec[0],
                "response_sign": spec[1],
                "latency_frames": int(latency_frames),
                "latency_seconds": round(latency_seconds, 6),
                "detected": detected,
            }
        )
    return {
        "events": events,
        "latencies_seconds": latencies_seconds,
        "missing_response_count": missing_response_count,
    }


class InteractionLatencyMetric(Metric):
    """Measure mean action latency or latency-budget pass rate from flow cues."""
    def __init__(
        self,
        metric_name: str,
        backend: str,
        normalization: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        """Validate action-flow backend and store latency runtime options."""
        super().__init__(metric_name, backend, normalization)
        self.runtime = dict(runtime or {})
        resolve_metric_backend(
            metric_name=metric_name,
            configured_backend=backend,
            auto_backend="action_flow",
            supported_backends={"action_flow"},
        )

    def compute(
        self,
        sample: BenchmarkSample,
        prediction: LoadedMedia,
        reference: LoadedMedia,
    ) -> MetricOutput:
        """Score action-response latency or budget pass rate from flow events."""
        del reference

        backend = resolve_metric_backend(
            metric_name=self.name,
            configured_backend=self.backend,
            auto_backend="action_flow",
            supported_backends={"action_flow"},
        )
        prediction_path = Path(prediction.path)
        record = _find_generation_record(sample, prediction_path)
        payload = _load_action_payload(record, prediction_path)
        if payload is None:
            return _not_applicable(
                self.name,
                backend,
                "action specification unavailable",
                prediction_path=str(prediction_path),
            )

        probe = _video_probe(prediction_path)
        fps = _coerce_float(probe.get("fps")) or _target_fps(
            runtime=self.runtime,
            record=record,
            sample=sample,
            probe_fps=None,
        )
        if fps is None or fps <= 0:
            return MetricOutput(
                raw=None,
                normalized=None,
                backend=backend,
                details={"prediction_path": str(prediction_path)},
                error="prediction FPS unavailable",
            )

        latency_payload = _measure_action_latencies(
            prediction_path=prediction_path,
            payload=payload,
            fps=fps,
            runtime=self.runtime,
        )
        latencies = [float(value) for value in latency_payload["latencies_seconds"]]
        if not latencies:
            return _not_applicable(
                self.name,
                backend,
                "no measurable action transitions",
                prediction_path=str(prediction_path),
                action_count=len(payload.get("actions") or []),
            )

        threshold = float(self.runtime.get("latency_threshold_seconds", 0.1))
        if self.name == "action_response_latency":
            raw = float(np.mean(latencies))
            normalized = normalize_score(
                raw,
                self.normalization or {"lower": 0.0, "upper": threshold, "higher_is_better": False},
            )
        else:
            raw = sum(value <= threshold for value in latencies) / float(len(latencies))
            normalized = normalize_score(raw, self.normalization or {"lower": 0.0, "upper": 1.0})

        return MetricOutput(
            raw=round(raw, 6),
            normalized=normalized,
            backend=backend,
            details={
                "prediction_path": str(prediction_path),
                "latency_threshold_seconds": threshold,
                "fps": fps,
                "action_count": len(payload.get("actions") or []),
                "measured_action_events": len(latencies),
                "missing_response_count": latency_payload["missing_response_count"],
                "latency_mean_seconds": round(float(np.mean(latencies)), 6),
                "latency_p50_seconds": round(float(np.percentile(latencies, 50)), 6),
                "latency_p95_seconds": round(float(np.percentile(latencies, 95)), 6),
                "latency_max_seconds": round(float(np.max(latencies)), 6),
                "events": latency_payload["events"],
            },
        )


def build_realtime_metric(
    *,
    metric_name: str,
    backend: str,
    normalization: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> Metric:
    """Factory that returns throughput or interaction latency metric instances."""
    if metric_name in REALTIME_THROUGHPUT_METRICS:
        return RealtimeThroughputMetric(
            metric_name=metric_name,
            backend=backend,
            normalization=normalization,
            runtime=runtime,
        )
    if metric_name in INTERACTION_LATENCY_METRICS:
        return InteractionLatencyMetric(
            metric_name=metric_name,
            backend=backend,
            normalization=normalization,
            runtime=runtime,
        )
    raise KeyError(f"unsupported realtime metric: {metric_name}")


__all__ = [
    "INTERACTION_LATENCY_METRICS",
    "REALTIME_METRICS",
    "REALTIME_THROUGHPUT_METRICS",
    "build_realtime_metric",
]
