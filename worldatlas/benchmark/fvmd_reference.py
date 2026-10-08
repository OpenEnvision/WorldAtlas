"""Model-independent, CPU-only eligibility preparation for FVMD references.

Never use this filter for predictions. A frozen eligible reference population
is strict at tracking time: new decode failures still fail the evaluation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2

from worldatlas.benchmark.distribution_clips import (
    ClipDecodeError, ClipSamplingSpec, VIDEO_EXTENSIONS, _clip_start_indices,
    atomic_write_text, clip_set_signature,
)
from worldatlas.common.progress import log_progress


@contextmanager
def cache_lock(path: Path):
    """Separate lock file: atomic replacement of data must not release the lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reference_probe_timeout(path: Path) -> float:
    """Seconds allowed for ``ffprobe -count_frames``.

    Typical clips finish well inside 600s. Multi-GB scenic references (tens of
    thousands of frames) do not, especially on NFS; aborting them is an
    operational failure, not an eligibility exclusion.
    """
    try:
        size = int(path.stat().st_size)
    except OSError:
        return 600.0
    return max(600.0, min(7200.0, size / (256 * 1024)))


def inspect_reference(path: Path, spec: ClipSamplingSpec, ffprobe: str) -> dict:
    # Missing files/timeouts/tool failures are operational problems, not grounds
    # for silently freezing a smaller corpus.
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ClipDecodeError(f"unsupported reference video extension: {path.suffix}")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-threads", "1", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=reference_probe_timeout(path), check=False,
    )
    if probe.returncode or probe.stderr.strip():
        message = probe.stderr.strip()
        if any(token in message.lower() for token in (
            "permission denied", "input/output error", "cannot allocate memory",
            "resource temporarily unavailable", "unknown decoder", "unrecognized option",
        )):
            raise RuntimeError(f"Reference probe infrastructure failure: {message}")
        raise ClipDecodeError(f"reference decode error: {message[:2000]}")
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams or not str(streams[0].get("nb_read_frames", "")).isdigit():
        raise ClipDecodeError("reference has no decodable video frames")
    decoded_count = int(streams[0]["nb_read_frames"])
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"ffprobe decoded reference but OpenCV cannot open it: {path}")
        declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ClipDecodeError("reference source frame rate unavailable")
    if spec.sample_fps is not None and source_fps < spec.sample_fps - 1e-6:
        raise ClipDecodeError(f"reference source fps {source_fps} below sample fps {spec.sample_fps}")
    sampled_count = (decoded_count if spec.sample_fps is None else
                     int(math.floor(decoded_count / (source_fps / spec.sample_fps))))
    if spec.max_source_frames is not None:
        sampled_count = min(sampled_count, spec.max_source_frames)
    clips = len(_clip_start_indices(sampled_count, spec))
    result = dict(decoded_frame_count=decoded_count, declared_frame_count=declared_count,
                  source_fps=source_fps, sampled_frame_count=sampled_count, clip_count=clips,
                  frame_count_corrected=declared_count != decoded_count)
    if not clips:
        return dict(result, eligible=False, reason="insufficient_sampled_frames")
    return dict(result, eligible=True)


def prepare_reference_selection(entries, spec: ClipSamplingSpec, cache_dir: Path, *, workers=8):
    """Freeze a content-bound roster before loading any GPU model; reuse it.

    Only expected video eligibility errors are exclusions. Operational errors
    propagate and never publish a partial roster. The key includes ALL inputs,
    including excluded videos, but no prediction/model identity.
    """
    entries = sorted(entries, key=lambda item: (str(item[0]), item[1]))
    if len({str(p) for p, _ in entries}) != len(entries):
        raise ValueError("FVMD reference population contains duplicate paths")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("FVMD reference preparation requires ffprobe on PATH")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature = clip_set_signature(entries, spec, extra=["fvmd_reference_eligibility", source_hash])
    destination = cache_dir / "reference_selection" / f"{signature}.json"
    with cache_lock(destination):
        if destination.is_file():
            payload = json.loads(destination.read_text())
            report = payload["selection"]
            if payload["sha256"] != _digest(report) or report["input_signature"] != signature:
                raise ValueError(f"FVMD reference selection integrity mismatch: {destination}")
        else:
            version = subprocess.run([ffprobe, "-version"], capture_output=True, text=True,
                                     check=True, timeout=30).stdout.splitlines()[0]
            log_progress("fvmd_reference_preflight", status="start", videos=len(entries), workers=workers)

            def inspect(entry):
                path, identifier = entry
                try:
                    details = inspect_reference(path, spec, ffprobe)
                except ClipDecodeError as exc:
                    details = dict(eligible=False, reason="decode_or_sampling_error", error=str(exc))
                return dict(path=str(path), id=identifier, **details)

            rows = []
            with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 16))) as pool:
                for row in pool.map(inspect, entries):
                    rows.append(row)
                    if len(rows) % 25 == 0 or len(rows) == len(entries):
                        log_progress("fvmd_reference_preflight", videos=f"{len(rows)}/{len(entries)}")
            eligible = [row for row in rows if row["eligible"]]
            report = dict(schema_version=1, input_signature=signature, clip_spec=spec.describe(),
                          ffprobe_version=version, candidate_video_count=len(rows),
                          eligible_video_count=len(eligible), excluded_video_count=len(rows)-len(eligible),
                          clip_count=sum(row["clip_count"] for row in eligible), videos=rows)
            payload = dict(selection=report, sha256=_digest(report))
            if clip_set_signature(entries, spec, extra=["fvmd_reference_eligibility", source_hash]) != signature:
                raise ValueError("Reference inputs changed during preparation; no roster published")
            atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not report["eligible_video_count"]:
        raise ValueError(f"No eligible FVMD references; inspect {destination}")
    log_progress("fvmd_reference_ready", eligible=report["eligible_video_count"],
                 excluded=report["excluded_video_count"], signature=payload["sha256"], manifest=destination)
    selected = [(Path(row["path"]), row["id"]) for row in report["videos"] if row["eligible"]]
    verified = {row["path"]: row["decoded_frame_count"] for row in report["videos"] if row["eligible"]}
    return selected, verified, payload
