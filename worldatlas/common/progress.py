"""Progress logging for model generation and metric evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence


_DEFAULT_HEARTBEAT_SECONDS = 120.0
_MAX_FIELD_CHARS = 240
_MAX_ERROR_CHARS = 200


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _compact_text(value: Any, *, limit: int = _MAX_FIELD_CHARS) -> str:
    text = " ".join(str(value).split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def heartbeat_interval_seconds(explicit: float | None = None) -> float:
    if explicit is not None:
        return explicit
    raw = os.environ.get("WORLDATLAS_PROGRESS_HEARTBEAT_SECONDS", str(_DEFAULT_HEARTBEAT_SECONDS))
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_HEARTBEAT_SECONDS


def rate_samples_per_hour(done: int, elapsed_s: float) -> str | None:
    if done <= 0 or elapsed_s <= 0:
        return None
    return f"{3600.0 * done / elapsed_s:.2f}"


def format_progress_line(event: str, **fields: Any) -> str:
    parts = [f"[{_timestamp()}]", "[progress]", event]
    for key, value in fields.items():
        if value is None or value == "":
            continue
        if key in {"prompt", "traceback"}:
            continue
        limit = _MAX_ERROR_CHARS if key == "error" else _MAX_FIELD_CHARS
        parts.append(f"{key}={_compact_text(value, limit=limit)}")
    return " ".join(parts)


def log_progress(event: str, **fields: Any) -> None:
    """Print one compact progress line and flush immediately."""
    print(format_progress_line(event, **fields), flush=True)


def log_exception(event: str, exc: BaseException, **fields: Any) -> None:
    """Print a progress line plus the real traceback, both flushed."""
    error = fields.pop("error", None) or f"{type(exc).__name__}: {exc}"
    log_progress(event, error=error, **fields)
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    sys.stdout.flush()
    sys.stderr.flush()


class ProgressHeartbeat:
    """Emit a heartbeat while a long generate/load/score call is in progress."""

    def __init__(self, *, interval_s: float | None = None, **fields: Any) -> None:
        self.interval_s = heartbeat_interval_seconds(interval_s)
        self.fields = fields
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def __enter__(self) -> ProgressHeartbeat:
        self._started = time.perf_counter()
        if self.interval_s > 0:
            self._thread = threading.Thread(
                target=self._run,
                name="worldatlas-progress-heartbeat",
                daemon=True,
            )
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            log_progress(
                "heartbeat",
                elapsed_s=f"{time.perf_counter() - self._started:.0f}",
                **self.fields,
            )

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def run_progress_subprocess(
    command: Sequence[str],
    *,
    cwd: str | os.PathLike[str],
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a metric worker, forwarding stdout live so shard logs are not silent."""
    merged_env = dict(os.environ if env is None else env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
    argv = [str(item) for item in command]
    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    chunks: list[str] = []
    stdout = process.stdout
    if stdout is not None:
        for line in stdout:
            chunks.append(line)
            print(line, end="", flush=True)
        stdout.close()
    code = process.wait()
    output = "".join(chunks)
    completed = subprocess.CompletedProcess(argv, code, stdout=output, stderr="")
    if check and code != 0:
        raise subprocess.CalledProcessError(code, argv, output=output, stderr="")
    return completed
