"""Supervise H3 progress without treating outer-job heartbeats as work."""

from __future__ import annotations

import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


PROGRESS_PREFIXES = (
    "H3_STAGE ", "PREPARE_READY ", "DEMO_PREPARED ", "PREPARED ",
    "MODEL_READY ", "CHUNK ", "ROLLOUT_STAGED ", "ROLLOUT_REUSED ",
    "ROLLOUT_PHASE_DONE ", "DECODER_READY ", "DECODE_PROGRESS ", "RESULT ",
)


def stop_process_group(process):
    # torchrun and all eight workers belong to this newly created session.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run_supervised(command, *, cwd, env, publish, diagnostics: Path,
                   stall_timeout_seconds=1800):
    timeout = float(stall_timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("SolarWM-H3 stall_timeout_seconds must be positive and finite")
    lines = queue.SimpleQueue()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)

    def read_output():
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    last_progress = time.monotonic()
    next_publish = last_progress
    last_stage = "worker startup"
    eof = False
    try:
        while True:
            try:
                line = lines.get(timeout=0.2)
                if line is None:
                    eof = True
                else:
                    print(line, end="", flush=True)
                    if "Fatal Python error:" in line:
                        # A peer may stay alive in native teardown, leaving
                        # torchrun to report a different rank's NCCL timeout.
                        # Preserve completed outputs and surface the first
                        # fatal error without waiting for that secondary hang.
                        publish()
                        raise RuntimeError(
                            f"SolarWM-H3 worker fatal error: {line.strip()}; "
                            f"last stage: {last_stage}; diagnostics: {diagnostics}"
                        )
                    if line.startswith(PROGRESS_PREFIXES):
                        last_progress = time.monotonic()
                        last_stage = line.strip()[:300]
            except queue.Empty:
                pass
            now = time.monotonic()
            if now >= next_publish:
                publish()
                next_publish = now + 5
            code = process.poll()
            if code is not None and eof:
                publish()
                if code:
                    raise subprocess.CalledProcessError(code, command)
                return
            if now - last_progress > timeout:
                raise TimeoutError(
                    f"SolarWM-H3 made no inference progress for {timeout:g}s; "
                    f"last stage: {last_stage}; per-rank stacks/NCCL logs: {diagnostics}"
                )
    finally:
        stop_process_group(process)
        reader.join(timeout=2)
        if not reader.is_alive():
            process.stdout.close()
