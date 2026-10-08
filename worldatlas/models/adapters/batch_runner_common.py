"""Shared helpers for model-specific batch runner subprocesses.

Batch runners consume JSONL batch specs and emit per-sample status lines on stdout.
These utilities cover argument parsing, spec loading, and safe output file copying.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from worldatlas.common.progress import log_progress

_tls = threading.local()


def add_common_batch_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags shared across all external batch runner modules."""
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--batch_spec_path", required=True, type=str)
    parser.add_argument("--generation_config_path", required=True, type=str)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_batch_spec(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL batch spec; each line is one generation request."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty batch spec: {path}")
    return rows


def copy_output(source: Path, destination: Path) -> None:
    """Atomically copy a generated artifact into the benchmark predictions tree."""
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing generated output: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination:
        temporary = destination.with_name(destination.name + ".tmp.syncing")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def latest_mp4(directory: Path) -> Path:
    """Pick the most recently written mp4 under a model output directory."""
    candidates = sorted(
        [path for path in directory.rglob("*.mp4") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no mp4 outputs under {directory}")
    return candidates[0]


def log_pipeline(event: str, **fields: Any) -> None:
    """Log checkpoint/runtime load events from a batch runner subprocess."""
    log_progress(event, **fields)


def begin_sample(
    sample_id: str,
    *,
    index: int,
    total: int,
    label: str | None = None,
    **fields: Any,
) -> float:
    """Log the start of a sample so a hung generation is visible in shard logs."""
    started_at = time.perf_counter()
    _tls.sample_id = sample_id
    _tls.index = index
    _tls.total = total
    _tls.started_at = started_at
    _tls.label = label
    log_progress(
        "sample_start",
        label=label,
        sample_id=sample_id,
        index=f"{index}/{total}",
        **fields,
    )
    return started_at


def print_status(sample_id: str, status: str, **payload: Any) -> None:
    """Emit compact human progress plus a machine-readable JSON status line."""
    traceback_text = payload.pop("traceback", None)
    compact = {key: value for key, value in payload.items() if key not in {"prompt"}}
    index = compact.pop("index", None)
    total = compact.pop("total", None)
    elapsed_s = compact.pop("elapsed_s", None)
    label = compact.pop("label", getattr(_tls, "label", None))
    if getattr(_tls, "sample_id", None) == sample_id:
        if index is None:
            index = getattr(_tls, "index", None)
        if total is None:
            total = getattr(_tls, "total", None)
        if elapsed_s is None and getattr(_tls, "started_at", None) is not None:
            elapsed_s = round(time.perf_counter() - _tls.started_at, 1)
    index_label = f"{index}/{total}" if index is not None and total is not None else None
    log_progress(
        "sample",
        label=label,
        sample_id=sample_id,
        index=index_label,
        status=status,
        elapsed_s=elapsed_s,
        error=compact.get("error"),
    )
    json_payload = {"sample_id": sample_id, "status": status, **compact}
    if index is not None:
        json_payload["index"] = index
    if total is not None:
        json_payload["total"] = total
    if elapsed_s is not None:
        json_payload["elapsed_s"] = elapsed_s
    print(json.dumps(json_payload, ensure_ascii=False), flush=True)
    if status == "failed" and traceback_text:
        print(traceback_text, flush=True)
