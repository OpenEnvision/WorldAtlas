"""Distinguish invalid measurements from a poisoned GPU worker context."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Any


CUDA_RESTART_EXIT_CODE = 75


class CudaContextError(RuntimeError):
    """The worker must exit; empty_cache cannot repair a failed CUDA context."""


def is_cuda_failure(error: Any) -> bool:
    message = str(error or "").lower()
    return any(token in message for token in (
        "cuda error", "cuda runtime error", "cuda out of memory", "cudnn_status_",
        "cublas_status_", "device-side assert", "unspecified launch failure",
        "illegal memory access", "cuda driver error",
    ))


def raise_if_cuda_failure(error: Any) -> None:
    if is_cuda_failure(error):
        raise CudaContextError(str(error))


def metric_output_failed(output: dict[str, Any]) -> bool:
    """NA from a real gate/support limit is healthy; backend errors are not NA."""
    if is_cuda_failure(output.get("error")):
        return True
    def finite_raw(value: Any) -> bool:
        # Raw measurements can be structured (e.g. VFIMamba's MSE/SSIM/LPIPS).
        # Validate every component, but do not require the container to be scalar.
        if value is None:
            return True
        if isinstance(value, dict):
            return all(finite_raw(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_raw(item) for item in value)
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

    if not finite_raw(output.get("raw")):
        return True
    normalized = output.get("normalized")
    if normalized is not None and (isinstance(normalized, bool) or not isinstance(normalized, (int, float)) or not math.isfinite(normalized)):
        return True
    if normalized is not None and not 0.0 <= normalized <= 1.0:
        return True
    if output.get("eligibility_status") == "not_applicable":
        return normalized is not None
    if output.get("error") and output.get("eligibility_status", "official") != "diagnostic":
        return True
    return normalized is None


def sample_score_failed(row: dict[str, Any], required: list[str] | None = None) -> bool:
    metrics = row.get("metrics") or {}
    return bool(
        row.get("error") or not row.get("sample_id") or (not metrics and required != [])
        or (required and not set(required) <= metrics.keys())
        or any(metric_output_failed(value) for value in metrics.values())
    )


def load_sample_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Load atomically saved, healthy samples from a source-pinned directory."""
    if not path.is_dir():
        return {}
    result = {}
    for file in path.glob("*.json"):
        row = json.loads(file.read_text(encoding="utf-8"))
        if not sample_score_failed(row):
            result[row["sample_id"]] = row
    return result


def append_sample_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    destination = path / (hashlib.sha256(row["sample_id"].encode()).hexdigest() + ".json")
    temporary = destination.with_suffix(f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
