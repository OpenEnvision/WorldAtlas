"""Worker entry point for long-sequence metric subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from worldatlas.common.progress import ProgressHeartbeat, log_exception, log_progress
from worldatlas.benchmark.long_sequence_runtime import compute_long_sequence_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Long-Sequence in a side runtime.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    request_path = Path(args.request)
    response_path = Path(args.response)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    metric_names = list(payload.get("metric_names") or [])
    started_at = time.perf_counter()
    log_progress(
        "worker_start",
        metric=",".join(metric_names) or "long_sequence",
        python=sys.executable,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        video_path=payload.get("video_path"),
    )
    try:
        with ProgressHeartbeat(metric=",".join(metric_names) or "long_sequence", stage="worker"):
            result = compute_long_sequence_metrics(
                payload["video_path"],
                metric_names=metric_names,
                runtime=dict(payload.get("runtime") or {}),
            )
        log_progress(
            "worker_done",
            metric=",".join(metric_names) or "long_sequence",
            elapsed_s=f"{time.perf_counter() - started_at:.1f}",
        )
        response = {"ok": True, "result": result}
    except Exception as exc:  # pragma: no cover - exercised through subprocess integration
        log_exception(
            "worker_fail",
            exc,
            metric=",".join(metric_names) or "long_sequence",
            elapsed_s=f"{time.perf_counter() - started_at:.1f}",
        )
        response = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "stdout": "",
            "stderr": "",
        }
    response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
