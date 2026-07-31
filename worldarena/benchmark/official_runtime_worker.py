"""Worker entry point for official metric subprocesses."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Callable

from worldarena.benchmark.official_runtime import (
    compute_camera_error,
    compute_motion_accuracy,
    compute_motion_magnitude,
    compute_motion_smoothness,
    compute_optical_flow_aepe,
    compute_reprojection_error,
)


METRIC_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "camera_error": compute_camera_error,
    "reprojection_error": compute_reprojection_error,
    "optical_flow_aepe": compute_optical_flow_aepe,
    "motion_magnitude": compute_motion_magnitude,
    "motion_smoothness": compute_motion_smoothness,
    "motion_accuracy": compute_motion_accuracy,
}


def _run_request(payload: dict[str, Any]) -> dict[str, Any]:
    metric_name = str(payload["metric"])
    try:
        handler = METRIC_HANDLERS[metric_name]
    except KeyError as exc:
        raise KeyError(f"unsupported official runtime metric: {metric_name}") from exc
    kwargs = dict(payload.get("kwargs") or {})
    return handler(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an official metric in a side runtime.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    request_path = Path(args.request)
    response_path = Path(args.response)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        result = _run_request(payload)
        response = {"ok": True, "result": result}
    except Exception as exc:  # pragma: no cover - exercised through subprocess integration
        response = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "stdout": "",
            "stderr": "",
        }
    response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
