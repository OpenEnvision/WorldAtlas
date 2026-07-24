"""Worker entry point for PULSE metric subprocesses."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from worldarena.benchmark.pulse_runtime import compute_pulse_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pulse-of-Motion in a side runtime.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    request_path = Path(args.request)
    response_path = Path(args.response)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        result = compute_pulse_metrics(
            payload["video_path"],
            runtime=dict(payload.get("runtime") or {}),
            fallback_meta_fps=payload.get("fallback_meta_fps"),
        )
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
