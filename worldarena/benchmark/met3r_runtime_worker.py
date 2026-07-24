"""Worker entry point for MeT3R metric subprocesses."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from worldarena.benchmark.met3r_runtime import compute_met3r_consistency_from_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MEt3R in a side runtime.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    request_path = Path(args.request)
    response_path = Path(args.response)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        result = compute_met3r_consistency_from_paths(
            payload["pairs"],
            runtime=dict(payload.get("runtime") or {}),
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
