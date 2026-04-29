from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from worldarena.benchmark.long_sequence_runtime import compute_long_sequence_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Long-Sequence in a side runtime.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    request_path = Path(args.request)
    response_path = Path(args.response)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        result = compute_long_sequence_metrics(
            payload["video_path"],
            metric_names=list(payload.get("metric_names") or []),
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
