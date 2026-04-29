from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.checkpoints import apply_checkpoint_env
from worldarena.models.adapters.lingbot_world import (
    _load_persistent_runtime,
    _run_persistent_generation,
)
from worldarena.models.config import load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena LingBot persistent batch runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--batch-spec-path", required=True, type=Path)
    parser.add_argument("--results-path", required=True, type=Path)
    parser.add_argument("--device-id", default=0, type=int)
    return parser.parse_args()


def _load_batch_spec(path: Path) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            for key in ("sample", "conditioning_image", "output_path", "prompt"):
                if key not in payload:
                    raise KeyError(f"{path}:{line_number} missing required key: {key}")
            if not isinstance(payload["sample"], dict):
                raise TypeError(f"{path}:{line_number} sample payload must be an object")
            requests.append(payload)
    return requests


def _write_result(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> None:
    args = parse_args()
    os.environ.update(apply_checkpoint_env())

    model_config = load_model_config(args.model_config.resolve())
    if model_config.family.lower() not in {"lingbot", "lingbot_world", "lingbot-world"}:
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")

    requests = _load_batch_spec(args.batch_spec_path.resolve())
    runtime = _load_persistent_runtime(model_config, device_id=args.device_id)

    results_path = args.results_path.resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_path.unlink()

    with results_path.open("w", encoding="utf-8") as handle:
        for payload in requests:
            sample = BenchmarkSample(**payload["sample"])
            conditioning_image = Path(payload["conditioning_image"]).expanduser().resolve()
            output_path = Path(payload["output_path"]).expanduser().resolve()
            prompt = str(payload["prompt"])
            result: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "prediction_path": str(output_path),
                "prompt": prompt,
            }
            try:
                generation_payload = _run_persistent_generation(
                    runtime,
                    sample=sample,
                    conditioning_image=conditioning_image,
                    output_path=output_path,
                    prompt=prompt,
                )
                result.update(generation_payload)
                result["status"] = "generated"
                result["error"] = None
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = str(exc)
            _write_result(handle, result)


if __name__ == "__main__":
    main()
