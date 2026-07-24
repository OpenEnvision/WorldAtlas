"""Distributed batch subprocess runner for LingBot-World multi-GPU inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.common.checkpoints import apply_checkpoint_env
from worldarena.models.adapters.lingbot_world import (
    _load_persistent_runtime,
    _run_persistent_generation,
)
from worldarena.models.config import load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Arena LingBot distributed persistent batch runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--batch-spec-path", required=True, type=Path)
    parser.add_argument("--results-path", required=True, type=Path)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--heartbeat-path", default=None, type=Path)
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


def _heartbeat(stage: str, **payload: Any) -> None:
    path_value = os.environ.get("LINGBOT_PERSISTENT_HEARTBEAT_LOG")
    if not path_value:
        return
    fields: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "rank": os.environ.get("RANK", "0"),
        "local_rank": os.environ.get("LOCAL_RANK", "0"),
        "stage": stage,
    }
    fields.update(payload)
    path = Path(path_value).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(f"{key}={value}" for key, value in fields.items()) + "\n")
            handle.flush()
    except Exception:
        return


def main() -> None:
    args = parse_args()
    os.environ.update(apply_checkpoint_env())
    if args.heartbeat_path is not None:
        os.environ["LINGBOT_PERSISTENT_HEARTBEAT_LOG"] = str(args.heartbeat_path.expanduser().resolve())
    _heartbeat("runner_start", batch_spec_path=args.batch_spec_path, results_path=args.results_path)

    model_config = load_model_config(args.model_config.resolve())
    if model_config.family.lower() not in {"lingbot", "lingbot_world", "lingbot-world"}:
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")

    requests = _load_batch_spec(args.batch_spec_path.resolve())
    _heartbeat("batch_spec_loaded", requests=len(requests))
    _heartbeat("runtime_load_start")
    runtime = _load_persistent_runtime(
        model_config,
        device_id=args.device_id,
        distributed=True,
    )
    _heartbeat("runtime_load_done", rank=runtime.rank, world_size=runtime.world_size)

    results_path = args.results_path.resolve()
    if runtime.rank == 0:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        if results_path.exists():
            results_path.unlink()
        result_handle = results_path.open("w", encoding="utf-8")
    else:
        result_handle = None

    try:
        total_requests = len(requests)
        for index, payload in enumerate(requests, start=1):
            sample = BenchmarkSample(**payload["sample"])
            conditioning_image = Path(payload["conditioning_image"]).expanduser().resolve()
            output_path = Path(payload["output_path"]).expanduser().resolve()
            prompt = str(payload["prompt"])
            if runtime.rank == 0:
                print(
                    f"[lingbot-distributed-batch] start {index}/{total_requests} {sample.sample_id}",
                    flush=True,
                )
            result: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "prediction_path": str(output_path),
                "prompt": prompt,
            }
            _heartbeat("sample_start", index=index, total=total_requests, sample_id=sample.sample_id)
            try:
                generation_payload = _run_persistent_generation(
                    runtime,
                    sample=sample,
                    conditioning_image=conditioning_image,
                    output_path=output_path,
                    prompt=prompt,
                )
            except Exception as exc:
                _heartbeat(
                    "sample_failed",
                    index=index,
                    total=total_requests,
                    sample_id=sample.sample_id,
                    error=repr(exc),
                )
                if runtime.rank == 0 and result_handle is not None:
                    result["status"] = "failed"
                    result["error"] = str(exc)
                    _write_result(result_handle, result)
                raise
            _heartbeat("sample_done", index=index, total=total_requests, sample_id=sample.sample_id)
            if runtime.rank == 0 and result_handle is not None:
                result.update(generation_payload)
                result["status"] = "generated"
                result["error"] = None
                _write_result(result_handle, result)
                print(
                    f"[lingbot-distributed-batch] finish {index}/{total_requests} {sample.sample_id}",
                    flush=True,
                )
    finally:
        _heartbeat("runner_finally_start")
        if result_handle is not None:
            result_handle.close()
        if runtime.dist is not None and runtime.dist.is_initialized():
            if sys.exc_info()[0] is None:
                try:
                    runtime.dist.barrier()
                finally:
                    runtime.dist.destroy_process_group()
            else:
                runtime.dist.destroy_process_group()
        _heartbeat("runner_finally_done")


if __name__ == "__main__":
    main()
