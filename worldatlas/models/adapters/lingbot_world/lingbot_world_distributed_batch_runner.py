"""Distributed batch subprocess runner for LingBot-World multi-GPU inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.common.checkpoints import apply_checkpoint_env
from worldatlas.models.adapters.batch_runner_common import begin_sample, print_status
from worldatlas.models.adapters.lingbot_world.common import LingBotWorldBaseAdapter, _persistent_heartbeat
from worldatlas.models.adapters.lingbot_world.lingbot_world_batch_runner import _load_batch_spec, _write_result
from worldatlas.models.registry import build_model_adapter
from worldatlas.models.config import load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas LingBot distributed persistent batch runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--batch-spec-path", required=True, type=Path)
    parser.add_argument("--results-path", required=True, type=Path)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--heartbeat-path", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.update(apply_checkpoint_env())
    if args.heartbeat_path is not None:
        os.environ["LINGBOT_PERSISTENT_HEARTBEAT_LOG"] = str(args.heartbeat_path.expanduser().resolve())
    _persistent_heartbeat("runner_start", batch_spec_path=args.batch_spec_path, results_path=args.results_path)

    model_config = load_model_config(args.model_config.resolve())
    adapter = build_model_adapter(model_config)
    if not isinstance(adapter, LingBotWorldBaseAdapter):
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")

    requests = _load_batch_spec(args.batch_spec_path.resolve())
    _persistent_heartbeat("batch_spec_loaded", requests=len(requests))
    _persistent_heartbeat("runtime_load_start")
    runtime = adapter._load_persistent_runtime(
        device_id=args.device_id,
        distributed=True,
    )
    _persistent_heartbeat("runtime_load_done", rank=runtime.rank, world_size=runtime.world_size)

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
                begin_sample(
                    sample.sample_id,
                    index=index,
                    total=total_requests,
                    label="LingBot-World",
                )
            result: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "prediction_path": str(output_path),
                "prompt": prompt,
            }
            _persistent_heartbeat("sample_start", index=index, total=total_requests, sample_id=sample.sample_id)
            try:
                generation_payload = adapter._run_persistent_generation(
                    runtime,
                    sample=sample,
                    conditioning_image=conditioning_image,
                    output_path=output_path,
                    prompt=prompt,
                )
            except Exception as exc:
                _persistent_heartbeat(
                    "sample_failed",
                    index=index,
                    total=total_requests,
                    sample_id=sample.sample_id,
                    error=repr(exc),
                )
                if runtime.rank == 0 and result_handle is not None:
                    result["status"] = "failed"
                    result["error"] = str(exc)
                    print_status(sample.sample_id, "failed", error=str(exc), label="LingBot-World")
                    _write_result(result_handle, result)
                raise
            _persistent_heartbeat("sample_done", index=index, total=total_requests, sample_id=sample.sample_id)
            if runtime.rank == 0 and result_handle is not None:
                result.update(generation_payload)
                result["status"] = "generated"
                result["error"] = None
                _write_result(result_handle, result)
                print_status(sample.sample_id, "generated", output_path=str(output_path), label="LingBot-World")
    finally:
        _persistent_heartbeat("runner_finally_start")
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
        _persistent_heartbeat("runner_finally_done")


if __name__ == "__main__":
    main()
