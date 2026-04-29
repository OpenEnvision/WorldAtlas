from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent LingBot batch generation runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suites", default=None, type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--visible-devices", default=None, type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _split_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def _prepare_cuda_visibility(args: argparse.Namespace) -> None:
    if args.visible_devices is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_devices)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    args = parse_args()
    _prepare_cuda_visibility(args)

    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.common.checkpoints import apply_checkpoint_env

    os.environ.update(apply_checkpoint_env())

    from worldarena.benchmark.manifest import load_manifest
    from worldarena.generation.runner import _conditioning_asset_for, _select_samples
    from worldarena.models.adapters.lingbot_world import (
        LingBotWorldAdapter,
        _load_persistent_runtime,
        _run_persistent_generation,
    )
    from worldarena.models.config import load_model_config

    model_config = load_model_config(args.model_config)
    if model_config.family.lower() not in {"lingbot", "lingbot_world", "lingbot-world"}:
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")
    if model_config.repo_root is None:
        raise ValueError("lingbot runner requires repo_root")

    manifest = load_manifest(args.manifest.resolve())
    suites = _split_arg(args.suites)
    selected = _select_samples(
        manifest,
        suites=suites,
        limit=args.limit,
        supported_suites=set(model_config.supported_suites),
    )
    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"shard_index must be within [0, {args.num_shards - 1}]")
    selected = selected[args.shard_index :: args.num_shards]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning_root = (output_dir / f"_conditioning_shard{args.shard_index}").resolve()
    conditioning_root.mkdir(parents=True, exist_ok=True)

    adapter = LingBotWorldAdapter(model_config)
    generation = dict(model_config.generation)
    runtime = _load_persistent_runtime(model_config, device_id=args.device_id)

    records_path = output_dir / f"generation_records.shard{args.shard_index}.jsonl"
    summary_path = output_dir / f"generation_manifest.shard{args.shard_index}.json"
    if records_path.exists():
        records_path.unlink()
    if summary_path.exists():
        summary_path.unlink()
    records: list[dict[str, Any]] = []

    for offset, sample in enumerate(selected, start=1):
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        prompt = adapter.prompt_for(sample)
        conditioning_image = _conditioning_asset_for(
            sample,
            reference_mode=model_config.reference_mode,
            cache_root=conditioning_root,
            min_video_frames=int(generation.get("conditioning_video_frames", 8)),
        )
        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(prediction_path),
            "conditioning_image": str(conditioning_image),
            "prompt": prompt,
            "status": "pending",
            "error": None,
            "shard_index": args.shard_index,
            "shard_position": offset,
            "shard_total": len(selected),
        }
        try:
            if prediction_path.exists() and not args.overwrite:
                record["status"] = "skipped_existing"
            else:
                payload = _run_persistent_generation(
                    runtime,
                    sample=sample,
                    conditioning_image=conditioning_image,
                    output_path=prediction_path,
                    prompt=prompt,
                )
                record.update(payload)
                record["status"] = "generated"
            records.append(record)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            records.append(record)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model_name": model_config.name,
        "family": model_config.family,
        "output_dir": str(output_dir),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": len(records),
        "generated": sum(1 for item in records if item["status"] == "generated"),
        "skipped_existing": sum(1 for item in records if item["status"] == "skipped_existing"),
        "failed": sum(1 for item in records if item["status"] == "failed"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
