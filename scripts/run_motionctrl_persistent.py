from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent MotionCtrl batch generation runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suites", default="image_static", type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--visible-devices", default=None, type=str)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _split_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def _prepare_environment(args: argparse.Namespace, env_overrides: dict[str, str]) -> None:
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_devices)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.update(env_overrides)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _path_from_generation(value: object, *, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    if base is not None:
        return (base / path).resolve()
    return (_project_root() / path).resolve()


def _summary(records: list[dict[str, Any]], *, model_config, output_dir: Path, args: argparse.Namespace, suites: set[str] | None, elapsed: float) -> dict[str, Any]:
    generated = sum(1 for item in records if item["status"] == "generated")
    skipped = sum(1 for item in records if item["status"] == "skipped_existing")
    failed = sum(1 for item in records if item["status"] == "failed")
    return {
        "model_name": model_config.name,
        "family": model_config.family,
        "output_dir": str(output_dir),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "records": len(records),
        "generated": generated,
        "skipped_existing": skipped,
        "failed": failed,
        "suites": sorted(suites) if suites else model_config.supported_suites,
        "elapsed_sec": elapsed,
        "videos_per_min": (generated / elapsed * 60.0) if elapsed > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.benchmark.manifest import load_manifest
    from worldarena.generation.runner import _conditioning_asset_for, _select_samples
    from worldarena.models.adapters.base import PreparedGenerationRequest
    from worldarena.models.adapters.motionctrl import MotionCtrlAdapter, _request_payload_for
    from worldarena.models.adapters.motionctrl_runner import MotionCtrlRuntime
    from worldarena.models.config import load_model_config

    model_config = load_model_config(args.model_config.resolve())
    _prepare_environment(args, model_config.env)
    if model_config.family.lower().replace("-", "_") != "motionctrl":
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")
    if model_config.repo_root is None:
        raise ValueError("MotionCtrl runner requires repo_root")

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

    generation = dict(model_config.generation)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning_root = (output_dir / f"_conditioning_shard{args.shard_index}").resolve()
    conditioning_root.mkdir(parents=True, exist_ok=True)
    runtime_root = project_root / "cache" / "model_runtime" / "motionctrl"
    runtime_root.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"generation_records.shard{args.shard_index}.jsonl"
    summary_path = output_dir / f"generation_manifest.shard{args.shard_index}.json"
    prompt_map_path = output_dir / f"prompt_map.shard{args.shard_index}.json"
    for path in (records_path, summary_path, prompt_map_path):
        path.unlink(missing_ok=True)

    adapter = MotionCtrlAdapter(model_config)
    records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    prompt_map: dict[str, str] = {}

    for offset, sample in enumerate(selected, start=1):
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        prompt = adapter.prompt_for(sample)
        record: dict[str, Any] = {
            "case_id": sample.sample_id,
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(prediction_path),
            "output_path": str(prediction_path),
            "conditioning_image": None,
            "conditioning_input": None,
            "prompt": prompt,
            "prompt_source": model_config.prompt_mode,
            "prompt_current": sample.prompt_current,
            "prompt_target": sample.prompt_target,
            "status": "pending",
            "error": None,
            "shard_index": args.shard_index,
            "shard_position": offset,
            "shard_total": len(selected),
        }
        prompt_map[str(prediction_path)] = prompt
        try:
            conditioning_image = _conditioning_asset_for(
                sample,
                reference_mode=model_config.reference_mode,
                cache_root=conditioning_root,
                min_video_frames=int(generation.get("conditioning_video_frames", 8)),
            )
            record["conditioning_image"] = str(conditioning_image)
            record["conditioning_input"] = str(conditioning_image)
            if prediction_path.exists() and not args.overwrite:
                record["status"] = "skipped_existing"
                records.append(record)
                _write_record(records_path, record)
                continue
            request = PreparedGenerationRequest(
                sample=sample,
                conditioning_image=conditioning_image,
                output_path=prediction_path,
                prompt=prompt,
            )
            request_payload = _request_payload_for(
                request,
                generation=generation,
                runtime_root=runtime_root,
            )
            pending.append({"record": record, "request_payload": request_payload})
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc) or repr(exc)
            record["traceback"] = traceback.format_exc()
            records.append(record)
            _write_record(records_path, record)

    ckpt_path = _path_from_generation(
        generation.get("checkpoint") or generation.get("ckpt_path") or "motionctrl.pth",
        base=model_config.checkpoint_dir or model_config.repo_root,
    )
    config_path = _path_from_generation(
        generation.get("config_path") or generation.get("base") or "configs/inference/config_both.yaml",
        base=model_config.repo_root,
    )
    openclip_pretrained_path = _path_from_generation(generation.get("openclip_pretrained_path"))
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"MotionCtrl checkpoint not found: {ckpt_path}")
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"MotionCtrl config not found: {config_path}")

    started = time.time()
    try:
        runtime = MotionCtrlRuntime(
            repo_root=model_config.repo_root,
            config_path=config_path,
            ckpt_path=ckpt_path,
            openclip_pretrained_path=str(openclip_pretrained_path) if openclip_pretrained_path is not None else None,
            height=int(generation.get("height", 256)),
            width=int(generation.get("width", 256)),
            frames=int(generation.get("frames", generation.get("video_length", 16))),
            device_id=args.device_id,
        )
    except Exception as exc:
        runtime_error = str(exc) or repr(exc)
        runtime_traceback = traceback.format_exc()
        for item in pending:
            record = item["record"]
            request_payload = item["request_payload"]
            record["status"] = "failed"
            record["error"] = runtime_error
            record["traceback"] = runtime_traceback
            record.update(
                {
                    "camera_path": request_payload.get("camera_path", []),
                    "camera_pose_file": request_payload.get("camera_pose_file"),
                    "control_source": request_payload.get("control_source"),
                    "image_conditioning_contract": request_payload.get("image_conditioning_contract"),
                }
            )
            records.append(record)
            _write_record(records_path, record)
        elapsed = time.time() - started
        prompt_map_path.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = _summary(records, model_config=model_config, output_dir=output_dir, args=args, suites=suites, elapsed=elapsed)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1

    total_pending = len(pending)
    batch_size = max(1, int(generation.get("batch_size", 1)))
    for batch_start in range(0, total_pending, batch_size):
        batch_items = pending[batch_start : batch_start + batch_size]
        request_payloads = [item["request_payload"] for item in batch_items]
        try:
            payloads = runtime.generate_many(
                request_payloads,
                seed=int(generation.get("seed", 1234)) + args.shard_index * 100000 + batch_start,
                fps=float(generation.get("fps", 10.0)),
                n_samples=int(generation.get("n_samples", 1)),
                ddim_steps=int(generation.get("ddim_steps", 50)),
                ddim_eta=float(generation.get("ddim_eta", 1.0)),
                unconditional_guidance_scale=float(generation.get("unconditional_guidance_scale", 7.5)),
                unconditional_guidance_scale_temporal=generation.get("unconditional_guidance_scale_temporal"),
                cond_T=int(generation.get("cond_T", 800)),
            )
            for item, payload in zip(batch_items, payloads):
                record = item["record"]
                record.update(payload)
                record["status"] = "generated"
        except Exception as exc:
            error_text = str(exc) or repr(exc)
            traceback_text = traceback.format_exc()
            for item in batch_items:
                record = item["record"]
                request_payload = item["request_payload"]
                record["status"] = "failed"
                record["error"] = error_text
                record["traceback"] = traceback_text
                record.update(
                    {
                        "camera_path": request_payload.get("camera_path", []),
                        "camera_pose_file": request_payload.get("camera_pose_file"),
                        "control_source": request_payload.get("control_source"),
                        "image_conditioning_contract": request_payload.get("image_conditioning_contract"),
                    }
                )
        for item in batch_items:
            record = item["record"]
            records.append(record)
            _write_record(records_path, record)
        elapsed = time.time() - started
        generated = sum(1 for row in records if row["status"] == "generated")
        failed = sum(1 for row in records if row["status"] == "failed")
        skipped = sum(1 for row in records if row["status"] == "skipped_existing")
        videos_per_min = (generated / elapsed * 60.0) if elapsed > 0 else 0.0
        completed = min(batch_start + len(batch_items), total_pending)
        remaining = max(total_pending - completed, 0)
        eta_min = (remaining / videos_per_min) if videos_per_min > 0 else None
        eta_text = f"{eta_min:.1f}" if eta_min is not None else "unknown"
        print(
            "[motionctrl] "
            f"shard={args.shard_index} done={completed}/{total_pending} "
            f"generated={generated} failed={failed} skipped={skipped} "
            f"batch_size={len(batch_items)} "
            f"videos/min={videos_per_min:.3f} "
            f"eta_min={eta_text}",
            flush=True,
        )

    elapsed = time.time() - started
    prompt_map_path.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = _summary(records, model_config=model_config, output_dir=output_dir, args=args, suites=suites, elapsed=elapsed)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
