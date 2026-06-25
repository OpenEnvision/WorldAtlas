#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


JUANXI_ROOT = Path("/mnt/cpfs/yangboxue/visual_generation/juanxi")
WORLDARENA_ROOT = JUANXI_ROOT / "WorldArena"
FOURDFY_ROOT = JUANXI_ROOT / "github_repos" / "4dfy"
PYTHON_BIN = JUANXI_ROOT / "conda" / "envs" / "4dfy" / "bin" / "python"
HF_HOME = JUANXI_ROOT / "conda" / "hf-home"
HF_HUB_CACHE = HF_HOME / "hub"
MVDREAM_CKPT = (
    HF_HUB_CACHE
    / "models--MVDream--MVDream"
    / "snapshots"
    / "d14ac9d78c48c266005729f2d5633f6c265da467"
    / "sd-v2.1-base-4view.pt"
)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_HUB_CACHE),
            "HUGGINGFACE_HUB_CACHE": str(HF_HUB_CACHE),
            "TRANSFORMERS_CACHE": str(HF_HUB_CACHE),
            "TORCH_HOME": str(JUANXI_ROOT / "conda" / "torch-home"),
            "XDG_CACHE_HOME": str(JUANXI_ROOT / "conda" / "xdg-cache"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def load_tasks(manifest_path: Path, limit: int | None) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("suite") != "image_static":
                continue
            tasks.append(row)
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def run_stage(
    *,
    gpu: int,
    sample_id: str,
    prompt: str,
    run_root: Path,
    config: str,
    stage_tag: str,
    weights: Path | None,
    log_dir: Path,
) -> Path:
    stage_root = run_root / config.replace(".yaml", "")
    trial_dir = stage_root / stage_tag
    ckpt = trial_dir / "ckpts" / "last.ckpt"
    if ckpt.is_file() and ckpt.stat().st_size > 0:
        return ckpt

    log_path = log_dir / f"{sample_id}.{stage_tag}.log"
    cmd = [
        str(PYTHON_BIN),
        "launch.py",
        "--config",
        f"configs/{config}",
        "--train",
        "--gpu",
        str(gpu),
        f"exp_root_dir={run_root}",
        "use_timestamp=false",
        f"tag={stage_tag}",
        "seed=0",
        f"system.prompt_processor.prompt={prompt}",
        f"system.guidance_multi_view.ckpt_path={MVDREAM_CKPT}",
        "system.loggers.wandb.enable=false",
    ]
    if weights is not None:
        cmd.append(f"system.weights={weights}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(
            cmd,
            cwd=str(FOURDFY_ROOT),
            env=build_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    if not ckpt.is_file() or ckpt.stat().st_size <= 0:
        raise FileNotFoundError(f"missing checkpoint after {stage_tag}: {ckpt}")
    return ckpt


def latest_prediction_video(stage3_trial: Path) -> Path:
    save_dir = stage3_trial / "save"
    candidates = [
        path
        for path in save_dir.glob("it*-test.mp4")
        if "depth" not in path.name and "static" not in path.name and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(f"no 4dfy test mp4 under {save_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def append_record(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORLDARENA_ROOT / "artifacts" / "benchmark" / "manifest" / "worldarena_manifest.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORLDARENA_ROOT / "artifacts" / "generations" / "4dfy_worldarena_image_static",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index out of range")

    tasks = load_tasks(args.manifest, args.limit)[args.shard_index :: args.num_shards]
    predictions_dir = args.output_root / "predictions"
    runs_dir = args.output_root / "runs"
    logs_dir = args.output_root / "logs"
    records_path = args.output_root / f"records.shard{args.shard_index}.jsonl"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    append_record(
        records_path,
        {
            "event": "worker_start",
            "gpu": args.gpu,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "tasks": len(tasks),
            "time": time.time(),
        },
    )

    for index, sample in enumerate(tasks):
        sample_id = str(sample["sample_id"])
        prompt = str(sample.get("prompt_target") or sample.get("prompt_current") or "")
        prediction_stem = str(sample["prediction_stem"])
        output_path = predictions_dir / f"{prediction_stem}.mp4"
        started = time.time()
        if output_path.is_file() and output_path.stat().st_size > 0:
            append_record(
                records_path,
                {
                    "sample_id": sample_id,
                    "status": "skipped_existing",
                    "prediction_path": str(output_path),
                    "gpu": args.gpu,
                    "shard_index": args.shard_index,
                    "index": index,
                },
            )
            continue

        try:
            run_root = runs_dir / sample_id
            ckpt1 = run_stage(
                gpu=args.gpu,
                sample_id=sample_id,
                prompt=prompt,
                run_root=run_root,
                config="fourdfy_stage_1.yaml",
                stage_tag=f"{sample_id}_stage1",
                weights=None,
                log_dir=logs_dir,
            )
            ckpt2 = run_stage(
                gpu=args.gpu,
                sample_id=sample_id,
                prompt=prompt,
                run_root=run_root,
                config="fourdfy_stage_2.yaml",
                stage_tag=f"{sample_id}_stage2",
                weights=ckpt1,
                log_dir=logs_dir,
            )
            ckpt3 = run_stage(
                gpu=args.gpu,
                sample_id=sample_id,
                prompt=prompt,
                run_root=run_root,
                config="fourdfy_stage_3.yaml",
                stage_tag=f"{sample_id}_stage3",
                weights=ckpt2,
                log_dir=logs_dir,
            )
            stage3_trial = ckpt3.parents[1]
            source_video = latest_prediction_video(stage3_trial)
            temporary = output_path.with_suffix(output_path.suffix + ".tmp")
            shutil.copy2(source_video, temporary)
            temporary.replace(output_path)
            append_record(
                records_path,
                {
                    "sample_id": sample_id,
                    "status": "generated",
                    "prediction_path": str(output_path),
                    "source_video": str(source_video),
                    "conditioning_image": sample.get("reference_path"),
                    "prompt": prompt,
                    "gpu": args.gpu,
                    "shard_index": args.shard_index,
                    "index": index,
                    "elapsed_seconds": round(time.time() - started, 3),
                },
            )
        except Exception as exc:
            append_record(
                records_path,
                {
                    "sample_id": sample_id,
                    "status": "failed",
                    "error": str(exc),
                    "prompt": prompt,
                    "gpu": args.gpu,
                    "shard_index": args.shard_index,
                    "index": index,
                    "elapsed_seconds": round(time.time() - started, 3),
                },
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
