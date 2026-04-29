from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worldarena.benchmark import load_manifest
from worldarena.generation.runner import _conditioning_asset_for, _select_samples
from worldarena.models.adapters.pose_synthesis import resolve_pose_annotation_path
from worldarena.models.adapters.worldfm import _panorama_prompt_for_sample
from worldarena.models.config import load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch decoupled WorldFM generation.")
    parser.add_argument("--model-config", default="config/models/worldfm.yaml")
    parser.add_argument("--manifest", default="artifacts/benchmark/manifest/worldarena_manifest.jsonl")
    parser.add_argument("--output-dir", default="/bench_output/worldfm_image_static")
    parser.add_argument("--suites", default="image_static")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["full", "prepare", "merge", "final", "prepare_merge"], default="full")
    parser.add_argument("--prepare-gpus", default="0,1,2,3")
    parser.add_argument("--final-gpus", default="0,1,2,3")
    parser.add_argument("--merge-workers", type=int, default=16)
    parser.add_argument("--threads-per-merge-worker", type=int, default=None)
    parser.add_argument("--threads-per-gpu-worker", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--spec-path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-after-final", action="store_true")
    return parser.parse_args()


def _split_csv(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return values


def _suite_set(raw: str | None) -> set[str] | None:
    values = _split_csv(raw or "")
    return set(values) if values else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_spec(args: argparse.Namespace, *, output_dir: Path, work_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    model_config = load_model_config(Path(args.model_config))
    manifest = load_manifest(Path(args.manifest).expanduser().resolve())
    selected = _select_samples(
        manifest,
        suites=_suite_set(args.suites),
        limit=args.limit,
        supported_suites=set(model_config.supported_suites),
    )
    cache_root = output_dir / "_conditioning"
    cache_root.mkdir(parents=True, exist_ok=True)
    generation = model_config.generation
    requests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for sample in selected:
        output_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(output_path),
            "conditioning_image": None,
            "prompt": "",
            "panorama_prompt": None,
            "panorama_negative_prompt": str(generation.get("panorama_negative_prompt", "")),
            "status": "pending",
            "error": None,
        }
        if output_path.exists() and not args.overwrite:
            record["status"] = "skipped_existing"
            records.append(record)
            continue
        try:
            conditioning_image = _conditioning_asset_for(
                sample,
                reference_mode=model_config.reference_mode,
                cache_root=cache_root,
                min_video_frames=int(generation.get("conditioning_video_frames", 8)),
            )
            pose_annotation_path, pose_source = resolve_pose_annotation_path(
                model_config,
                sample,
                conditioning_image.expanduser().resolve(),
                frame_count=int(generation.get("trajectory_frames", generation.get("num_frames", 81))),
                namespace="worldfm",
                focal_scale=float(generation.get("synthetic_focal_scale", 0.6)),
            )
            panorama_prompt = _panorama_prompt_for_sample(sample, generation)
            request = {
                "sample_id": sample.sample_id,
                "prediction_stem": sample.prediction_stem,
                "conditioning_image": str(conditioning_image.expanduser().resolve()),
                "output_path": str(output_path),
                "prompt": "",
                "panorama_prompt": panorama_prompt,
                "panorama_negative_prompt": str(generation.get("panorama_negative_prompt", "")),
                "annotation_path": str(pose_annotation_path),
                "pose_source": pose_source,
            }
            requests.append(request)
            record.update(
                {
                    "conditioning_image": request["conditioning_image"],
                    "panorama_prompt": panorama_prompt,
                    "panorama_negative_prompt": request["panorama_negative_prompt"],
                    "pose_source": pose_source,
                    "pose_annotation_path": str(pose_annotation_path),
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep manifest build going.
            record["status"] = "failed"
            record["error"] = str(exc)
        records.append(record)

    spec_path = Path(args.spec_path).expanduser().resolve() if args.spec_path else work_root / "requests.jsonl"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with spec_path.open("w", encoding="utf-8") as file:
        for request in requests:
            file.write(json.dumps(request, ensure_ascii=False) + "\n")
    with (work_root / "initial_records.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json(
        work_root / "launch_manifest.json",
        {
            "model_config": str(Path(args.model_config).resolve()),
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "output_dir": str(output_dir),
            "work_root": str(work_root),
            "requests": len(requests),
            "selected": len(selected),
            "mode": args.mode,
        },
    )
    return spec_path, records


def _model_args(args: argparse.Namespace, model_config_path: Path) -> list[str]:
    model_config = load_model_config(model_config_path)
    generation = model_config.generation
    command = [
        "--repo_root",
        str(model_config.repo_root),
        "--checkpoint_dir",
        str(model_config.checkpoint_dir),
        "--model_filename",
        str(generation.get("model_filename", "worldfm_2-step.pth")),
        "--vae_subdir",
        str(generation.get("vae_subdir", "vae")),
        "--step",
        str(int(generation.get("step", 2))),
        "--image_size",
        str(int(generation.get("image_size", 512))),
        "--cfg_scale",
        str(float(generation.get("cfg_scale", 4.5))),
        "--render_size",
        str(int(generation.get("render_size", 512))),
        "--fps",
        str(int(generation.get("fps", 30))),
        "--video_crf",
        str(int(generation.get("video_crf", 14))),
        "--video_preset",
        str(generation.get("video_preset", "medium")),
    ]
    for option_name in ("config_path", "hw_path", "moge_path", "moge_pretrained"):
        option_value = generation.get(option_name)
        if option_value:
            command.extend([f"--{option_name}", str(option_value)])
    return command


def _thread_env(env: dict[str, str], threads: int) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = str(max(int(threads), 1))


def _start_worker(
    *,
    stage: str,
    worker_name: str,
    log_path: Path,
    env: dict[str, str],
    base_command: list[str],
    gpu_id: str | None,
    threads: int,
    delete_after_final: bool,
) -> tuple[str, subprocess.Popen, object]:
    worker_env = env.copy()
    worker_env["PYTHONUNBUFFERED"] = "1"
    worker_env["TOKENIZERS_PARALLELISM"] = "false"
    if gpu_id is None:
        worker_env["CUDA_VISIBLE_DEVICES"] = ""
        gpu_index = "-1"
    else:
        worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_index = "0"
    _thread_env(worker_env, threads)
    command = [
        sys.executable,
        "-m",
        "worldarena.models.adapters.worldfm_decoupled_runner",
        "--stage",
        stage,
        *base_command,
        "--gpu_index",
        gpu_index,
        "--worker_name",
        worker_name,
    ]
    if delete_after_final and stage == "final":
        command.append("--delete_after_final")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    print(f"[launch_worldfm_decoupled] start {worker_name} log={log_path}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=worker_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return worker_name, process, log_file


def _wait(processes: list[tuple[str, subprocess.Popen, object]]) -> bool:
    failed = False
    remaining = list(processes)
    try:
        while remaining:
            active: list[tuple[str, subprocess.Popen, object]] = []
            for name, process, log_file in remaining:
                code = process.poll()
                if code is None:
                    active.append((name, process, log_file))
                    continue
                log_file.close()
                print(f"[launch_worldfm_decoupled] {name} exited code={code}", flush=True)
                failed = failed or code != 0
            remaining = active
            if remaining:
                time.sleep(15)
    finally:
        for name, process, log_file in remaining:
            if process.poll() is None:
                print(f"[launch_worldfm_decoupled] terminating {name}", flush=True)
                process.terminate()
            log_file.close()
    return not failed


def _summarize(output_dir: Path, work_root: Path, records: list[dict[str, Any]]) -> None:
    updated: list[dict[str, Any]] = []
    for record in records:
        if record["status"] == "skipped_existing":
            updated.append(record)
            continue
        prediction_path = Path(str(record["prediction_path"]))
        stem = prediction_path.stem
        final_done = work_root / "final" / stem / "done.json"
        final_failed = work_root / "final" / stem / "failed.json"
        if prediction_path.exists():
            record["status"] = "generated"
            record["error"] = None
        elif final_failed.exists():
            record["status"] = "failed"
            try:
                record["error"] = json.loads(final_failed.read_text(encoding="utf-8")).get("error")
            except Exception:
                record["error"] = "final stage failed"
        elif final_done.exists():
            record["status"] = "failed"
            record["error"] = f"final stage done but output missing: {prediction_path}"
        updated.append(record)
    with (output_dir / "generation_records.jsonl").open("w", encoding="utf-8") as file:
        for record in updated:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "model_name": "worldfm",
        "family": "worldfm",
        "output_dir": str(output_dir),
        "records": len(updated),
        "generated": sum(1 for item in updated if item["status"] == "generated"),
        "skipped_existing": sum(1 for item in updated if item["status"] == "skipped_existing"),
        "failed": sum(1 for item in updated if item["status"] == "failed"),
        "work_root": str(work_root),
    }
    _write_json(output_dir / "generation_manifest.json", manifest)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root).expanduser().resolve() if args.work_root else output_dir / "_worldfm_decoupled"
    run_root = work_root / f"run_{uuid4().hex[:10]}"
    if args.spec_path:
        run_root = work_root
    run_root.mkdir(parents=True, exist_ok=True)
    spec_path, records = _build_spec(args, output_dir=output_dir, work_root=run_root)
    if not spec_path.exists() or spec_path.stat().st_size == 0:
        _summarize(output_dir, run_root, records)
        print("[launch_worldfm_decoupled] no runnable requests", flush=True)
        return 0

    model_args = _model_args(args, Path(args.model_config))
    base_command = [
        *model_args,
        "--batch_spec_path",
        str(spec_path),
        "--work_root",
        str(run_root),
        "--poll_seconds",
        str(float(args.poll_seconds)),
    ]
    if args.overwrite:
        base_command.append("--overwrite")

    log_dir = output_dir / "logs"
    env = os.environ.copy()
    prepare_gpus = _split_csv(args.prepare_gpus)
    final_gpus = _split_csv(args.final_gpus)
    merge_threads = int(args.threads_per_merge_worker or max(1, (os.cpu_count() or 1) // max(int(args.merge_workers), 1)))

    merge_processes: list[tuple[str, subprocess.Popen, object]] = []
    if args.mode in {"full", "merge", "prepare_merge"}:
        for index in range(max(int(args.merge_workers), 1)):
            merge_processes.append(
                _start_worker(
                    stage="merge",
                    worker_name=f"merge{index}",
                    log_path=log_dir / f"worldfm_decoupled_merge{index}.log",
                    env=env,
                    base_command=base_command,
                    gpu_id=None,
                    threads=merge_threads,
                    delete_after_final=False,
                )
            )

    ok = True
    if args.mode in {"full", "prepare", "prepare_merge"}:
        prepare_processes = [
            _start_worker(
                stage="prepare",
                worker_name=f"prepare{index}_gpu{gpu_id}",
                log_path=log_dir / f"worldfm_decoupled_prepare{index}_gpu{gpu_id}.log",
                env=env,
                base_command=base_command,
                gpu_id=gpu_id,
                threads=int(args.threads_per_gpu_worker),
                delete_after_final=False,
            )
            for index, gpu_id in enumerate(prepare_gpus)
        ]
        ok = _wait(prepare_processes) and ok

    if args.mode in {"full", "final"}:
        final_processes = [
            _start_worker(
                stage="final",
                worker_name=f"final{index}_gpu{gpu_id}",
                log_path=log_dir / f"worldfm_decoupled_final{index}_gpu{gpu_id}.log",
                env=env,
                base_command=base_command,
                gpu_id=gpu_id,
                threads=int(args.threads_per_gpu_worker),
                delete_after_final=bool(args.delete_after_final),
            )
            for index, gpu_id in enumerate(final_gpus)
        ]
        ok = _wait(final_processes) and ok

    if merge_processes:
        ok = _wait(merge_processes) and ok
    _summarize(output_dir, run_root, records)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
