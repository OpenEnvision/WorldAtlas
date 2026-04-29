from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent FlashWorld batch generation runner.")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suites", default=None, type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--visible-devices", default=None, type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_cli_command(
    *,
    model_config,
    repo_root: Path,
    checkpoint_path: Path | None,
) -> list[str]:
    cli_path = (repo_root / "cli.py").resolve()
    if not cli_path.exists():
        raise FileNotFoundError(f"FlashWorld CLI entrypoint not found: {cli_path}")

    generation = model_config.generation
    command = [
        model_config.python_bin,
        str(cli_path),
    ]
    if checkpoint_path is not None:
        command.extend(["--ckpt", str(checkpoint_path)])
    for flag_name in (
        "offload_t5",
        "offload_vae",
        "offload_transformer_during_vae",
        "export_ply",
        "export_spz",
    ):
        if generation.get(flag_name, False):
            command.append(f"--{flag_name}")
    return command


def _subprocess_error_text(exc: subprocess.CalledProcessError) -> str:
    return f"exit code {exc.returncode}"


def main() -> None:
    args = parse_args()
    _prepare_cuda_visibility(args)

    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.benchmark.manifest import load_manifest
    from worldarena.generation.runner import _conditioning_asset_for, _select_samples
    from worldarena.models.adapters.common import build_env
    from worldarena.models.adapters.flashworld import FlashWorldAdapter, _resolved_annotation_path
    from worldarena.models.adapters.flashworld_runner import _build_request_payload, _resolve_checkpoint_path
    from worldarena.models.config import load_model_config

    model_config = load_model_config(args.model_config.resolve())
    if model_config.family.lower() not in {"flashworld", "flash_world", "flash-world"}:
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")
    if model_config.repo_root is None:
        raise ValueError("flashworld runner requires repo_root")

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
    work_root = (output_dir / f"_flashworld_work_shard{args.shard_index}").resolve()
    input_dir = work_root / "inputs"
    runtime_output_dir = work_root / "outputs"

    records_path = output_dir / f"generation_records.shard{args.shard_index}.jsonl"
    summary_path = output_dir / f"generation_manifest.shard{args.shard_index}.json"
    prompt_map_path = output_dir / f"prompt_map.shard{args.shard_index}.json"

    adapter = FlashWorldAdapter(model_config)
    repo_root = model_config.repo_root.resolve()
    generation = dict(model_config.generation)
    checkpoint_path = _resolve_checkpoint_path(str(model_config.checkpoint_dir)) if model_config.checkpoint_dir else None
    frame_count = int(generation.get("num_frames", 24))

    pending: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    prompt_map: dict[str, str] = {}

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
            "conditioning_input": str(conditioning_image),
            "prompt": prompt,
            "status": "pending",
            "error": None,
            "shard_index": args.shard_index,
            "shard_position": offset,
            "shard_total": len(selected),
        }
        prompt_map[str(prediction_path)] = prompt
        if prediction_path.exists() and not args.overwrite:
            record["status"] = "skipped_existing"
            records.append(record)
            continue

        annotation_path = _resolved_annotation_path(
            model_config,
            sample,
            conditioning_image.expanduser().resolve(),
            frame_count=frame_count,
        )
        payload = _build_request_payload(
            annotation_path=str(annotation_path),
            conditioning_image=conditioning_image.expanduser().resolve(),
            prompt=prompt,
            num_frames=frame_count,
            image_height=int(generation.get("image_height", 480)),
            image_width=int(generation.get("image_width", 704)),
            image_index=int(generation.get("image_index", 0)),
        )
        pending.append(
            {
                "sample": sample,
                "record": record,
                "payload": payload,
                "prediction_path": prediction_path,
            }
        )

    if records_path.exists():
        records_path.unlink()
    if summary_path.exists():
        summary_path.unlink()
    if prompt_map_path.exists():
        prompt_map_path.unlink()

    cli_error: str | None = None
    if pending:
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        runtime_output_dir.mkdir(parents=True, exist_ok=True)

        for item in pending:
            sample = item["sample"]
            request_path = input_dir / f"{sample.prediction_stem}.json"
            _write_json(request_path, item["payload"])

        command = _resolve_cli_command(
            model_config=model_config,
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
        )
        command.extend(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(runtime_output_dir),
                "--video",
                "--video_fps",
                str(int(generation.get("video_fps", 15))),
            ]
        )

        env = build_env(
            extra_pythonpaths=[project_root],
            overrides=model_config.env,
        )
        try:
            subprocess.run(
                command,
                check=True,
                cwd=str(repo_root),
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            cli_error = _subprocess_error_text(exc)

        for item in pending:
            sample = item["sample"]
            record = item["record"]
            prediction_path = item["prediction_path"]
            generated_root = runtime_output_dir / sample.prediction_stem
            generated_video = generated_root / "video.mp4"
            if generated_video.exists():
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                prediction_path.unlink(missing_ok=True)
                shutil.move(str(generated_video), str(prediction_path))
                record["status"] = "generated"
                record["command"] = command
                if generation.get("export_ply", False):
                    generated_ply = generated_root / "gaussians.ply"
                    if generated_ply.exists():
                        prediction_path.with_suffix(".ply").unlink(missing_ok=True)
                        shutil.move(str(generated_ply), str(prediction_path.with_suffix(".ply")))
                        record["ply_path"] = str(prediction_path.with_suffix(".ply"))
                if generation.get("export_spz", False):
                    generated_spz = generated_root / "gaussians.spz"
                    if generated_spz.exists():
                        prediction_path.with_suffix(".spz").unlink(missing_ok=True)
                        shutil.move(str(generated_spz), str(prediction_path.with_suffix(".spz")))
                        record["spz_path"] = str(prediction_path.with_suffix(".spz"))
            else:
                record["status"] = "failed"
                record["error"] = cli_error or f"FlashWorld output was not written: {generated_video}"
            records.append(record)

        if not args.keep_workdir:
            shutil.rmtree(work_root, ignore_errors=True)

    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    prompt_map_path.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")

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
    if cli_error:
        summary["cli_error"] = cli_error
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
