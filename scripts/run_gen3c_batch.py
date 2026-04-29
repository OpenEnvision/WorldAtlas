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
    parser = argparse.ArgumentParser(description="Batch GEN3C generation runner for one benchmark shard.")
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
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_devices)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _python_env_root(python_bin: str) -> Path | None:
    path = Path(python_bin).expanduser()
    if not path.is_absolute() or path.parent.name != "bin":
        return None
    return path.parent.parent


def _build_persistent_command(
    *,
    model_config,
    generation: dict[str, Any],
    checkpoints_root: Path,
    batch_input_path: Path,
) -> list[str]:
    num_gpus = int(generation.get("num_gpus", 1))
    if num_gpus != 1:
        raise ValueError("GEN3C persistent batch runner currently expects generation.num_gpus=1")
    command = [model_config.python_bin]
    command.extend(
        [
            "-m",
            "worldarena.models.adapters.gen3c_persistent_runner",
            "--checkpoint_dir",
            str(checkpoints_root),
            "--batch_input_path",
            str(batch_input_path),
            "--num_video_frames",
            str(int(generation.get("num_video_frames", 121))),
            "--height",
            str(int(generation.get("height", 704))),
            "--width",
            str(int(generation.get("width", 1280))),
            "--fps",
            str(int(generation.get("fps", 24))),
            "--seed",
            str(int(generation.get("seed", 42))),
            "--guidance",
            str(float(generation.get("guidance", 1.0))),
            "--num_steps",
            str(int(generation.get("num_steps", 35))),
            "--trajectory",
            str(generation.get("trajectory", "left")),
            "--camera_rotation",
            str(generation.get("camera_rotation", "center_facing")),
            "--movement_distance",
            str(float(generation.get("movement_distance", 0.3))),
            "--filter_points_threshold",
            str(float(generation.get("filter_points_threshold", 0.05))),
        ]
    )
    if generation.get("negative_prompt"):
        command.extend(["--negative_prompt", str(generation["negative_prompt"])])
    if generation.get("prompt_upsampler_dir"):
        command.extend(["--prompt_upsampler_dir", str(generation["prompt_upsampler_dir"])])
    for flag_name in (
        "save_buffer",
        "foreground_masking",
        "offload_diffusion_transformer",
        "offload_tokenizer",
        "offload_text_encoder_model",
        "offload_prompt_upsampler",
        "offload_guardrail_models",
        "disable_prompt_encoder",
    ):
        if generation.get(flag_name, False):
            command.append(f"--{flag_name}")
    return command


def main() -> None:
    args = parse_args()
    _prepare_cuda_visibility(args)

    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldarena.benchmark.manifest import load_manifest
    from worldarena.common.checkpoints import filter_checkpoint_env_overrides, hf_local_dir
    from worldarena.generation.runner import _conditioning_asset_for, _select_samples
    from worldarena.models.adapters.gen3c import Gen3CAdapter, _generation_for_sample
    from worldarena.models.adapters.gen3c_runner import (
        _build_env,
        _checkpoint_layout,
        _link_repo_alias,
        _prepare_checkpoint_root,
        _require_repo_layout,
    )
    from worldarena.models.config import load_model_config

    model_config = load_model_config(args.model_config.resolve())
    if model_config.family.lower() != "gen3c":
        raise ValueError(f"unsupported model family for this runner: {model_config.family}")
    if model_config.repo_root is None or model_config.checkpoint_dir is None:
        raise ValueError("GEN3C batch runner requires repo_root and checkpoint_dir")

    manifest = load_manifest(args.manifest.resolve())
    selected = _select_samples(
        manifest,
        suites=_split_arg(args.suites),
        limit=args.limit,
        supported_suites=set(model_config.supported_suites),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning_root = (output_dir / f"_conditioning_shard{args.shard_index}").resolve()
    conditioning_root.mkdir(parents=True, exist_ok=True)
    work_root = (output_dir / f"_gen3c_work_shard{args.shard_index}").resolve()
    workspace_root = work_root / "workspace"

    records_path = output_dir / f"generation_records.shard{args.shard_index}.jsonl"
    summary_path = output_dir / f"generation_manifest.shard{args.shard_index}.json"
    prompt_map_path = output_dir / f"prompt_map.shard{args.shard_index}.json"

    adapter = Gen3CAdapter(model_config)
    pending: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    prompt_map: dict[str, str] = {}
    base_generation = dict(model_config.generation)

    for offset, sample in enumerate(selected, start=1):
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        prompt = adapter.prompt_for(sample)
        sample_generation = _generation_for_sample(base_generation, sample)
        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(prediction_path),
            "conditioning_image": None,
            "conditioning_input": None,
            "prompt": prompt,
            "status": "pending",
            "error": None,
            "shard_index": args.shard_index,
            "shard_position": offset,
            "shard_total": len(selected),
            "camera_path": sample_generation.get("camera_path"),
            "camera_path_primary": sample_generation.get("camera_path_primary"),
            "trajectory": sample_generation.get("trajectory"),
            "camera_rotation": sample_generation.get("camera_rotation"),
        }
        prompt_map[str(prediction_path)] = prompt
        records.append(record)

        if prediction_path.exists() and not args.overwrite:
            record["status"] = "skipped_existing"
            continue
        if prediction_path.exists() and args.overwrite:
            prediction_path.unlink()

        try:
            conditioning_image = _conditioning_asset_for(
                sample,
                reference_mode=model_config.reference_mode,
                cache_root=conditioning_root,
                min_video_frames=int(sample_generation.get("conditioning_video_frames", 8)),
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            continue

        record["conditioning_image"] = str(conditioning_image)
        record["conditioning_input"] = str(conditioning_image)
        pending.append(
            {
                "sample": sample,
                "record": record,
                "prediction_path": prediction_path,
                "conditioning_image": conditioning_image,
                "prompt": prompt,
                "generation": sample_generation,
            }
        )

    if work_root.exists() and pending:
        shutil.rmtree(work_root, ignore_errors=True)
    workspace_root.mkdir(parents=True, exist_ok=True)

    if pending:
        repo_root = model_config.repo_root.resolve()
        _require_repo_layout(repo_root)
        checkpoint_layout = _checkpoint_layout(model_config.checkpoint_dir)
        checkpoints_root = _prepare_checkpoint_root(
            work_root,
            checkpoint_layout,
        )
        moge_checkpoint = hf_local_dir("Ruicheng/moge-vitl", required=True) / "model.pt"
        if not moge_checkpoint.is_file():
            raise FileNotFoundError(f"GEN3C MoGe checkpoint not found: {moge_checkpoint}")
        _link_repo_alias(workspace_root, "Ruicheng/moge-vitl", moge_checkpoint)

        env = _build_env(repo_root, workspace_root)
        env.update(filter_checkpoint_env_overrides(model_config.env))
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(project_root)]
        )
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env_root = _python_env_root(model_config.python_bin)
        if env_root is not None:
            env.setdefault("CONDA_PREFIX", str(env_root))
            env.setdefault("CUDA_HOME", str(env_root))

        batch_input_path = work_root / "batch_inputs.jsonl"
        error_root = work_root / "errors"
        error_root.mkdir(parents=True, exist_ok=True)
        with batch_input_path.open("w", encoding="utf-8") as handle:
            for index, item in enumerate(pending):
                generation = dict(item["generation"])
                payload = {
                    "prompt": item["prompt"],
                    "visual_input": str(Path(item["conditioning_image"]).expanduser().resolve()),
                    "output_path": str(Path(item["prediction_path"]).expanduser().resolve()),
                    "error_path": str(error_root / f"{index}.txt"),
                    "trajectory": generation.get("trajectory"),
                    "camera_rotation": generation.get("camera_rotation"),
                    "movement_distance": generation.get("movement_distance"),
                    "filter_points_threshold": generation.get("filter_points_threshold"),
                    "foreground_masking": generation.get("foreground_masking", False),
                    "negative_prompt": generation.get("negative_prompt"),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        command = _build_persistent_command(
            model_config=model_config,
            generation=base_generation,
            checkpoints_root=checkpoints_root,
            batch_input_path=batch_input_path,
        )
        batch_error: str | None = None
        try:
            subprocess.run(command, check=True, cwd=str(workspace_root), env=env)
        except subprocess.CalledProcessError as exc:
            batch_error = f"exit code {exc.returncode}"

        for index, item in enumerate(pending):
            record = item["record"]
            prediction_path = Path(item["prediction_path"])
            error_path = error_root / f"{index}.txt"
            record["command"] = command
            if prediction_path.exists():
                record["status"] = "generated"
                record["prediction_path"] = str(prediction_path)
            else:
                record["status"] = "failed"
                record["error"] = error_path.read_text(encoding="utf-8") if error_path.exists() else batch_error
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
        "supported_suites": model_config.supported_suites,
    }
    _write_json(summary_path, summary)
    _write_jsonl(records_path, records)
    _write_json(prompt_map_path, prompt_map)

    if not args.keep_workdir and pending:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
