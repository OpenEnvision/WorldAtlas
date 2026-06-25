from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any

import yaml

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena LongLive batch runner.")
    add_common_batch_args(parser)
    parser.add_argument("--entrypoint", default="inference.py", type=str)
    return parser.parse_args()


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "ref"


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _materialize_git_ref(repo_root: Path, git_ref: str, cache_root: Path) -> Path:
    commit = _git_output(repo_root, "rev-parse", git_ref)
    destination = cache_root / f"{repo_root.name}_{_safe_token(git_ref)}_{commit[:12]}"
    marker = destination / ".worldarena_git_ref"
    if (
        destination.is_dir()
        and marker.is_file()
        and marker.read_text(encoding="utf-8", errors="ignore") == commit
    ):
        return destination

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_root / f".tmp_{destination.name}_{os.getpid()}"
    archive_path = cache_root / f"{temporary.name}.tar"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "archive",
                "--format=tar",
                "--output",
                str(archive_path),
                git_ref,
            ],
            check=True,
        )
        subprocess.run(["tar", "-xf", str(archive_path), "-C", str(temporary)], check=True)
        (temporary / ".worldarena_git_ref").write_text(commit, encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    finally:
        archive_path.unlink(missing_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return destination


def _resolve_source_root(repo_root: Path, generation: dict[str, Any], work_dir: Path) -> Path:
    explicit_root = generation.get("source_root")
    if explicit_root:
        path = Path(str(explicit_root)).expanduser()
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    git_ref = str(generation.get("git_ref", "") or "").strip()
    if not git_ref:
        return repo_root

    materialized_root = generation.get("materialized_repo_root")
    if materialized_root:
        path = Path(str(materialized_root)).expanduser()
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    cache_root = Path(str(generation.get("repo_cache_dir") or work_dir / "_longlive_sources"))
    cache_root = cache_root.expanduser().resolve()
    return _materialize_git_ref(repo_root, git_ref, cache_root)


def _load_official_config(source_root: Path, generation: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    config_name = str(generation.get("config_path") or "configs/inference.yaml")
    config_path = Path(config_name).expanduser()
    if not config_path.is_absolute():
        config_path = source_root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"LongLive official config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"LongLive config must be a mapping: {config_path}")
    return config_path, payload


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _write_prompt_file(rows: list[dict[str, Any]], work_dir: Path, generation: dict[str, Any]) -> Path:
    prompt_format = str(generation.get("prompt_file_format", "txt")).strip().lower()
    if prompt_format == "jsonl":
        prompt_path = work_dir / "longlive_prompts.jsonl"
        with prompt_path.open("w", encoding="utf-8") as file:
            for row in rows:
                prompt_sequence = row.get("prompt_sequence") or [str(row["prompt"])]
                file.write(json.dumps({"prompts": prompt_sequence}, ensure_ascii=False) + "\n")
        return prompt_path

    prompt_path = work_dir / "longlive_prompts.txt"
    with prompt_path.open("w", encoding="utf-8") as file:
        for row in rows:
            prompt = str(row["prompt"]).replace("\r", " ").replace("\n", " ").strip()
            file.write(prompt + "\n")
    return prompt_path


def _runtime_config_for_batch(
    *,
    official_config: dict[str, Any],
    rows: list[dict[str, Any]],
    prompt_path: Path,
    output_folder: Path,
    generation: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(official_config))
    config_overrides = generation.get("config_overrides") or {}
    if not isinstance(config_overrides, dict):
        raise ValueError("LongLive generation.config_overrides must be a mapping")
    _deep_merge(payload, dict(config_overrides))

    version = str(generation.get("version", "")).strip()
    is_v1 = version.startswith("1") or "data_path" in payload
    if is_v1:
        payload["data_path"] = str(prompt_path)
    else:
        payload.setdefault("data", {})
        if not isinstance(payload["data"], dict):
            raise ValueError("LongLive config data section must be a mapping")
        payload["data"]["data_path"] = str(prompt_path)

    payload["output_folder"] = str(output_folder)
    if isinstance(payload.get("inference"), dict) and "output_folder" in payload["inference"]:
        payload["inference"]["output_folder"] = str(output_folder)
    payload["save_with_index"] = True
    payload["inference_iter"] = len(rows) - 1
    payload["save_latents_only"] = False
    if isinstance(payload.get("inference"), dict) and "save_latents_only" in payload["inference"]:
        payload["inference"]["save_latents_only"] = False
    return payload


def _write_runtime_config(
    *,
    official_config: dict[str, Any],
    rows: list[dict[str, Any]],
    prompt_path: Path,
    output_folder: Path,
    generation: dict[str, Any],
    work_dir: Path,
) -> Path:
    runtime_config = _runtime_config_for_batch(
        official_config=official_config,
        rows=rows,
        prompt_path=prompt_path,
        output_folder=output_folder,
        generation=generation,
    )
    runtime_config_path = work_dir / "longlive_worldarena_config.yaml"
    runtime_config_path.write_text(
        yaml.safe_dump(runtime_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return runtime_config_path


def _torchrun_path(python_bin: str) -> str:
    candidate = Path(python_bin).expanduser().with_name("torchrun")
    return str(candidate) if candidate.exists() else "torchrun"


def _build_upstream_command(
    *,
    python_bin: str,
    entrypoint: str,
    runtime_config_path: Path,
    generation: dict[str, Any],
) -> list[str]:
    nproc_per_node = int(generation.get("nproc_per_node", 1) or 1)
    use_torchrun = bool(generation.get("use_torchrun", False)) or nproc_per_node > 1
    if use_torchrun:
        command = [
            _torchrun_path(python_bin),
            "--nproc_per_node",
            str(nproc_per_node),
        ]
        master_port = generation.get("master_port")
        if master_port is not None:
            command.extend(["--master_port", str(int(master_port))])
        command.append(entrypoint)
    else:
        command = [python_bin, entrypoint]
    command.extend(["--config_path", str(runtime_config_path)])
    return command


def _generated_video_for(output_folder: Path, row_index: int) -> Path:
    patterns = [
        f"rank*-{row_index}-0_*.mp4",
        f"rank*-{row_index}-*.mp4",
        f"*{row_index}*.mp4",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in output_folder.glob(pattern) if path.is_file())
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"LongLive did not write an mp4 for batch index {row_index}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _copy_outputs(rows: list[dict[str, Any]], output_folder: Path) -> int:
    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            generated_path = _generated_video_for(output_folder, row_index)
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            copy_output(generated_path, output_path)
            print_status(sample_id, "generated", output_path=str(output_path), source_path=str(generated_path))
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return failed


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    work_dir = Path(args.batch_spec_path).expanduser().resolve().parent
    source_root = _resolve_source_root(repo_root, generation, work_dir)
    official_config_path, official_config = _load_official_config(source_root, generation)
    prompt_path = _write_prompt_file(rows, work_dir, generation)
    output_folder = (work_dir / "longlive_outputs").resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    runtime_config_path = _write_runtime_config(
        official_config=official_config,
        rows=rows,
        prompt_path=prompt_path,
        output_folder=output_folder,
        generation=generation,
        work_dir=work_dir,
    )
    entrypoint = str(generation.get("model_entrypoint") or args.entrypoint)
    command = _build_upstream_command(
        python_bin=sys.executable,
        entrypoint=entrypoint,
        runtime_config_path=runtime_config_path,
        generation=generation,
    )

    if os.environ.get("WORLDARENA_LONGLIVE_DRY_RUN") == "1":
        print_status(
            "longlive",
            "dry_run",
            dry_run=True,
            rows=len(rows),
            source_root=str(source_root),
            official_config_path=str(official_config_path),
            runtime_config_path=str(runtime_config_path),
            prompt_path=str(prompt_path),
            output_folder=str(output_folder),
            command=command,
        )
        return 0

    subprocess.run(command, cwd=str(source_root), check=True)
    return 1 if _copy_outputs(rows, output_folder) else 0


if __name__ == "__main__":
    raise SystemExit(main())
