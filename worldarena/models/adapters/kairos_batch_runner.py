from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    load_batch_spec,
    load_json,
    print_status,
)


DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards, contorted human joints, objects "
    "floating against natural forces, abrupt shot changes"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Kairos Sensenova batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: Any, *, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base is not None:
        return (base / path).resolve()
    return path.resolve()


def _default_config_path(repo_root: Path, generation: dict[str, Any]) -> Path:
    variant = str(generation.get("variant", "robot-480p-distilled")).lower()
    name = "kairos_4b_config.py" if variant in {"720p", "4b-720p", "pretrained"} else "kairos_4b_config_DMD.py"
    return repo_root / "kairos" / "configs" / name


def _patched_config_path(
    *,
    repo_root: Path,
    models_root: Path,
    generation: dict[str, Any],
    spec_dir: Path,
) -> Path:
    source = _resolve_path(generation.get("config_path"), base=repo_root) or _default_config_path(repo_root, generation)
    text = source.read_text(encoding="utf-8")
    models_root_text = str(models_root.resolve())
    text = text.replace("KAIROS_MODEL_DIR = 'models'", f"KAIROS_MODEL_DIR = {models_root_text!r}")
    text = text.replace('KAIROS_MODEL_DIR = "models"', f"KAIROS_MODEL_DIR = {models_root_text!r}")
    overrides = ["", "# WorldArena runtime overrides"]
    if generation.get("pretrained_dit"):
        overrides.append(f"pipeline['pretrained_dit'] = {str(generation['pretrained_dit'])!r}")
    if generation.get("text_encoder_path"):
        overrides.append(f"pipeline['pipeline_args']['text_encoder_path'] = {str(generation['text_encoder_path'])!r}")
    if generation.get("vae_path"):
        overrides.append(f"pipeline['pipeline_args']['vae_path'] = {str(generation['vae_path'])!r}")
    if generation.get("prompt_rewriter_path"):
        overrides.append(f"prompt_rewriter_path = {str(generation['prompt_rewriter_path'])!r}")
    target = spec_dir / "kairos_config.py"
    target.write_text(text + "\n".join(overrides) + "\n", encoding="utf-8")
    return target


def _input_json_for_row(
    *,
    row: dict[str, Any],
    generation: dict[str, Any],
    output_dir: Path,
    spec_dir: Path,
    row_index: int,
) -> Path:
    input_image = ""
    configured_image = row.get("input_image") or row.get("conditioning_image")
    if configured_image and Path(str(configured_image)).is_file() and not _as_bool(generation.get("force_t2v", False)):
        input_image = str(Path(str(configured_image)).expanduser().resolve())
    payload = {
        "prompt": str(row.get("prompt") or generation.get("prompt") or ""),
        "input_image": input_image,
        "negative_prompt": str(generation.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT),
        "seed": int(generation.get("seed", 0)) + row_index,
        "tiled": _as_bool(generation.get("tiled", True)),
        "height": int(generation.get("height", 480)),
        "width": int(generation.get("width", 640)),
        "num_frames": int(generation.get("num_frames", 81)),
        "cfg_scale": float(generation.get("cfg_scale", 5)),
        "use_prompt_rewriter": _as_bool(generation.get("use_prompt_rewriter", False)),
        "output_dir": str(output_dir),
    }
    for key in ("tea_cache_l1_thresh", "shift", "sample_solver"):
        if key in generation and generation[key] is not None:
            payload[key] = generation[key]
    path = spec_dir / f"kairos_input_{row_index:05d}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _run_manage_libs(repo_root: Path, generation: dict[str, Any], env: dict[str, str]) -> None:
    if not _as_bool(generation.get("run_manage_libs", True)):
        return
    python_executable = str(generation.get("python_executable") or sys.executable)
    subprocess.run(
        [python_executable, str(repo_root / "kairos" / "third_party" / "manage_libs.py")],
        check=True,
        cwd=str(repo_root),
        env=env,
    )


def _run_kairos_command(
    *,
    repo_root: Path,
    input_json: Path,
    config_path: Path,
    generation: dict[str, Any],
    env: dict[str, str],
    log_path: Path,
) -> None:
    command = [
        str(generation.get("torchrun_executable") or "torchrun"),
        "--nnodes=1",
        "--master_port",
        str(int(generation.get("master_port", 29556))),
        "--nproc-per-node",
        str(int(generation.get("nproc_per_node", 1))),
        str((repo_root / "examples" / "inference.py").resolve()),
        "--input_file",
        str(input_json),
        "--config_file",
        str(config_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            check=True,
            cwd=str(repo_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    models_root = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else repo_root / "models"
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    spec_dir = Path(args.batch_spec_path).expanduser().resolve().parent
    config_path = _patched_config_path(
        repo_root=repo_root,
        models_root=models_root,
        generation=generation,
        spec_dir=spec_dir,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")])
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")

    failed = 0
    try:
        _run_manage_libs(repo_root, generation, env)
    except Exception as exc:
        for row in rows:
            print_status(str(row.get("sample_id", "")), "failed", error=str(exc))
        return 1

    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        output_path = Path(str(row["output_path"])).expanduser().resolve()
        raw_dir = output_path.parent / "_kairos_raw" / output_path.stem
        raw_dir.mkdir(parents=True, exist_ok=True)
        input_json = _input_json_for_row(
            row=row,
            generation=generation,
            output_dir=raw_dir,
            spec_dir=spec_dir,
            row_index=row_index,
        )
        log_path = spec_dir / f"kairos_{row_index:05d}.log"
        try:
            _run_kairos_command(
                repo_root=repo_root,
                input_json=input_json,
                config_path=config_path,
                generation=generation,
                env=env,
                log_path=log_path,
            )
            generated = raw_dir / "output.mp4"
            if not generated.is_file():
                generated = sorted(raw_dir.rglob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)[0]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generated, output_path)
            print_status(sample_id, "generated", output_path=str(output_path), raw_output_path=str(generated))
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc(), log_path=str(log_path))
    return 1 if failed == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
