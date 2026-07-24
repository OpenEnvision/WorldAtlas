"""Subprocess runner for HunyuanWorld inference inside the upstream model repo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from worldarena.common.checkpoints import apply_checkpoint_env, checkpoint_path, resolve_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Arena HunyuanWorld subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--conditioning_image", default=None, type=str)
    parser.add_argument("--annotation_path", default=None, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--classes", default="outdoor", type=str)
    parser.add_argument("--labels_fg1", nargs="*", default=None)
    parser.add_argument("--labels_fg2", nargs="*", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu_index", default=0, type=int)
    parser.add_argument("--fp8_attention", action="store_true")
    parser.add_argument("--fp8_gemm", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--export_drc", action="store_true")
    return parser.parse_args()


def _pythonpath_entries(repo_root: Path) -> list[str]:
    entries = [str(repo_root)]
    for dependency_name in ("Real-ESRGAN", "ZIM", "MoGe"):
        dependency_root = repo_root / dependency_name
        if dependency_root.exists():
            entries.append(str(dependency_root))
    return entries


def _build_env(repo_root: Path, *, gpu_index: int) -> dict[str, str]:
    env = apply_checkpoint_env()
    pythonpaths = _pythonpath_entries(repo_root)
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    if gpu_index >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def _normalize_labels(values: list[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _require_repo_layout(repo_root: Path) -> None:
    for required_file in ("demo_panogen.py", "demo_scenegen.py"):
        path = repo_root / required_file
        if not path.exists():
            raise FileNotFoundError(f"HunyuanWorld entrypoint not found: {path}")


def _stage_zim_checkpoint(repo_root: Path) -> Path:
    zim_source_root = repo_root / "ZIM"
    if not zim_source_root.exists():
        raise FileNotFoundError(f"HunyuanWorld dependency checkout not found: {zim_source_root}")

    checkpoint_dir = checkpoint_path("ZIM", "zim_vit_l_2092", kind="dir", required=True)
    target = zim_source_root / "zim_vit_l_2092"
    if target.is_symlink():
        if target.resolve() != checkpoint_dir.resolve():
            raise FileExistsError(f"HunyuanWorld ZIM checkpoint symlink points elsewhere: {target}")
        return target
    if target.exists():
        if target.resolve() != checkpoint_dir.resolve():
            raise FileExistsError(f"HunyuanWorld ZIM checkpoint path is occupied: {target}")
        return target
    target.symlink_to(checkpoint_dir, target_is_directory=True)
    return target


def _archive_directory(source_dir: Path, output_path: Path) -> list[str]:
    members: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(source_dir)
            archive.write(path, arcname=str(relative_path))
            members.append(str(relative_path))
    return members


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("HunyuanWorld runner requires checkpoint_dir")
    output_path = Path(args.output_path).expanduser().resolve()
    _require_repo_layout(repo_root)
    _stage_zim_checkpoint(repo_root)
    env = _build_env(repo_root, gpu_index=args.gpu_index)

    labels_fg1 = _normalize_labels(args.labels_fg1)
    labels_fg2 = _normalize_labels(args.labels_fg2)
    prompt = str(args.prompt).strip()

    conditioning_image = (
        Path(args.conditioning_image).expanduser().resolve()
        if args.conditioning_image
        else None
    )
    if conditioning_image is None and not prompt:
        raise ValueError("HunyuanWorld runner requires either --conditioning_image or a non-empty --prompt")

    with tempfile.TemporaryDirectory(prefix="worldarena_hunyuanworld_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        work_dir = temp_dir / "output"
        work_dir.mkdir(parents=True, exist_ok=True)

        local_conditioning_path: Path | None = None
        if conditioning_image is not None:
            suffix = conditioning_image.suffix or ".png"
            local_conditioning_path = temp_dir / f"conditioning{suffix}"
            shutil.copy2(conditioning_image, local_conditioning_path)

        panogen_command = [
            sys.executable,
            str((repo_root / "demo_panogen.py").resolve()),
            "--prompt",
            prompt,
            "--negative_prompt",
            str(args.negative_prompt),
            "--seed",
            str(args.seed),
            "--output_path",
            str(work_dir),
        ]
        if local_conditioning_path is not None:
            panogen_command.extend(["--image_path", str(local_conditioning_path)])
        for flag in ("fp8_attention", "fp8_gemm", "cache"):
            if getattr(args, flag):
                panogen_command.append(f"--{flag}")

        subprocess.run(
            panogen_command,
            check=True,
            cwd=str(repo_root),
            env=env,
        )

        panorama_path = work_dir / "panorama.png"
        if not panorama_path.exists():
            raise FileNotFoundError(f"HunyuanWorld panorama output was not written: {panorama_path}")

        scenegen_command = [
            sys.executable,
            str((repo_root / "demo_scenegen.py").resolve()),
            "--image_path",
            str(panorama_path),
            "--classes",
            str(args.classes),
            "--seed",
            str(args.seed),
            "--output_path",
            str(work_dir),
            "--export_drc",
            "True" if args.export_drc else "False",
        ]
        if labels_fg1:
            scenegen_command.extend(["--labels_fg1", *labels_fg1])
        if labels_fg2:
            scenegen_command.extend(["--labels_fg2", *labels_fg2])
        for flag in ("fp8_attention", "fp8_gemm", "cache"):
            if getattr(args, flag):
                scenegen_command.append(f"--{flag}")

        subprocess.run(
            scenegen_command,
            check=True,
            cwd=str(repo_root),
            env=env,
        )

        mesh_paths = sorted(work_dir.glob("mesh_layer*.ply"))
        if not mesh_paths:
            raise FileNotFoundError(f"HunyuanWorld mesh outputs were not written under: {work_dir}")

        manifest_path = work_dir / "worldarena_hunyuanworld_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "negative_prompt": str(args.negative_prompt),
                    "classes": str(args.classes),
                    "labels_fg1": labels_fg1,
                    "labels_fg2": labels_fg2,
                    "annotation_path": args.annotation_path,
                    "conditioning_image": str(conditioning_image) if conditioning_image else None,
                    "input_mode": "image_to_world" if conditioning_image is not None else "text_to_world",
                    "mesh_files": [path.name for path in mesh_paths],
                    "export_drc": bool(args.export_drc),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        archive_members = _archive_directory(work_dir, output_path)
        if not archive_members:
            raise FileNotFoundError(f"HunyuanWorld archive is empty: {output_path}")


if __name__ == "__main__":
    main()
