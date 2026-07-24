"""Subprocess runner for InSpatio-World inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InSpatio-World subprocess runner for WorldAtlas Arena.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--input_video", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--traj_txt_path", default="traj/x_y_circle_cycle.txt", type=str)
    parser.add_argument("--config_path", default="configs/inference_1.3b.yaml", type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--da3_repo_root", default="thirdparty/Depth-Anything-3", type=str)
    parser.add_argument("--da3_model_path", default=None, type=str)
    parser.add_argument("--step2_gpus", default="0", type=str)
    parser.add_argument("--step3_gpus", default="0", type=str)
    parser.add_argument("--step3_nproc", default=1, type=int)
    parser.add_argument("--master_port", default=29513, type=int)
    parser.add_argument("--relative_to_source", default=False, type=_parse_bool)
    parser.add_argument("--rotation_only", default=False, type=_parse_bool)
    parser.add_argument("--adaptive_frame", default=True, type=_parse_bool)
    parser.add_argument("--freeze_repeat", default=0, type=int)
    parser.add_argument("--freeze_frame", default=None, type=int)
    parser.add_argument("--use_tae", default=False, type=_parse_bool)
    parser.add_argument("--tae_checkpoint_path", default=None, type=str)
    parser.add_argument("--compile_dit", default=False, type=_parse_bool)
    parser.add_argument("--radius_ratio", default=1.0, type=float)
    parser.add_argument("--keep_work_dir", default=False, type=_parse_bool)
    return parser.parse_args()


def _resolve_repo_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_checkpoint_path(checkpoint_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    token = str(value)
    if token.startswith(("ckpt/", "./", "../", "/", "~")):
        return resolve_checkpoint_path(token)
    resolved = (checkpoint_root / Path(token)).resolve()
    if not resolved.is_relative_to(checkpoint_root):
        raise ValueError(f"InSpatio-World checkpoint path must stay under {checkpoint_root}: {resolved}")
    return resolved


def _pythonpath_entries_for_da3(repo_root: Path, value: str | None) -> list[Path]:
    if not value:
        return []
    da3_root = _resolve_repo_path(repo_root, value)
    if da3_root is None or not da3_root.exists():
        raise FileNotFoundError(f"Depth-Anything-3 source root not found: {da3_root}")
    entries = []
    src_root = da3_root / "src"
    if src_root.exists():
        entries.append(src_root)
    entries.append(da3_root)
    return entries


def _prepare_single_video_input(
    *,
    work_dir: Path,
    input_video: Path,
    prompt: str,
    radius_ratio: float,
) -> tuple[Path, Path]:
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    video_path = input_dir / f"source{input_video.suffix or '.mp4'}"
    try:
        video_path.symlink_to(input_video)
    except OSError:
        shutil.copy2(input_video, video_path)

    vggt_depth_path = input_dir / "new_vggt" / video_path.stem
    payload = [
        {
            "video_path": str(video_path),
            "vggt_depth_path": str(vggt_depth_path),
            "vggt_extrinsics_path": str(vggt_depth_path / "extrinsics.txt"),
            "radius_ratio": float(radius_ratio),
            "text": prompt,
        }
    ]
    json_path = input_dir / "new.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return input_dir, json_path


def _find_prediction_video(output_root: Path) -> Path:
    candidates = sorted(output_root.rglob("*-pred_video_rank*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"no InSpatio-World prediction video found under: {output_root}")
    if len(candidates) > 1:
        candidate_list = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"multiple InSpatio-World prediction videos found: {candidate_list}")
    return candidates[0]


def _script_env(*, extra_pythonpaths: list[Path] | None = None) -> dict[str, str]:
    env = apply_checkpoint_env()
    runtime_bin = Path(os.path.abspath(sys.executable)).parent
    path_entries = [str(runtime_bin)]
    if env.get("PATH"):
        path_entries.append(env["PATH"])
    env["PATH"] = os.pathsep.join(path_entries)
    pythonpath_entries = [str(path) for path in (extra_pythonpaths or []) if str(path)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    if pythonpath_entries:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_root = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_root is None:
        raise ValueError("InSpatio-World runner requires checkpoint_dir")
    input_video = Path(args.input_video).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        raise FileNotFoundError(f"InSpatio-World input video not found: {input_video}")

    pipeline_script = (repo_root / "run_test_pipeline.sh").resolve()
    if not pipeline_script.exists():
        raise FileNotFoundError(f"InSpatio-World pipeline script not found: {pipeline_script}")

    checkpoint_path = _resolve_checkpoint_path(
        checkpoint_root,
        args.checkpoint_path or "InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors",
    )
    config_path = _resolve_repo_path(repo_root, args.config_path)
    traj_txt_path = _resolve_repo_path(repo_root, args.traj_txt_path)
    da3_pythonpaths = _pythonpath_entries_for_da3(repo_root, args.da3_repo_root)
    da3_model_path = _resolve_checkpoint_path(
        checkpoint_root,
        args.da3_model_path or "DA3",
    )
    tae_checkpoint_path = _resolve_checkpoint_path(
        checkpoint_root,
        args.tae_checkpoint_path or "taehv/taew2_1.pth",
    )

    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"InSpatio-World checkpoint not found: {checkpoint_path}")
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"InSpatio-World config not found: {config_path}")
    if traj_txt_path is None or not traj_txt_path.exists():
        raise FileNotFoundError(f"InSpatio-World trajectory file not found: {traj_txt_path}")
    if da3_model_path is None or not da3_model_path.exists():
        raise FileNotFoundError(f"InSpatio-World DA3 model path not found: {da3_model_path}")
    if args.use_tae and (tae_checkpoint_path is None or not tae_checkpoint_path.exists()):
        raise FileNotFoundError(f"InSpatio-World TAE checkpoint not found: {tae_checkpoint_path}")

    with tempfile.TemporaryDirectory(prefix="worldarena_inspatio_") as temp_dir_raw:
        work_dir = Path(temp_dir_raw)
        input_dir, _ = _prepare_single_video_input(
            work_dir=work_dir,
            input_video=input_video,
            prompt=args.prompt,
            radius_ratio=args.radius_ratio,
        )
        output_root = work_dir / "output"

        command = [
            "bash",
            str(pipeline_script),
            "--input_dir",
            str(input_dir),
            "--traj_txt_path",
            str(traj_txt_path),
            "--checkpoint_path",
            str(checkpoint_path),
            "--config_path",
            str(config_path),
            "--da3_model_path",
            str(da3_model_path),
            "--step2_gpus",
            str(args.step2_gpus),
            "--step3_gpus",
            str(args.step3_gpus),
            "--step3_nproc",
            str(args.step3_nproc),
            "--output_folder",
            str(output_root),
            "--master_port",
            str(args.master_port),
            "--skip_step1",
        ]
        if args.relative_to_source:
            command.append("--relative_to_source")
        if args.rotation_only:
            command.append("--rotation_only")
        if not args.adaptive_frame:
            command.append("--disable_adaptive_frame")
        if args.freeze_repeat > 0:
            command.extend(["--freeze_repeat", str(args.freeze_repeat)])
        if args.freeze_frame is not None:
            command.extend(["--freeze_frame", str(args.freeze_frame)])
        if args.use_tae:
            command.append("--use_tae")
            command.extend(["--tae_checkpoint_path", str(tae_checkpoint_path)])
        if args.compile_dit:
            command.append("--compile_dit")

        subprocess.run(
            command,
            check=True,
            cwd=str(repo_root),
            env=_script_env(extra_pythonpaths=da3_pythonpaths),
        )

        predicted_video = _find_prediction_video(output_root)
        shutil.copy2(predicted_video, output_path)

        if args.keep_work_dir:
            preserved = output_path.parent / f"{output_path.stem}_inspatio_workdir"
            if preserved.exists():
                shutil.rmtree(preserved)
            shutil.copytree(work_dir, preserved)


if __name__ == "__main__":
    main()
