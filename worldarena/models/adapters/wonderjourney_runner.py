from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from worldarena.common.checkpoints import apply_checkpoint_env


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
    parser = argparse.ArgumentParser(description="WorldArena WonderJourney subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--entrypoint", default="run.py", type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--scene_name", required=True, type=str)
    parser.add_argument("--entities", nargs="*", default=None)
    parser.add_argument("--style_prompt", default="photorealistic", type=str)
    parser.add_argument("--background_prompt", default="", type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--midas_checkpoint", required=True, type=str)
    parser.add_argument("--stable_diffusion_checkpoint", default=None, type=str)
    parser.add_argument("--stable_diffusion_revision", default=None, type=str)
    parser.add_argument("--rotation_path", default=None, type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--frames", default=5, type=int)
    parser.add_argument("--num_scenes", default=1, type=int)
    parser.add_argument("--num_keyframes", default=2, type=int)
    parser.add_argument("--save_fps", default=10, type=int)
    parser.add_argument("--run_timeout", default=None, type=float)
    parser.add_argument("--depth_model", default="midas_v3.1", type=str)
    parser.add_argument("--camera_speed", default=0.0005, type=float)
    parser.add_argument("--rotation_range", default=0.0, type=float)
    parser.add_argument("--init_focal_length", default=500.0, type=float)
    parser.add_argument("--inpainting_resolution_gen", default=512, type=int)
    parser.add_argument("--inpainting_resolution_interp", default=512, type=int)
    parser.add_argument("--kf2_upsample_coef", default=4, type=int)
    parser.add_argument("--fg_depth_range", default=0.0015, type=float)
    parser.add_argument("--depth_shift", default=0.0001, type=float)
    parser.add_argument("--regenerate_times", default=3, type=int)
    parser.add_argument("--num_finetune_decoder_steps", default=100, type=int)
    parser.add_argument("--num_finetune_decoder_steps_interp", default=30, type=int)
    parser.add_argument("--camera_speed_multiplier_rotation", default=0.0, type=float)
    parser.add_argument("--use_gpt", default=False, type=_parse_bool)
    parser.add_argument("--debug", default=False, type=_parse_bool)
    parser.add_argument("--skip_interp", default=False, type=_parse_bool)
    parser.add_argument("--skip_gen", default=False, type=_parse_bool)
    parser.add_argument("--enable_regenerate", default=False, type=_parse_bool)
    parser.add_argument("--finetune_decoder_gen", default=True, type=_parse_bool)
    parser.add_argument("--finetune_decoder_interp", default=False, type=_parse_bool)
    parser.add_argument("--finetune_depth_model", default=True, type=_parse_bool)
    parser.add_argument("--keep_work_dir", default=False, type=_parse_bool)
    parser.add_argument("--archive_sidecars", default=True, type=_parse_bool)
    return parser.parse_args()


@contextlib.contextmanager
def _work_dir(keep_work_dir: bool):
    if keep_work_dir:
        path = Path(tempfile.mkdtemp(prefix="worldarena_wonderjourney_"))
        try:
            yield path
        finally:
            pass
        return

    with tempfile.TemporaryDirectory(prefix="worldarena_wonderjourney_") as temp_dir_raw:
        yield Path(temp_dir_raw)


def _slugify(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "sample"


def _normalized_env(repo_root: Path) -> dict[str, str]:
    env = apply_checkpoint_env()
    pythonpaths = [str(repo_root), str(repo_root / "midas_module")]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _require_repo_layout(repo_root: Path, entrypoint: str) -> tuple[Path, Path]:
    entrypoint_path = (repo_root / entrypoint).resolve()
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"WonderJourney entrypoint not found: {entrypoint_path}")
    base_config_path = (repo_root / "config" / "base-config.yaml").resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"WonderJourney base config not found: {base_config_path}")
    return entrypoint_path, base_config_path


def _runtime_entrypoint(entrypoint_path: Path, work_dir: Path, revision: str | None) -> Path:
    source = entrypoint_path.read_text(encoding="utf-8")
    marker = 'revision="fp16",'
    if marker not in source:
        return entrypoint_path
    replacement = f'revision="{revision}",' if revision else ""
    patched_path = work_dir / "worldarena_wonderjourney_entrypoint.py"
    patched_path.write_text(source.replace(marker, replacement), encoding="utf-8")
    return patched_path


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _parse_rotation_path(value: str | None, *, num_scenes: int, num_keyframes: int) -> list[int]:
    required = max(int(num_scenes), 1) * max(int(num_keyframes), 1)
    if value is None or not str(value).strip():
        return [0] * required
    text = str(value).strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in text.split(",") if item.strip()]
    if not isinstance(parsed, list):
        raise ValueError(f"rotation_path must be a list or comma-separated string: {value!r}")
    rotation_path = [int(item) for item in parsed]
    if len(rotation_path) < required:
        rotation_path.extend([0] * (required - len(rotation_path)))
    return rotation_path


def _content_prompt(scene_name: str, entities: list[str]) -> str:
    values = [scene_name, *entities]
    return ", ".join(value for value in values if value)


def _write_example_yaml(
    path: Path,
    *,
    example_name: str,
    image_filename: str,
    style_prompt: str,
    content_prompt: str,
    negative_prompt: str,
    background_prompt: str,
) -> Path:
    payload = [
        {
            "name": example_name,
            "image_filepath": f"examples/images/{image_filename}",
            "style_prompt": style_prompt,
            "content_prompt": content_prompt,
            "negative_prompt": negative_prompt,
            "background": background_prompt,
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _write_example_config(path: Path, *, args: argparse.Namespace, example_name: str) -> Path:
    payload: dict[str, Any] = {
        "runs_dir": f"output/{example_name}",
        "example_name": example_name,
        "seed": int(args.seed),
        "frames": int(args.frames),
        "num_scenes": int(args.num_scenes),
        "num_keyframes": int(args.num_keyframes),
        "use_gpt": bool(args.use_gpt),
        "debug": bool(args.debug),
        "skip_interp": bool(args.skip_interp),
        "skip_gen": bool(args.skip_gen),
        "enable_regenerate": bool(args.enable_regenerate),
        "finetune_decoder_gen": bool(args.finetune_decoder_gen),
        "finetune_decoder_interp": bool(args.finetune_decoder_interp),
        "finetune_depth_model": bool(args.finetune_depth_model),
        "depth_model": str(args.depth_model),
        "camera_speed": float(args.camera_speed),
        "rotation_range": float(args.rotation_range),
        "init_focal_length": float(args.init_focal_length),
        "inpainting_resolution_gen": int(args.inpainting_resolution_gen),
        "inpainting_resolution_interp": int(args.inpainting_resolution_interp),
        "kf2_upsample_coef": int(args.kf2_upsample_coef),
        "fg_depth_range": float(args.fg_depth_range),
        "depth_shift": float(args.depth_shift),
        "regenerate_times": int(args.regenerate_times),
        "num_finetune_decoder_steps": int(args.num_finetune_decoder_steps),
        "num_finetune_decoder_steps_interp": int(args.num_finetune_decoder_steps_interp),
        "camera_speed_multiplier_rotation": float(args.camera_speed_multiplier_rotation),
        "save_fps": int(args.save_fps),
        "rotation_path": _parse_rotation_path(
            args.rotation_path,
            num_scenes=int(args.num_scenes),
            num_keyframes=int(args.num_keyframes),
        ),
    }
    if args.stable_diffusion_checkpoint:
        payload["stable_diffusion_checkpoint"] = str(args.stable_diffusion_checkpoint)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _latest_merged_dir(runs_root: Path) -> Path | None:
    candidates = [path for path in runs_root.glob("*_merged") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _copy_prediction_outputs(
    *,
    output_path: Path,
    merged_dir: Path,
    archive_sidecars: bool,
) -> dict[str, Path]:
    generated_video = merged_dir / "output.mp4"
    if not generated_video.exists():
        raise FileNotFoundError(f"WonderJourney output video was not written: {generated_video}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    shutil.copy2(generated_video, output_path)
    copied = {"prediction": output_path}

    if not archive_sidecars:
        return copied
    for filename, suffix in (
        ("output_reverse.mp4", ".reverse"),
        ("keyframes.mp4", ".keyframes"),
        ("keyframes_reverse.mp4", ".keyframes_reverse"),
    ):
        source = merged_dir / filename
        if not source.exists():
            continue
        destination = output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
        destination.unlink(missing_ok=True)
        shutil.copy2(source, destination)
        copied[suffix.strip(".")] = destination
    return copied


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    conditioning_image = Path(args.conditioning_image).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    midas_checkpoint = Path(args.midas_checkpoint).expanduser().resolve()

    if not conditioning_image.exists():
        raise FileNotFoundError(f"WonderJourney conditioning image not found: {conditioning_image}")
    if not midas_checkpoint.exists():
        raise FileNotFoundError(f"WonderJourney MiDaS checkpoint not found: {midas_checkpoint}")

    entrypoint_path, base_config_path = _require_repo_layout(repo_root, args.entrypoint)
    example_name = _slugify(args.sample_name)
    entities = [str(value) for value in (args.entities or []) if str(value).strip()]
    content_prompt = _content_prompt(args.scene_name, entities)

    with _work_dir(bool(args.keep_work_dir)) as work_dir:
        examples_root = work_dir / "examples"
        image_suffix = conditioning_image.suffix or ".png"
        local_conditioning = examples_root / "images" / f"{example_name}{image_suffix}"
        local_conditioning.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(conditioning_image, local_conditioning)
        (work_dir / "output").mkdir(parents=True, exist_ok=True)
        _link_or_copy(midas_checkpoint, work_dir / "dpt_beit_large_512.pt")

        _write_example_yaml(
            examples_root / "examples.yaml",
            example_name=example_name,
            image_filename=local_conditioning.name,
            style_prompt=str(args.style_prompt),
            content_prompt=content_prompt,
            negative_prompt=str(args.negative_prompt),
            background_prompt=str(args.background_prompt),
        )
        example_config_path = _write_example_config(
            work_dir / "config" / "worldarena_example.yaml",
            args=args,
            example_name=example_name,
        )

        runtime_entrypoint = _runtime_entrypoint(
            entrypoint_path,
            work_dir,
            str(args.stable_diffusion_revision).strip() or None,
        )
        command = [
            sys.executable,
            str(runtime_entrypoint),
            "--base-config",
            str(base_config_path),
            "--example_config",
            str(example_config_path),
        ]
        subprocess.run(
            command,
            check=True,
            cwd=str(work_dir),
            env=_normalized_env(repo_root),
            timeout=args.run_timeout,
        )

        merged_dir = _latest_merged_dir(work_dir / "output" / example_name)
        if merged_dir is None:
            raise FileNotFoundError(
                f"WonderJourney merged output directory was not written under {work_dir}"
            )
        _copy_prediction_outputs(
            output_path=output_path,
            merged_dir=merged_dir,
            archive_sidecars=bool(args.archive_sidecars),
        )


if __name__ == "__main__":
    main()
