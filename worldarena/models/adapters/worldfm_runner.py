"""Subprocess runner for WorldFM single-sample inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from worldarena.benchmark.annotations import load_camera_matrices, load_intrinsics_sequence
from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Arena WorldFM subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--annotation_path", required=True, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--config_path", default=None, type=str)
    parser.add_argument("--hw_path", default=None, type=str)
    parser.add_argument("--moge_path", default=None, type=str)
    parser.add_argument("--moge_pretrained", default=None, type=str)
    parser.add_argument("--model_filename", default="worldfm_2-step.pth", type=str)
    parser.add_argument("--vae_subdir", default="vae", type=str)
    parser.add_argument("--step", default=2, type=int)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--cfg_scale", default=4.5, type=float)
    parser.add_argument("--render_size", default=512, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--gpu_index", default=0, type=int)
    return parser.parse_args()


def _resolve_repo_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_checkpoint_path(checkpoint_dir: Path, value: str) -> Path:
    checkpoint_root = checkpoint_dir.resolve()
    token = str(value)
    if token.startswith(("ckpt/", "./", "../", "/", "~")):
        resolved = resolve_checkpoint_path(token)
        if resolved is None:
            raise FileNotFoundError(f"WorldFM checkpoint path is missing: {token}")
        return resolved
    resolved = (checkpoint_root / Path(token)).resolve()
    if not resolved.is_relative_to(checkpoint_root):
        raise ValueError(f"WorldFM checkpoint path must stay under {checkpoint_root}: {resolved}")
    return resolved


def _resolve_model_reference(value: str | None) -> str | None:
    if not value:
        return None
    token = str(value).strip()
    if token.startswith("ckpt/"):
        resolved = resolve_checkpoint_path(token, kind="dir", required=True)
        if resolved is None:
            raise FileNotFoundError(f"WorldFM model reference is missing: {token}")
        return str(resolved)
    path = Path(token).expanduser()
    if path.is_absolute() or token.startswith(("./", "../", "~")):
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"WorldFM model reference is missing: {resolved}")
        return str(resolved)
    return token


def _intrinsics_vector_to_matrix(intrinsics: np.ndarray) -> list[list[float]]:
    values = np.asarray(intrinsics, dtype=np.float32)
    if values.shape != (4,):
        raise ValueError(f"expected intrinsics vector with shape (4,), got {values.shape}")
    fx, fy, cx, cy = [float(value) for value in values]
    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def _build_meta_payload(
    *,
    sample_name: str,
    image_name: str,
    annotation_path: str,
) -> dict[str, object]:
    camera_matrices = load_camera_matrices(annotation_path)
    if camera_matrices is None:
        raise FileNotFoundError(f"poses.npy not found under annotation path: {annotation_path}")

    intrinsics = load_intrinsics_sequence(
        annotation_path,
        target_frames=len(camera_matrices),
    )
    if intrinsics is None:
        raise FileNotFoundError(f"intrinsics.npy not found under annotation path: {annotation_path}")

    if len(intrinsics) > 1 and not np.allclose(intrinsics, intrinsics[0], atol=1e-4):
        print(
            "[WorldAtlas Arena][WorldFM] intrinsics vary across frames; using the first matrix for all poses.",
            file=sys.stderr,
            flush=True,
        )

    return {
        "name": sample_name,
        "image": image_name,
        "K": _intrinsics_vector_to_matrix(intrinsics[0]),
        "c2w": camera_matrices.astype(np.float32).tolist(),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("WorldFM runner requires checkpoint_dir")
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = _resolve_repo_path(repo_root, args.config_path)
    hw_path = _resolve_repo_path(repo_root, args.hw_path)
    moge_path = _resolve_repo_path(repo_root, args.moge_path)
    model_path = _resolve_checkpoint_path(checkpoint_dir, args.model_filename)
    vae_path = _resolve_checkpoint_path(checkpoint_dir, args.vae_subdir)
    moge_pretrained = _resolve_model_reference(args.moge_pretrained)

    if not model_path.exists():
        raise FileNotFoundError(f"WorldFM checkpoint not found: {model_path}")
    if not vae_path.exists():
        raise FileNotFoundError(f"WorldFM VAE path not found: {vae_path}")

    with tempfile.TemporaryDirectory(prefix="worldarena_worldfm_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        image_source = Path(args.conditioning_image).expanduser().resolve()
        image_copy = temp_dir / f"conditioning{image_source.suffix or '.png'}"
        shutil.copy2(image_source, image_copy)

        meta_payload = _build_meta_payload(
            sample_name=args.sample_name,
            image_name=image_copy.name,
            annotation_path=args.annotation_path,
        )
        meta_path = temp_dir / "meta.json"
        meta_path.write_text(
            json.dumps(meta_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        output_root = temp_dir / "outputs"
        command = [
            sys.executable,
            str((repo_root / "run_pipeline.py").resolve()),
            "--meta",
            str(meta_path),
            "--output_dir",
            str(output_root),
            "--model_path",
            str(model_path),
            "--vae_path",
            str(vae_path),
            "--step",
            str(args.step),
            "--image_size",
            str(args.image_size),
            "--cfg_scale",
            str(args.cfg_scale),
            "--render_size",
            str(args.render_size),
            "--gpu_index",
            str(args.gpu_index),
            "--save_mode",
            "video",
            "--fps",
            str(args.fps),
        ]
        if config_path is not None:
            command.extend(["--config", str(config_path)])
        if hw_path is not None:
            command.extend(["--hw_path", str(hw_path)])
        if moge_path is not None:
            command.extend(["--moge_path", str(moge_path)])
        if moge_pretrained:
            command.extend(["--moge_pretrained", moge_pretrained])

        subprocess.run(
            command,
            check=True,
            cwd=str(repo_root),
            env=apply_checkpoint_env(),
        )

        generated_video = output_root / args.sample_name / "output.mp4"
        if not generated_video.exists():
            raise FileNotFoundError(f"WorldFM output video was not written: {generated_video}")
        shutil.copy2(generated_video, output_path)


if __name__ == "__main__":
    main()
