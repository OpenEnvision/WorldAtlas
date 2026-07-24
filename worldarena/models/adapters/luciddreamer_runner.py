"""Subprocess runner for LucidDreamer single-sample inference."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

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
    parser = argparse.ArgumentParser(description="WorldAtlas Arena LucidDreamer subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument(
        "--campath_gen",
        default="lookdown",
        choices=("lookdown", "lookaround", "rotate360"),
        type=str,
    )
    parser.add_argument(
        "--campath_render",
        default="llff",
        type=str,
    )
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--diff_steps", default=50, type=int)
    parser.add_argument("--keep_work_dir", default=False, type=_parse_bool)
    return parser.parse_args()


@contextlib.contextmanager
def _work_dir(keep_work_dir: bool):
    if keep_work_dir:
        path = Path(tempfile.mkdtemp(prefix="worldarena_luciddreamer_"))
        try:
            yield path
        finally:
            pass
        return

    with tempfile.TemporaryDirectory(prefix="worldarena_luciddreamer_") as temp_dir_raw:
        yield Path(temp_dir_raw)


def _require_repo_layout(repo_root: Path) -> Path:
    entrypoint = repo_root / "run.py"
    if not entrypoint.exists():
        raise FileNotFoundError(f"LucidDreamer entrypoint not found: {entrypoint}")
    return entrypoint.resolve()


def _write_text(path: Path, text: str) -> Path:
    path.write_text(str(text), encoding="utf-8")
    return path


def _resolved_model_name(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    token = str(model_name).strip()
    return token or None


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    entrypoint = _require_repo_layout(repo_root)
    conditioning_image = Path(args.conditioning_image).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not conditioning_image.exists():
        raise FileNotFoundError(f"LucidDreamer conditioning image not found: {conditioning_image}")

    with _work_dir(bool(args.keep_work_dir)) as work_dir:
        input_dir = work_dir / "inputs"
        save_dir = work_dir / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        local_image = input_dir / f"conditioning{conditioning_image.suffix or '.png'}"
        shutil.copy2(conditioning_image, local_image)
        prompt_path = _write_text(input_dir / "prompt.txt", args.prompt)
        negative_prompt_path = _write_text(input_dir / "negative_prompt.txt", args.negative_prompt)

        command = [
            sys.executable,
            str(entrypoint),
            "--image",
            str(local_image),
            "--text",
            str(prompt_path),
            "--neg_text",
            str(negative_prompt_path),
            "--campath_gen",
            str(args.campath_gen),
            "--campath_render",
            str(args.campath_render),
            "--seed",
            str(int(args.seed)),
            "--diff_steps",
            str(int(args.diff_steps)),
            "--save_dir",
            str(save_dir),
        ]
        model_name = _resolved_model_name(args.model_name)
        if model_name is not None:
            command.extend(["--model_name", model_name])

        subprocess.run(
            command,
            check=True,
            cwd=str(repo_root),
            env=apply_checkpoint_env(),
        )

        generated_video = save_dir / f"{args.campath_render}.mp4"
        generated_depth_video = save_dir / f"depth_{args.campath_render}.mp4"
        generated_ply = save_dir / "gsplat.ply"
        if not generated_video.exists():
            raise FileNotFoundError(f"LucidDreamer output video was not written: {generated_video}")
        if not generated_ply.exists():
            raise FileNotFoundError(f"LucidDreamer Gaussian output was not written: {generated_ply}")

        output_path.unlink(missing_ok=True)
        shutil.copy2(generated_video, output_path)

        ply_output = output_path.with_suffix(".ply")
        ply_output.unlink(missing_ok=True)
        shutil.copy2(generated_ply, ply_output)

        if generated_depth_video.exists():
            depth_output = output_path.with_name(f"{output_path.stem}.depth{output_path.suffix}")
            depth_output.unlink(missing_ok=True)
            shutil.copy2(generated_depth_video, depth_output)


if __name__ == "__main__":
    main()
