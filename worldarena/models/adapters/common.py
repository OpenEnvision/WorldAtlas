from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
from PIL import Image


def build_env(
    *,
    extra_pythonpaths: Iterable[Path] = (),
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpaths = [str(path) for path in extra_pythonpaths if str(path)]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    if pythonpaths:
        env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    if overrides:
        env.update({key: str(value) for key, value in overrides.items()})
    return env


def first_frame_to_image(video_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video for conditioning: {video_path}")
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"failed to decode first frame: {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(output_path)
    return output_path


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(
        command,
        check=True,
        cwd=str(cwd),
        env=env,
    )
