from __future__ import annotations

import base64
import io
import os
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Mapping

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


def extend_command_with_options(
    command: list[str],
    *,
    payload: Mapping[str, object],
    value_options: Mapping[str, Callable[[object], object]] | None = None,
    bool_value_options: Iterable[str] = (),
    flag_options: Iterable[str] = (),
) -> None:
    for name, cast in (value_options or {}).items():
        value = payload.get(name)
        if value is None:
            continue
        command.extend([f"--{name}", str(cast(value))])
    for name in bool_value_options:
        if name not in payload or payload[name] is None:
            continue
        command.extend([f"--{name}", "true" if bool(payload[name]) else "false"])
    for name in flag_options:
        if payload.get(name, False):
            command.append(f"--{name}")


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


def image_path_to_data_url(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{content}"


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
