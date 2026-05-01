from __future__ import annotations

import base64
import io
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

import cv2
from PIL import Image

from worldarena.common.checkpoints import apply_checkpoint_env, filter_checkpoint_env_overrides

if TYPE_CHECKING:
    from worldarena.models.adapters.base import ModelAdapter, PreparedGenerationRequest


def build_env(
    *,
    extra_pythonpaths: Iterable[Path] = (),
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    env = apply_checkpoint_env()
    pythonpaths = [str(path) for path in extra_pythonpaths if str(path)]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    if pythonpaths:
        env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    if overrides:
        env.update(filter_checkpoint_env_overrides(overrides))
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


def run_sequential_generate_batch(
    adapter: "ModelAdapter",
    requests: list["PreparedGenerationRequest"],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for request in requests:
        started_at = time.perf_counter()
        try:
            payload = dict(
                adapter.generate(
                    sample=request.sample,
                    conditioning_image=request.conditioning_image,
                    output_path=request.output_path,
                    prompt=request.prompt,
                )
            )
            payload.setdefault(
                "generation_wall_time_seconds",
                round(time.perf_counter() - started_at, 6),
            )
            payload.setdefault("status", "generated")
            payload.setdefault("prediction_path", str(request.output_path))
            payload.setdefault("prompt", request.prompt)
        except Exception as exc:
            payload = {
                "status": "failed",
                "error": str(exc),
                "prediction_path": str(request.output_path),
                "prompt": request.prompt,
                "generation_wall_time_seconds": round(time.perf_counter() - started_at, 6),
            }
        results[request.sample.sample_id] = payload
    return results
