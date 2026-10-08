"""Shared subprocess, media, and batch-generation helpers for model adapters."""

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

from worldatlas.common.checkpoints import (
    CHECKPOINT_CACHE_ROOT_ENV,
    CHECKPOINT_ROOT_ENV,
    apply_checkpoint_env,
    filter_checkpoint_env_overrides,
    project_root as worldatlas_project_root,
)

if TYPE_CHECKING:
    from worldatlas.models.adapters.base import ModelAdapter, PreparedGenerationRequest


def build_env(
    *,
    extra_pythonpaths: Iterable[Path] = (),
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a subprocess environment with checkpoint paths and PYTHONPATH set."""
    base_env = os.environ.copy()
    for name in (CHECKPOINT_ROOT_ENV, CHECKPOINT_CACHE_ROOT_ENV):
        if overrides and name in overrides:
            base_env[name] = str(overrides[name])
    env = apply_checkpoint_env(base_env)
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
    """Append CLI flags derived from a generation config dict."""
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
    """Extract the first decoded frame from a video as a PNG conditioning image."""
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
    child_env = dict(env) if env is not None else dict(os.environ)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(
        command,
        check=True,
        cwd=str(cwd),
        env=child_env,
    )


def run_sequential_generate_batch(
    adapter: "ModelAdapter",
    requests: list["PreparedGenerationRequest"],
) -> dict[str, dict[str, Any]]:
    """Fallback batch path that calls ``generate`` once per sample in-process."""
    from worldatlas.common.progress import log_progress, rate_samples_per_hour

    results: dict[str, dict[str, Any]] = {}
    total = len(requests)
    batch_started_at = time.perf_counter()
    generated = 0
    failed = 0
    for index, request in enumerate(requests, start=1):
        sample_id = request.sample.sample_id
        index_label = f"{index}/{total}"
        log_progress("sample_start", sample_id=sample_id, index=index_label)
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
        if payload.get("status") == "failed":
            failed += 1
        else:
            generated += 1
        elapsed_s = float(payload.get("generation_wall_time_seconds") or 0.0)
        total_elapsed = time.perf_counter() - batch_started_at
        log_progress(
            "sample",
            sample_id=sample_id,
            index=index_label,
            status=payload.get("status"),
            elapsed_s=f"{elapsed_s:.1f}",
            generated=generated,
            failed=failed,
            rate_samples_per_hour=rate_samples_per_hour(generated + failed, total_elapsed),
            error=payload.get("error"),
        )
        results[sample_id] = payload
    return results
