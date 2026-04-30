from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from typing import Any, Sequence

from worldarena.common.checkpoints import apply_checkpoint_env, filter_checkpoint_env_overrides
from worldarena.benchmark.long_sequence_assets import inspect_long_sequence_layout


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LONG_SEQUENCE_RUNTIME_WORKER_ENV = "WORLDARENA_LONG_SEQUENCE_RUNTIME_WORKER"
LONG_SEQUENCE_DIMENSION_MAP = {
    "long_sequence_motion_smoothness": "motion_smoothness",
    "long_sequence_dynamic_degree": "dynamic_degree",
    "long_sequence_aesthetic_quality": "aesthetic_quality",
    "long_sequence_imaging_quality": "imaging_quality",
}


def _runtime_cache_dir(name: str) -> Path:
    path = PROJECT_ROOT / ".cache" / "benchmark" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_long_sequence_root() -> Path:
    return Path(inspect_long_sequence_layout()["module_root"]).resolve()


def _resolve_runtime_python_bin(runtime: dict[str, Any] | None) -> str | None:
    payload = dict(runtime or {})
    raw = str(payload.get("python_bin") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if raw.startswith(".") or "/" in raw:
        return str((PROJECT_ROOT / path).absolute())
    return raw


def resolve_long_sequence_runtime_python(runtime: dict[str, Any] | None = None) -> str | None:
    return _resolve_runtime_python_bin(runtime)


def _coerce_runtime_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dedupe_paths(entries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)
    return deduped


def _subprocess_runtime_env(runtime: dict[str, Any] | None) -> dict[str, str]:
    env = apply_checkpoint_env()
    env[LONG_SEQUENCE_RUNTIME_WORKER_ENV] = "1"
    pythonpath_entries = [str(PROJECT_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(_dedupe_paths(pythonpath_entries))
    for key, value in filter_checkpoint_env_overrides(dict((runtime or {}).get("env", {}))).items():
        env[str(key)] = str(value)
    return env


def _metric_dimensions(metric_names: Sequence[str]) -> list[str]:
    dimensions: list[str] = []
    seen: set[str] = set()
    for metric_name in metric_names:
        try:
            dimension = LONG_SEQUENCE_DIMENSION_MAP[metric_name]
        except KeyError as exc:
            raise KeyError(f"unsupported long-sequence metric: {metric_name}") from exc
        if dimension in seen:
            continue
        seen.add(dimension)
        dimensions.append(dimension)
    return dimensions


def _metric_runtime_kwargs(runtime: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(runtime or {})
    return {
        "device": str(payload.get("device") or _default_device()),
        "read_frame": _coerce_runtime_bool(payload.get("read_frame"), default=False),
        "load_ckpt_from_local": _coerce_runtime_bool(payload.get("load_ckpt_from_local"), default=False),
        "use_semantic_splitting": _coerce_runtime_bool(
            payload.get("use_semantic_splitting"),
            default=False,
        ),
        "threshold": float(payload.get("threshold", 35.0)),
        "imaging_quality_preprocessing_mode": str(
            payload.get("imaging_quality_preprocessing_mode") or "longer"
        ),
        "clip_duration_seconds": float(payload.get("clip_duration_seconds", 2.0)),
        "clip_fps": int(payload.get("clip_fps", 8)),
        "clip_length_config": str(payload.get("clip_length_config") or "clip_length_mix.yaml"),
        "dev_flag": _coerce_runtime_bool(payload.get("dev_flag"), default=True),
    }


def _symlink_or_copy_video(source_path: Path, target_path: Path) -> None:
    try:
        target_path.symlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _extract_dimension_payload(
    metric_name: str,
    payload: Any,
) -> tuple[float | None, dict[str, Any]]:
    raw = payload
    clip_results: list[dict[str, Any]] = []
    video_results: list[dict[str, Any]] = []
    if isinstance(payload, (list, tuple)):
        raw = payload[0] if len(payload) > 0 else None
        if len(payload) > 1 and isinstance(payload[1], list):
            clip_results = [item for item in payload[1] if isinstance(item, dict)]
        if len(payload) > 2 and isinstance(payload[2], list):
            video_results = [item for item in payload[2] if isinstance(item, dict)]
    numeric_video_scores = [
        item for item in video_results if isinstance(item.get("video_results"), (int, float))
    ]
    score = None if raw is None else float(raw)
    return score, {
        "dimension": LONG_SEQUENCE_DIMENSION_MAP[metric_name],
        "clip_result_count": len(clip_results),
        "video_count": len(numeric_video_scores),
        "per_video_scores": numeric_video_scores,
    }


def _run_local_long_sequence(
    video_path: str,
    *,
    metric_names: Sequence[str],
    runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    layout = inspect_long_sequence_layout()
    if not layout["ready"]:
        raise FileNotFoundError(str(layout["error"]))

    source_path = Path(video_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"prediction video not found: {source_path}")

    token = uuid.uuid4().hex
    workspace = _runtime_cache_dir("long_sequence_workspace") / token
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_video_path = input_dir / source_path.name
    _symlink_or_copy_video(source_path, input_video_path)

    options = _metric_runtime_kwargs(runtime)
    keep_workspace = _coerce_runtime_bool((runtime or {}).get("keep_workspace"), default=False)
    run_name = f"worldarena_{token}"
    try:
        import torch
        from worldarena.benchmark.long_sequence_dispatch import evaluate_long_sequence_dimensions

        result_payload = evaluate_long_sequence_dimensions(
            videos_path=input_dir,
            output_path=output_dir,
            name=run_name,
            dimensions=_metric_dimensions(metric_names),
            device=torch.device(options["device"]),
            threshold=options["threshold"],
            use_semantic_splitting=bool(options["use_semantic_splitting"]),
            clip_duration=options["clip_duration_seconds"],
            clip_fps=options["clip_fps"],
            imaging_quality_preprocessing_mode=options["imaging_quality_preprocessing_mode"],
            clip_length_config=options["clip_length_config"],
            dev_flag=bool(options["dev_flag"]),
            static_filter_flag=False,
            num_of_samples_per_prompt=1,
        )
        metrics: dict[str, float | None] = {}
        metric_details: dict[str, dict[str, Any]] = {}
        for metric_name in metric_names:
            dimension = LONG_SEQUENCE_DIMENSION_MAP[metric_name]
            score, details = _extract_dimension_payload(metric_name, result_payload.get(dimension))
            metrics[metric_name] = score
            metric_details[metric_name] = details
        return {
            "backend": "long_sequence",
            "metrics": metrics,
            "metric_details": metric_details,
            "source_root": layout["module_root"],
            "runtime_device": options["device"],
        }
    finally:
        if not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def _maybe_run_in_subprocess(
    video_path: str,
    *,
    metric_names: Sequence[str],
    runtime: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if os.getenv(LONG_SEQUENCE_RUNTIME_WORKER_ENV) == "1":
        return None
    python_bin = _resolve_runtime_python_bin(runtime)
    if python_bin is None:
        return None

    ipc_dir = _runtime_cache_dir("long_sequence_runtime_ipc")
    token = uuid.uuid4().hex
    request_path = ipc_dir / f"long_sequence.{token}.request.json"
    response_path = ipc_dir / f"long_sequence.{token}.response.json"
    request_path.write_text(
        json.dumps(
            {
                "video_path": video_path,
                "metric_names": list(metric_names),
                "runtime": dict(runtime or {}),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    command = [
        python_bin,
        "-m",
        "worldarena.benchmark.long_sequence_runtime_worker",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=_subprocess_runtime_env(runtime),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise RuntimeError(
            " | ".join(
                part
                for part in ("Long-sequence runtime subprocess failed", stderr, stdout)
                if part
            )
        ) from exc
    if not response_path.exists():
        logs = " | ".join(
            part for part in ((completed.stderr or "").strip(), (completed.stdout or "").strip()) if part
        )
        raise RuntimeError(f"Long-sequence runtime subprocess did not write a response: {logs}")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if not _coerce_runtime_bool((runtime or {}).get("keep_ipc"), default=False):
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
    if not bool(response.get("ok")):
        error = str(response.get("error") or "Long-sequence runtime subprocess failed")
        traceback_text = str(response.get("traceback") or "").strip()
        stderr = str(response.get("stderr") or "").strip()
        stdout = str(response.get("stdout") or "").strip()
        raise RuntimeError(" | ".join(part for part in (error, traceback_text, stderr, stdout) if part))
    return dict(response["result"])


def compute_long_sequence_metrics(
    video_path: str,
    *,
    metric_names: Sequence[str],
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = _maybe_run_in_subprocess(
        video_path,
        metric_names=metric_names,
        runtime=runtime,
    )
    if result is not None:
        return result
    return _run_local_long_sequence(
        video_path,
        metric_names=metric_names,
        runtime=runtime,
    )


__all__ = [
    "LONG_SEQUENCE_DIMENSION_MAP",
    "compute_long_sequence_metrics",
    "inspect_long_sequence_layout",
    "resolve_long_sequence_root",
    "resolve_long_sequence_runtime_python",
]
