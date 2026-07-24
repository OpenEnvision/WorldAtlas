"""Native (in-process) long-sequence metric computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from worldarena.benchmark.flow_backends import (
    compute_amt_motion_smoothness,
    compute_raft_dynamic_degree,
)
from worldarena.benchmark.quality_backends import (
    compute_musiq_scores,
    compute_quality_component_scores,
)
from worldarena.common.video_io import read_video_frames

LONG_SEQUENCE_NATIVE_DIMENSIONS = {
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
}


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _round_list(values: Sequence[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def _load_manifest_entries(
    json_dir: str | Path,
    *,
    dimension: str,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(json_dir).read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    for item in payload:
        dimensions = list(item.get("dimension") or [])
        if dimension not in dimensions:
            continue
        clip_paths = [
            str(path)
            for path in (item.get("video_list") or [])
            if Path(str(path)).suffix.lower() in {".mp4", ".avi", ".mov"}
        ]
        if not clip_paths:
            continue
        entries.append(
            {
                "prompt": str(item.get("prompt_en") or item.get("prompt") or ""),
                "clip_paths": clip_paths,
            }
        )
    return entries


def _long_video_path(prompt: str, clip_paths: Sequence[str]) -> str:
    first_clip = Path(clip_paths[0]).resolve()
    split_clip_root = first_clip.parents[1]
    candidate = split_clip_root.parent / f"{prompt}.mp4"
    return str(candidate if candidate.exists() else candidate)


def _clip_quality_score(
    clip_path: str,
    *,
    scorer: Callable[[Sequence[np.ndarray]], dict[str, Any]],
) -> tuple[float | None, dict[str, Any]]:
    frames = read_video_frames(Path(clip_path))
    result = scorer(frames)
    raw = result.get("raw")
    clip_payload = {
        "video_path": clip_path,
        "video_results": raw,
        "frame_count": len(frames),
        "quality_backend": result.get("backend"),
    }
    for key in (
        "frame_scores",
        "native_frame_scores",
        "native_score_range",
        "preprocess_mode",
        "weights_path",
    ):
        if key in result:
            clip_payload[key] = result[key]
    return (None if raw is None else float(raw)), clip_payload


def _clip_backend_score(
    clip_path: str,
    *,
    scorer: Callable[[str], dict[str, Any]],
) -> tuple[float | None, dict[str, Any]]:
    result = scorer(clip_path)
    raw = result.get("raw")
    clip_payload = {
        "video_path": clip_path,
        "video_results": raw,
    }
    for key, value in result.items():
        if key == "raw":
            continue
        clip_payload[key] = value
    return (None if raw is None else float(raw)), clip_payload


def _native_long_quality(
    json_dir: str | Path,
    *,
    dimension: str,
    scorer: Callable[[Sequence[np.ndarray]], dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    entries = _load_manifest_entries(json_dir, dimension=dimension)
    clip_results: list[dict[str, Any]] = []
    video_results: list[dict[str, Any]] = []
    overall_scores: list[float] = []

    for entry in entries:
        prompt = str(entry["prompt"])
        clip_paths = list(entry["clip_paths"])

        clip_scores: list[float] = []
        quality_backends: list[str] = []
        for clip_path in clip_paths:
            clip_score, clip_payload = _clip_quality_score(clip_path, scorer=scorer)
            clip_payload["long_video_prompt"] = prompt
            clip_results.append(clip_payload)
            if clip_score is not None:
                clip_scores.append(float(clip_score))
            backend_name = clip_payload.get("quality_backend")
            if isinstance(backend_name, str) and backend_name:
                quality_backends.append(backend_name)

        video_score = _mean(clip_scores)
        if video_score is not None:
            overall_scores.append(float(video_score))

        video_results.append(
            {
                "video_path": _long_video_path(prompt, clip_paths),
                "video_results": video_score,
                "clip_count": len(clip_paths),
                "clip_scores": _round_list(clip_scores),
                "quality_backends": sorted(set(quality_backends)),
            }
        )

    return _mean(overall_scores), clip_results, video_results


def _native_long_video_metric(
    json_dir: str | Path,
    *,
    dimension: str,
    scorer: Callable[[str], dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    entries = _load_manifest_entries(json_dir, dimension=dimension)
    clip_results: list[dict[str, Any]] = []
    video_results: list[dict[str, Any]] = []
    overall_scores: list[float] = []

    for entry in entries:
        prompt = str(entry["prompt"])
        clip_paths = list(entry["clip_paths"])

        clip_scores: list[float] = []
        backends: list[str] = []
        for clip_path in clip_paths:
            clip_score, clip_payload = _clip_backend_score(clip_path, scorer=scorer)
            clip_payload["long_video_prompt"] = prompt
            clip_results.append(clip_payload)
            if clip_score is not None:
                clip_scores.append(float(clip_score))
            backend_name = clip_payload.get("backend")
            if isinstance(backend_name, str) and backend_name:
                backends.append(backend_name)

        video_score = _mean(clip_scores)
        if video_score is not None:
            overall_scores.append(float(video_score))
        video_results.append(
            {
                "video_path": _long_video_path(prompt, clip_paths),
                "video_results": video_score,
                "clip_count": len(clip_paths),
                "clip_scores": _round_list(clip_scores),
                "backends": sorted(set(backends)),
            }
        )

    return _mean(overall_scores), clip_results, video_results


def compute_long_motion_smoothness(
    json_dir: str | Path,
    device: Any,
    submodules_list: Any,
    **kwargs: Any,
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    config_path = kwargs.get("amt_config_path")
    ckpt_path = kwargs.get("amt_checkpoint_path")
    if isinstance(submodules_list, dict):
        config_path = config_path or submodules_list.get("config")
        ckpt_path = ckpt_path or submodules_list.get("ckpt")

    return _native_long_video_metric(
        json_dir,
        dimension="motion_smoothness",
        scorer=lambda clip_path: compute_amt_motion_smoothness(
            clip_path,
            device=device,
            config_path=None if config_path is None else str(config_path),
            ckpt_path=None if ckpt_path is None else str(ckpt_path),
        ),
    )


def compute_long_dynamic_degree(
    json_dir: str | Path,
    device: Any,
    submodules_list: Any,
    **kwargs: Any,
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    model_path = kwargs.get("raft_model_path")
    if isinstance(submodules_list, dict):
        model_path = model_path or submodules_list.get("model")

    return _native_long_video_metric(
        json_dir,
        dimension="dynamic_degree",
        scorer=lambda clip_path: compute_raft_dynamic_degree(
            clip_path,
            device=device,
            model_path=None if model_path is None else str(model_path),
        ),
    )


def compute_long_aesthetic_quality(
    json_dir: str | Path,
    device: Any,
    submodules_list: Any,
    **kwargs: Any,
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    del device
    del submodules_list
    del kwargs

    return _native_long_quality(
        json_dir,
        dimension="aesthetic_quality",
        scorer=lambda frames: compute_quality_component_scores(
            frames,
            component_name="aesthetic",
        ),
    )


def compute_long_imaging_quality(
    json_dir: str | Path,
    device: Any,
    submodules_list: Any,
    **kwargs: Any,
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    del device

    model_path = kwargs.get("musiq_model_path")
    if model_path is None and isinstance(submodules_list, dict):
        model_path = submodules_list.get("model_path")
    preprocess_mode = str(kwargs.get("imaging_quality_preprocessing_mode") or "longer")
    return _native_long_quality(
        json_dir,
        dimension="imaging_quality",
        scorer=lambda frames: compute_musiq_scores(
            frames,
            model_path=None if model_path is None else str(model_path),
            preprocess_mode=preprocess_mode,
        ),
    )


__all__ = [
    "LONG_SEQUENCE_NATIVE_DIMENSIONS",
    "compute_long_aesthetic_quality",
    "compute_long_dynamic_degree",
    "compute_long_imaging_quality",
    "compute_long_motion_smoothness",
]
