from __future__ import annotations

from bisect import bisect_left
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from worldarena.benchmark.clip_backend import clip_image_features
from worldarena.benchmark.consistency_backends import (
    dinov2_image_features,
    dreamsim_image_features,
    transition_scores,
)
from worldarena.benchmark.flow_backends import (
    compute_amt_motion_smoothness,
    compute_raft_dynamic_degree,
)
from worldarena.benchmark.long_sequence_params import (
    default_long_sequence_mapping_table,
    default_long_sequence_slow_fast_params,
)
from worldarena.benchmark.quality_backends import (
    compute_musiq_scores,
    compute_quality_component_scores,
)
from worldarena.common.video_io import read_video_frames

LONG_SEQUENCE_NATIVE_DIMENSIONS = {
    "subject_consistency",
    "background_consistency",
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


def _mapping_table(
    path: str | Path | None,
    *,
    dimension: str,
) -> dict[float, float]:
    if path is None:
        return default_long_sequence_mapping_table(dimension)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {float(key): float(value) for key, value in raw.items()}


def _mapped_clip2clip_score(score: float, mapping_table: dict[float, float]) -> float:
    keys = sorted(mapping_table)
    if not keys:
        return float(score)
    idx = bisect_left(keys, score)
    if idx == 0:
        return mapping_table[keys[0]]
    if idx == len(keys):
        return mapping_table[keys[-1]]
    k0, k1 = keys[idx - 1], keys[idx]
    if np.isclose(k0, k1):
        return mapping_table[k0]
    y0 = mapping_table[k0]
    y1 = mapping_table[k1]
    return float(y0 + (y1 - y0) * (score - k0) / (k1 - k0))


def _slow_fast_params(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return default_long_sequence_slow_fast_params()
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict(payload)


def _clip_sequence_score(
    clip_path: str,
    *,
    extractor: Callable[[Sequence[np.ndarray]], tuple[np.ndarray, str]],
) -> tuple[float | None, dict[str, Any], np.ndarray | None]:
    frames = read_video_frames(Path(clip_path))
    if len(frames) < 2:
        return (
            None,
            {
                "video_path": clip_path,
                "video_results": None,
                "frame_count": len(frames),
                "error": "not enough frames in split clip",
            },
            frames[0] if frames else None,
        )
    features, backend_name = extractor(frames)
    scores = transition_scores(features)
    raw = _mean(scores)
    return (
        raw,
        {
            "video_path": clip_path,
            "video_results": raw,
            "frame_count": len(frames),
            "feature_backend": backend_name,
            "transition_scores": _round_list(scores),
        },
        frames[0],
    )


def _frames_sequence_score(
    frames: Sequence[np.ndarray],
    *,
    extractor: Callable[[Sequence[np.ndarray]], tuple[np.ndarray, str]],
) -> tuple[float | None, str | None, list[float]]:
    if len(frames) < 2:
        return None, None, []
    features, backend_name = extractor(frames)
    scores = transition_scores(features)
    return _mean(scores), backend_name, scores


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


def _subject_inclip_extractor(
    frames: Sequence[np.ndarray],
) -> tuple[np.ndarray, str]:
    return dinov2_image_features(frames), "dinov2"


def _subject_clip2clip_extractor(
    extractor_name: str,
) -> Callable[[Sequence[np.ndarray]], tuple[np.ndarray, str]]:
    if extractor_name in {"dino", "dinov2"}:
        return _subject_inclip_extractor
    if extractor_name == "dreamsim":
        return lambda frames: (dreamsim_image_features(frames), "dreamsim")
    raise KeyError(f"unsupported long-sequence subject clip2clip extractor: {extractor_name}")


def _background_inclip_extractor(
    frames: Sequence[np.ndarray],
) -> tuple[np.ndarray, str]:
    features, loader_name = clip_image_features(frames)
    return features, f"clip:{loader_name}"


def _background_clip2clip_extractor(
    extractor_name: str,
) -> Callable[[Sequence[np.ndarray]], tuple[np.ndarray, str]]:
    if extractor_name == "dreamsim":
        return lambda frames: (dreamsim_image_features(frames), "dreamsim")
    if extractor_name == "clip":
        return _background_inclip_extractor
    raise KeyError(f"unsupported long-sequence background clip2clip extractor: {extractor_name}")


def _native_long_consistency(
    json_dir: str | Path,
    *,
    dimension: str,
    inclip_extractor: Callable[[Sequence[np.ndarray]], tuple[np.ndarray, str]],
    clip2clip_extractor: Callable[[Sequence[np.ndarray]], tuple[np.ndarray, str]],
    mapping_file_path: str | Path,
    slow_fast_eval_config: str | Path | None,
    w_inclip: float,
    w_clip2clip: float,
    dev_flag: bool,
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    entries = _load_manifest_entries(json_dir, dimension=dimension)
    mapping_table = _mapping_table(mapping_file_path, dimension=dimension)
    params = _slow_fast_params(slow_fast_eval_config)
    if dev_flag:
        postfix = "sb" if dimension == "subject_consistency" else "bg"
        w_inclip = float(params.get(f"w_inclip_{postfix}", w_inclip))
        w_clip2clip = float(params.get(f"w_clip2clip_{postfix}", w_clip2clip))

    clip_results: list[dict[str, Any]] = []
    video_results: list[dict[str, Any]] = []
    overall_scores: list[float] = []
    for entry in entries:
        prompt = str(entry["prompt"])
        clip_paths = list(entry["clip_paths"])

        inclip_scores: list[float] = []
        first_frames: list[np.ndarray] = []
        for clip_path in clip_paths:
            clip_score, clip_payload, first_frame = _clip_sequence_score(
                clip_path,
                extractor=inclip_extractor,
            )
            clip_payload["long_video_prompt"] = prompt
            clip_results.append(clip_payload)
            if clip_score is not None:
                inclip_scores.append(float(clip_score))
            if first_frame is not None:
                first_frames.append(first_frame)

        inclip_score = _mean(inclip_scores)
        clip2clip_score, clip2clip_backend, clip2clip_transitions = _frames_sequence_score(
            first_frames,
            extractor=clip2clip_extractor,
        )
        mapped_clip2clip_score = (
            None
            if clip2clip_score is None
            else _mapped_clip2clip_score(float(clip2clip_score), mapping_table)
        )
        fused_score = inclip_score
        if inclip_score is not None and mapped_clip2clip_score not in (None, 0.0):
            fused_score = float(inclip_score) * float(w_inclip) + float(mapped_clip2clip_score) * float(w_clip2clip)
        if fused_score is not None:
            overall_scores.append(float(fused_score))

        video_results.append(
            {
                "video_path": _long_video_path(prompt, clip_paths),
                "video_results": fused_score,
                "inclip_score": inclip_score,
                "clip2clip_score": clip2clip_score,
                "mapped_clip2clip_score": mapped_clip2clip_score,
                "clip_count": len(clip_paths),
                "clip2clip_backend": clip2clip_backend,
                "clip2clip_transition_scores": _round_list(clip2clip_transitions),
                "w_inclip": float(w_inclip),
                "w_clip2clip": float(w_clip2clip),
            }
        )

    return _mean(overall_scores), clip_results, video_results


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


def compute_long_subject_consistency(
    json_dir: str | Path,
    device: Any,
    submodules_list: Any,
    **kwargs: Any,
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    del device
    del submodules_list

    extractor_name = str(kwargs.get("sb_clip2clip_feat_extractor") or "dinov2").strip().lower()
    mapping_path = kwargs.get("sb_mapping_file_path")
    return _native_long_consistency(
        json_dir,
        dimension="subject_consistency",
        inclip_extractor=_subject_inclip_extractor,
        clip2clip_extractor=_subject_clip2clip_extractor(extractor_name),
        mapping_file_path=mapping_path,
        slow_fast_eval_config=kwargs.get("slow_fast_eval_config"),
        w_inclip=float(kwargs.get("w_inclip", 1.0)),
        w_clip2clip=float(kwargs.get("w_clip2clip", 0.0)),
        dev_flag=bool(kwargs.get("dev_flag", True)),
    )


def compute_long_background_consistency(
    json_dir: str | Path,
    device: Any,
    submodules_list: Any,
    **kwargs: Any,
) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    del device
    del submodules_list

    extractor_name = str(kwargs.get("bg_clip2clip_feat_extractor") or "clip").strip().lower()
    mapping_path = kwargs.get("bg_mapping_file_path")
    return _native_long_consistency(
        json_dir,
        dimension="background_consistency",
        inclip_extractor=_background_inclip_extractor,
        clip2clip_extractor=_background_clip2clip_extractor(extractor_name),
        mapping_file_path=mapping_path,
        slow_fast_eval_config=kwargs.get("slow_fast_eval_config"),
        w_inclip=float(kwargs.get("w_inclip", 1.0)),
        w_clip2clip=float(kwargs.get("w_clip2clip", 0.0)),
        dev_flag=bool(kwargs.get("dev_flag", True)),
    )


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
    "compute_long_background_consistency",
    "compute_long_dynamic_degree",
    "compute_long_imaging_quality",
    "compute_long_motion_smoothness",
    "compute_long_subject_consistency",
]
