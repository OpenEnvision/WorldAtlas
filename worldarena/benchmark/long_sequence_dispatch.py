"""Dispatch long-sequence metric jobs across worker processes."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from worldarena.benchmark.long_video import build_long_video_manifest, prepare_long_video_clips


LONG_SEQUENCE_EVALUATOR_SPECS: dict[str, tuple[str, str]] = {
    "motion_smoothness": (
        "worldarena.benchmark.long_sequence_native",
        "compute_long_motion_smoothness",
    ),
    "dynamic_degree": (
        "worldarena.benchmark.long_sequence_native",
        "compute_long_dynamic_degree",
    ),
    "aesthetic_quality": (
        "worldarena.benchmark.long_sequence_native",
        "compute_long_aesthetic_quality",
    ),
    "imaging_quality": (
        "worldarena.benchmark.long_sequence_native",
        "compute_long_imaging_quality",
    ),
}
SUPPORTED_LONG_DIMENSIONS: tuple[str, ...] = tuple(LONG_SEQUENCE_EVALUATOR_SPECS)


def resolve_long_sequence_evaluator(
    dimension: str,
) -> Callable[[str, Any, Any], object]:
    try:
        module_name, attr_name = LONG_SEQUENCE_EVALUATOR_SPECS[dimension]
    except KeyError as exc:
        raise KeyError(f"unsupported WorldAtlas Arena long-sequence dimension: {dimension}") from exc
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _save_json(data: object, path: Path) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def evaluate_long_sequence_dimensions(
    *,
    videos_path: str | Path,
    output_path: str | Path,
    name: str,
    dimensions: Sequence[str],
    device: Any,
    threshold: float = 35.0,
    use_semantic_splitting: bool = False,
    clip_duration: float = 2.0,
    clip_fps: int = 8,
    **kwargs: Any,
) -> dict[str, object]:
    requested_dimensions = list(dict.fromkeys(str(dimension) for dimension in dimensions))
    unsupported = sorted(set(requested_dimensions) - set(SUPPORTED_LONG_DIMENSIONS))
    if unsupported:
        raise KeyError(f"unsupported WorldAtlas Arena long-sequence dimensions: {unsupported}")

    videos_root = Path(videos_path)
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    split_clip_root = prepare_long_video_clips(
        videos_root,
        threshold=threshold,
        use_semantic_splitting=use_semantic_splitting,
        clip_duration=clip_duration,
        clip_fps=clip_fps,
    )
    manifest_path = build_long_video_manifest(
        split_clip_root,
        output_path=output_root,
        name=name,
        dimension_list=requested_dimensions,
    )

    results_dict: dict[str, object] = {}
    for dimension in requested_dimensions:
        evaluate_func = resolve_long_sequence_evaluator(dimension)
        results_dict[dimension] = evaluate_func(
            str(manifest_path),
            device,
            None,
            **kwargs,
        )

    _save_json(results_dict, output_root / f"{name}_eval_results.json")
    return results_dict


__all__ = [
    "LONG_SEQUENCE_EVALUATOR_SPECS",
    "SUPPORTED_LONG_DIMENSIONS",
    "evaluate_long_sequence_dimensions",
    "resolve_long_sequence_evaluator",
]
