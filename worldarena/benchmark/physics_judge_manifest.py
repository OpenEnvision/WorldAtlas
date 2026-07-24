"""Build manifests for physics judge evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from worldarena.benchmark.physics_judge_taxonomy import build_judge_card
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.taxonomy import (
    artifact_type_for_modality,
    benchmark_split,
    control_signals_for_sample,
    task_family_for_suite,
)
from worldarena.common.serialization import ensure_dir, write_json


def _read_records(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of records in {index_path}")
    return [record for record in payload if isinstance(record, dict)]


def _resolve_video_path(data_root: Path, record: dict[str, Any]) -> Path:
    raw_path = Path(str(record.get("path", "")).strip())
    if raw_path.is_file():
        return raw_path.resolve()
    fallback = (
        data_root
        / "videos_by_category"
        / str(record.get("group", "")).strip()
        / str(record.get("dimension", "")).strip()
        / str(record.get("video_name", "")).strip()
    )
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"failed to resolve local video for {record.get('video_name')}")


def _resolve_first_frame_path(data_root: Path, record: dict[str, Any]) -> Path | None:
    order = record.get("order")
    dimension = str(record.get("dimension", "")).strip()
    if order is None or not dimension:
        return None
    candidate = data_root / "first_frame" / f"{int(order):03d}_{dimension}_{int(record.get('rank_within_dimension', 0)):02d}.jpg"
    if candidate.exists():
        return candidate.resolve()
    return None


def _probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    fps = fps if fps > 0 else 24.0
    duration_seconds = frame_count / fps if frame_count > 0 else 0.0
    return {
        "width": width or None,
        "height": height or None,
        "fps": fps,
        "duration_seconds": duration_seconds or None,
    }


def build_wmbench_physics_judge_manifest(
    *,
    data_root: Path,
    index_name: str = "benchmark_physics_subset_extended.json",
    limit: int | None = None,
) -> list[BenchmarkSample]:
    index_path = (data_root / index_name).resolve()
    records = _read_records(index_path)
    if limit is not None:
        records = records[:limit]

    manifest: list[BenchmarkSample] = []
    for record in records:
        video_path = _resolve_video_path(data_root, record)
        first_frame_path = _resolve_first_frame_path(data_root, record)
        video_info = _probe_video(video_path)
        group = str(record.get("group", "")).strip()
        dimension = str(record.get("dimension", "")).strip()
        video_stem = video_path.stem.replace("-", "_")
        sample_id = f"physics_judge__{group}__{dimension}__{video_stem}"
        caption = str(record.get("captions", "")).strip()
        prompt_target = caption or f"Generate a video showing {dimension.replace('_', ' ')}."
        try:
            relative_path = str(video_path.relative_to(data_root))
        except ValueError:
            relative_path = video_path.name
        manifest.append(
            BenchmarkSample(
                sample_id=sample_id,
                suite="experimental",
                split=benchmark_split(),
                asset_level="video_dynamic",
                modality="video",
                is_formal=False,
                eligible_for_official=False,
                category_path=f"physics/{group}/{dimension}",
                path=str(video_path),
                relative_path=relative_path,
                reference_path=str(video_path),
                prediction_stem=sample_id,
                conditioning_strategy="first_frame",
                prompt_current=caption,
                prompt_target=prompt_target,
                style="photorealistic",
                environment="physics",
                scene=dimension,
                motion_category=dimension,
                source_name="wmbench_physics",
                source_type="reference_video",
                source_group_id=f"wmbench_physics::{group}::{dimension}::{record.get('family_id', video_stem)}",
                license_bucket="unknown",
                prompt_sequence=[caption] if caption else [],
                camera_path=[],
                generation_mode="text_to_video",
                benchmark_family="worldarena",
                task_family=task_family_for_suite("experimental", track="physics_judge"),
                artifact_type=artifact_type_for_modality("video"),
                control_signals=control_signals_for_sample(
                    modality="video",
                    conditioning_strategy="first_frame",
                    has_pose=False,
                ),
                annotation_path=None,
                mask_path=None,
                has_annotation=False,
                has_instruction=False,
                has_pose=False,
                has_mask=False,
                mask_count=0,
                width=video_info["width"],
                height=video_info["height"],
                fps=video_info["fps"],
                duration_seconds=video_info["duration_seconds"],
                duration_bucket=None,
                conditioning_path=str(first_frame_path) if first_frame_path else None,
                conditioning_frame_count=1 if first_frame_path else None,
                evaluation_start_ratio=0.0,
                track="physics_judge",
                physics_spec={
                    "group": group,
                    "dimension": dimension,
                    "reference_label": str(record.get("label", "")).strip(),
                    "reference_caption": caption,
                    "selection_note": str(record.get("selection_note", "")).strip(),
                    "judge_card": build_judge_card(
                        group=group,
                        dimension=dimension,
                        caption=caption,
                        label=str(record.get("label", "")).strip() or None,
                    ),
                },
                physics_case={
                    "index_path": str(index_path),
                    "video_path": str(video_path),
                    "first_frame_path": str(first_frame_path) if first_frame_path else None,
                },
            )
        )
    return manifest


def write_physics_judge_manifest_artifacts(
    *,
    manifest: list[BenchmarkSample],
    output_dir: Path,
    snapshot_name: str,
    source_root: Path,
) -> Path:
    ensure_dir(output_dir)
    manifest_path = output_dir / "worldarena_physics_judge_manifest.jsonl"
    lines = [json.dumps(sample.to_dict(), ensure_ascii=False) for sample in manifest]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "worldarena_physics_judge_summary.json",
        {
            "snapshot_name": snapshot_name,
            "source_root": str(source_root),
            "items": len(manifest),
            "groups": sorted({sample.physics_spec.get("group", "") for sample in manifest if sample.physics_spec}),
            "dimensions": sorted(
                {sample.physics_spec.get("dimension", "") for sample in manifest if sample.physics_spec}
            ),
        },
    )
    return manifest_path


__all__ = [
    "build_wmbench_physics_judge_manifest",
    "write_physics_judge_manifest_artifacts",
]
