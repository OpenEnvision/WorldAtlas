from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
from PIL import Image

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.taxonomy import (
    artifact_type_for_modality,
    assign_split,
    control_signals_for_sample,
    task_family_for_suite,
)
from worldarena.common.serialization import ensure_dir, write_json


TRACK_TO_METRIC_GROUPS: dict[str, list[str]] = {
    "i2v": [
        "video_physics",
        "geometry_3d",
        "temporal_consistency",
        "world_4d",
    ],
}

CORE_METRICS_BY_GROUP: dict[str, tuple[str, ...]] = {
    "video_physics": ("mse", "st_iou", "s_iou", "ws_iou"),
    "geometry_3d": ("abs_rel", "rmse", "delta1", "delta2", "delta3"),
    "temporal_consistency": (
        "depth_warp_l1",
        "depth_warp_charb",
        "rgb_warp_charb",
        "epe",
        "fl_all",
        "one_px_out",
    ),
    "world_4d": (
        "chamfer_4d",
        "worldline_l2_error",
        "worldline_mean_drift",
        "worldline_final_drift",
        "worldline_fail_rate",
        "worldline_length",
    ),
}

OPTIONAL_METRICS_BY_GROUP: dict[str, tuple[str, ...]] = {
    "temporal_consistency": ("rgb_warp_lpips",),
    "world_4d": (
        "novel_time_depth_l1",
        "novel_time_depth_charb",
        "novel_time_rgb_l1",
        "novel_time_rgb_charb",
    ),
}


def flatten_metric_groups(metric_groups: dict[str, Sequence[str]]) -> list[str]:
    return [
        metric_name
        for group_name in TRACK_TO_METRIC_GROUPS["i2v"]
        for metric_name in metric_groups.get(group_name, ())
    ]


REQUIRED_CORE_METRICS: tuple[str, ...] = tuple(flatten_metric_groups(CORE_METRICS_BY_GROUP))
OPTIONAL_PHYSICS_METRICS: tuple[str, ...] = tuple(
    flatten_metric_groups(OPTIONAL_METRICS_BY_GROUP)
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_video_info(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if frame_count <= 0:
        raise RuntimeError(f"video contains no decodable frames: {video_path}")
    return {
        "fps": fps if fps > 0 else 8.0,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / (fps if fps > 0 else 8.0),
    }


def _export_first_frame(video_path: Path, image_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"failed to read first frame from: {video_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(image_path)


def _discover_sample_ids(assets_root: Path, sample_ids: Sequence[str] | None) -> list[str]:
    if sample_ids:
        return [str(sample_id) for sample_id in sample_ids]
    discovered = sorted(path.stem for path in assets_root.glob("*.json"))
    if discovered:
        return discovered
    return []


def _resolve_physics_data_root(physics_pipeline_root: Path) -> Path:
    override_root = os.environ.get("WORLDARENA_PHYSICS_DATA_ROOT")
    if override_root:
        return Path(override_root).expanduser().resolve()

    benchmark_root = physics_pipeline_root.parent.parent
    return (benchmark_root / ".runtime" / "physics_pipeline" / "data_root").resolve()


def _case_paths(output_root: Path, benchmark_sample_id: str) -> dict[str, Path]:
    case_root = output_root / "cases" / benchmark_sample_id
    simulator_root = case_root / "simulator"
    world_gt_root = case_root / "world_gt"
    return {
        "case_root": case_root,
        "simulator_root": simulator_root,
        "submitted_video_path": simulator_root / "submitted_video.mp4",
        "metadata_template_path": simulator_root / "metadata_template.json",
        "world_gt_root": world_gt_root,
    }


def _build_metadata_template(
    meta: dict[str, Any],
    *,
    submitted_video_path: Path,
) -> dict[str, Any]:
    return {
        "conditioning_tracks": ["i2v"],
        "video_prompt": str(meta.get("video_prompt", "")).strip(),
        "fg_prompt": str(meta.get("fg_prompt", "")).strip(),
        "primary": str(meta.get("primary", "")).strip(),
        "secondary": str(meta.get("secondary", "")).strip(),
        "primary_obj_rot_axis": meta.get("primary_obj_rot_axis"),
        "submitted_video_path": str(submitted_video_path.resolve()),
        "notes": (
            "WorldArena simulator-backed physics case. "
            "Populate submitted_video.mp4 with a generated rollout before evaluation."
        ),
    }


def _build_sample(
    *,
    output_root: Path,
    conditioning_root: Path,
    physics_pipeline_root: Path,
    physics_data_root: Path,
    assets_root: Path,
    source_sample_id: str,
    meta: dict[str, Any],
    video_path: Path,
    video_info: dict[str, Any],
    split_ratios: dict[str, float],
) -> BenchmarkSample:
    benchmark_sample_id = f"physics__{source_sample_id}__i2v"
    paths = _case_paths(output_root, benchmark_sample_id)
    ensure_dir(paths["simulator_root"])
    ensure_dir(paths["world_gt_root"])

    conditioning_path = conditioning_root / f"{source_sample_id}.png"
    if not conditioning_path.exists():
        _export_first_frame(video_path, conditioning_path)

    metadata_template = _build_metadata_template(
        meta,
        submitted_video_path=paths["submitted_video_path"],
    )
    paths["metadata_template_path"].write_text(
        json.dumps(metadata_template, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    prompt = str(meta.get("video_prompt", "")).strip()
    if not prompt:
        raise ValueError(f"physics asset {source_sample_id} is missing video_prompt")

    source_group_id = f"physics::{source_sample_id}"
    return BenchmarkSample(
        sample_id=benchmark_sample_id,
        suite="experimental",
        split=assign_split(source_group_id, split_ratios),
        asset_level="video_nonformal",
        modality="video",
        is_formal=False,
        eligible_for_official=False,
        category_path="physics/i2v",
        path=str(video_path.resolve()),
        relative_path=str(video_path.resolve().relative_to(assets_root.parent.resolve())),
        reference_path=str(video_path.resolve()),
        prediction_stem=benchmark_sample_id,
        conditioning_strategy="external_image",
        prompt_current=prompt,
        prompt_target=prompt,
        style=None,
        environment=None,
        scene=None,
        motion_category=str(meta.get("primary", "")).strip() or "physics",
        source_name="worldarena_physics",
        source_type="simulator_video",
        source_group_id=source_group_id,
        license_bucket="internal",
        benchmark_family="worldarena_physics",
        task_family=task_family_for_suite("experimental", track="physics_i2v"),
        artifact_type=artifact_type_for_modality("video"),
        control_signals=control_signals_for_sample(
            modality="video",
            conditioning_strategy="external_image",
            has_pose=False,
        ),
        has_annotation=True,
        has_instruction=True,
        has_pose=False,
        has_mask=False,
        mask_count=0,
        width=int(video_info["width"]) or None,
        height=int(video_info["height"]) or None,
        fps=float(video_info["fps"]),
        duration_seconds=float(video_info["duration_seconds"]),
        duration_bucket=None,
        conditioning_path=str(conditioning_path.resolve()),
        track="physics_i2v",
        physics_spec={
            "benchmark_family": "worldarena_physics",
            "backend": "simulator_reference",
            "source_sample_id": source_sample_id,
            "frame_count": int(video_info["frame_count"]),
            "resolution": [int(video_info["width"]), int(video_info["height"])],
            "metrics_design": {
                group_name: list(CORE_METRICS_BY_GROUP[group_name])
                for group_name in TRACK_TO_METRIC_GROUPS["i2v"]
                if group_name in CORE_METRICS_BY_GROUP
            },
            "optional_metrics_design": {
                group_name: list(OPTIONAL_METRICS_BY_GROUP[group_name])
                for group_name in TRACK_TO_METRIC_GROUPS["i2v"]
                if group_name in OPTIONAL_METRICS_BY_GROUP
            },
            "metric_contract": {
                "required_core_metrics": list(REQUIRED_CORE_METRICS),
                "optional_metrics": list(OPTIONAL_PHYSICS_METRICS),
                "require_full_frame_coverage": True,
                "require_pred_world_artifacts": True,
            },
        },
        physics_case={
            "physics_pipeline_root": str(physics_pipeline_root.resolve()),
            "physics_data_root": str(physics_data_root.resolve()),
            "source_sample_id": source_sample_id,
            "benchmark_sample_id": benchmark_sample_id,
            "case_root": str(paths["case_root"].resolve()),
            "simulator_root": str(paths["simulator_root"].resolve()),
            "submitted_video_path": str(paths["submitted_video_path"].resolve()),
            "metadata_template_path": str(paths["metadata_template_path"].resolve()),
            "world_gt_root": str(paths["world_gt_root"].resolve()),
            "track": "i2v",
            "physics_export_benchmark_name": "worldarena_physics",
            "physics_render_split": "original_length",
        },
    )


def write_physics_manifest_artifacts(
    manifest: list[BenchmarkSample],
    output_root: Path,
) -> Path:
    ensure_dir(output_root)
    manifest_path = output_root / "worldarena_physics_manifest.jsonl"
    lines = [json.dumps(sample.to_dict(), ensure_ascii=False) for sample in manifest]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_root / "worldarena_physics_manifest_summary.json",
        {
            "generated_at": utc_now_iso(),
            "items": len(manifest),
            "suite_counts": {
                suite: sum(1 for sample in manifest if sample.suite == suite)
                for suite in sorted({sample.suite for sample in manifest})
            },
            "track_counts": {
                track: sum(1 for sample in manifest if sample.track == track)
                for track in sorted({sample.track for sample in manifest if sample.track})
            },
            "required_core_metrics": list(REQUIRED_CORE_METRICS),
        },
    )
    return manifest_path


def build_worldarena_physics_manifest(
    *,
    physics_pipeline_root: str | Path,
    output_root: str | Path,
    sample_ids: Sequence[str] | None = None,
    split_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    physics_pipeline_root = Path(physics_pipeline_root).expanduser().resolve()
    if not physics_pipeline_root.exists():
        raise FileNotFoundError(f"physics pipeline root not found: {physics_pipeline_root}")

    output_root = Path(output_root).expanduser().resolve()
    assets_root = physics_pipeline_root / "assets"
    if not assets_root.exists():
        raise FileNotFoundError(f"physics assets root not found: {assets_root}")
    physics_data_root = _resolve_physics_data_root(physics_pipeline_root)
    conditioning_root = ensure_dir(output_root / "conditioning")
    ensure_dir(output_root / "cases")

    resolved_split_ratios = split_ratios or {
        "dev_public": 0.2,
        "dev_private": 0.1,
        "test_hidden": 0.7,
    }
    resolved_sample_ids = _discover_sample_ids(assets_root, sample_ids)
    manifest: list[BenchmarkSample] = []

    for source_sample_id in resolved_sample_ids:
        video_path = assets_root / f"{source_sample_id}.mp4"
        meta_path = assets_root / f"{source_sample_id}.json"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)

        meta = _read_json(meta_path)
        video_info = _load_video_info(video_path)
        manifest.append(
            _build_sample(
                output_root=output_root,
                conditioning_root=conditioning_root,
                physics_pipeline_root=physics_pipeline_root,
                physics_data_root=physics_data_root,
                assets_root=assets_root,
                source_sample_id=source_sample_id,
                meta=meta,
                video_path=video_path,
                video_info=video_info,
                split_ratios=resolved_split_ratios,
            )
        )

    manifest_path = write_physics_manifest_artifacts(manifest, output_root)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "physics_pipeline_root": str(physics_pipeline_root),
        "physics_data_root": str(physics_data_root.resolve()),
        "conditioning_root": str(conditioning_root.resolve()),
        "cases_root": str((output_root / "cases").resolve()),
        "items": len(manifest),
        "sample_ids": resolved_sample_ids,
        "required_core_metrics": list(REQUIRED_CORE_METRICS),
    }
