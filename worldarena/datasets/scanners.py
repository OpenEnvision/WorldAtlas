"""Scan dataset collection roots and discover benchmark-eligible assets."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from worldarena.common.annotation_index import build_annotation_reference, load_annotation_entry
from worldarena.datasets.config import CollectionConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ANNOTATION_FILES = [
    "caption.json",
    "instructions.json",
    "indexes.txt",
    "intrinsics.npy",
    "poses.npy",
    "dyn_masks.npz",
]
VIPE_ANNOTATION_ENV = "WORLD_ARENA_VIPE_ANNOTATION_ROOT"
VIPE_ANNOTATION_FILES = [
    "camera_type.txt",
    "frame_indices.npy",
    "poses_c2w.npy",
    "poses_w2c.npy",
    "quality_report.json",
    "timestamps.npy",
]


def should_skip_path(path: Path, prefixes: list[str]) -> bool:
    return any(part.startswith(prefix) for part in path.parts for prefix in prefixes)


def bucketize_duration(duration_seconds: float | None) -> str | None:
    if duration_seconds is None:
        return None
    if duration_seconds < 5:
        return "0-5s"
    if duration_seconds < 10:
        return "5-10s"
    if duration_seconds < 20:
        return "10-20s"
    return "20s+"


def _base_record(collection: CollectionConfig, path: Path) -> dict[str, Any]:
    relative_path = path.relative_to(collection.root)
    return {
        "collection": collection.name,
        "group": collection.group,
        "modality": collection.modality,
        "path": str(path),
        "relative_path": str(relative_path),
        "filename": path.name,
        "stem": path.stem,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_instruction_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "instruction_segment_count": 0,
            "instruction_labels": None,
            "instruction_total_frames": None,
        }

    labels: list[str] = []
    total_frames = 0
    for segment, values in payload.items():
        labels.extend(str(value) for value in values if value)
        match = re.match(r"(\d+)->(\d+)", str(segment))
        if match:
            total_frames += int(match.group(2)) - int(match.group(1)) + 1

    deduped_labels = list(dict.fromkeys(labels))
    return {
        "instruction_segment_count": len(payload),
        "instruction_labels": " | ".join(deduped_labels) if deduped_labels else None,
        "instruction_total_frames": total_frames or None,
    }


def _extract_caption_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "has_caption": False,
            "tag_scene_type_primary": None,
            "tag_scene_type_secondary": None,
            "tag_brightness": None,
            "tag_time_of_day": None,
            "tag_weather": None,
            "tag_crowd_density": None,
        }

    tags = payload.get("CategoryTags", {}) if isinstance(payload, dict) else {}
    if not isinstance(tags, dict):
        tags = {}

    scene_type = tags.get("sceneType", {})
    if not isinstance(scene_type, dict):
        scene_type = {}

    return {
        "has_caption": True,
        "tag_scene_type_primary": scene_type.get("first"),
        "tag_scene_type_secondary": scene_type.get("second"),
        "tag_brightness": tags.get("brightness"),
        "tag_time_of_day": tags.get("timeOfDay"),
        "tag_weather": tags.get("weather"),
        "tag_crowd_density": tags.get("crowdDensity"),
    }


def read_annotation_bundle(annotation_dir: Path) -> dict[str, Any]:
    has_annotation = annotation_dir.exists()
    present = [name for name in ANNOTATION_FILES if (annotation_dir / name).exists()]
    missing = [name for name in ANNOTATION_FILES if name not in present]
    caption_payload = _read_json(annotation_dir / "caption.json") if has_annotation else None
    instruction_payload = _read_json(annotation_dir / "instructions.json") if has_annotation else None

    payload = {
        "annotation_path": str(annotation_dir) if has_annotation else None,
        "has_annotation": has_annotation,
        "annotation_complete": has_annotation and len(present) == len(ANNOTATION_FILES),
        "annotation_present_count": len(present),
        "annotation_missing_files": " | ".join(missing) if missing else None,
        "has_mask": (annotation_dir / "dyn_masks.npz").exists(),
        "mask_count": 1 if (annotation_dir / "dyn_masks.npz").exists() else 0,
        "mask_kind": "npz_archive" if (annotation_dir / "dyn_masks.npz").exists() else None,
    }
    payload.update(_extract_caption_metadata(caption_payload))
    payload.update(_extract_instruction_metadata(instruction_payload))
    return payload


def read_annotation_payloads(
    *,
    annotation_path: str | None,
    caption_payload: dict[str, Any] | None,
    instruction_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    has_annotation = isinstance(caption_payload, dict) or isinstance(instruction_payload, dict)
    present: list[str] = []
    if isinstance(caption_payload, dict):
        present.append("caption.json")
    if isinstance(instruction_payload, dict):
        present.append("instructions.json")
    missing = [name for name in ANNOTATION_FILES if name not in present]

    payload = {
        "annotation_path": annotation_path if has_annotation else None,
        "has_annotation": has_annotation,
        "annotation_complete": has_annotation and len(present) == len(ANNOTATION_FILES),
        "annotation_present_count": len(present),
        "annotation_missing_files": " | ".join(missing) if missing else None,
        "has_mask": False,
        "mask_count": 0,
        "mask_kind": None,
    }
    payload.update(_extract_caption_metadata(caption_payload))
    payload.update(_extract_instruction_metadata(instruction_payload))
    return payload


def _count_mask_images(mask_dir: Path) -> int:
    if not mask_dir.exists():
        return 0
    return sum(1 for path in mask_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _image_annotation_dir(path: Path) -> Path:
    return path.parent / "annotations" / f"{path.stem}_ann"


def _has_annotation_payload(annotation_dir: Path) -> bool:
    if not annotation_dir.is_dir():
        return False
    return any((annotation_dir / name).exists() for name in (*ANNOTATION_FILES, *VIPE_ANNOTATION_FILES))


def _first_annotation_dir_with_payload(*annotation_dirs: Path | None) -> Path | None:
    for annotation_dir in annotation_dirs:
        if annotation_dir is not None and _has_annotation_payload(annotation_dir):
            return annotation_dir
    return None


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _configured_vipe_annotation_roots(static_root: Path) -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get(VIPE_ANNOTATION_ENV)
    if env_root:
        roots.append(Path(env_root).expanduser())

    if static_root.name == "static":
        video_root = static_root.parent
        roots.append(video_root.with_name(f"{video_root.name}_vipe_ann"))
    else:
        roots.append(static_root.with_name(f"{static_root.name}_vipe_ann"))
    return _unique_paths(roots)


def _matching_vipe_annotation_dir(annotation_parent: Path, stem: str) -> Path | None:
    for name in (f"{stem}_vipe_ann", f"{stem}_ann", stem):
        candidate = annotation_parent / name
        if _has_annotation_payload(candidate):
            return candidate

    if not annotation_parent.is_dir():
        return None

    matches = [
        child
        for child in sorted(annotation_parent.iterdir())
        if child.is_dir()
        and child.name.startswith(stem)
        and child.name.endswith("_vipe_ann")
        and _has_annotation_payload(child)
    ]
    return matches[0] if matches else None


def _static_video_vipe_annotation_dir(static_root: Path, path: Path) -> Path | None:
    relative_parent = path.relative_to(static_root).parent
    candidate_parents: list[Path] = []
    for vipe_root in _configured_vipe_annotation_roots(static_root):
        candidate_parents.extend(
            [
                vipe_root / relative_parent / "annotations_vipe",
                vipe_root / "static" / relative_parent / "annotations_vipe",
            ]
        )
        if not relative_parent.parts or relative_parent.parts[0] not in {"photorealistic", "stylized"}:
            for style in ("photorealistic", "stylized"):
                candidate_parents.append(vipe_root / "static" / style / relative_parent / "annotations_vipe")

    for annotation_parent in _unique_paths(candidate_parents):
        annotation_dir = _matching_vipe_annotation_dir(annotation_parent, path.stem)
        if annotation_dir is not None:
            return annotation_dir
    return None


def _image_annotation_entry(collection: CollectionConfig, path: Path) -> tuple[str, dict[str, Any]] | None:
    dataset_root = collection.root.parent
    index_path = dataset_root / "image_annotations.json"
    relative_path = path.relative_to(dataset_root)
    entry = load_annotation_entry(index_path, relative_path)
    if entry is None:
        return None
    return build_annotation_reference(index_path, relative_path), entry


def scan_static_image(collection: CollectionConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(collection.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_path = path.relative_to(collection.root)
        if should_skip_path(relative_path, collection.exclude_dir_prefixes):
            continue
        parts = relative_path.parts
        if len(parts) < 4:
            continue

        record = _base_record(collection, path)
        record.update(
            {
                "source_type": "static_image",
                "motion_regime": "static",
                "style": parts[0],
                "environment": parts[1],
                "scene": "/".join(parts[2:-1]),
                "has_mask": False,
                "mask_count": 0,
                "mask_kind": None,
            }
        )
        annotation_entry = _image_annotation_entry(collection, path)
        if annotation_entry is not None:
            annotation_path, payload = annotation_entry
            record.update(
                read_annotation_payloads(
                    annotation_path=annotation_path,
                    caption_payload=payload.get("caption") if isinstance(payload.get("caption"), dict) else None,
                    instruction_payload=payload.get("instructions")
                    if isinstance(payload.get("instructions"), dict)
                    else None,
                )
            )
        else:
            record.update(read_annotation_bundle(_image_annotation_dir(path)))
        records.append(record)
    return records


def scan_dynamic_image(collection: CollectionConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(collection.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_path = path.relative_to(collection.root)
        if should_skip_path(relative_path, collection.exclude_dir_prefixes):
            continue
        parts = relative_path.parts
        if len(parts) < 4 or parts[1] != "images":
            continue

        style, _, motion_category = parts[:3]
        mask_dir = collection.root / style / "masks" / motion_category / path.stem
        mask_count = _count_mask_images(mask_dir)

        record = _base_record(collection, path)
        record.update(
            {
                "source_type": "dynamic_image",
                "motion_regime": "dynamic",
                "style": style,
                "motion_category": motion_category,
                "has_mask": mask_count > 0,
                "mask_count": mask_count,
                "mask_kind": "mask_directory" if mask_count > 0 else None,
                "mask_path": str(mask_dir) if mask_dir.exists() else None,
            }
        )
        annotation_entry = _image_annotation_entry(collection, path)
        if annotation_entry is not None:
            annotation_path, payload = annotation_entry
            annotation_payload = read_annotation_payloads(
                annotation_path=annotation_path,
                caption_payload=payload.get("caption") if isinstance(payload.get("caption"), dict) else None,
                instruction_payload=payload.get("instructions")
                if isinstance(payload.get("instructions"), dict)
                else None,
            )
        else:
            annotation_payload = read_annotation_bundle(_image_annotation_dir(path))
        if mask_count > 0:
            annotation_payload["has_mask"] = True
            annotation_payload["mask_count"] = mask_count
            annotation_payload["mask_kind"] = "mask_directory"
        record.update(annotation_payload)
        records.append(record)
    return records


def scan_static_video(collection: CollectionConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(collection.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative_path = path.relative_to(collection.root)
        if should_skip_path(relative_path, collection.exclude_dir_prefixes):
            continue
        parts = relative_path.parts
        if len(parts) < 5:
            continue

        spatialvid_annotation_dir = path.parent / "annotations" / f"{path.stem}_SpatialVID_ann"
        vipe_annotation_dir = _static_video_vipe_annotation_dir(collection.root, path)
        annotation_dir = (
            _first_annotation_dir_with_payload(vipe_annotation_dir, spatialvid_annotation_dir)
            or spatialvid_annotation_dir
        )
        record = _base_record(collection, path)
        record.update(
            {
                "source_type": "static_video",
                "motion_regime": "static",
                "style": parts[0],
                "environment": parts[1],
                "scene": parts[2],
                "duration_bucket": parts[3],
            }
        )
        record.update(read_annotation_bundle(annotation_dir))
        records.append(record)
    return records


def scan_dynamic_video(collection: CollectionConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    annotation_root = collection.root / "annotations"
    for path in sorted(collection.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative_path = path.relative_to(collection.root)
        if should_skip_path(relative_path, collection.exclude_dir_prefixes):
            continue
        parts = relative_path.parts
        if len(parts) < 4 or parts[1] != "videos":
            continue

        style, _, motion_category = parts[:3]
        annotation_dir = annotation_root / motion_category / f"{path.stem}_ann"
        mask_dir = collection.root / style / "masks" / motion_category / path.stem / "masks"
        mask_count = _count_mask_images(mask_dir)

        record = _base_record(collection, path)
        record.update(
            {
                "source_type": "dynamic_video",
                "motion_regime": "dynamic",
                "style": style,
                "motion_category": motion_category,
                "mask_path": str(mask_dir) if mask_dir.exists() else None,
                "mask_count": mask_count,
                "has_mask": mask_count > 0,
                "mask_kind": "mask_directory" if mask_count > 0 else None,
            }
        )
        annotation_payload = read_annotation_bundle(annotation_dir)
        if mask_count > 0:
            annotation_payload["has_mask"] = True
            annotation_payload["mask_count"] = mask_count
            annotation_payload["mask_kind"] = "mask_directory"
        record.update(annotation_payload)
        records.append(record)
    return records


def scan_aaa_games(collection: CollectionConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    videos_root = collection.root / "videos"
    annotations_root = collection.root / "annotations"
    for path in sorted(videos_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        record = _base_record(collection, path)
        annotation_dir = annotations_root / path.stem.zfill(3)
        record.update(
            {
                "source_type": "aaa_games",
                "motion_regime": "gameplay",
                "style": "synthetic",
                "environment": "gameplay",
                "scene": "AAA_Games",
            }
        )
        record.update(read_annotation_bundle(annotation_dir))
        records.append(record)
    return records


def scan_review_video(collection: CollectionConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(collection.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative_path = path.relative_to(collection.root)
        if should_skip_path(relative_path, collection.exclude_dir_prefixes):
            continue
        parts = relative_path.parts
        record = _base_record(collection, path)
        record.update(
            {
                "source_type": "review_video",
                "motion_regime": "auxiliary",
                "review_bucket": parts[0] if parts else "misc",
                "style": None,
                "scene": None,
                "has_annotation": False,
                "annotation_complete": False,
                "annotation_present_count": 0,
                "annotation_missing_files": None,
                "has_mask": False,
                "mask_count": 0,
                "mask_kind": None,
            }
        )
        records.append(record)
    return records


def _load_physics_subset_metadata(root: Path) -> dict[str, dict[str, Any]]:
    metadata_path = root / "benchmark_physics_subset_extended.csv"
    if not metadata_path.exists():
        return {}

    def number(value: str | None) -> int | float | None:
        if value is None or value == "":
            return None
        try:
            numeric = float(value)
        except ValueError:
            return None
        return int(numeric) if numeric.is_integer() else numeric

    rows: dict[str, dict[str, Any]] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video_name = row.get("video_name")
            if not video_name:
                continue
            subset_stem = Path(row.get("symlink_path") or video_name).with_suffix(".jpg").name
            first_frame_path = root / "first_frame" / subset_stem
            rows[video_name] = {
                "physics_subset_order": number(row.get("order")),
                "physics_rank_within_dimension": number(row.get("rank_within_dimension")),
                "physics_quota": number(row.get("quota")),
                "physics_label": row.get("label") or None,
                "caption_text": row.get("captions") or None,
                "visual_quality_score": number(row.get("visual_quality_score")),
                "motion_score_v2": number(row.get("motion_score_v2")),
                "text_bbox_ratio": number(row.get("text_bbox_ratio")),
                "selection_score": number(row.get("selection_score")),
                "match_type": row.get("match_type") or None,
                "annotation_labels": row.get("annotation_labels") or None,
                "family_id": row.get("family_id") or None,
                "all_matching_dimensions": row.get("all_matching_dimensions") or None,
                "selection_note": row.get("selection_note") or None,
                "metadata_width": number(row.get("width")),
                "metadata_height": number(row.get("height")),
                "metadata_duration_seconds": number(row.get("duration")),
                "metadata_fps": number(row.get("fps")),
                "first_frame_path": str(first_frame_path) if first_frame_path.exists() else None,
            }
    return rows


def scan_physics_video(collection: CollectionConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    metadata_by_video = _load_physics_subset_metadata(collection.root)
    videos_root = collection.root / "videos_by_category"

    for path in sorted(videos_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative_path = path.relative_to(collection.root)
        if should_skip_path(relative_path, collection.exclude_dir_prefixes):
            continue
        parts = relative_path.parts
        if len(parts) < 4 or parts[0] != "videos_by_category":
            continue

        physics_group, physics_dimension = parts[1], parts[2]
        metadata = metadata_by_video.get(path.name, {})
        record = _base_record(collection, path)
        record.update(
            {
                "source_type": "physics_video",
                "motion_regime": "physics",
                "style": physics_group,
                "scene": physics_dimension,
                "physics_group": physics_group,
                "physics_dimension": physics_dimension,
                "has_physics_metadata": bool(metadata),
                "has_caption": bool(metadata.get("caption_text")),
                "has_annotation": False,
                "annotation_complete": False,
                "annotation_present_count": 0,
                "annotation_missing_files": None,
                "has_mask": False,
                "mask_count": 0,
                "mask_kind": None,
            }
        )
        record.update(metadata)
        records.append(record)
    return records


SCANNERS: dict[str, Callable[[CollectionConfig], list[dict[str, Any]]]] = {
    "static_image": scan_static_image,
    "dynamic_image": scan_dynamic_image,
    "static_video": scan_static_video,
    "dynamic_video": scan_dynamic_video,
    "aaa_games": scan_aaa_games,
    "review_video": scan_review_video,
    "physics_video": scan_physics_video,
}


def scan_collections(collections: list[CollectionConfig]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for collection in collections:
        scanner = SCANNERS[collection.scanner]
        records.extend(scanner(collection))
    return records
