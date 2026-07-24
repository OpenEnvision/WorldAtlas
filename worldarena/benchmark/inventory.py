"""Scan dataset roots and build benchmark inventory artifacts."""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worldarena.common.annotation_index import build_annotation_reference, load_annotation_entry
from worldarena.common.media import probe_media
from worldarena.common.serialization import ensure_dir, write_json
from worldarena.datasets.scanners import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from worldarena.benchmark.config import BenchmarkConfig
from worldarena.benchmark.schemas import DiscoveredAsset

VIPE_ANNOTATION_ENV = "WORLD_ARENA_VIPE_ANNOTATION_ROOT"


def _count_mask_images(mask_dir: Path) -> int:
    if not mask_dir.exists():
        return 0
    return sum(
        1
        for path in mask_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _should_skip_hidden(relative_path: Path) -> bool:
    return any(part.startswith(".") for part in relative_path.parts)


def _image_annotation_dir(path: Path) -> Path:
    return path.parent / "annotations" / f"{path.stem}_ann"


def _image_annotation_index_path(image_root: Path) -> Path:
    return image_root / "image_annotations.json"


def _image_annotation_state(path: Path, image_root: Path) -> tuple[str | None, bool, bool, bool]:
    relative_path = path.relative_to(image_root)
    index_path = _image_annotation_index_path(image_root)
    entry = load_annotation_entry(index_path, relative_path)
    if entry is not None:
        has_annotation = isinstance(entry.get("caption"), dict)
        has_instruction = isinstance(entry.get("instructions"), dict)
        has_pose = False
        annotation_path = build_annotation_reference(index_path, relative_path) if has_annotation else None
        return annotation_path, has_annotation, has_instruction, has_pose

    annotation_dir = _image_annotation_dir(path)
    has_annotation, has_instruction, has_pose = _annotation_flags(annotation_dir)
    annotation_path = str(annotation_dir) if has_annotation else None
    return annotation_path, has_annotation, has_instruction, has_pose


def _discover_image_asset(path: Path, image_root: Path) -> DiscoveredAsset | None:
    relative_path = path.relative_to(image_root)
    if _should_skip_hidden(relative_path):
        return None
    parts = relative_path.parts
    if not parts:
        return None

    if parts[0] == "static" and len(parts) >= 5:
        _, style, environment, scene, _ = parts[:5]
        category_path = "/".join(parts[:-1])
        annotation_path, has_annotation, has_instruction, has_pose = _image_annotation_state(path, image_root)
        return DiscoveredAsset(
            asset_level="image_static",
            modality="image",
            is_formal=True,
            eligible_for_official=True,
            path=str(path),
            relative_path=str(relative_path),
            category_path=category_path,
            source_name="worldarena_formal_image",
            source_type="curated_image",
            source_group_id=f"{category_path}::{path.stem}",
            license_bucket="C",
            style=style,
            environment=environment,
            scene=scene,
            fine_class=scene,
            annotation_path=annotation_path,
            has_annotation=has_annotation,
            has_instruction=has_instruction,
            has_pose=has_pose,
            size_bytes=path.stat().st_size,
        )

    if parts[0] == "dynamic" and len(parts) >= 5:
        _, style = parts[:2]
        if parts[2] == "images":
            motion_category = parts[3]
            category_path = "/".join(parts[:-1])
            mask_dir = image_root / "dynamic" / style / "masks" / motion_category / path.stem
            mask_count = _count_mask_images(mask_dir)
            annotation_path, has_annotation, has_instruction, has_pose = _image_annotation_state(path, image_root)
            return DiscoveredAsset(
                asset_level="image_dynamic",
                modality="image",
                is_formal=True,
                eligible_for_official=True,
                path=str(path),
                relative_path=str(relative_path),
                category_path=category_path,
                source_name="worldarena_formal_image",
                source_type="curated_image",
                source_group_id=f"{category_path}::{path.stem}",
                license_bucket="C",
                style=style,
                motion_category=motion_category,
                fine_class=motion_category,
                annotation_path=annotation_path,
                mask_path=str(mask_dir) if mask_dir.exists() else None,
                has_annotation=has_annotation,
                has_instruction=has_instruction,
                has_pose=has_pose,
                has_mask=mask_count > 0,
                mask_count=mask_count,
                size_bytes=path.stat().st_size,
            )
        if parts[2].startswith("_backup") and len(parts) >= 6 and parts[3] == "images":
            motion_category = parts[4]
            category_path = "/".join(parts[:-1])
            annotation_path, has_annotation, has_instruction, has_pose = _image_annotation_state(path, image_root)
            return DiscoveredAsset(
                asset_level="image_nonformal",
                modality="image",
                is_formal=False,
                eligible_for_official=False,
                path=str(path),
                relative_path=str(relative_path),
                category_path=category_path,
                source_name="worldarena_backup_image",
                source_type="curated_image",
                source_group_id=f"{category_path}::{path.stem}",
                license_bucket="C",
                style=style,
                motion_category=motion_category,
                fine_class=motion_category,
                annotation_path=annotation_path,
                has_annotation=has_annotation,
                has_instruction=has_instruction,
                has_pose=has_pose,
                size_bytes=path.stat().st_size,
            )
    return None


def _annotation_flags(annotation_dir: Path) -> tuple[bool, bool, bool]:
    has_annotation = annotation_dir.exists()
    has_instruction = (annotation_dir / "instructions.json").exists()
    has_pose = (annotation_dir / "poses.npy").exists() and (annotation_dir / "intrinsics.npy").exists()
    return has_annotation, has_instruction, has_pose


def _has_annotation_payload(annotation_dir: Path) -> bool:
    if not annotation_dir.is_dir():
        return False
    return any(
        (annotation_dir / name).exists()
        for name in (
            "caption.json",
            "instructions.json",
            "generation_spec.json",
            "poses.npy",
            "intrinsics.npy",
            "dyn_masks.npz",
        )
    )


def _annotation_dir_with_payload(default_dir: Path, *candidate_dirs: Path) -> Path:
    for annotation_dir in (default_dir, *candidate_dirs):
        if _has_annotation_payload(annotation_dir):
            return annotation_dir
    return default_dir


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


def _configured_vipe_annotation_roots(video_root: Path) -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get(VIPE_ANNOTATION_ENV)
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(video_root.with_name(f"{video_root.name}_vipe_ann"))
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


def _static_video_vipe_annotation_dir(video_root: Path, path: Path) -> Path | None:
    relative_parent = path.relative_to(video_root).parent
    relative_without_static = Path(*relative_parent.parts[1:]) if relative_parent.parts[:1] == ("static",) else None
    candidate_parents: list[Path] = []
    for vipe_root in _configured_vipe_annotation_roots(video_root):
        candidate_parents.append(vipe_root / relative_parent / "annotations_vipe")
        if relative_without_static is not None:
            candidate_parents.append(vipe_root / relative_without_static / "annotations_vipe")
        else:
            candidate_parents.append(vipe_root / "static" / relative_parent / "annotations_vipe")
            if not relative_parent.parts or relative_parent.parts[0] not in {"photorealistic", "stylized"}:
                for style in ("photorealistic", "stylized"):
                    candidate_parents.append(vipe_root / "static" / style / relative_parent / "annotations_vipe")

    for annotation_parent in _unique_paths(candidate_parents):
        annotation_dir = _matching_vipe_annotation_dir(annotation_parent, path.stem)
        if annotation_dir is not None:
            return annotation_dir
    return None


def _discover_video_asset(path: Path, video_root: Path) -> DiscoveredAsset | None:
    relative_path = path.relative_to(video_root)
    if _should_skip_hidden(relative_path):
        return None
    parts = relative_path.parts
    if not parts:
        return None

    if parts[0] == "static" and len(parts) >= 6:
        _, style, environment, scene, duration_bucket, _ = parts[:6]
        category_path = "/".join(parts[:-1])
        spatialvid_annotation_dir = path.parent / "annotations" / f"{path.stem}_SpatialVID_ann"
        vipe_annotation_dir = _static_video_vipe_annotation_dir(video_root, path)
        annotation_dir = (
            _first_annotation_dir_with_payload(vipe_annotation_dir, spatialvid_annotation_dir)
            or spatialvid_annotation_dir
        )
        has_annotation, has_instruction, has_pose = _annotation_flags(annotation_dir)
        has_mask = (annotation_dir / "dyn_masks.npz").exists()
        return DiscoveredAsset(
            asset_level="video_static",
            modality="video",
            is_formal=True,
            eligible_for_official=True,
            path=str(path),
            relative_path=str(relative_path),
            category_path=category_path,
            source_name="worldarena_formal_video",
            source_type="curated_video",
            source_group_id=f"{category_path}::{path.stem}",
            license_bucket="C",
            style=style,
            environment=environment,
            scene=scene,
            duration_bucket=duration_bucket,
            fine_class=scene,
            annotation_path=str(annotation_dir) if has_annotation else None,
            has_annotation=has_annotation,
            has_instruction=has_instruction,
            has_pose=has_pose,
            has_mask=has_mask,
            mask_count=1 if has_mask else 0,
            size_bytes=path.stat().st_size,
        )

    if parts[0] == "dynamic" and len(parts) >= 5 and parts[2] == "videos":
        _, style, _, motion_category, _ = parts[:5]
        category_path = "/".join(parts[:-1])
        annotation_dir = _annotation_dir_with_payload(
            video_root / "dynamic" / "annotations" / motion_category / f"{path.stem}_ann",
            path.parent / "annotations_vipe" / path.stem,
            path.parent / "annotations_vipe" / f"{path.stem}_ann",
        )
        mask_dir = video_root / "dynamic" / style / "masks" / motion_category / path.stem / "masks"
        mask_count = _count_mask_images(mask_dir)
        has_annotation, has_instruction, has_pose = _annotation_flags(annotation_dir)
        return DiscoveredAsset(
            asset_level="video_dynamic",
            modality="video",
            is_formal=True,
            eligible_for_official=True,
            path=str(path),
            relative_path=str(relative_path),
            category_path=category_path,
            source_name="worldarena_formal_video",
            source_type="curated_video",
            source_group_id=f"{category_path}::{path.stem}",
            license_bucket="C",
            style=style,
            motion_category=motion_category,
            fine_class=motion_category,
            annotation_path=str(annotation_dir) if has_annotation else None,
            mask_path=str(mask_dir) if mask_dir.exists() else None,
            has_annotation=has_annotation,
            has_instruction=has_instruction,
            has_pose=has_pose,
            has_mask=mask_count > 0,
            mask_count=mask_count,
            size_bytes=path.stat().st_size,
        )

    if parts[0] == "AAA_Games" and len(parts) >= 3 and parts[1] == "videos":
        category_path = "/".join(parts[:-1])
        annotation_dir = video_root / "AAA_Games" / "annotations" / path.stem.zfill(3)
        has_annotation, has_instruction, has_pose = _annotation_flags(annotation_dir)
        has_mask = (annotation_dir / "dyn_masks.npz").exists()
        return DiscoveredAsset(
            asset_level="video_nonformal",
            modality="video",
            is_formal=False,
            eligible_for_official=False,
            path=str(path),
            relative_path=str(relative_path),
            category_path=category_path,
            source_name="worldarena_game_world",
            source_type="game_world",
            source_group_id=f"{category_path}::{path.stem}",
            license_bucket="C",
            style="synthetic",
            environment="gameplay",
            scene="AAA_Games",
            fine_class="AAA_Games",
            annotation_path=str(annotation_dir) if has_annotation else None,
            has_annotation=has_annotation,
            has_instruction=has_instruction,
            has_pose=has_pose,
            has_mask=has_mask,
            mask_count=1 if has_mask else 0,
            size_bytes=path.stat().st_size,
        )
    return None


def discover_assets(config: BenchmarkConfig) -> list[DiscoveredAsset]:
    assets: list[DiscoveredAsset] = []

    for path in sorted(config.paths.image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        asset = _discover_image_asset(path, config.paths.image_root)
        if asset is not None:
            assets.append(asset)

    for path in sorted(config.paths.video_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        asset = _discover_video_asset(path, config.paths.video_root)
        if asset is not None:
            assets.append(asset)

    image_paths = [Path(asset.path) for asset in assets if asset.modality == "image"]
    video_paths = [Path(asset.path) for asset in assets if asset.modality == "video"]
    media_lookup = probe_media(
        image_paths=image_paths,
        video_paths=video_paths,
        cache_path=config.paths.cache_dir / "media_probe_cache.json",
        max_workers=8,
    )

    for asset in assets:
        probe = media_lookup.get(asset.path, {})
        asset.width = probe.get("width")
        asset.height = probe.get("height")
        asset.fps = probe.get("fps")
        asset.duration_seconds = probe.get("duration_seconds")
        asset.probe_error = probe.get("probe_error")
        asset.usable = asset.width is not None and asset.height is not None and not asset.probe_error

    return assets


def _snapshot_hash(assets: list[DiscoveredAsset]) -> str:
    hasher = hashlib.sha256()
    for asset in sorted(assets, key=lambda item: item.relative_path):
        hasher.update(
            (
                f"{asset.relative_path}|{asset.size_bytes}|{asset.width}|{asset.height}|"
                f"{asset.fps}|{asset.duration_seconds}\n"
            ).encode("utf-8")
        )
    return hasher.hexdigest()


def _count_key(asset: DiscoveredAsset) -> str:
    if asset.asset_level == "image_static":
        return "image_static_formal"
    if asset.asset_level == "image_dynamic":
        return "image_dynamic_formal"
    if asset.asset_level == "video_static":
        return "video_static_formal"
    if asset.asset_level == "video_dynamic":
        return "video_dynamic_formal"
    if asset.asset_level == "video_nonformal":
        return "video_nonformal"
    return "image_nonformal"


def build_inventory(config: BenchmarkConfig) -> dict[str, Any]:
    assets = discover_assets(config)
    print("Discovered assets count:", len(assets))
    counts = {
        key: {
            "planned": config.planned_counts.get(key),
            "scanned": 0,
            "usable": 0,
        }
        for key in (
            "image_static_formal",
            "image_dynamic_formal",
            "video_static_formal",
            "video_dynamic_formal",
            "video_nonformal",
            "image_nonformal",
        )
    }

    missing_modalities = {
        "missing_rgb": 0,
        "missing_pose": 0,
        "missing_masks": 0,
        "missing_instruction": 0,
    }
    coverage_by_asset_level: dict[str, dict[str, int]] = defaultdict(
        lambda: {"items": 0, "with_mask": 0, "with_annotation": 0, "with_instruction": 0, "with_pose": 0}
    )

    for asset in assets:
        key = _count_key(asset)
        counts[key]["scanned"] += 1
        counts[key]["usable"] += int(asset.usable)
        missing_modalities["missing_rgb"] += int(not asset.usable)
        if asset.modality == "video":
            missing_modalities["missing_pose"] += int(not asset.has_pose)
            missing_modalities["missing_instruction"] += int(not asset.has_instruction)
        if asset.asset_level in {"image_dynamic", "video_dynamic", "image_nonformal", "video_nonformal"}:
            missing_modalities["missing_masks"] += int(not asset.has_mask)

        bucket = coverage_by_asset_level[asset.asset_level]
        bucket["items"] += 1
        bucket["with_mask"] += int(asset.has_mask)
        bucket["with_annotation"] += int(asset.has_annotation)
        bucket["with_instruction"] += int(asset.has_instruction)
        bucket["with_pose"] += int(asset.has_pose)

    mismatch_rows = []
    for key, payload in counts.items():
        planned = payload["planned"]
        scanned = payload["scanned"]
        if planned is None:
            status = "not_configured"
            delta = None
        else:
            delta = scanned - planned
            status = "match" if delta == 0 else "mismatch"
        mismatch_rows.append(
            {
                "bucket": key,
                "planned": planned,
                "scanned": scanned,
                "delta": delta,
                "status": status,
            }
        )

    if not assets:
        records_frame = pd.DataFrame(columns=["asset_level", "category_path", "relative_path", "modality"])
    else:
        records_frame = pd.DataFrame.from_records([asset.to_dict() for asset in assets]).sort_values(["asset_level", "category_path", "relative_path"])
    mismatch_frame = pd.DataFrame.from_records(mismatch_rows)

    summary = {
        "snapshot_name": config.snapshot_name,
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "roots": {
            "image_root": str(config.paths.image_root),
            "video_root": str(config.paths.video_root),
        },
        "snapshot_hash": _snapshot_hash(assets),
        "counts": counts,
        "missing_modalities": missing_modalities,
        "coverage_by_asset_level": coverage_by_asset_level,
        "totals": {
            "records": int(records_frame.shape[0]),
            "images": int((records_frame["modality"] == "image").sum()) if not records_frame.empty else 0,
            "videos": int((records_frame["modality"] == "video").sum()) if not records_frame.empty else 0,
        },
    }

    return {
        "assets": assets,
        "records_frame": records_frame,
        "mismatch_frame": mismatch_frame,
        "summary": summary,
    }


def write_inventory_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    ensure_dir(output_dir)
    records_frame: pd.DataFrame = result["records_frame"]
    mismatch_frame: pd.DataFrame = result["mismatch_frame"]
    summary = result["summary"]

    records_frame.to_csv(output_dir / "worldarena_inventory.csv", index=False)
    mismatch_frame.to_csv(output_dir / "count_mismatch_report.csv", index=False)
    write_json(output_dir / "worldarena_inventory.json", summary)
    write_json(output_dir / "missing_modalities_report.json", summary["missing_modalities"])
