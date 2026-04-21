from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from worldarena.benchmark import build_inventory, build_manifest, write_inventory_artifacts, write_manifest_artifacts
from worldarena.benchmark.config import BenchmarkConfig, BenchmarkPaths


def _make_config(tmp_path: Path, image_root: Path, video_root: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        config_path=tmp_path / "benchmark.yaml",
        paths=BenchmarkPaths(
            image_root=image_root,
            video_root=video_root,
            inventory_dir=tmp_path / "artifacts" / "inventory",
            manifest_dir=tmp_path / "artifacts" / "manifest",
            reports_dir=tmp_path / "artifacts" / "reports",
            cache_dir=tmp_path / ".cache" / "benchmark",
        ),
        snapshot_name="test_snapshot",
        target_fps=24,
        anchor_positions=[0.0, 0.5, 1.0],
        motion_frame_count=8,
        split_ratios={"dev_public": 0.2, "dev_private": 0.1, "test_hidden": 0.7},
        aggregate_weights={
            "image_static": 0.15,
            "image_dynamic": 0.15,
            "video_static": 0.30,
            "video_dynamic": 0.40,
        },
        metrics_by_suite={
            "image_static": ["scene_alignment"],
            "image_dynamic": ["dynamic_alignment"],
            "video_static": ["rollout_quality"],
            "video_dynamic": ["motion_magnitude"],
            "auxiliary": ["dynamic_alignment"],
            "experimental": ["dynamic_alignment"],
        },
        metric_backends={
            "scene_alignment": "reference_proxy",
            "dynamic_alignment": "reference_proxy",
            "rollout_quality": "reference_proxy",
            "motion_magnitude": "pixel_diff",
        },
        normalization={
            "scene_alignment": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "dynamic_alignment": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "rollout_quality": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
            "motion_magnitude": {"lower": 0.0, "upper": 1.0, "higher_is_better": True},
        },
    )


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), color).save(path)


def _write_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        6.0,
        (48, 32),
    )
    assert writer.isOpened()
    for color in colors:
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[:, :] = color
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _touch_video_sidecar(annotation_dir: Path, include_pose: bool = True) -> None:
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "instructions.json").write_text(json.dumps({"0->3": ["pan left"]}), encoding="utf-8")
    (annotation_dir / "caption.json").write_text(json.dumps({"CategoryTags": {}}), encoding="utf-8")
    if include_pose:
        np.save(annotation_dir / "poses.npy", np.zeros((2, 4, 4), dtype=np.float32))
        np.save(annotation_dir / "intrinsics.npy", np.eye(3, dtype=np.float32))


def test_benchmark_inventory_and_manifest_cover_formal_and_nonformal_assets(tmp_path: Path) -> None:
    image_root = tmp_path / "image"
    video_root = tmp_path / "videos"
    config = _make_config(tmp_path, image_root, video_root)

    _write_image(image_root / "static" / "photorealistic" / "indoor" / "living_spaces" / "0001.png", (120, 40, 20))
    _write_image(image_root / "dynamic" / "photorealistic" / "images" / "fluid" / "0001.png", (40, 120, 20))
    _write_image(image_root / "dynamic" / "photorealistic" / "masks" / "fluid" / "0001" / "mask.png", (255, 255, 255))
    _write_image(
        image_root / "dynamic" / "photorealistic" / "_backup_before_refresh" / "images" / "rigid" / "0001.png",
        (20, 40, 120),
    )

    static_video = video_root / "static" / "photorealistic" / "indoor" / "living_spaces" / "0-5s" / "0001.avi"
    dynamic_video = video_root / "dynamic" / "photorealistic" / "videos" / "fluid" / "0001.avi"
    aaa_video = video_root / "AAA_Games" / "videos" / "001.avi"
    _write_video(static_video, [(0, 0, 0), (32, 32, 32), (64, 64, 64)])
    _write_video(dynamic_video, [(0, 40, 90), (20, 60, 120), (40, 80, 150)])
    _write_video(aaa_video, [(90, 0, 0), (120, 20, 20), (150, 40, 40)])
    _touch_video_sidecar(static_video.parent / "annotations" / "0001_SpatialVID_ann")
    _touch_video_sidecar(video_root / "dynamic" / "annotations" / "fluid" / "0001_ann")
    _touch_video_sidecar(video_root / "AAA_Games" / "annotations" / "001", include_pose=False)

    inventory = build_inventory(config)
    write_inventory_artifacts(inventory, config.paths.inventory_dir)

    counts = inventory["summary"]["counts"]
    assert counts["image_static_formal"]["scanned"] == 1
    assert counts["image_dynamic_formal"]["scanned"] == 1
    assert counts["video_static_formal"]["scanned"] == 1
    assert counts["video_dynamic_formal"]["scanned"] == 1
    assert counts["video_nonformal"]["scanned"] == 1
    assert counts["image_nonformal"]["scanned"] == 1

    manifest = build_manifest(inventory["assets"], config)
    manifest_path = write_manifest_artifacts(manifest, config.paths.manifest_dir, config.snapshot_name)
    assert manifest_path.exists()
    assert len(manifest) == 6
    assert any(sample.suite == "auxiliary" for sample in manifest)
    assert any(sample.suite == "experimental" for sample in manifest)
    assert any(sample.suite == "image_static" for sample in manifest)
    assert any(sample.suite == "video_dynamic" for sample in manifest)

    summary_path = config.paths.manifest_dir / "worldarena_manifest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["items"] == 6
