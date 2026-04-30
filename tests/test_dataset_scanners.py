from __future__ import annotations

from pathlib import Path

from worldarena.datasets.config import CollectionConfig
from worldarena.datasets.scanners import scan_physics_video


def test_scan_physics_video_reads_category_and_subset_metadata(tmp_path: Path) -> None:
    root = tmp_path / "physics"
    video_path = (
        root
        / "videos_by_category"
        / "basic_physics_consistency"
        / "collision"
        / "sample.mp4"
    )
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    first_frame = root / "first_frame" / "001_collision_01.jpg"
    first_frame.parent.mkdir()
    first_frame.write_bytes(b"image")
    (root / "benchmark_physics_subset_extended.csv").write_text(
        "\n".join(
            [
                "order,rank_within_dimension,group,dimension,quota,video_name,path,symlink_path,label,captions,visual_quality_score,motion_score_v2,text_bbox_ratio,width,height,duration,fps,selection_score,match_type,annotation_labels,family_id,all_matching_dimensions,selection_note",
                "1,1,basic_physics_consistency,collision,12,sample.mp4,/data/sample.mp4,/subset/001_collision_01.mp4,rigid body motion,A collision clip,5.5,0.7,0.0,1920,1080,4.2,30.0,9.1,caption_pattern,collision,family,collision,clear collision",
            ]
        ),
        encoding="utf-8",
    )

    records = scan_physics_video(
        CollectionConfig(
            name="physics_videos",
            scanner="physics_video",
            modality="physics",
            group="benchmark",
            root=root,
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record["modality"] == "physics"
    assert record["physics_group"] == "basic_physics_consistency"
    assert record["physics_dimension"] == "collision"
    assert record["has_physics_metadata"] is True
    assert record["has_caption"] is True
    assert record["caption_text"] == "A collision clip"
    assert record["first_frame_path"] == str(first_frame)
