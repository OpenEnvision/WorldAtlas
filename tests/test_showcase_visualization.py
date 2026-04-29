from __future__ import annotations

import json
from pathlib import Path

from worldarena.visualization.showcase import build_showcase, discover_showcase_media


def _write_media(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"media")
    return path


def test_build_showcase_uses_generation_records(tmp_path: Path) -> None:
    astra_root = tmp_path / "astra"
    yume_root = tmp_path / "yume"
    astra_video = _write_media(astra_root / "sample_001.mp4")
    yume_video = _write_media(yume_root / "sample_001.mp4")
    for root, video in [(astra_root, astra_video), (yume_root, yume_video)]:
        record = {
            "sample_id": "sample-001",
            "suite": "image_static",
            "prediction_path": str(video),
            "prompt": "a target world",
            "status": "generated",
        }
        (root / "generation_records.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    result = build_showcase(
        {"astra": astra_root, "yume": yume_root},
        output_dir=tmp_path / "showcase",
        mode="method_compare",
    )

    assert result["summary"]["samples"] == 1
    assert result["summary"]["methods"] == 2
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert {item["method"] for item in manifest["items"]} == {"astra", "yume"}
    assert {item["sample_id"] for item in manifest["items"]} == {"sample-001"}
    assert Path(result["html_path"]).exists()
    for item in manifest["items"]:
        assert (Path(result["output_dir"]) / item["url"]).exists()


def test_discover_showcase_media_groups_multiview_tree(tmp_path: Path) -> None:
    root = tmp_path / "method"
    _write_media(root / "case_001" / "CAM_FRONT.mp4")
    _write_media(root / "case_001" / "CAM_BACK.mp4")

    entries = discover_showcase_media({"method": root}, mode="multi_view")

    assert {entry.sample_id for entry in entries} == {"case_001"}
    assert {entry.view for entry in entries} == {"CAM_FRONT", "CAM_BACK"}
    assert {entry.method for entry in entries} == {"method"}


def test_gif_compare_filters_video_files(tmp_path: Path) -> None:
    root = tmp_path / "method"
    _write_media(root / "sample.gif")
    _write_media(root / "sample.mp4")

    entries = discover_showcase_media({"method": root}, mode="gif_compare")

    assert len(entries) == 1
    assert entries[0].media_type == "gif"
