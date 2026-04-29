from __future__ import annotations

import json
from pathlib import Path

from worldarena.benchmark.long_video import build_long_video_manifest, long_video_prompt_name


def test_long_video_prompt_name_normalizes_split_suffixes() -> None:
    assert long_video_prompt_name("forest-0.mp4") == "forest"
    assert long_video_prompt_name("forest-0_000.mp4") == "forest"
    assert long_video_prompt_name("forest-Scene-0_000.mp4") == "forest"
    assert long_video_prompt_name("forest.mp4") == "forest"


def test_build_long_video_manifest_groups_split_variants(tmp_path: Path) -> None:
    split_root = tmp_path / "split_clip"
    forest_split = split_root / "forest-0"
    forest_scene = split_root / "forest-Scene-1"
    forest_split.mkdir(parents=True)
    forest_scene.mkdir(parents=True)
    (forest_split / "forest-0_000.mp4").write_bytes(b"")
    (forest_scene / "forest-Scene-1_000.mp4").write_bytes(b"")
    (forest_scene / "ignore.txt").write_text("x", encoding="utf-8")

    manifest_path = build_long_video_manifest(
        split_root,
        output_path=tmp_path / "output",
        name="demo",
        dimension_list=["subject_consistency", "background_consistency"],
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["prompt_en"] == "forest"
    assert payload[0]["dimension"] == ["subject_consistency", "background_consistency"]
    assert payload[0]["video_list"] == [
        str(forest_split / "forest-0_000.mp4"),
        str(forest_scene / "forest-Scene-1_000.mp4"),
    ]
