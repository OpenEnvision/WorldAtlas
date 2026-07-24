"""Long-video chunking and stitching utilities for benchmark metrics."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Sequence

from worldarena.common.video_io import probe_video_fps, read_video_frames, write_video_frames


LONG_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}


def is_long_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in LONG_VIDEO_SUFFIXES


def iter_long_video_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if is_long_video_file(path))


def long_video_prompt_name(path: str) -> str:
    """Normalize split clip / scene filenames back to the original long-video prompt name."""

    prompt = Path(path).stem
    pattern = re.compile(r"(-Scene-\d+|-\d+)_\d+$")
    prompt = re.sub(pattern, "", prompt)

    trailing_split_suffix = r"(-Scene-\d+|-\d+)$"
    if re.search(trailing_split_suffix, prompt):
        return re.sub(trailing_split_suffix, "", prompt)
    return prompt


def _split_clip_inventory_ready(split_clip_root: Path, raw_videos: Sequence[Path]) -> bool:
    if not split_clip_root.is_dir() or not raw_videos:
        return False
    split_folders = [path for path in split_clip_root.iterdir() if path.is_dir()]
    return len(split_folders) >= len(raw_videos)


def _split_video_into_scenes(video_path: Path, output_dir: Path, threshold: float) -> bool:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "scenedetect is required for semantic long-video splitting"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        return False

    frames = read_video_frames(video_path)
    fps = probe_video_fps(video_path, default=8.0)
    video_name = video_path.stem
    for index, (start, end) in enumerate(scene_list):
        start_frame = max(int(start.get_frames()), 0)
        end_frame = max(int(end.get_frames()), start_frame + 1)
        scene_frames = frames[start_frame:end_frame]
        if not scene_frames:
            continue
        write_video_frames(
            output_dir / f"{video_name}-Scene-{index}.mp4",
            scene_frames,
            fps=fps,
        )
    return True


def _split_video_into_clips(
    video_path: Path,
    output_root: Path,
    *,
    duration: float,
) -> Path:
    frames = read_video_frames(video_path)
    fps = probe_video_fps(video_path, default=8.0)
    segment_frame_count = max(int(round(float(fps) * float(duration))), 1)

    video_name = video_path.stem
    output_dir = output_root / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(frames) < segment_frame_count:
        write_video_frames(output_dir / f"{video_name}_full.mp4", frames, fps=fps)
        return output_dir

    total_segments = len(frames) // segment_frame_count
    remaining_frames = len(frames) % segment_frame_count
    segment_index = 0
    for idx in range(total_segments):
        start = idx * segment_frame_count
        end = start + segment_frame_count
        write_video_frames(
            output_dir / f"{video_name}_{segment_index:03d}.mp4",
            frames[start:end],
            fps=fps,
        )
        segment_index += 1

    if remaining_frames > 0:
        extended_start = max(0, (total_segments * segment_frame_count) - (segment_frame_count - remaining_frames))
        write_video_frames(
            output_dir / f"{video_name}_{segment_index:03d}.mp4",
            frames[extended_start:],
            fps=fps,
        )
    return output_dir


def prepare_long_video_clips(
    videos_path: Path,
    *,
    threshold: float = 35.0,
    use_semantic_splitting: bool = False,
    clip_duration: float = 2.0,
    clip_fps: int = 8,
) -> Path:
    """Create the shared `split_clip/` layout consumed by long-video metrics."""

    videos_root = Path(videos_path)
    split_clip_root = videos_root / "split_clip"
    raw_videos = iter_long_video_files(videos_root)
    if _split_clip_inventory_ready(split_clip_root, raw_videos):
        return split_clip_root

    split_scene_video_paths: set[Path] = set()
    if use_semantic_splitting:
        for video_path in raw_videos:
            output_dir = videos_root / "split_scene" / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            scene_found = _split_video_into_scenes(video_path, output_dir, threshold)
            if scene_found:
                split_scene_video_paths.add(video_path.resolve())

    split_clip_root.mkdir(parents=True, exist_ok=True)
    for video_path in raw_videos:
        if video_path.resolve() in split_scene_video_paths:
            scene_dir = videos_root / "split_scene" / video_path.stem
            for scene_path in sorted(scene_dir.iterdir()):
                if not is_long_video_file(scene_path):
                    continue
                _split_video_into_clips(scene_path, split_clip_root, duration=clip_duration)
            continue
        _split_video_into_clips(video_path, split_clip_root, duration=clip_duration)
    return split_clip_root


def build_long_video_manifest(
    split_clip_root: Path,
    *,
    output_path: Path,
    name: str,
    dimension_list: Sequence[str],
) -> Path:
    """Build the minimal manifest shared by the long-video algorithm implementations."""

    split_root = Path(split_clip_root)
    if not split_root.exists():
        raise FileNotFoundError(f"split clip root not found: {split_root}")

    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    dimensions = list(dict.fromkeys(str(dimension) for dimension in dimension_list))

    prompt_payload: dict[str, dict[str, object]] = {}
    for prompt_folder in sorted(split_root.iterdir()):
        if not prompt_folder.is_dir():
            continue
        base_prompt = long_video_prompt_name(prompt_folder.name)
        entry = prompt_payload.setdefault(
            base_prompt,
            {
                "prompt_en": base_prompt,
                "dimension": list(dimensions),
                "video_list": [],
            },
        )
        video_list = entry["video_list"]
        assert isinstance(video_list, list)
        for video_file in sorted(prompt_folder.iterdir()):
            if not is_long_video_file(video_file):
                continue
            video_list.append(str(video_file))

    manifest_path = output_root / f"{name}_full_info.json"
    manifest_path.write_text(
        json.dumps(list(prompt_payload.values()), ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "LONG_VIDEO_SUFFIXES",
    "build_long_video_manifest",
    "is_long_video_file",
    "iter_long_video_files",
    "long_video_prompt_name",
    "prepare_long_video_clips",
]
