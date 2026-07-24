from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worldarena.benchmark.schemas import BenchmarkSample


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a video_static manifest from VIPE annotations.")
    parser.add_argument(
        "--video-root",
        default="data/video",
        type=Path,
    )
    parser.add_argument(
        "--vipe-root",
        default="data/video_vipe_ann",
        type=Path,
    )
    parser.add_argument(
        "--extra-video-root",
        default="data/stylized_by_category/video",
        type=Path,
        help="Optional supplemental split root used for stylized static videos.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/benchmark/manifest/worldarena_video_static_vipe_4971.jsonl",
        type=Path,
    )
    return parser.parse_args()


def _video_candidates(
    video_root: Path,
    extra_video_root: Path | None,
    *,
    style: str,
    environment: str,
    scene: str,
    duration_bucket: str,
) -> list[Path]:
    candidates = [video_root / environment / scene / duration_bucket]
    if extra_video_root is not None:
        candidates.extend(
            [
                extra_video_root / "static" / style / environment / scene / duration_bucket,
                extra_video_root / style / environment / scene / duration_bucket,
                extra_video_root / environment / scene / duration_bucket,
            ]
        )
    return candidates


def _video_for_annotation(
    video_root: Path,
    extra_video_root: Path | None,
    *,
    style: str,
    environment: str,
    scene: str,
    duration_bucket: str,
    stem: str,
    cache: dict[Path, dict[str, Path]],
) -> Path | None:
    for video_dir in _video_candidates(
        video_root,
        extra_video_root,
        style=style,
        environment=environment,
        scene=scene,
        duration_bucket=duration_bucket,
    ):
        if not video_dir.is_dir():
            continue
        directory_index = cache.get(video_dir)
        if directory_index is None:
            directory_index = {
                path.stem: path
                for path in sorted(video_dir.iterdir())
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            }
            cache[video_dir] = directory_index
        video_path = directory_index.get(stem)
        if video_path is not None:
            return video_path
    return None


def _quality_report(annotation_dir: Path) -> dict[str, Any]:
    path = annotation_dir / "quality_report.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _frame_count(annotation_dir: Path, quality: dict[str, Any]) -> int | None:
    value = quality.get("source_video_frame_count") or quality.get("frame_count")
    if value is not None:
        return int(value)
    frame_indices = annotation_dir / "frame_indices.npy"
    if frame_indices.is_file():
        return int(np.load(frame_indices).shape[0])
    return None


def _default_prompt(*, style: str, environment: str, scene: str) -> str:
    style_text = "stylized" if style == "stylized" else "photorealistic"
    scene_text = scene.replace("_", " ")
    return f"A {style_text} {environment} {scene_text} scene."


def _sample_id(
    *,
    style: str,
    environment: str,
    scene: str,
    counters: dict[tuple[str, str, str], int],
) -> str:
    key = (style, environment, scene)
    counters[key] += 1
    return f"video_static_{style}_{environment}_{scene}_{counters[key]:04d}"


def _annotation_dirs(vipe_root: Path) -> list[Path]:
    root = vipe_root / "static"
    return sorted(path.parent for path in root.rglob("poses_c2w.npy") if path.parent.name.endswith("_vipe_ann"))


def _relative_video_path(path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def build_manifest(
    video_root: Path,
    vipe_root: Path,
    extra_video_root: Path | None,
) -> tuple[list[BenchmarkSample], dict[str, Any]]:
    annotation_dirs = _annotation_dirs(vipe_root)
    counters: dict[tuple[str, str, str], int] = defaultdict(int)
    samples: list[BenchmarkSample] = []
    missing_video = 0
    duplicate_video = 0
    seen_videos: set[Path] = set()
    video_cache: dict[Path, dict[str, Path]] = {}

    for annotation_dir in annotation_dirs:
        stem = annotation_dir.name.removesuffix("_vipe_ann")
        relative_annotation = annotation_dir.relative_to(vipe_root)
        parts = relative_annotation.parts
        if len(parts) < 6 or parts[0] != "static":
            continue
        _, style, environment, scene, duration_bucket, _, _ = parts[:7]
        video_path = _video_for_annotation(
            video_root,
            extra_video_root,
            style=style,
            environment=environment,
            scene=scene,
            duration_bucket=duration_bucket,
            stem=stem,
            cache=video_cache,
        )
        if video_path is None:
            missing_video += 1
            continue
        if video_path in seen_videos:
            duplicate_video += 1
            continue
        seen_videos.add(video_path)
        quality = _quality_report(annotation_dir)
        fps = quality.get("fps")
        frame_count = _frame_count(annotation_dir, quality)
        duration_seconds = None
        if frame_count is not None and fps:
            duration_seconds = float(frame_count) / float(fps)

        fallback_prompt = _default_prompt(style=style, environment=environment, scene=scene)
        prompt_current = fallback_prompt
        prompt_target = fallback_prompt

        sample_id = _sample_id(
            style=style,
            environment=environment,
            scene=scene,
            counters=counters,
        )
        relative_video = _relative_video_path(
            video_path,
            [video_root] + ([extra_video_root] if extra_video_root is not None else []),
        )
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                suite="video_static",
                split="test",
                asset_level="video_static",
                modality="video",
                is_formal=True,
                eligible_for_official=True,
                category_path="/".join((style, environment, scene, duration_bucket)),
                path=str(video_path),
                relative_path=relative_video,
                reference_path=str(video_path),
                prediction_stem=sample_id,
                conditioning_strategy="first_65_frames",
                prompt_current=prompt_current,
                prompt_target=prompt_target,
                style=style,
                environment=environment,
                scene=scene,
                motion_category=None,
                source_name="worldarena_formal_video",
                source_type="curated_video",
                source_group_id=f"{style}/{environment}/{scene}/{duration_bucket}::{video_path.stem}",
                license_bucket="C",
                prompt_sequence=[],
                camera_path=[],
                generation_mode="static",
                benchmark_family="worldarena",
                task_family="video_generation",
                artifact_type="video",
                control_signals=["text", "image", "pose"],
                annotation_path=str(annotation_dir),
                mask_path=None,
                has_annotation=True,
                has_instruction=(annotation_dir / "instructions.json").is_file(),
                has_pose=True,
                has_mask=(annotation_dir / "dyn_masks.npz").is_file(),
                mask_count=1 if (annotation_dir / "dyn_masks.npz").is_file() else 0,
                width=int(quality["width"]) if quality.get("width") else None,
                height=int(quality["height"]) if quality.get("height") else None,
                fps=float(fps) if fps else None,
                duration_seconds=duration_seconds,
                duration_bucket=duration_bucket,
                conditioning_frame_count=65,
                conditioning_end_ratio=None,
                evaluation_start_ratio=0.0,
                track="camera_conditioned_scene_exploration",
            )
        )

    summary = {
        "video_root": str(video_root),
        "extra_video_root": str(extra_video_root) if extra_video_root is not None else None,
        "vipe_root": str(vipe_root),
        "indexed_videos": len(seen_videos),
        "samples": len(samples),
        "missing_video_for_annotation": missing_video,
        "duplicate_video_annotations": duplicate_video,
    }
    return samples, summary


def main() -> int:
    args = parse_args()
    extra_video_root = args.extra_video_root.resolve() if args.extra_video_root else None
    samples, summary = build_manifest(
        args.video_root.resolve(),
        args.vipe_root.resolve(),
        extra_video_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(sample.to_dict(), ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**summary, "output": str(args.output), "summary_path": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
