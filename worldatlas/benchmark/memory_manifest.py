"""Derive closed-loop memory samples from static image assets.

Memory samples reuse the still images already in the benchmark and pair each one with a
closed camera itinerary, so the track needs no new capture. Every sample is a single
conditioning image plus a loop that provably returns to its anchor, which is what makes
revisit scoring possible without any ground-truth long video.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from worldatlas.benchmark.loop_trajectories import (
    SUPPORTED_LOOP_IDS,
    loop_camera_path,
    loop_definition,
)
from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.benchmark.taxonomy import MEMORY_SUITE


# Ordered by how hard the revisit is: in-place rotation first, then out-and-back
# translation, then circuits that never retrace, then repeated revisits.
DEFAULT_MEMORY_LOOPS: tuple[str, ...] = (
    "yaw_loop_360",
    "yaw_loop_180",
    "dolly_loop",
    "square_loop",
    "palindrome_loop",
    "double_yaw_loop",
)

LOOP_DIFFICULTY: dict[str, str] = {
    "yaw_loop_360": "easy",
    "yaw_loop_180": "easy",
    "dolly_loop": "medium",
    "palindrome_loop": "medium",
    "square_loop": "hard",
    "double_yaw_loop": "hard",
}

_LOOP_PROMPTS: dict[str, str] = {
    "yaw_loop_360": "Rotate the camera a full turn in place and come back to the starting view.",
    "yaw_loop_180": "Turn the camera to the right, then turn back to the starting view.",
    "dolly_loop": "Move the camera forward through the scene, then back to where it started.",
    "square_loop": "Walk a closed square around the scene and return to the starting view.",
    "palindrome_loop": "Move forward, turn and continue, then retrace the whole path back.",
    "double_yaw_loop": "Turn away and back to the start twice in a row.",
}


def memory_sample_id(base_sample_id: str, loop_id: str) -> str:
    """Build a stable identifier for one image-and-loop combination."""
    return f"memory-{loop_id}-{base_sample_id}"


def build_memory_sample(sample: BenchmarkSample, loop_id: str) -> BenchmarkSample:
    """Turn one static image sample into a closed-loop memory sample."""
    if loop_id not in SUPPORTED_LOOP_IDS:
        raise ValueError(f"unsupported loop trajectory {loop_id!r}; expected {SUPPORTED_LOOP_IDS}")
    definition = loop_definition(loop_id)
    sample_id = memory_sample_id(sample.sample_id, loop_id)
    prompt_target = _LOOP_PROMPTS.get(loop_id, f"Follow the {loop_id} camera itinerary.")

    return BenchmarkSample(
        sample_id=sample_id,
        suite=MEMORY_SUITE,
        split=sample.split,
        asset_level=sample.asset_level,
        modality="image",
        is_formal=sample.is_formal,
        eligible_for_official=sample.eligible_for_official,
        category_path=sample.category_path,
        path=sample.path,
        relative_path=sample.relative_path,
        reference_path=sample.reference_path,
        prediction_stem=sample_id,
        conditioning_strategy="reference_image",
        prompt_current=sample.prompt_current,
        prompt_target=prompt_target,
        style=sample.style,
        environment=sample.environment,
        scene=sample.scene,
        motion_category=sample.motion_category,
        source_name=sample.source_name,
        source_type=sample.source_type,
        source_group_id=sample.source_group_id,
        license_bucket=sample.license_bucket,
        prompt_sequence=[prompt_target],
        camera_path=list(loop_camera_path(loop_id)),
        generation_mode="static",
        benchmark_family="worldatlas",
        task_family="world_model",
        artifact_type="video",
        control_signals=["image", "camera_pose"],
        annotation_path=sample.annotation_path,
        mask_path=sample.mask_path,
        has_annotation=sample.has_annotation,
        has_instruction=sample.has_instruction,
        has_pose=sample.has_pose,
        has_mask=sample.has_mask,
        mask_count=sample.mask_count,
        width=sample.width,
        height=sample.height,
        fps=sample.fps,
        duration_seconds=sample.duration_seconds,
        duration_bucket=sample.duration_bucket,
        conditioning_frame_count=1,
        conditioning_end_ratio=None,
        evaluation_start_ratio=0.0,
        track=f"memory_{definition.family}_loop",
    )


def _category_key(sample: BenchmarkSample) -> str:
    """Stable stratum key; prefer the path taxonomy, fall back to style/env/scene."""
    path = str(sample.category_path or "").strip()
    if path:
        return path
    return "/".join(
        [
            str(sample.style or "unknown"),
            str(sample.environment or "unknown"),
            str(sample.scene or "unknown"),
        ]
    )


def select_memory_sources(
    sources: Sequence[BenchmarkSample],
    *,
    limit: int | None,
) -> list[BenchmarkSample]:
    """Pick a deterministic, stratified subset of source images for every loop to share.

    Images are bucketed by ``category_path`` (the 20 style×environment×scene cells of
    ``image_static``). Within each bucket they stay sorted by ``sample_id``. Selection
    then round-robins across sorted buckets so a small ``limit`` still spans the
    taxonomy instead of collapsing onto the first alphabetical scene.

    When ``limit`` is ``None`` the full source list is returned (still sorted). The
    same selected list is reused by every loop so trajectory difficulty and scene
    difficulty stay factorially separable.
    """
    ordered = sorted(sources, key=lambda item: item.sample_id)
    if limit is None:
        return list(ordered)

    cap = max(int(limit), 0)
    if cap == 0 or not ordered:
        return []
    if cap >= len(ordered):
        return list(ordered)

    buckets: dict[str, list[BenchmarkSample]] = defaultdict(list)
    for sample in ordered:
        buckets[_category_key(sample)].append(sample)

    bucket_keys = sorted(buckets)
    selected: list[BenchmarkSample] = []
    cursors = {key: 0 for key in bucket_keys}
    while len(selected) < cap:
        progressed = False
        for key in bucket_keys:
            index = cursors[key]
            bucket = buckets[key]
            if index >= len(bucket):
                continue
            selected.append(bucket[index])
            cursors[key] = index + 1
            progressed = True
            if len(selected) >= cap:
                break
        if not progressed:
            break
    return selected


def build_memory_manifest(
    manifest: Iterable[BenchmarkSample],
    *,
    loops: Sequence[str] = DEFAULT_MEMORY_LOOPS,
    source_suite: str = "image_static",
    limit_per_loop: int | None = None,
) -> list[BenchmarkSample]:
    """Expand static image samples into memory samples, one per requested loop.

    Sources are drawn from the static image suite because a closed itinerary needs a
    scene the camera can leave and come back to, and because starting from a still
    image keeps the whole track free of ground-truth video. When ``limit_per_loop`` is
    set, every loop expands over the *same* stratified image subset so paired
    trajectory comparisons stay valid.
    """
    sources = [
        sample
        for sample in manifest
        if sample.suite == source_suite and sample.modality == "image"
    ]
    selected = select_memory_sources(sources, limit=limit_per_loop)

    samples: list[BenchmarkSample] = []
    for loop_id in loops:
        samples.extend(build_memory_sample(sample, loop_id) for sample in selected)
    return samples


def memory_manifest_summary(samples: Sequence[BenchmarkSample]) -> dict[str, object]:
    """Summarize a memory manifest by loop identifier and difficulty."""
    by_loop: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_category: dict[str, int] = {}
    source_ids: set[str] = set()
    for sample in samples:
        matched = False
        for loop_id in SUPPORTED_LOOP_IDS:
            prefix = f"memory-{loop_id}-"
            if sample.sample_id.startswith(prefix):
                by_loop[loop_id] = by_loop.get(loop_id, 0) + 1
                difficulty = LOOP_DIFFICULTY.get(loop_id, "unknown")
                by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
                source_ids.add(sample.sample_id[len(prefix) :])
                matched = True
                break
        if matched:
            key = _category_key(sample)
            by_category[key] = by_category.get(key, 0) + 1
    return {
        "total": len(samples),
        "unique_source_images": len(source_ids),
        "by_loop": dict(sorted(by_loop.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_category_path": dict(sorted(by_category.items())),
    }


__all__ = [
    "DEFAULT_MEMORY_LOOPS",
    "LOOP_DIFFICULTY",
    "build_memory_manifest",
    "build_memory_sample",
    "memory_manifest_summary",
    "memory_sample_id",
    "select_memory_sources",
]
