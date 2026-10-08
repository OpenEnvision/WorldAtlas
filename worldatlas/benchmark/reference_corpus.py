"""Reference video corpus for suite-level distribution metrics.

Distribution metrics (JEDi, FVMD) compare a set of predictions against a set of
real videos. Pairing each prediction with its own ground-truth clip caps the
reference distribution at the number of predictions, which is far below what a
kernel-MMD or Frechet estimate needs, and leaves image-conditioned suites with
no reference at all because their reference asset is a still image.

This module indexes the collected WorldAtlas video corpus instead, so every
suite scores against the same fixed real-video distribution. The corpus is a
benchmark constant: its signature is recorded with each run because changing
the corpus changes what the numbers mean.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from worldatlas.benchmark.distribution_clips import VIDEO_EXTENSIONS


STATIC_KIND = "static"
DYNAMIC_KIND = "dynamic"

STATIC_SUITES = frozenset({"image_static", "video_static"})
DYNAMIC_SUITES = frozenset({"image_dynamic", "video_dynamic"})

# Directory segments that hold derived artifacts rather than corpus videos.
_EXCLUDED_SEGMENT_PREFIXES = ("annotation", "mask")


@dataclass(frozen=True, slots=True)
class ReferenceGroupKey:
    """Identifies one reference distribution.

    Static assets carry ``style`` and ``environment`` so predictions are only
    compared against real videos of the same look and scene domain. Dynamic
    assets have no ``environment`` in the corpus layout and are far fewer, so
    they form a single unstratified group.
    """

    kind: str
    style: str | None = None
    environment: str | None = None

    @property
    def label(self) -> str:
        parts = [part for part in (self.style, self.environment) if part]
        return "__".join([self.kind, *parts]) if parts else self.kind

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "style": self.style,
            "environment": self.environment,
            "label": self.label,
        }


@dataclass(slots=True)
class ReferenceGroup:
    key: ReferenceGroupKey
    videos: tuple[Path, ...]
    signature: str
    total_available: int

    def entries(self) -> list[tuple[Path, str]]:
        return [(path, path.name) for path in self.videos]

    def describe(self) -> dict[str, Any]:
        return {
            **self.key.describe(),
            "video_count": len(self.videos),
            "total_available": self.total_available,
            "signature": self.signature,
        }


@dataclass(slots=True)
class ReferenceIndex:
    root: Path
    groups: dict[ReferenceGroupKey, ReferenceGroup]
    signature: str

    def group_for_suite(
        self,
        suite: str,
        *,
        style: str | None,
        environment: str | None,
    ) -> ReferenceGroup | None:
        key = group_key_for_suite(suite, style=style, environment=environment)
        return self.groups.get(key) if key is not None else None

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "signature": self.signature,
            "groups": [group.describe() for group in self.groups.values()],
        }


def group_key_for_suite(
    suite: str,
    *,
    style: str | None,
    environment: str | None,
) -> ReferenceGroupKey | None:
    """Reference group a prediction from ``suite`` should be scored against."""
    if suite in STATIC_SUITES:
        if not style or not environment:
            return None
        return ReferenceGroupKey(kind=STATIC_KIND, style=style, environment=environment)
    if suite in DYNAMIC_SUITES:
        return ReferenceGroupKey(kind=DYNAMIC_KIND)
    return None


@dataclass(frozen=True, slots=True)
class PredictionEntry:
    """One prediction video plus the metadata used to place it in a stratum."""

    path: Path
    sample_id: str
    style: str | None = None
    environment: str | None = None


@dataclass(slots=True)
class DistributionTask:
    """One prediction set scored against one reference set."""

    label: str
    predictions: list[PredictionEntry]
    reference: list[tuple[Path, str]]
    group_key: ReferenceGroupKey | None
    reference_signature: str

    @property
    def is_overall(self) -> bool:
        return self.group_key is None

    def prediction_entries(self) -> list[tuple[Path, str]]:
        return [(entry.path, entry.sample_id) for entry in self.predictions]


OVERALL_LABEL = "overall"


def kind_for_suite(suite: str) -> str | None:
    if suite in STATIC_SUITES:
        return STATIC_KIND
    if suite in DYNAMIC_SUITES:
        return DYNAMIC_KIND
    return None


def groups_for_kind(index: ReferenceIndex, kind: str) -> list[ReferenceGroup]:
    return [group for group in index.groups.values() if group.key.kind == kind]


def build_distribution_tasks(
    *,
    suite: str,
    predictions: Sequence[PredictionEntry],
    index: ReferenceIndex,
    stratified: bool = True,
) -> list[DistributionTask]:
    """Overall task first, then one task per stratum.

    The overall score is the headline number: it uses every prediction against
    the whole reference kind, so it stays stable even when an individual stratum
    is too small to score. Strata are diagnostics layered on top.
    """
    kind = kind_for_suite(suite)
    if kind is None:
        return []
    groups = groups_for_kind(index, kind)
    if not groups or not predictions:
        return []

    overall_reference = iter_group_entries(groups)
    overall_signature = hashlib.sha256(
        "".join(group.signature for group in sorted(groups, key=lambda item: item.key.label)).encode(
            "ascii"
        )
    ).hexdigest()[:20]
    tasks = [
        DistributionTask(
            label=OVERALL_LABEL,
            predictions=list(predictions),
            reference=overall_reference,
            group_key=None,
            reference_signature=overall_signature,
        )
    ]

    # A single group means the stratum is the overall set; scoring it twice would
    # only burn GPU time for an identical number.
    if not stratified or len(groups) <= 1:
        return tasks

    for group in sorted(groups, key=lambda item: item.key.label):
        matching = [
            entry
            for entry in predictions
            if entry.style == group.key.style and entry.environment == group.key.environment
        ]
        if not matching:
            continue
        tasks.append(
            DistributionTask(
                label=group.key.label,
                predictions=matching,
                reference=group.entries(),
                group_key=group.key,
                reference_signature=group.signature,
            )
        )
    return tasks


def _is_excluded_segment(part: str) -> bool:
    lowered = part.lower()
    return lowered.startswith(".") or lowered.startswith(_EXCLUDED_SEGMENT_PREFIXES)


def _is_excluded(parts: Sequence[str]) -> bool:
    return any(_is_excluded_segment(part) for part in parts)


# Corpus videos sit at a known depth, so the walk can stop there instead of
# descending into the per-video annotation and mask trees that live beside them.
# Those trees hold far more entries than the videos themselves, which makes an
# unpruned walk dominate the cost of building the index on a network filesystem.
_LEAF_DEPTH = {STATIC_KIND: 5, DYNAMIC_KIND: 4}


def _iter_corpus_videos(root: Path):
    """Yield ``(path, group_key)`` for every corpus video, pruning as it walks."""
    for kind, leaf_depth in _LEAF_DEPTH.items():
        base = root / kind
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base, topdown=True, followlinks=True):
            relative_dir = Path(dirpath).relative_to(root)
            if len(relative_dir.parts) >= leaf_depth:
                dirnames[:] = []
            else:
                dirnames[:] = sorted(
                    name for name in dirnames if not _is_excluded_segment(name)
                )
            for name in sorted(filenames):
                if Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                key = classify_corpus_video(relative_dir / name)
                if key is not None:
                    yield root / relative_dir / name, key


def classify_corpus_video(relative_path: Path) -> ReferenceGroupKey | None:
    """Map a corpus-relative video path to its reference group.

    Layouts recognised:
      ``static/{style}/{environment}/{scene}/{duration_bucket}/{file}``
      ``dynamic/videos/{motion_category}/{file}``
      ``dynamic/{style}/videos/{motion_category}/{file}``

    ``AAA_Games`` gameplay footage is deliberately excluded: it is not formal
    benchmark data and would shift the reference distribution.
    """
    parts = relative_path.parts
    if _is_excluded(parts):
        return None

    if parts[0] == STATIC_KIND and len(parts) == 6:
        _, style, environment, _scene, _duration, _ = parts
        return ReferenceGroupKey(kind=STATIC_KIND, style=style, environment=environment)

    if parts[0] == DYNAMIC_KIND and (
        (len(parts) == 4 and parts[1] == "videos")
        or (len(parts) == 5 and parts[2] == "videos")
    ):
        return ReferenceGroupKey(kind=DYNAMIC_KIND)

    return None


def _group_signature(videos: Sequence[Path], root: Path) -> str:
    from concurrent.futures import ThreadPoolExecutor
    from worldatlas.benchmark.integrity import path_content
    digest = hashlib.sha256()
    digest.update(b"reference_content_v2")
    # Preserve roster order and the exact serial signature. Bounded parallel
    # reads avoid one network-filesystem round trip per video serially.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, content in zip(videos, pool.map(path_content, videos)):
            if content.get("missing") or "sha256" not in content:
                raise FileNotFoundError(f"Reference video is missing or not a file: {path}")
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(json.dumps(content, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:20]


def build_reference_index(
    root: Path,
    *,
    max_videos_per_group: int | None = None,
    seed: int = 17,
) -> ReferenceIndex:
    """Scan ``root`` and group every corpus video into a reference distribution."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"reference corpus root does not exist: {root}")

    buckets: dict[ReferenceGroupKey, list[Path]] = {}
    for path, key in _iter_corpus_videos(root):
        buckets.setdefault(key, []).append(path)

    groups: dict[ReferenceGroupKey, ReferenceGroup] = {}
    for key, videos in buckets.items():
        selected = sorted(videos)
        total_available = len(selected)
        if max_videos_per_group is not None and total_available > max_videos_per_group:
            selected = sorted(random.Random(f"{seed}:{key.label}").sample(selected, max_videos_per_group))
        groups[key] = ReferenceGroup(
            key=key,
            videos=tuple(selected),
            signature=_group_signature(selected, root),
            total_available=total_available,
        )

    corpus_digest = hashlib.sha256()
    for key in sorted(groups, key=lambda item: item.label):
        corpus_digest.update(key.label.encode("utf-8"))
        corpus_digest.update(groups[key].signature.encode("ascii"))

    return ReferenceIndex(
        root=root,
        groups=dict(sorted(groups.items(), key=lambda item: item[0].label)),
        signature=corpus_digest.hexdigest()[:20],
    )


def write_reference_index(index: ReferenceIndex, path: Path) -> None:
    payload = {
        "root": str(index.root),
        "signature": index.signature,
        "groups": [
            {
                **group.key.describe(),
                "signature": group.signature,
                "total_available": group.total_available,
                "videos": [str(video.relative_to(index.root)) for video in group.videos],
            }
            for group in index.groups.values()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_reference_index(path: Path) -> ReferenceIndex:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    root = Path(payload["root"])
    groups: dict[ReferenceGroupKey, ReferenceGroup] = {}
    for entry in payload.get("groups", []):
        key = ReferenceGroupKey(
            kind=str(entry["kind"]),
            style=entry.get("style"),
            environment=entry.get("environment"),
        )
        videos = tuple(root / relative for relative in entry.get("videos", []))
        groups[key] = ReferenceGroup(
            key=key,
            videos=videos,
            signature=_group_signature(videos, root),
            total_available=int(entry.get("total_available", len(videos))),
        )
    # A frozen roster does not freeze the bytes at its paths. Bind the content
    # actually read today without modifying the historical index on disk.
    digest = hashlib.sha256()
    for key in sorted(groups, key=lambda item: item.label):
        digest.update(key.label.encode('utf-8'))
        digest.update(groups[key].signature.encode('ascii'))
    return ReferenceIndex(root=root, groups=groups, signature=digest.hexdigest()[:20])


def resolve_reference_index(config: Any) -> ReferenceIndex | None:
    """Load the configured reference corpus, preferring a prebuilt index file."""
    corpus = getattr(config, "reference_corpus", None)
    if corpus is None or not corpus.enabled:
        return None

    if corpus.index_path is not None and Path(corpus.index_path).is_file():
        return load_reference_index(Path(corpus.index_path))

    if corpus.root is None:
        raise ValueError(
            "benchmark.reference_corpus is enabled but neither root nor an existing "
            "index_path was provided"
        )
    return build_reference_index(
        Path(corpus.root),
        max_videos_per_group=corpus.max_videos_per_group,
        seed=corpus.seed,
    )


def split_group_for_calibration(
    group: ReferenceGroup,
    *,
    seed: int = 17,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Halve a reference group to measure the metric's real-versus-real floor.

    The score between two halves of the same corpus should sit near zero; a
    large value means the pipeline or the sample size is wrong, not that a model
    is bad.
    """
    videos = list(group.videos)
    random.Random(f"{seed}:{group.key.label}:calibration").shuffle(videos)
    midpoint = len(videos) // 2
    first = [(path, path.name) for path in sorted(videos[:midpoint])]
    second = [(path, path.name) for path in sorted(videos[midpoint:])]
    return first, second


def iter_group_entries(groups: Iterable[ReferenceGroup]) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for group in groups:
        entries.extend(group.entries())
    return entries


__all__ = [
    "DYNAMIC_KIND",
    "DYNAMIC_SUITES",
    "OVERALL_LABEL",
    "STATIC_KIND",
    "STATIC_SUITES",
    "DistributionTask",
    "PredictionEntry",
    "ReferenceGroup",
    "ReferenceGroupKey",
    "ReferenceIndex",
    "build_distribution_tasks",
    "build_reference_index",
    "classify_corpus_video",
    "group_key_for_suite",
    "groups_for_kind",
    "iter_group_entries",
    "kind_for_suite",
    "load_reference_index",
    "resolve_reference_index",
    "split_group_for_calibration",
    "write_reference_index",
]
