"""The dedicated still-image pool behind the closed-loop memory track.

The memory track does not reuse ``image_static``. It draws from a separate pool under
``data/image/memory``: 480 stills picked out of a larger non-WorldAtlas set by an
image-quality score, so a revisit is judged on frames that were sharp and detailed to
begin with. Keeping the pool disjoint also means a memory score can never be confused
with a score on one of the four generation suites.

``name_mapping.csv`` records where each renamed file came from, which is the only place
the style / environment / scene taxonomy survives; ``selection.json`` adds the pixel
dimensions and the ranking score.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Sequence

from worldatlas.benchmark.memory_manifest import DEFAULT_MEMORY_LOOPS, build_memory_sample
from worldatlas.benchmark.schemas import BenchmarkSample

MEMORY_POOL_DIRNAME = "memory"
NAME_MAPPING_FILE = "name_mapping.csv"
SELECTION_FILE = "selection.json"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True, slots=True)
class MemoryPoolImage:
    """One curated still, with the taxonomy recovered from its original path."""

    name: str
    path: Path
    relative_path: str
    style: str
    environment: str
    scene: str
    rank: int
    score: float
    width: int | None
    height: int | None

    @property
    def category_path(self) -> str:
        """Taxonomy cell, matching the shape used elsewhere in the manifest."""
        return f"{MEMORY_POOL_DIRNAME}/{self.style}/{self.environment}/{self.scene}"

    @property
    def stem(self) -> str:
        """File stem, e.g. ``000``."""
        return Path(self.name).stem


def _parse_old_rel(old_rel: str) -> tuple[str, str, str]:
    parts = [part for part in str(old_rel).strip().split("/") if part]
    if len(parts) < 4:
        raise ValueError(f"cannot read style/environment/scene from old_rel {old_rel!r}")
    return parts[0], parts[1], parts[2]


def _selection_sizes(pool_dir: Path) -> dict[str, tuple[int | None, int | None]]:
    selection = pool_dir / SELECTION_FILE
    if not selection.is_file():
        return {}
    payload = json.loads(selection.read_text(encoding="utf-8")) or {}
    sizes: dict[str, tuple[int | None, int | None]] = {}
    for entry in payload.get("selected") or []:
        name = str(entry.get("name") or entry.get("rel") or "").strip()
        if not name:
            continue
        width = entry.get("w")
        height = entry.get("h")
        sizes[name] = (
            int(width) if width is not None else None,
            int(height) if height is not None else None,
        )
    return sizes


def load_memory_pool(image_root: Path, *, duration_seconds: int | None = None) -> list[MemoryPoolImage]:
    """Read the curated pool, ordered by its selection rank.

    Paths are resolved because generation runs from job working directories that do
    not share this process's cwd.
    """
    image_root = Path(image_root).expanduser().resolve()
    pool_dir = image_root / MEMORY_POOL_DIRNAME
    if duration_seconds is not None:
        if duration_seconds not in (30, 60):
            raise ValueError("memory duration must be 30 or 60 seconds")
        pool_dir = pool_dir / f"{duration_seconds}s"
    elif (pool_dir / "60s" / NAME_MAPPING_FILE).is_file():
        pool_dir = pool_dir / "60s"
    mapping_path = pool_dir / NAME_MAPPING_FILE
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"memory pool needs {NAME_MAPPING_FILE} to recover the taxonomy: {mapping_path}"
        )
    sizes = _selection_sizes(pool_dir)

    images: list[MemoryPoolImage] = []
    with mapping_path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            name = str(row.get("new_name") or "").strip()
            if not name:
                raise ValueError(f"missing new_name at {mapping_path}:{line_number}")
            path = pool_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"memory pool image is missing: {path}")
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"unsupported memory pool image type: {path}")
            style, environment, scene = _parse_old_rel(row.get("old_rel", ""))
            width, height = sizes.get(name, (None, None))
            images.append(
                MemoryPoolImage(
                    name=name,
                    path=path,
                    relative_path=path.relative_to(image_root).as_posix(),
                    style=style,
                    environment=environment,
                    scene=scene,
                    rank=int(row.get("rank") or line_number),
                    score=float(row.get("score") or 0.0),
                    width=width,
                    height=height,
                )
            )

    if not images:
        raise ValueError(f"memory pool is empty: {pool_dir}")
    names = [image.name for image in images]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate memory pool entries: {duplicates}")
    images.sort(key=lambda item: item.rank)
    return images


def _source_sample(image: MemoryPoolImage) -> BenchmarkSample:
    """Wrap one pool image so the shared memory sample builder can shape it."""
    sample_id = f"memory_pool_{image.stem}"
    return BenchmarkSample(
        sample_id=sample_id,
        suite="memory_pool",
        split="test",
        asset_level="image_memory",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path=image.category_path,
        path=str(image.path),
        relative_path=image.relative_path,
        reference_path=str(image.path),
        prediction_stem=sample_id,
        conditioning_strategy="reference_image",
        # The pool carries no captions; the itinerary prompt is what the model is
        # actually asked to follow, and build_memory_sample supplies that.
        prompt_current="",
        prompt_target="",
        style=image.style,
        environment=image.environment,
        scene=image.scene,
        motion_category=None,
        source_name="worldatlas_memory_pool",
        source_type="curated_image",
        source_group_id=f"{image.category_path}::{image.stem}",
        license_bucket="C",
        width=image.width,
        height=image.height,
    )


def assign_pool_loops(
    images: Sequence[MemoryPoolImage],
    loops: Sequence[str] = DEFAULT_MEMORY_LOOPS,
) -> list[tuple[MemoryPoolImage, str]]:
    """Give each image exactly one loop, round-robin within taxonomy cells.

    The pool was ranked purely on image quality, so its scene mix is uneven. Walking
    the images grouped by taxonomy cell and handing out loops in turn keeps every loop
    at the same size and gives each one the same scene mix, which is what makes loop
    scores comparable when no image is shared between them.
    """
    if not loops:
        raise ValueError("assign_pool_loops needs at least one loop")
    ordered = sorted(images, key=lambda item: (item.category_path, item.rank))
    return [(image, loops[index % len(loops)]) for index, image in enumerate(ordered)]


def build_memory_pool_manifest(
    image_root: Path,
    *,
    loops: Sequence[str] = DEFAULT_MEMORY_LOOPS,
    limit: int | None = None,
    duration_seconds: int | None = None,
) -> list[BenchmarkSample]:
    """Turn the curated pool into memory_loop samples, one loop per image."""
    images = load_memory_pool(image_root, duration_seconds=duration_seconds)
    if limit is not None:
        images = images[: max(int(limit), 0)]
    samples = []
    for image, loop_id in assign_pool_loops(images, loops):
        source = _source_sample(image)
        if duration_seconds is not None:
            # Retain every published 60s identity; new images get a disjoint namespace.
            if duration_seconds == 30:
                source = replace(
                    source, sample_id=f"memory_pool_30s_{image.stem}",
                    source_group_id=f"memory/30s/{image.style}/{image.environment}/{image.scene}::{image.stem}",
                )
            source = replace(source, duration_seconds=float(duration_seconds), duration_bucket=f"{duration_seconds}s")
        samples.append(build_memory_sample(source, loop_id))
    samples.sort(key=lambda item: item.sample_id)
    return samples


__all__ = [
    "IMAGE_SUFFIXES",
    "MEMORY_POOL_DIRNAME",
    "MemoryPoolImage",
    "assign_pool_loops",
    "build_memory_pool_manifest",
    "load_memory_pool",
]
