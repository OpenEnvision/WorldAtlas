"""Find prediction sidecars without losing a symlink's public name/location."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def prediction_sidecar_candidates(
    prediction_path: Path,
    patterns: Iterable[str],
    hints: Iterable[object] = (),
) -> list[Path]:
    """Prefer explicit hints, then entry-side files, then target-side fallbacks.

    Exporters put annotations beside the public prediction, which may link to a
    differently named video in a model's work directory. Preserve that entry and
    its stem; resolving the video first discards both. Also retain the historical
    target-directory/entry-stem lookup for existing action-sidecar layouts.
    """
    entry = Path(prediction_path).expanduser().absolute()
    target = entry.resolve()
    locations = list(dict.fromkeys([
        (entry.parent, entry.stem),
        (target.parent, target.stem),
        (target.parent, entry.stem),
    ]))
    parents = list(dict.fromkeys(parent for parent, _ in locations))
    candidates: list[Path] = []
    for value in hints:
        if not value:
            continue
        hint = Path(str(value)).expanduser()
        candidates.extend([hint] if hint.is_absolute() else [parent / hint for parent in parents])
    patterns = tuple(patterns)
    for parent, stem in locations:
        candidates.extend(parent / pattern.format(stem=stem) for pattern in patterns)
    return list(dict.fromkeys(candidates))
