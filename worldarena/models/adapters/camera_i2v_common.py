"""Shared camera-path helpers for CamI2V and RealCam-I2V runners.

Both upstream demos consume the RealEstate10K text layout: seven metadata
columns followed by a flattened 3x4 world-to-camera matrix.  WorldArena stores
camera intent as one or more named path segments and scores against relative
camera-to-world matrices, so the conversion belongs at this adapter boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
from typing import Any, Iterator, Sequence

import numpy as np

from worldarena.benchmark.synthetic_camera import synthetic_camera_matrices


OFFICIAL_CAMERA_METADATA = np.asarray(
    [114514.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
    dtype=np.float64,
)


def as_bool(value: Any, *, default: bool = False) -> bool:
    """Coerce YAML/CLI-style values without treating ``"false"`` as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_camera_path(value: Any) -> list[str]:
    """Normalize one camera token or a sequence, defaulting images to fixed."""
    raw_values: Sequence[Any]
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raw_values = []

    tokens = [
        str(item).strip().lower().replace("-", "_").replace(" ", "_")
        for item in raw_values
        if str(item).strip()
    ]
    return tokens or ["fixed"]


def camera_path_from_row(row: dict[str, Any]) -> list[str]:
    """Read the normalized WorldArena camera path from a batch-spec row."""
    return normalize_camera_path(row.get("camera_path"))


def official_camera_rows(
    camera_path: Sequence[str],
    *,
    target_frames: int,
    runtime: dict[str, Any] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convert a WorldArena path to the numeric matrix rows upstream expects."""
    trajectory_runtime = {
        key: value
        for key, value in dict(runtime or {}).items()
        if key.startswith("synthetic_camera_")
    }
    trajectory = synthetic_camera_matrices(
        list(camera_path),
        target_frames=int(target_frames),
        runtime=trajectory_runtime,
    )
    c2ws = np.asarray(trajectory.matrices, dtype=np.float64)
    if c2ws.shape != (int(target_frames), 4, 4):
        raise ValueError(
            f"unexpected synthetic camera shape {c2ws.shape}; "
            f"expected {(int(target_frames), 4, 4)}"
        )
    if not np.isfinite(c2ws).all():
        raise ValueError("synthetic camera trajectory contains non-finite values")

    # Upstream immediately reconstructs 4x4 w2c matrices and inverts them.
    # Serializing the inverse here preserves WorldArena's c2w convention exactly.
    w2cs = np.linalg.inv(c2ws)
    metadata = np.repeat(
        OFFICIAL_CAMERA_METADATA[None, :],
        repeats=len(w2cs),
        axis=0,
    )
    rows = np.concatenate([metadata, w2cs[:, :3, :].reshape(len(w2cs), 12)], axis=1)
    if rows.shape[1] != 19:
        raise AssertionError(f"official camera rows must have 19 columns, got {rows.shape[1]}")
    return rows, trajectory.camera_path


def write_official_camera_file(
    path: Path,
    camera_path: Sequence[str],
    *,
    target_frames: int,
    runtime: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Write one upstream-compatible camera text file and return its path tokens."""
    rows, normalized_path = official_camera_rows(
        camera_path,
        target_frames=target_frames,
        runtime=runtime,
    )
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        # The official files use an ``https`` header and load with
        # ``np.loadtxt(..., comments="https")``.
        handle.write("https://worldarena.generated/camera-path\n")
        np.savetxt(handle, rows, fmt="%.9f")
    return normalized_path


def resolve_runtime_path(
    value: Any,
    *,
    bases: Sequence[Path],
    default: str,
) -> Path:
    """Resolve a configured artifact against ordered model/checkpoint roots."""
    raw = str(value or default)
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    resolved_candidates = [(base / candidate).resolve() for base in bases]
    for resolved in resolved_candidates:
        if resolved.exists():
            return resolved
    if not resolved_candidates:
        return candidate.resolve()
    # Return the canonical first location so any later error names the expected
    # file, even before checkpoints have been downloaded.
    return resolved_candidates[0]


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """Temporarily switch cwd for upstream code that uses relative model paths."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


__all__ = [
    "as_bool",
    "camera_path_from_row",
    "normalize_camera_path",
    "official_camera_rows",
    "resolve_runtime_path",
    "working_directory",
    "write_official_camera_file",
]
