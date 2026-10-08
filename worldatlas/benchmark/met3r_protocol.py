"""MEt3R CVPR 2025 definition and WorldAtlas's explicit video input contract.

Eq. (1): average both directions of the author's two-image scorer. Sec. 5.1:
consecutive generated views, not input-image/anchor comparisons. Decode the
entire supplied prediction; the paper's dataset-specific 48/80-view limits are
not a license to truncate a different benchmark's videos silently.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2

from worldatlas.common.video_io import declared_frame_count_off_by_one

PROTOCOL = "met3r_paper_consecutive_symmetric_v1"
RESIZE_POLICY = "area_256_squared_height32_width16_bilinear_antialias"
UPSTREAM_COMMIT = "ee0e1752898559e1a3e85e2e151d3edeb9b55f73"
UPSTREAM_FILES = {
    "met3r/met3r.py": "863713845c0bfd908c2bd5f3326dce4f91dbb68f72bb1a61d0b19b87f41b0dfe",
    "met3r/__init__.py": "02100fbc3d163a8576db78b847315d2f8b22b081edb111c09d450358456bc802",
}


def verify_upstream(root: Path) -> None:
    for relative, expected in UPSTREAM_FILES.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"MEt3R official source differs from reviewed commit {UPSTREAM_COMMIT}: {path}")


def paper_input_shape(height: int, width: int) -> tuple[int, int]:
    """Approximately 256^2 pixels, preserving aspect up to the patch grid.

The paper specifies area/aspect, not integer rounding. Round ideal height to
32 pixels and width to 16 pixels (ties upward). Official FeatUp's DINO
PatchEmbed crops a row AND column when the patch-grid height is odd, without
updating its positional encoding. An even patch-grid height avoids that path;
width need not be even. Previously compatible height/width choices are kept.
This is an input-adapter convention, not the author's unpublished resizer.
"""
    if height <= 0 or width <= 0:
        raise ValueError("MEt3R requires positive frame dimensions")
    scale = 256.0 / math.sqrt(height * width)
    return tuple(max(grid, grid * math.floor(side * scale / grid + .5))
                 for side, grid in ((height, 32), (width, 16)))


def native_video_pair_batches(path: Path, batch_size: int, metadata: dict):
    """Yield every (t, t+1) pair once, with bounded memory and full decode checks."""
    if batch_size < 1:
        raise ValueError("MEt3R batch size must be positive")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open MEt3R prediction video: {path}")
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if expected < 2 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("MEt3R needs a video with at least two frames and valid frame metadata")
        previous, shape, count, batch = None, None, 0, []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Invalid decoded MEt3R frame")
            if shape is None:
                shape = frame.shape
            if frame.shape != shape:
                raise ValueError("MEt3R video dimensions changed during decode")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if previous is not None:
                batch.append((previous, frame))
            previous = frame
            count += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
        if count != expected and not declared_frame_count_off_by_one(expected, count):
            raise ValueError(f"Incomplete MEt3R video decode: {count}/{expected} frames: {path}")
        if batch:
            yield batch
        metadata.update(native_frame_count=count, frame_count=count,
                        declared_frame_count=expected,
                        frame_count_corrected=declared_frame_count_off_by_one(expected, count),
                        native_fps=fps, expected_pair_count=count - 1,
                        native_size=[shape[1], shape[0]],
                        metric_input_size=list(reversed(paper_input_shape(*shape[:2]))),
                        resize_policy=RESIZE_POLICY,
                        frame_scope="entire_prediction_video", frame_stride=1,
                        first_frame_index=0, last_frame_index=count - 1,
                        input_image_used=False, temporal_resampling=False)
    finally:
        capture.release()
