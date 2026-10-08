"""Overlap-window rollout for models that cannot allocate a full minute of latents.

Bidirectional full-sequence diffusion (SANA-WM, Matrix-Game 1) dies when the
memory track asks for ~961 frames in one forward. Official short clips still
fit, so the runners tile those clips with a one-sided overlap: the last frames
of window *i* become the conditioning image of window *i+1*, and the stitcher
drops the regenerated overlap before concatenating.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def plan_overlapping_windows(
    total_frames: int,
    chunk_frames: int,
    overlap_frames: int,
) -> list[tuple[int, int]]:
    """Cover ``[0, total_frames)`` with inclusive ``(start, length)`` windows.

    Subsequent windows start ``overlap_frames`` before the previous window ended
    so the next call can condition on already-generated frames. A leftover
    shorter than ``chunk_frames`` is emitted as a tail window; if that tail
    cannot start at the current cursor, the last window is pulled back so it
    still ends on ``total_frames``.
    """
    if total_frames < 1:
        raise ValueError(f"total_frames must be >= 1, got {total_frames}")
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}")
    if overlap_frames < 0:
        raise ValueError(f"overlap_frames must be >= 0, got {overlap_frames}")
    if chunk_frames <= overlap_frames:
        raise ValueError(
            f"chunk_frames={chunk_frames} must be greater than overlap_frames={overlap_frames}"
        )
    if total_frames <= chunk_frames:
        return [(0, total_frames)]

    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        length = min(chunk_frames, total_frames - start)
        if start + length < total_frames and length < chunk_frames:
            start = max(0, total_frames - chunk_frames)
            length = total_frames - start
        windows.append((start, length))
        covered = start + length
        if covered >= total_frames:
            break
        start = covered - overlap_frames
    return windows


def align_hunyuan_884(frames: int) -> int:
    """Smallest Hunyuan 884 ``video_length`` that is 1 or ``4k+1`` and >= frames."""
    if frames <= 1:
        return 1
    remainder = (frames - 1) % 4
    return frames if remainder == 0 else frames + (4 - remainder)


def apply_length_aligner(
    windows: list[tuple[int, int]],
    align: Callable[[int], int],
    total_frames: int,
) -> list[tuple[int, int]]:
    """Re-align window lengths after a lattice snap, keeping coverage of total_frames."""
    aligned: list[tuple[int, int]] = []
    for start, length in windows:
        legal = align(length)
        if legal != length:
            start = max(0, start - (legal - length))
            if start + legal > total_frames:
                start = max(0, total_frames - legal)
            length = legal
        aligned.append((start, length))
    return aligned


def plan_matrix_game_1_windows(
    total_frames: int,
    *,
    chunk_frames: int = 65,
    overlap_frames: int = 5,
) -> list[tuple[int, int]]:
    """Matrix-Game 1 windows. Each length is Hunyuan-legal (1 or 4k+1)."""
    chunk_frames = align_hunyuan_884(chunk_frames)
    windows = plan_overlapping_windows(total_frames, chunk_frames, overlap_frames)
    return apply_length_aligner(windows, align_hunyuan_884, total_frames)


def align_matrix_game_2_latent_frames(
    num_output_frames: int,
    num_frame_per_block: int = 3,
) -> int:
    """Smallest latent count >= ``num_output_frames`` that is a multiple of the block.

    Matrix-Game 2's ``CausalInferencePipeline`` asserts
    ``noise.shape[2] % num_frame_per_block == 0``. Universal / GTA / TempleRun
    configs all use block size 3. ``plan_rollout`` for 60 s at 12 fps yields 181,
    which is not divisible by 3; snap up to 183 (729 pixels, 60.75 s).
    """
    if num_output_frames < 1:
        raise ValueError(f"num_output_frames must be >= 1, got {num_output_frames}")
    if num_frame_per_block < 1:
        raise ValueError(f"num_frame_per_block must be >= 1, got {num_frame_per_block}")
    remainder = num_output_frames % num_frame_per_block
    if remainder == 0:
        return num_output_frames
    return num_output_frames + (num_frame_per_block - remainder)


def snap_stride_plus_one(frames: int, stride: int = 8) -> int:
    """Nearest ``stride*k + 1`` count, ties rounding up (SANA LTX-2 VAE)."""
    if frames < 1:
        return 1
    if (frames - 1) % stride == 0:
        return frames
    floor_cand = frames - ((frames - 1) % stride)
    ceil_cand = floor_cand + stride
    return floor_cand if (frames - floor_cand) < (ceil_cand - frames) else ceil_cand


def plan_sana_wm_windows(
    total_frames: int,
    *,
    chunk_frames: int = 161,
    overlap_frames: int = 1,
    stride: int = 8,
) -> list[tuple[int, int]]:
    """SANA-WM windows snapped to the LTX-2 ``8k+1`` lattice."""
    chunk_frames = snap_stride_plus_one(chunk_frames, stride)
    total_frames = snap_stride_plus_one(total_frames, stride)
    return plan_overlapping_windows(total_frames, chunk_frames, overlap_frames)


def stitch_windowed_frames(
    chunks: Sequence[np.ndarray],
    windows: Sequence[tuple[int, int]],
    *,
    pose_offset: int = 0,
    target_frames: int | None = None,
) -> np.ndarray:
    """Concatenate window videos, dropping regenerated overlap.

    ``pose_offset`` is how many leading poses each chunk already omitted
    (SANA's refiner drops the sink / condition frame, so that is 1).
    Frame ``i`` of a chunk is treated as pose ``start + pose_offset + i``.
    """
    if len(chunks) != len(windows):
        raise ValueError(f"chunk/window count mismatch: {len(chunks)} vs {len(windows)}")
    if not chunks:
        raise ValueError("stitch_windowed_frames requires at least one chunk")

    pieces: list[np.ndarray] = []
    covered = 0
    for (start, _length), video in zip(windows, chunks):
        if video.ndim < 1:
            raise ValueError(f"expected a time-major array, got {video.shape}")
        pose0 = start + pose_offset
        drop = max(0, covered - pose0)
        if drop >= int(video.shape[0]):
            continue
        pieces.append(np.asarray(video[drop:]))
        covered = pose0 + int(video.shape[0])
    if not pieces:
        raise ValueError("stitch_windowed_frames produced no frames")
    stacked = np.concatenate(pieces, axis=0)
    if target_frames is not None:
        stacked = stacked[:target_frames]
    return np.ascontiguousarray(stacked)
