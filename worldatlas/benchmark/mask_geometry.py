"""Explicit source-to-conditioning geometry shared with mask evaluation."""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps

from worldatlas.benchmark.mask_provenance import mask_files, validate_mask_metadata

SUPPORTED_TRANSFORM_TYPES = frozenset({
    'resize',
    'pil_fit',
    'resize_center_crop',
    'resize_center_crop_floor',
    'resize_center_crop_ceil',
    'resize_center_crop_trunc_floor',
    'ceil_floor',
    'resize_center_crop_ceil_floor',
    'even_trim_ceil_floor',
    'source_crop_resize',
    'source_crop_th_tw_resize',
    'ffmpeg_runtime_crop',
    'ffmpeg_yuv420p_scale_crop',
})
_RECORDED_FFMPEG_CROP = frozenset({
    'ffmpeg_runtime_crop',
    'ffmpeg_yuv420p_scale_crop',
})
_RESIZE_THEN_CROP = frozenset({
    'resize_center_crop',
    'resize_center_crop_floor',
    'resize_center_crop_ceil',
    'resize_center_crop_trunc_floor',
    'ceil_floor',
    'resize_center_crop_ceil_floor',
})
_CEIL_FLOOR = frozenset({'ceil_floor', 'resize_center_crop_ceil_floor'})


def _positive_int_pair(value, name):
    if value is None or len(value) != 2 or any(type(n) is not int or n <= 0 for n in value):
        raise ValueError(f'{name} must be a pair of positive integers')
    return int(value[0]), int(value[1])


def _int_box(value, name):
    if value is None or len(value) != 4 or any(type(n) is not int for n in value):
        raise ValueError(f'{name} must be an integer box [left, top, right, bottom]')
    left, top, right, bottom = (int(n) for n in value)
    if left >= right or top >= bottom:
        raise ValueError(f'{name} is empty')
    return left, top, right, bottom


def _geometry_mapping(geometry, key):
    if geometry.get(key) is not None:
        return geometry.get(key)
    layout = geometry.get('layout')
    if isinstance(layout, dict) and layout.get(key) is not None:
        return layout.get(key)
    evidence = geometry.get('evidence')
    if isinstance(evidence, dict):
        aliases = {
            'resize': ('resized_size', 'resize'),
            'crop': ('crop_box', 'crop'),
            'even_trim_size': ('even_trim_size',),
            'even_trim_box': ('even_trim_box',),
        }
        for name in aliases.get(key, (key,)):
            if evidence.get(name) is not None:
                return evidence.get(name)
    return None


def _require_identity_exif(geometry):
    orientation = geometry.get('exif_orientation', 1)
    evidence = geometry.get('evidence')
    if isinstance(evidence, dict) and evidence.get('exif_orientation') is not None:
        orientation = evidence['exif_orientation']
    if orientation not in (1, None):
        raise ValueError('Nontrivial EXIF orientation requires an explicit verified transform')


def even_trim_layout(source_size):
    """Drop the last odd row/column before LTX's ceil cover-crop.

    Echo-WM / LTX ``encode_single_frame`` floors each side to a multiple of two
    for H.264, then ``resize_and_center_crop`` runs on that trimmed tensor.
    Masks are annotated on the original PIL size, so keep ``source_size`` and
    apply this trim inside the transform.
    """
    width, height = _positive_int_pair(source_size, 'source_size')
    even_w, even_h = (width // 2) * 2, (height // 2) * 2
    if even_w <= 0 or even_h <= 0:
        raise ValueError('Even trim is empty')
    return (even_w, even_h), (0, 0, even_w, even_h)


def source_crop_layout(source_size, output_size):
    """Matrix-Game-2: crop in source pixels; PIL rounds each edge independently."""
    width, height = _positive_int_pair(source_size, 'source_size')
    tw, th = _positive_int_pair(output_size, 'output_size')
    if height / width > th / tw:
        crop_w, crop_h = int(width), int(width * th / tw)
    else:
        crop_h, crop_w = int(height), int(height * tw / th)
    left = round((width - crop_w) / 2)
    top = round((height - crop_h) / 2)
    right = round((width + crop_w) / 2)
    bottom = round((height + crop_h) / 2)
    if right <= left or bottom <= top:
        raise ValueError('Source crop is empty')
    return (right - left, bottom - top), (left, top, right, bottom)


def source_crop_th_tw_layout(source_size, output_size, align=32):
    """Matrix-Game-3 CenterCropResizeVideo: int(w*tr)/int(h/tr), then int(round) offsets."""
    width, height = _positive_int_pair(source_size, 'source_size')
    tw, th = _positive_int_pair(output_size, 'output_size')
    th_a = int(round(th / align) * align)
    tw_a = int(round(tw / align) * align)
    if th_a <= 0 or tw_a <= 0:
        raise ValueError('Aligned th/tw target is empty')
    tr = th_a / tw_a
    if height / width > tr:
        new_h, new_w = int(width * tr), width
    else:
        new_h, new_w = height, int(height / tr)
    top = int(round((height - new_h) / 2.0))
    left = int(round((width - new_w) / 2.0))
    box = (left, top, left + new_w, top + new_h)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('th/tw source crop is empty')
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError('th/tw source crop is outside the source image')
    return (new_w, new_h), box


def recorded_resize_crop_layout(geometry):
    """Apply executed resize/crop coordinates; do not re-derive ffmpeg letterbox."""
    output_size = _positive_int_pair(geometry.get('output_size'), 'output_size')
    resized = _positive_int_pair(_geometry_mapping(geometry, 'resize'), 'recorded resize')
    crop = _int_box(_geometry_mapping(geometry, 'crop'), 'recorded crop')
    left, top, right, bottom = crop
    rw, rh = resized
    tw, th = output_size
    if (right - left, bottom - top) != (tw, th):
        raise ValueError('Recorded crop size differs from output_size')
    if not (0 <= left < right <= rw and 0 <= top < bottom <= rh):
        raise ValueError('Recorded crop is outside resized image')
    return resized, crop


def conditioning_crop_layout(source_size, output_size, operation):
    """Match recorded upstream resize arithmetic before choosing crop offsets."""
    width, height = source_size
    tw, th = output_size
    if any(type(n) is not int or n <= 0 for n in (width, height, tw, th)):
        raise ValueError('Conditioning dimensions must be positive integers')
    scale = max(tw / width, th / height)
    if operation in _CEIL_FLOOR:
        # LTX / SVC default crop: ceil(src*scale) then floor crop offsets.
        # This is not resize_center_crop_ceil (round crop; max(ceil, target)).
        rw, rh = math.ceil(width * scale), math.ceil(height * scale)
    elif operation == 'resize_center_crop_ceil':
        # LingBot Video: interpolate(ceil(H*scale), ceil(W*scale)), round crop.
        rw, rh = max(math.ceil(width * scale), tw), max(math.ceil(height * scale), th)
    elif operation == 'resize_center_crop_trunc_floor':
        # RealCam: preserve the exact multiply/divide order upstream;
        # int(width*scale) is not equivalent at floating-point boundaries.
        if width / height > tw / th:
            rw, rh = int(width * th / height), th
        else:
            rw, rh = tw, int(height * tw / width)
    elif operation in {'resize_center_crop', 'resize_center_crop_floor'}:
        rw, rh = round(width * scale), round(height * scale)
    else:
        raise ValueError(f'Unsupported crop operation: {operation}')
    if operation in {'resize_center_crop_floor', 'resize_center_crop_trunc_floor'} or operation in _CEIL_FLOOR:
        left, top = (rw - tw) // 2, (rh - th) // 2
    else:
        left, top = round((rw - tw) / 2), round((rh - th) / 2)
    if rw < tw or rh < th:
        raise ValueError('Conditioning resize cannot cover target')
    return (rw, rh), (left, top, left + tw, top + th)


def transform_mask(mask: Image.Image, geometry: dict, target_size: tuple[int, int]) -> Image.Image:
    if list(mask.size) != geometry.get('source_size'):
        raise ValueError('Mask dimensions differ from verified source dimensions')
    output_size = tuple(geometry.get('output_size', ()))
    if len(output_size) != 2 or any(not isinstance(n, int) or n <= 0 for n in output_size):
        raise ValueError('Missing or invalid conditioning output_size')
    if not geometry.get('evidence'):
        raise ValueError('Conditioning geometry requires verification evidence')
    _require_identity_exif(geometry)
    operation = geometry.get('type')
    if operation == 'resize':
        transformed = mask.resize(output_size, Image.Resampling.NEAREST)
    elif operation == 'pil_fit':
        # ABot: PIL.ImageOps.fit crops the source using fractional bounds.
        transformed = ImageOps.fit(mask, output_size, method=Image.Resampling.NEAREST, centering=(0.5, 0.5))
    elif operation == 'even_trim_ceil_floor':
        even_size, even_box = even_trim_layout(mask.size)
        recorded_even = _geometry_mapping(geometry, 'even_trim_size')
        if recorded_even is not None and _positive_int_pair(recorded_even, 'even_trim_size') != even_size:
            raise ValueError('Recorded even trim size does not match H.264 even-floor arithmetic')
        recorded_box = _geometry_mapping(geometry, 'even_trim_box')
        if recorded_box is not None and _int_box(recorded_box, 'even_trim_box') != even_box:
            raise ValueError('Recorded even trim box does not match H.264 even-floor arithmetic')
        (rw, rh), crop = conditioning_crop_layout(even_size, output_size, 'ceil_floor')
        recorded_resize = _geometry_mapping(geometry, 'resize')
        if recorded_resize is not None and _positive_int_pair(recorded_resize, 'recorded resize') != (rw, rh):
            raise ValueError('Recorded even-trim resize does not match ceil_floor arithmetic')
        recorded_crop = _geometry_mapping(geometry, 'crop')
        if recorded_crop is not None and _int_box(recorded_crop, 'recorded crop') != crop:
            raise ValueError('Recorded even-trim crop does not match ceil_floor arithmetic')
        transformed = mask.crop(even_box).resize((rw, rh), Image.Resampling.NEAREST).crop(crop)
    elif operation in _RESIZE_THEN_CROP:
        # Alaya: torchvision resize uses round(scale*size), center_crop uses
        # round((resized-target)/2), including Python's tie-to-even rule.
        (rw, rh), crop = conditioning_crop_layout(mask.size, output_size, operation)
        transformed = mask.resize((rw, rh), Image.Resampling.NEAREST)
        transformed = transformed.crop(crop)
    elif operation == 'source_crop_resize':
        # Matrix-Game-2: crop the source first (PIL float-edge rounding), then
        # nearest-neighbor resize the cropped patch. Not a full-frame stretch.
        _, crop = source_crop_layout(mask.size, output_size)
        recorded = _geometry_mapping(geometry, 'crop')
        if recorded is not None and _int_box(recorded, 'recorded source crop') != crop:
            raise ValueError('Recorded source crop does not match source_crop_resize arithmetic')
        transformed = mask.crop(crop).resize(output_size, Image.Resampling.NEAREST)
    elif operation == 'source_crop_th_tw_resize':
        # Matrix-Game-3: crop with int(w*tr)/int(h/tr) and int(round) offsets,
        # then nearest-neighbor resize. Not MG2's independent-edge PIL box.
        _, crop = source_crop_th_tw_layout(mask.size, output_size)
        recorded = _geometry_mapping(geometry, 'crop')
        if recorded is not None and _int_box(recorded, 'recorded th/tw crop') != crop:
            raise ValueError('Recorded source crop does not match source_crop_th_tw_resize arithmetic')
        transformed = mask.crop(crop).resize(output_size, Image.Resampling.NEAREST)
    elif operation in _RECORDED_FFMPEG_CROP:
        # SolarWM: use the executed yuv420p scale/crop coordinates. Cover-scale
        # approximations are a different geometry and must not be substituted.
        (rw, rh), crop = recorded_resize_crop_layout(geometry)
        transformed = mask.resize((rw, rh), Image.Resampling.NEAREST).crop(crop)
    else:
        raise ValueError(f'Unsupported/unverified conditioning transform: {operation}')
    return transformed.resize(target_size, Image.Resampling.NEAREST)


@contextmanager
def aligned_mask_paths(mask_dir: str, *, source_path: str, relative_path: str,
                       geometry: dict, target_size: tuple[int, int]):
    directory = Path(mask_dir)
    validate_mask_metadata(directory, relative_path)
    with Image.open(source_path) as source:
        if list(source.size) != geometry.get('source_size'):
            raise ValueError('Geometry belongs to a different source size')
    with tempfile.TemporaryDirectory(prefix='worldatlas-aligned-masks-') as temporary:
        paths = []
        for index, path in enumerate(mask_files(directory)):
            with Image.open(path) as image:
                mask = Image.fromarray((np.asarray(image.convert('L')) > 0).astype(np.uint8) * 255)
            if not np.asarray(mask).any():
                raise ValueError(f'Saved annotation mask is empty before conditioning crop: {path}')
            transformed = transform_mask(mask, geometry, target_size)
            if not np.asarray(transformed).any():
                # An annotated object outside the executed crop is not an
                # object to seed into SAM2. Other visible instances remain.
                continue
            output = Path(temporary) / f'{index:04d}.png'
            transformed.save(output)
            paths.append(str(output))
        if not paths:
            raise ValueError('No visible foreground masks; conditioning support N/A must be handled upstream')
        yield paths


DEGENERATE_MASK_POLICY = 'skip_degenerate_pairs'
LEGACY_DEGENERATE_MASK_POLICY = 'error_no_score'
PROPAGATED_MASK_NA_REASON = 'propagated_mask_has_no_distinct_object'
PROPAGATED_MASK_EMPTY_REASON = 'propagated_mask_is_empty'
PROPAGATED_MASK_ALL_FOREGROUND_REASON = 'propagated_mask_is_all_foreground'
PROPAGATED_MASK_SKIP_REASONS = frozenset({
    PROPAGATED_MASK_NA_REASON,
    PROPAGATED_MASK_EMPTY_REASON,
    PROPAGATED_MASK_ALL_FOREGROUND_REASON,
})


def propagated_mask_skip_reason(mask: np.ndarray) -> str | None:
    """All-fg or empty/all-bg propagated masks are ineligible, not errors."""
    mask = np.asarray(mask).astype(bool, copy=False)
    if mask.size == 0 or not mask.any():
        return PROPAGATED_MASK_EMPTY_REASON
    if mask.all():
        return PROPAGATED_MASK_ALL_FOREGROUND_REASON
    return None


def flow_region_contrast(flow_magnitude: np.ndarray, mask: np.ndarray) -> float | None:
    if flow_magnitude.shape != mask.shape or not np.isfinite(flow_magnitude).all():
        raise ValueError('Invalid optical flow shape or non-finite values')
    if propagated_mask_skip_reason(mask) is not None:
        return None
    mask = mask.astype(bool)
    return float(flow_magnitude[mask].max() - flow_magnitude[~mask].max())


def summarize_motion_pair_scores(
    scores: list[float],
    *,
    expected_pair_count: int,
    skipped: list[dict] | None = None,
    foreground_fractions: list[float] | None = None,
    dilation_pixels: int = 10,
) -> tuple[float | None, dict]:
    """Mean official contrast over remaining pairs; all-skip is N/A, not zero."""
    skipped = list(skipped or [])
    if expected_pair_count < 1:
        raise ValueError('Motion accuracy requires at least one adjacent frame pair')
    if len(scores) + len(skipped) != expected_pair_count:
        raise ValueError('Motion pair skip bookkeeping does not cover every adjacent pair')
    details = {
        'flow_pair_count': len(scores),
        'expected_flow_pair_count': expected_pair_count,
        'skipped_degenerate_pair_count': len(skipped),
        'skipped_degenerate_pairs': skipped,
        'degenerate_mask_policy': DEGENERATE_MASK_POLICY,
        'dilation_pixels': dilation_pixels,
        'raw_units': 'decoded_pixels_per_frame',
        'reducer': 'mean(max_fg-max_bg)',
    }
    if foreground_fractions:
        details['foreground_fraction_min'] = min(foreground_fractions)
        details['foreground_fraction_max'] = max(foreground_fractions)
    if not scores:
        details['not_applicable_reason'] = PROPAGATED_MASK_NA_REASON
        return None, details
    return float(sum(scores) / len(scores)), details
