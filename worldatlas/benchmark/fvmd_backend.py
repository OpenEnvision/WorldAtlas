"""Frechet Video Motion Distance (FVMD) backend.

Upstream's ``fvmd.fvmd.fvmd()`` entry point cannot be used here. It drives both
sides through a single loop bounded by ``len(gen_dataset)`` and then reshapes the
reference histograms with the prediction's window count, so an independently
sized reference corpus is either silently truncated, silently duplicated, or a
shape error. This module therefore drives the decoupled upstream subfunctions
(``tracking_fullseq`` -> ``calc_hist`` -> Frechet distance) once per side.

``tests/test_fvmd_metric.py`` pins the orchestration against upstream
``fvmd()`` on equal-sized input so the split cannot silently drift.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from worldatlas.benchmark.distribution_clips import (
    ClipBuildReport,
    ClipSamplingSpec,
    atomic_save_npy,
    atomic_write_text,
    clip_set_signature,
    clips_for_video,
    distribution_cache_ready,
    distribution_cache_metadata,
    require_complete_clip_report,
    validate_distribution_features,
    reference_runtime,
)
from worldatlas.common.progress import log_progress
from worldatlas.benchmark.metrics.base import normalize_score
from worldatlas.common.checkpoints import checkpoint_env, filter_checkpoint_env_overrides
from worldatlas.benchmark.fvmd_streaming import ordered_prefetch
from worldatlas.benchmark.fvmd_reference import cache_lock, prepare_reference_selection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FVMD_REPO_CANDIDATES = (
    PROJECT_ROOT / "thirdparty" / "FVMD-frechet-video-motion-distance",
)
DEFAULT_ANALYSIS_SIZE = 256

# Upstream's histogram layout is fixed: 400 tracked points fold into a 20x20
# grid of 4x4 cells, 16 frames fold into 4 temporal cubes, and each cell carries
# 8 angle bins. Velocity and acceleration each contribute 4*4*4*8 = 512 dims.
FVMD_TRACKED_POINTS = 400
FVMD_CLIP_FRAMES = 16
FVMD_FEATURE_DIM = 1024
_WEIGHT_LOCK = threading.Lock()
_WEIGHT_CACHE: dict[str, Any] = {}


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _resolve_fvmd_repo_root(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("repo_root")
    if configured:
        repo_root = Path(str(configured)).expanduser()
        if not repo_root.is_absolute():
            repo_root = PROJECT_ROOT / repo_root
        if not repo_root.exists():
            raise FileNotFoundError(f"FVMD repo_root does not exist: {repo_root}")
        if not (repo_root / "fvmd").is_dir():
            raise FileNotFoundError(
                f"FVMD repo_root does not contain an fvmd package: {repo_root}"
            )
        return repo_root.resolve()

    for candidate in DEFAULT_FVMD_REPO_CANDIDATES:
        if (candidate / "fvmd").is_dir():
            return candidate.resolve()
    return None


def _load_fvmd_modules(runtime: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    """Import the upstream subfunctions this backend orchestrates."""
    repo_root = _resolve_fvmd_repo_root(runtime)
    if repo_root is not None:
        repo_path = str(repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    try:
        tracking = importlib.import_module("fvmd.keypoint_tracking")
        features = importlib.import_module("fvmd.extract_motion_features")
        frechet = importlib.import_module("fvmd.frechet_distance")
        datasets = importlib.import_module("fvmd.datasets.video_datasets")
    except Exception as exc:  # pragma: no cover - depends on optional FVMD runtime
        raise ModuleNotFoundError(
            "FVMD package is unavailable; install fvmd or set "
            "benchmark.metric_runtime.fvmd.repo_root to the cloned repository. "
            f"Import error: {exc.__class__.__name__}: {exc}"
        ) from exc

    return (
        {
            "tracking": tracking,
            "calc_hist": features.calc_hist,
            "activation_statistics": frechet.calculate_activation_statistics,
            "frechet_distance": frechet.calculate_frechet_distance,
            "VideoDatasetNP": datasets.VideoDatasetNP,
        },
        repo_root,
    )


def _prepare_runtime_environment(runtime: dict[str, Any]) -> None:
    os.environ.update(checkpoint_env())
    os.environ.update(filter_checkpoint_env_overrides(dict(runtime.get("env", {}))))


def _visible_cuda_tokens() -> list[str]:
    raw = os.environ.get("WORLDATLAS_FVMD_DEVICE_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not str(raw).strip() or str(raw).strip() == "-1":
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _resolve_device_ids(runtime: dict[str, Any]) -> list[int]:
    """Local CUDA indices for PIPs tracking.

    Official ``run_tracking`` asserts ``B == 1``, so ``DataParallel`` cannot
    split a clip. Multiple visible devices therefore mean independent trackers,
    not a different Frechet formula. ``device_ids`` in runtime wins; otherwise a
    multi-id ``CUDA_VISIBLE_DEVICES`` is remapped to ``0..n-1``.
    """
    configured = runtime.get("device_ids")
    if configured:
        return [int(value) for value in configured]
    tokens = _visible_cuda_tokens()
    if len(tokens) > 1:
        return list(range(len(tokens)))
    return [0]


def _contiguous_partitions(items: Sequence[Any], parts: int) -> list[Sequence[Any]]:
    if parts < 1:
        raise ValueError("FVMD tracking shard count must be positive")
    if not items:
        return []
    actual = min(int(parts), len(items))
    base, extra = divmod(len(items), actual)
    partitions: list[Sequence[Any]] = []
    start = 0
    for index in range(actual):
        size = base + (1 if index < extra else 0)
        partitions.append(items[start : start + size])
        start += size
    return partitions


def _merge_clip_reports(parts: Sequence[ClipBuildReport], *, candidate_video_count: int) -> ClipBuildReport:
    merged = ClipBuildReport(candidate_video_count=candidate_video_count)
    for report in parts:
        merged.used_video_count += report.used_video_count
        merged.clip_count += report.clip_count
        merged.skipped_video_count += report.skipped_video_count
        merged.source_fps_observed.extend(report.source_fps_observed)
        space = 20 - len(merged.used_ids_preview)
        if space > 0:
            merged.used_ids_preview.extend(report.used_ids_preview[:space])
        space = 20 - len(merged.skipped_videos_preview)
        if space > 0:
            merged.skipped_videos_preview.extend(report.skipped_videos_preview[:space])
    return merged


def _cached_pips_weights(tracking: Any) -> Any:
    url = str(getattr(tracking, "PIPS_WEIGHTS", "pips2_weights.pth"))
    with _WEIGHT_LOCK:
        cached = _WEIGHT_CACHE.get(url)
        if cached is None:
            cached = tracking.load_state_dict_from_url(tracking.PIPS_WEIGHTS, progress=False)
            _WEIGHT_CACHE[url] = cached
        return cached


def _build_tracker(modules: dict[str, Any], runtime: dict[str, Any]) -> Any:
    import torch

    tracking = modules["tracking"]
    device_ids = [int(value) for value in runtime.get("device_ids", [0])]
    if not device_ids:
        raise ValueError("FVMD tracker device_ids must not be empty")
    if not torch.cuda.is_available():
        raise RuntimeError("FVMD keypoint tracking requires a CUDA device")

    # CUDA's current device is thread-local. Upstream utilities may use bare
    # .cuda(); keep each replica's allocations on its assigned GPU, not GPU 0.
    torch.cuda.set_device(device_ids[0])

    model = tracking.Pips(stride=_as_positive_int(runtime.get("tracking_stride"), 8))
    model = model.to(f"cuda:{device_ids[0]}")
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.module.load_state_dict(_cached_pips_weights(tracking)["model_state_dict"])
    model.eval()
    return model


def _motion_features(
    clips: np.ndarray,
    *,
    model: Any,
    modules: dict[str, Any],
    spec: ClipSamplingSpec,
    runtime: dict[str, Any],
    loader_workers: int = 1,
) -> np.ndarray:
    """Track one clip set and fold it into ``(clips, 1024)`` motion histograms."""
    import torch
    from torch.utils.data import DataLoader

    tracking = modules["tracking"]
    calc_hist = modules["calc_hist"]
    points = _as_positive_int(runtime.get("tracking_points"), FVMD_TRACKED_POINTS)
    iters = _as_positive_int(runtime.get("tracking_iters"), 16)

    # seq_len/stride are pinned to the clip length so each clip yields exactly one
    # tracking window; upstream's stride=1 default would emit overlapping windows.
    dataset = modules["VideoDatasetNP"](
        clips,
        img_size=spec.analysis_size,
        seq_len=spec.clip_frame_count,
        stride=spec.clip_frame_count,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=max(0, int(loader_workers))
    )

    velocities: list[Any] = []
    accelerations: list[Any] = []
    with torch.no_grad():
        for sample in loader:
            _, velocity, acceleration = tracking.tracking_fullseq(
                model,
                sample,
                None,
                N=points,
                iters=iters,
                S_max=spec.clip_frame_count,
            )
            velocities.extend(velocity)
            accelerations.extend(acceleration)

    if not velocities:
        raise ValueError("FVMD keypoint tracking produced no motion windows")

    velocity_array = torch.cat(velocities, dim=0).cpu().numpy()
    acceleration_array = torch.cat(accelerations, dim=0).cpu().numpy()
    window_count = velocity_array.shape[0]
    return np.concatenate(
        (
            calc_hist(velocity_array).reshape(window_count, -1),
            calc_hist(acceleration_array).reshape(window_count, -1),
        ),
        axis=1,
    )


def _validate_spec(spec: ClipSamplingSpec) -> None:
    if spec.clip_frame_count != FVMD_CLIP_FRAMES:
        raise ValueError(
            "FVMD's histogram layout is defined for "
            f"{FVMD_CLIP_FRAMES}-frame windows; clip_frame_count={spec.clip_frame_count} "
            "would change the feature dimension and make scores incomparable"
        )
    if spec.analysis_size % 32 != 0:
        raise ValueError(
            f"FVMD keypoint tracking requires analysis_size divisible by 32, got {spec.analysis_size}"
        )


def _features_for_entries(
    entries: Sequence[tuple[Path, str]],
    *,
    model: Any,
    modules: dict[str, Any],
    spec: ClipSamplingSpec,
    runtime: dict[str, Any],
    label: str,
) -> tuple[np.ndarray, ClipBuildReport]:
    """Track in input order while a bounded CPU queue decodes upcoming videos."""
    report = ClipBuildReport(candidate_video_count=len(entries))
    chunks: list[np.ndarray] = []
    ahead = runtime.get("decode_prefetch_videos", 2)
    progress_every = _as_positive_int(runtime.get("progress_every"), 10)
    device = runtime.get("device_ids", [0])[0]
    started = time.monotonic()
    decode_wait = track_seconds = 0.0

    def decode(entry):
        path, identifier = entry
        try:
            if label == "reference":
                count = runtime.get("_reference_verified_counts", {}).get(str(path))
                video_clips, source_fps = clips_for_video(path, spec, verified_frame_count=count)
            else:
                video_clips, source_fps = clips_for_video(path, spec)
        except Exception as exc:
            return path, identifier, None, None, str(exc)
        return path, identifier, video_clips, source_fps, None

    with closing(ordered_prefetch(entries, decode, ahead=ahead)) as decoded:
        while True:
            waiting = time.monotonic()
            try:
                path, identifier, video_clips, source_fps, error = next(decoded)
            except StopIteration:
                break
            decode_wait += time.monotonic() - waiting
            if error is not None:
                report.skipped_video_count += 1
                if len(report.skipped_videos_preview) < 20:
                    report.skipped_videos_preview.append(
                        {"id": identifier, "path": str(path), "reason": error}
                    )
                log_progress("fvmd_decode", label=label, device=device, sample_id=identifier,
                             status="fail", error=error)
                continue
            clip_array = np.stack(video_clips, axis=0)
            tracking_started = time.monotonic()
            chunks.append(
                _motion_features(
                    clip_array,
                    model=model,
                    modules=modules,
                    spec=spec,
                    runtime=runtime,
                    loader_workers=0,
                )
            )
            track_seconds += time.monotonic() - tracking_started
            report.used_video_count += 1
            report.clip_count += int(clip_array.shape[0])
            if source_fps is not None:
                report.source_fps_observed.append(source_fps)
            if len(report.used_ids_preview) < 20:
                report.used_ids_preview.append(identifier)
            done = report.used_video_count + report.skipped_video_count
            if report.used_video_count == 1 or done % progress_every == 0 or done == len(entries):
                log_progress("fvmd_track_progress", label=label, device=device,
                             videos=f"{done}/{len(entries)}", clips=report.clip_count,
                             decode_prefetch_videos=ahead, decode_wait_s=round(decode_wait, 2),
                             tracking_s=round(track_seconds, 2), elapsed_s=round(time.monotonic() - started, 2))
            del clip_array, video_clips
    if not chunks:
        raise ValueError(f"{label} side produced no FVMD motion windows")
    return np.concatenate(chunks, axis=0), report


def _extract_motion_side(
    entries: Sequence[tuple[Path, str]],
    *,
    modules: dict[str, Any],
    spec: ClipSamplingSpec,
    runtime: dict[str, Any],
    device_ids: Sequence[int],
    label: str,
) -> tuple[np.ndarray, ClipBuildReport, str]:
    """Official per-clip tracking, optionally one PIPs replica per visible GPU.

    Each lane overlaps bounded CPU video decoding with GPU tracking, without
    materializing the full suite's clip stack before tracking can begin.
    """
    if len(device_ids) <= 1 or len(entries) <= 1:
        model = _build_tracker(modules, {**runtime, "device_ids": [int(device_ids[0])]})
        features, report = _features_for_entries(
            entries,
            model=model,
            modules=modules,
            spec=spec,
            runtime={**runtime, "device_ids": [int(device_ids[0])]},
            label=label,
        )
        return features, report, "per_video"

    partitions = _contiguous_partitions(list(entries), len(device_ids))
    log_progress(
        "fvmd_track_start",
        label=label,
        videos=len(entries),
        devices=",".join(str(device) for device in device_ids[: len(partitions)]),
        shards=len(partitions),
    )

    def _shard(item: tuple[int, Sequence[tuple[Path, str]]]) -> tuple[int, np.ndarray, ClipBuildReport]:
        shard_index, shard_entries = item
        device_id = int(device_ids[shard_index])
        model = _build_tracker(modules, {**runtime, "device_ids": [device_id]})
        features, report = _features_for_entries(
            shard_entries,
            model=model,
            modules=modules,
            spec=spec,
            runtime={**runtime, "device_ids": [device_id]},
            label=label,
        )
        log_progress(
            "fvmd_track_shard",
            label=label,
            device=device_id,
            videos=len(shard_entries),
            clips=report.clip_count,
            status="ok",
        )
        return shard_index, features, report

    with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
        finished = list(pool.map(_shard, enumerate(partitions)))
    finished.sort(key=lambda item: item[0])
    features = np.concatenate([item[1] for item in finished], axis=0)
    report = _merge_clip_reports([item[2] for item in finished], candidate_video_count=len(entries))
    return features, report, "sharded_per_video"


def compute_frechet_video_motion_distance(
    *,
    prediction_entries: Sequence[tuple[Path, str]],
    reference_entries: Sequence[tuple[Path, str]],
    normalization: dict[str, Any],
    runtime: dict[str, Any],
    output_dir: Path,
    reference_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Frechet distance between prediction and reference motion distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_spec = ClipSamplingSpec.from_runtime(
        runtime, default_analysis_size=DEFAULT_ANALYSIS_SIZE
    )
    reference_spec = ClipSamplingSpec.from_runtime(
        reference_runtime(runtime), default_analysis_size=DEFAULT_ANALYSIS_SIZE
    )
    _validate_spec(prediction_spec)
    _validate_spec(reference_spec)

    cache_dir = reference_cache_dir or (output_dir / "reference_cache")
    reference_entries, verified_counts, selection = prepare_reference_selection(
        reference_entries, reference_spec, cache_dir,
        workers=runtime.get("reference_preflight_workers", 8),
    )
    atomic_write_text(output_dir / "reference_selection.json", json.dumps(selection, indent=2, sort_keys=True) + "\n")
    reference_floor = max(reference_spec.min_clip_count, int(runtime.get("min_reference_clips", 0)))
    if selection["selection"]["clip_count"] < reference_floor:
        raise ValueError(f"Eligible FVMD reference clips below minimum {reference_floor}; inspect reference_selection.json")

    _prepare_runtime_environment(runtime)
    modules, repo_root = _load_fvmd_modules(runtime)

    from worldatlas.benchmark.integrity import path_content
    from urllib.parse import urlparse
    import torch
    weights_name = Path(urlparse(str(getattr(modules['tracking'], 'PIPS_WEIGHTS', 'pips2_weights.pth'))).path).name
    dependencies = [json.dumps(runtime, sort_keys=True, default=str),
                    json.dumps(path_content(Path(__file__)), sort_keys=True),
                    json.dumps(path_content(Path(__file__).with_name('fvmd_streaming.py')), sort_keys=True),
                    json.dumps(path_content(Path(__file__).with_name('fvmd_reference.py')), sort_keys=True),
                    json.dumps(path_content(Path(torch.hub.get_dir()) / 'checkpoints' / weights_name), sort_keys=True)]
    if repo_root is not None:
        dependencies.extend(json.dumps(path_content(p), sort_keys=True) for p in sorted((repo_root / 'fvmd').rglob('*.py')))
    reference_signature = clip_set_signature(reference_entries, reference_spec, extra=[*dependencies, selection["sha256"]])
    prediction_signature = clip_set_signature(prediction_entries, prediction_spec, extra=dependencies)
    cached_features = cache_dir / f"{reference_signature}.npy"
    cached_report = cache_dir / f"{reference_signature}.json"

    device_ids = _resolve_device_ids(runtime)
    details: dict[str, Any] = {
        "reference_cache_hit": False,
        "reference_signature": reference_signature,
        "prediction_signature": prediction_signature,
        "prediction_clip_spec": prediction_spec.describe(),
        "reference_clip_spec": reference_spec.describe(),
        "tracking_device_ids": device_ids,
        "decode_prefetch_videos": runtime.get("decode_prefetch_videos", 2),
        "reference_selection_sha256": selection["sha256"],
        "reference_population": {key: selection["selection"][key] for key in
                                 ("candidate_video_count", "eligible_video_count", "excluded_video_count", "clip_count")},
    }

    prediction_features, prediction_report, prediction_mode = _extract_motion_side(
        prediction_entries,
        modules=modules,
        spec=prediction_spec,
        runtime=runtime,
        device_ids=device_ids,
        label="prediction",
    )
    details["prediction"] = prediction_report.describe()
    details["prediction_feature_extraction"] = prediction_mode
    require_complete_clip_report(details["prediction"], len(prediction_entries), label='prediction')
    validate_distribution_features(
        prediction_features, int(details["prediction"]["clip_count"]), label='FVMD prediction'
    )

    # Concurrent model jobs share one reference extraction, not competing writes.
    with cache_lock(cached_features):
        reference_cache_hit = distribution_cache_ready(cached_features)
        details["reference_cache_hit"] = reference_cache_hit
        if reference_cache_hit:
            reference_features = np.load(cached_features)
            details["reference"] = json.loads(cached_report.read_text(encoding="utf-8"))
            details["reference_feature_extraction"] = "cache"
        else:
            if cached_features.exists():
                raise ValueError(f"Incomplete reference feature cache requires review: {cached_features}")
            reference_features, reference_report, reference_mode = _extract_motion_side(
                reference_entries,
                modules=modules,
                spec=reference_spec,
                runtime={**runtime, "_reference_verified_counts": verified_counts},
                device_ids=device_ids,
                label="reference",
            )
            details["reference"] = reference_report.describe()
            details["reference_feature_extraction"] = reference_mode
            require_complete_clip_report(details["reference"], len(reference_entries), label='reference')
            if details["reference"]["clip_count"] != selection["selection"]["clip_count"]:
                raise ValueError("FVMD reference clip count changed after preparation")
            validate_distribution_features(
                reference_features, int(details["reference"]["clip_count"]), label='FVMD reference'
            )
            atomic_save_npy(cached_features, reference_features)
            atomic_write_text(
                cached_report, distribution_cache_metadata(cached_features, details["reference"])
            )
    require_complete_clip_report(details["reference"], len(reference_entries), label='reference')
    if details["reference"]["clip_count"] != selection["selection"]["clip_count"]:
        raise ValueError("FVMD cached reference clip count differs from frozen selection")

    prediction_mu, prediction_sigma = modules["activation_statistics"](prediction_features)
    reference_mu, reference_sigma = modules["activation_statistics"](reference_features)
    raw = float(
        modules["frechet_distance"](
            prediction_mu, prediction_sigma, reference_mu, reference_sigma
        )
    )
    if not np.isfinite(raw):
        raise ValueError(f"FVMD returned a non-finite score: {raw}")

    prediction_count = int(prediction_features.shape[0])
    reference_count = int(reference_features.shape[0])
    feature_dim = int(prediction_features.shape[1])
    details.update(
        {
            "reference_cache_dir": str(cache_dir),
            "repo_root": str(repo_root) if repo_root is not None else None,
            "feature_dim": feature_dim,
            "prediction_window_count": prediction_count,
            "reference_window_count": reference_count,
            # A Frechet distance estimates a full covariance, so a side with fewer
            # windows than dimensions has a singular covariance and the score is
            # only meaningful when compared at matching window counts.
            "singular_covariance_sides": [
                side
                for side, count in (
                    ("prediction", prediction_count),
                    ("reference", reference_count),
                )
                if count <= feature_dim
            ],
            "torch_home": os.environ.get("TORCH_HOME"),
            "raw_direction": "lower_is_better",
        }
    )
    return {
        "raw": raw,
        "normalized": normalize_score(raw, normalization),
        "backend": "fvmd",
        "details": details,
    }


__all__ = ["FVMD_FEATURE_DIM", "compute_frechet_video_motion_distance"]
