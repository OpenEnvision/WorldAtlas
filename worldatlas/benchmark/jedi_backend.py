"""JEDi (JEPA Embedding Distance) backend for distribution-level video quality."""

from __future__ import annotations

import importlib
import fcntl
from contextlib import contextmanager
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from worldatlas.benchmark.distribution_clips import (
    ClipBuildReport,
    ClipSamplingSpec,
    atomic_write_text,
    build_clip_array,
    clip_set_signature,
    distribution_cache_ready,
    distribution_cache_metadata,
    require_complete_clip_report,
    validate_distribution_features,
    reference_runtime,
)
from worldatlas.benchmark.jedi_reference import prepare_reference_selection
from worldatlas.benchmark.metrics.base import normalize_score
from worldatlas.common.checkpoints import checkpoint_env, filter_checkpoint_env_overrides
from worldatlas.common.progress import log_progress


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JEDI_REPO_CANDIDATES = (PROJECT_ROOT / "thirdparty" / "JEDi",)
DEFAULT_ANALYSIS_SIZE = 224


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _resolve_jedi_repo_root(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("repo_root")
    if configured:
        repo_root = Path(str(configured)).expanduser()
        if not repo_root.is_absolute():
            repo_root = PROJECT_ROOT / repo_root
        if not repo_root.exists():
            raise FileNotFoundError(f"JEDi repo_root does not exist: {repo_root}")
        if not (repo_root / "videojedi").is_dir():
            raise FileNotFoundError(
                f"JEDi repo_root does not contain a videojedi package: {repo_root}"
            )
        return repo_root.resolve()

    for candidate in DEFAULT_JEDI_REPO_CANDIDATES:
        if (candidate / "videojedi").is_dir():
            return candidate.resolve()
    return None


def _load_jedi_metric_class(runtime: dict[str, Any]) -> tuple[Any, Path | None]:
    repo_root = _resolve_jedi_repo_root(runtime)
    if repo_root is not None:
        repo_path = str(repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    try:
        module = importlib.import_module("videojedi")
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise ModuleNotFoundError(
            "JEDi package is unavailable; install videojedi or set "
            "benchmark.metric_runtime.jedi.repo_root to the cloned repository. "
            f"Import error: {exc.__class__.__name__}: {exc}"
        ) from exc

    metric_class = getattr(module, "JEDiMetric", None)
    if metric_class is None:
        raise AttributeError("videojedi package does not expose JEDiMetric")
    return metric_class, repo_root


def _prepare_runtime_environment(runtime: dict[str, Any]) -> None:
    os.environ.update(checkpoint_env())
    os.environ.update(filter_checkpoint_env_overrides(dict(runtime.get("env", {}))))


def _resolve_runtime_path(value: Any, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _require_configured_path(path: str | None, *, label: str, kind: str) -> None:
    if path is None:
        return
    candidate = Path(path)
    exists = candidate.is_dir() if kind == "directory" else candidate.is_file()
    if not exists:
        raise FileNotFoundError(f"JEDi {label} {kind} does not exist: {candidate}")


def _dependency_tokens(*, model_dir: str, config_path: str | None) -> list[str]:
    """Checkpoint identity, so a weight change invalidates cached features."""
    from worldatlas.benchmark.integrity import path_content
    tokens: list[str] = []
    candidates = [Path(model_dir) / "vith16.pth.tar", Path(model_dir) / "ssv2-probe.pth.tar"]
    if config_path is not None:
        candidates.append(Path(config_path))
    for path in candidates:
        tokens.append(json.dumps(path_content(path), sort_keys=True))
    return tokens


def _collate_clips(batch: Any) -> Any:
    from torch.utils.data import default_collate
    clips, labels = default_collate(batch)
    return clips.float() / 255.0, labels


def _make_loader(clips: np.ndarray, batch_size: int) -> Any:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Keep the full corpus in uint8. Materialize float32 only for the current
    # batch, with identical ordering and normalization to the eager loader.
    tensor = torch.from_numpy(clips).permute(0, 1, 4, 2, 3)
    labels = torch.zeros((tensor.shape[0],), dtype=torch.long)
    return DataLoader(TensorDataset(tensor, labels), batch_size=batch_size,
                      shuffle=False, collate_fn=_collate_clips)


def _materialize(source: Path, destination: Path) -> None:
    """Place ``source`` at ``destination`` without ever exposing a partial file.

    ``os.link`` is already atomic. The copy fallback stages into the destination
    directory first, so a concurrent run cannot load a half-written cache entry.
    """
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return
    except OSError:
        pass
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, staging)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)


def _reference_cache_ready(features: Path) -> bool:
    return distribution_cache_ready(features)


@contextmanager
def _reference_cache_guard(features: Path):
    """One cold-cache producer across jobs; warm readers proceed concurrently."""
    if _reference_cache_ready(features):
        yield
        return
    features.parent.mkdir(parents=True, exist_ok=True)
    with features.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        held = not _reference_cache_ready(features)
        if not held:
            fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            yield
        finally:
            if held:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _visible_cuda_tokens() -> list[str]:
    raw = os.environ.get("WORLDATLAS_JEDI_DEVICE_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not str(raw).strip() or str(raw).strip() == "-1":
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _parent_cuda_tokens() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not str(raw).strip() or str(raw).strip() == "-1":
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _resolve_device_ids(runtime: dict[str, Any]) -> list[int]:
    """Local CUDA indices for V-JEPA feature extraction.

    V-JEPA is hardcoded to ``cuda:0`` plus a process group, so multiple visible
    devices become independent workers (each sees one GPU as ``cuda:0``), not a
    different MMD. ``device_ids`` in runtime wins; otherwise a multi-id
    ``CUDA_VISIBLE_DEVICES`` is remapped to ``0..n-1``.
    """
    configured = runtime.get("device_ids")
    if configured:
        return [int(value) for value in configured]
    tokens = _visible_cuda_tokens()
    if len(tokens) > 1:
        return list(range(len(tokens)))
    return [0]


def _cuda_visible_for_local_index(local_index: int) -> str:
    """Map a local ``device_ids`` entry onto the parent process GPU token.

    Spawned workers reset ``CUDA_VISIBLE_DEVICES`` to a single token so V-JEPA's
    ``cuda:0`` is that card. Local index ``0`` with parent
    ``CUDA_VISIBLE_DEVICES=4,5,6,7`` must become ``4``, not physical GPU 0.
    """
    tokens = _parent_cuda_tokens()
    index = int(local_index)
    if tokens and 0 <= index < len(tokens):
        return tokens[index]
    return str(index)


def _contiguous_partitions(items: Sequence[Any], parts: int) -> list[Sequence[Any]]:
    if parts < 1:
        raise ValueError("JEDi feature shard count must be positive")
    if not items:
        return []
    actual = min(int(parts), len(items))
    base, extra = divmod(len(items), actual)
    partitions: list[Sequence[Any]] = []
    start = 0
    for index in range(actual):
        size = base + (1 if index < extra else 0)
        partitions.append(items[start:start + size])
        start += size
    return partitions


def _spec_from_describe(payload: dict[str, Any]) -> ClipSamplingSpec:
    return ClipSamplingSpec(
        clip_frame_count=int(payload["clip_frame_count"]),
        analysis_size=int(payload["analysis_size"]),
        sample_fps=payload.get("sample_fps"),
        max_clips_per_video=int(payload["max_clips_per_video"]),
        clip_stride=payload.get("clip_stride"),
        preserve_aspect_ratio=bool(payload.get("preserve_aspect_ratio", False)),
        max_source_frames=payload.get("max_source_frames"),
        min_clip_count=int(payload.get("min_clip_count", 2)),
    )


def _merge_described_reports(
    parts: Sequence[dict[str, Any]],
    *,
    candidate_video_count: int,
    source_fps: Sequence[float],
) -> ClipBuildReport:
    merged = ClipBuildReport(candidate_video_count=candidate_video_count)
    for report in parts:
        merged.used_video_count += int(report.get("used_video_count", 0))
        merged.clip_count += int(report.get("clip_count", 0))
        merged.skipped_video_count += int(report.get("skipped_video_count", 0))
        space = 20 - len(merged.used_ids_preview)
        if space > 0:
            merged.used_ids_preview.extend(list(report.get("used_ids_preview") or [])[:space])
        space = 20 - len(merged.skipped_videos_preview)
        if space > 0:
            merged.skipped_videos_preview.extend(list(report.get("skipped_videos_preview") or [])[:space])
    merged.source_fps_observed.extend(source_fps)
    return merged


def _encode_clip_array_to_npy(array: np.ndarray, runtime: dict[str, Any], dest: Path) -> np.ndarray:
    metric_class, repo_root = _load_jedi_metric_class(runtime)
    model_base = repo_root or PROJECT_ROOT
    model_dir = _resolve_runtime_path(runtime.get("model_dir"), PROJECT_ROOT) or str(model_base)
    config_path = _resolve_runtime_path(runtime.get("config_path"), model_base)
    work = dest.parent / f".encode_{dest.stem}_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        metric = metric_class(feature_path=str(work), model_dir=model_dir, config_path=config_path)
        np.save(work / "train.npy", np.zeros((1, 1), dtype=np.float32))
        metric.load_features(
            train_loader=None,
            test_loader=_make_loader(array, _as_positive_int(runtime.get("batch_size"), 4)),
            num_samples=int(array.shape[0]),
        )
        features = np.asarray(metric.test_features)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, features)
        return features
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _extract_shard_payload(payload: dict[str, Any]) -> dict[str, Any]:
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["device"])
    try:
        _prepare_runtime_environment(payload["runtime"])
        entries = [(Path(path), identifier) for path, identifier in payload["entries"]]
        verified = payload.get("verified_counts")
        array, report = build_clip_array(
            entries,
            _spec_from_describe(payload["spec"]),
            label=payload["label"],
            verified_counts=verified,
        )
        dest = Path(payload["feature_npy"])
        _encode_clip_array_to_npy(array, payload["runtime"], dest)
        return {
            "feature_npy": str(dest),
            "report": report.describe(),
            "source_fps_observed": list(report.source_fps_observed),
        }
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _inprocess_shards() -> bool:
    return os.environ.get("WORLDATLAS_JEDI_INPROCESS_SHARDS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _map_feature_shards(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(payloads) == 1 or _inprocess_shards():
        return [_extract_shard_payload(item) for item in payloads]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(len(payloads)) as pool:
        return pool.map(_extract_shard_payload, payloads)


def _extract_jedi_side(
    entries: Sequence[tuple[Path, str]],
    spec: ClipSamplingSpec,
    runtime: dict[str, Any],
    device_ids: Sequence[int],
    *,
    label: str,
    work_dir: Path,
    verified_counts: dict[str, int] | None = None,
) -> tuple[np.ndarray, ClipBuildReport, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    if len(device_ids) <= 1 or len(entries) <= 1:
        array, report = build_clip_array(
            entries, spec, label=label, verified_counts=verified_counts
        )
        features = _encode_clip_array_to_npy(array, runtime, work_dir / f"{label}.npy")
        return features, report, "per_video"

    partitions = _contiguous_partitions(list(entries), len(device_ids))
    log_progress(
        "jedi_feature_start",
        label=label,
        videos=len(entries),
        devices=",".join(str(device) for device in device_ids[: len(partitions)]),
        shards=len(partitions),
    )
    payloads = []
    for index, part in enumerate(partitions):
        payloads.append(
            {
                "device": _cuda_visible_for_local_index(device_ids[index]),
                "entries": [(str(path), identifier) for path, identifier in part],
                "spec": spec.describe(),
                "runtime": {key: value for key, value in runtime.items() if key != "device_ids"},
                "label": label,
                "feature_npy": str(work_dir / f"{label}_shard{index}.npy"),
                "verified_counts": (
                    None
                    if verified_counts is None
                    else {str(path): verified_counts[str(path)] for path, _ in part if str(path) in verified_counts}
                ),
            }
        )
    finished = _map_feature_shards(payloads)
    features = np.concatenate([np.load(item["feature_npy"]) for item in finished], axis=0)
    report = _merge_described_reports(
        [item["report"] for item in finished],
        candidate_video_count=len(entries),
        source_fps=[value for item in finished for value in item.get("source_fps_observed") or []],
    )
    return features, report, "sharded_per_video"


def compute_jedi_distance(
    *,
    prediction_entries: Sequence[tuple[Path, str]],
    reference_entries: Sequence[tuple[Path, str]],
    normalization: dict[str, Any],
    runtime: dict[str, Any],
    output_dir: Path,
    reference_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """MMD between the prediction and reference clip distributions.

    The two sides are independent sets: JEDi's estimator has separate
    ``1/m^2``, ``1/n^2`` and ``1/(mn)`` terms and never pairs samples, so the
    reference corpus can be much larger than the prediction set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_spec = ClipSamplingSpec.from_runtime(
        runtime, default_analysis_size=DEFAULT_ANALYSIS_SIZE
    )
    reference_spec = ClipSamplingSpec.from_runtime(
        reference_runtime(runtime), default_analysis_size=DEFAULT_ANALYSIS_SIZE
    )

    cache_dir = reference_cache_dir or (output_dir / "reference_cache")
    reference_entries, verified_counts, selection = prepare_reference_selection(
        reference_entries, reference_spec, cache_dir,
        workers=runtime.get("reference_preflight_workers", 8),
    )
    atomic_write_text(output_dir / "reference_selection.json", json.dumps(selection, indent=2, sort_keys=True) + "\n")
    reference_floor = max(reference_spec.min_clip_count, int(runtime.get("min_reference_clips", 0)))
    if selection["selection"]["clip_count"] < reference_floor:
        raise ValueError(
            f"Eligible JEDi reference clips below minimum {reference_floor}; inspect reference_selection.json"
        )

    _prepare_runtime_environment(runtime)
    metric_class, repo_root = _load_jedi_metric_class(runtime)
    model_base = repo_root or PROJECT_ROOT
    model_dir = _resolve_runtime_path(runtime.get("model_dir"), PROJECT_ROOT) or str(model_base)
    config_path = _resolve_runtime_path(runtime.get("config_path"), model_base)
    if runtime.get("model_dir") is not None:
        _require_configured_path(model_dir, label="model_dir", kind="directory")
    if runtime.get("config_path") is not None:
        _require_configured_path(config_path, label="config_path", kind="file")

    dependency_tokens = _dependency_tokens(model_dir=model_dir, config_path=config_path)
    from worldatlas.benchmark.integrity import path_content
    dependency_tokens.append(json.dumps(path_content(Path(__file__)), sort_keys=True))
    dependency_tokens.append(json.dumps(path_content(Path(__file__).with_name('jedi_reference.py')), sort_keys=True))
    if repo_root is not None:
        dependency_tokens.extend(json.dumps(path_content(p), sort_keys=True) for p in sorted((repo_root / 'videojedi').rglob('*.py')))
    reference_signature = clip_set_signature(
        reference_entries, reference_spec, extra=[*dependency_tokens, selection["sha256"]]
    )
    prediction_signature = clip_set_signature(
        prediction_entries, prediction_spec, extra=dependency_tokens
    )

    cached_features = cache_dir / f"{reference_signature}.npy"
    cached_report = cache_dir / f"{reference_signature}.json"

    # Upstream trusts train.npy/test.npy by name alone. Predictions are private
    # to this invocation; only verified shared reference features may be reused.
    feature_dir = output_dir / "features" / f"{reference_signature}__{prediction_signature}" / uuid.uuid4().hex
    feature_dir.mkdir(parents=True, exist_ok=True)

    device_ids = _resolve_device_ids(runtime)
    use_shards = len(device_ids) > 1 and (
        len(prediction_entries) > 1 or len(reference_entries) > 1
    )
    details: dict[str, Any] = {
        "reference_signature": reference_signature,
        "prediction_signature": prediction_signature,
        "prediction_clip_spec": prediction_spec.describe(),
        "reference_clip_spec": reference_spec.describe(),
        "feature_device_ids": list(device_ids),
        "reference_selection_sha256": selection["sha256"],
        "reference_population": {key: selection["selection"][key] for key in
                                 ("candidate_video_count", "eligible_video_count", "excluded_video_count", "clip_count")},
    }

    prediction_array: np.ndarray | None = None
    prediction_features: np.ndarray | None = None
    if use_shards:
        prediction_features, prediction_report, prediction_mode = _extract_jedi_side(
            prediction_entries,
            prediction_spec,
            runtime,
            device_ids,
            label="prediction",
            work_dir=feature_dir / "pred_shards",
        )
        details["prediction_feature_extraction"] = prediction_mode
    else:
        prediction_array, prediction_report = build_clip_array(
            prediction_entries, prediction_spec, label="prediction"
        )
        details["prediction_feature_extraction"] = "per_video"
    details["prediction"] = prediction_report.describe()
    require_complete_clip_report(details["prediction"], len(prediction_entries), label='prediction')

    # Predictions decode concurrently across models before any cache wait.
    # Recheck the shared cache afterwards: another model may have completed it.
    with _reference_cache_guard(cached_features):
        reference_cache_hit = _reference_cache_ready(cached_features)
        details['reference_cache_hit'] = reference_cache_hit
        reference_array: np.ndarray | None = None
        reference_features: np.ndarray | None = None
        if reference_cache_hit:
            _materialize(cached_features, feature_dir / "train.npy")
            if cached_report.is_file():
                details["reference"] = json.loads(cached_report.read_text(encoding="utf-8"))
            else:
                details["reference"] = {"candidate_video_count": len(reference_entries)}
            details["reference_feature_extraction"] = "cache"
        else:
            if cached_features.exists():
                raise ValueError(f"Incomplete reference feature cache requires review: {cached_features}")
            if use_shards:
                reference_features, reference_report, reference_mode = _extract_jedi_side(
                    reference_entries,
                    reference_spec,
                    runtime,
                    device_ids,
                    label="reference",
                    work_dir=feature_dir / "ref_shards",
                    verified_counts=verified_counts,
                )
                details["reference_feature_extraction"] = reference_mode
            else:
                reference_array, reference_report = build_clip_array(
                    reference_entries,
                    reference_spec,
                    label="reference",
                    verified_counts=verified_counts,
                )
                details["reference_feature_extraction"] = "per_video"
            details["reference"] = reference_report.describe()
        require_complete_clip_report(details["reference"], len(reference_entries), label='reference')
        if details["reference"].get("clip_count") != selection["selection"]["clip_count"]:
            raise ValueError("JEDi reference clip count changed after preparation")

        prediction_clip_count = int(
            prediction_features.shape[0]
            if prediction_features is not None
            else prediction_array.shape[0]
        )
        reference_clip_count = int(
            reference_features.shape[0]
            if reference_features is not None
            else (
                reference_array.shape[0]
                if reference_array is not None
                else details["reference"].get("clip_count", 0)
            )
        )
        batch_size = _as_positive_int(runtime.get("batch_size"), 4)
        # load_features applies one num_samples cap to both sides, so anything below
        # the larger side would truncate the reference corpus back down to the
        # prediction count and undo the point of a fixed corpus.
        num_samples = _as_positive_int(
            runtime.get("num_samples"),
            max(reference_clip_count, prediction_clip_count),
        )
        if num_samples < max(reference_clip_count, prediction_clip_count):
            raise ValueError("JEDi num_samples cannot truncate either fixed distribution")

        metric = metric_class(
            feature_path=str(feature_dir),
            model_dir=model_dir,
            config_path=config_path,
        )
        if use_shards:
            if prediction_features is None:
                raise ValueError("JEDi sharded prediction features are missing")
            np.save(feature_dir / "test.npy", prediction_features)
            if not reference_cache_hit:
                if reference_features is None:
                    raise ValueError("JEDi sharded reference features are missing")
                np.save(feature_dir / "train.npy", reference_features)
            metric.load_features(train_loader=None, test_loader=None, num_samples=num_samples)
        else:
            metric.load_features(
                train_loader=(
                    None if reference_array is None else _make_loader(reference_array, batch_size)
                ),
                test_loader=_make_loader(prediction_array, batch_size),
                num_samples=num_samples,
            )
        validate_distribution_features(getattr(metric, 'train_features', None), reference_clip_count, label='JEDi reference')
        validate_distribution_features(getattr(metric, 'test_features', None), prediction_clip_count, label='JEDi prediction')
        raw = float(metric.compute_metric())
        if not np.isfinite(raw):
            raise ValueError(f"JEDi returned a non-finite score: {raw}")

        produced_reference = feature_dir / "train.npy"
        if not reference_cache_hit and produced_reference.is_file():
            _materialize(produced_reference, cached_features)
            atomic_write_text(
                cached_report, distribution_cache_metadata(cached_features, details["reference"])
            )

        train_features = getattr(metric, "train_features", None)
        test_features = getattr(metric, "test_features", None)
        details.update(
            {
                "feature_dir": str(feature_dir),
                "reference_cache_dir": str(cache_dir),
                "repo_root": str(repo_root) if repo_root is not None else None,
                "model_dir": model_dir,
                "config_path": config_path,
                "batch_size": batch_size,
                "num_samples": num_samples,
                "reference_feature_count": (
                    None if train_features is None else int(np.asarray(train_features).shape[0])
                ),
                "prediction_feature_count": (
                    None if test_features is None else int(np.asarray(test_features).shape[0])
                ),
                "torch_home": os.environ.get("TORCH_HOME"),
                "raw_direction": "lower_is_better",
            }
        )
        return {
            "raw": raw,
            "normalized": normalize_score(raw, normalization),
            "backend": "jedi",
            "details": details,
        }


__all__ = ["compute_jedi_distance"]
