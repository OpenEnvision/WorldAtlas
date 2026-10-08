"""Benchmark evaluation orchestrator.

Resolves prediction paths, loads media, runs configured metrics per sample,
and writes aggregated scores through ``report.write_run_outputs``.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worldatlas.benchmark.aggregates import aggregate_results, refresh_aggregate_indexes
from worldatlas.benchmark.config import BenchmarkConfig
from worldatlas.benchmark.distribution_clips import reference_runtime
from worldatlas.benchmark.fvmd_backend import compute_frechet_video_motion_distance
from worldatlas.benchmark.jedi_backend import compute_jedi_distance
from worldatlas.benchmark.media import load_media, probe_media_frame_count
from worldatlas.benchmark.metric_errors import MetricLoadError
from worldatlas.benchmark.metrics.registry import build_metrics, split_suite_level_metrics
from worldatlas.benchmark.reference_corpus import (
    DistributionTask,
    PredictionEntry,
    ReferenceIndex,
    build_distribution_tasks,
    resolve_reference_index,
)
from worldatlas.benchmark.report import write_run_outputs
from worldatlas.benchmark.sample_identity import cache_file_digests
from worldatlas.benchmark.schemas import BenchmarkSample, MetricOutput, SampleScore
from worldatlas.benchmark.runtime_health import (
    CudaContextError, append_sample_checkpoint, load_sample_checkpoint,
    raise_if_cuda_failure, sample_score_failed,
)
from worldatlas.benchmark.taxonomy import prediction_extensions, prediction_modality_for_task
from worldatlas.common.progress import (
    ProgressHeartbeat,
    log_exception,
    log_progress,
    rate_samples_per_hour,
)


def _prediction_modality(sample: BenchmarkSample) -> str:
    return prediction_modality_for_task(
        suite=sample.suite,
        source_modality=sample.modality,
        task_family=sample.task_family,
        artifact_type=sample.artifact_type,
    )


def _resolve_prediction_path(predictions_root: Path, sample: BenchmarkSample) -> Path | None:
    from worldatlas.benchmark.camera_video_identity import POLICY as CAMERA_POLICY, resolve_video as resolve_camera
    if sample.sample_id_policy == CAMERA_POLICY:
        return resolve_camera(sample, predictions_root)
    from worldatlas.benchmark.video_metric_identity import POLICY, resolve_video
    if sample.sample_id_policy == POLICY:
        return resolve_video(sample, predictions_root)
    _validate_image_identity_policy(sample)
    if sample.sample_id_policy:
        from worldatlas.benchmark.sample_identity import resolve_bound_prediction
        return resolve_bound_prediction(sample, predictions_root)
    candidates: list[Path] = []
    for extension in prediction_extensions(_prediction_modality(sample)):
        candidates.extend(
            [
                predictions_root / sample.suite / f"{sample.prediction_stem}{extension}",
                predictions_root / f"{sample.prediction_stem}{extension}",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _validate_image_identity_policy(sample: BenchmarkSample, *, required: bool = False) -> None:
    from worldatlas.benchmark.video_metric_identity import POLICY
    if sample.sample_id_policy == POLICY and not required:
        return  # run_benchmark separately enforces the video-only metric allowlist.
    from worldatlas.benchmark.sample_identity import IMAGE_LEVELS, SOURCE_FILENAME_POLICY
    if sample.suite in IMAGE_LEVELS and (required or sample.source_name == "worldatlas_formal_image"):
        if sample.sample_id_policy != SOURCE_FILENAME_POLICY:
            raise ValueError("Formal image evaluation requires a verified filename-ID manifest; legacy ordinal inputs are disabled")


def _evaluation_start_ratio(sample: BenchmarkSample) -> float:
    raw = float(sample.evaluation_start_ratio or 0.0)
    return min(max(raw, 0.0), 0.95)


def _prediction_window_start_ratio(sample: BenchmarkSample, prediction_path: Path) -> float:
    start_ratio = _evaluation_start_ratio(sample)
    prediction_modality = _prediction_modality(sample)
    if prediction_modality != "video" or start_ratio <= 0.0:
        return 0.0
    try:
        reference_total = probe_media_frame_count(Path(sample.reference_path), sample.modality)
        prediction_total = probe_media_frame_count(prediction_path, prediction_modality)
    except Exception:
        return 0.0
    rollout_reference = max(int(round(reference_total * (1.0 - start_ratio))), 1)
    if rollout_reference <= 0:
        return 0.0
    observed_ratio = float(prediction_total) / float(rollout_reference)
    full_ratio = 1.0 / max(1.0 - start_ratio, 1e-6)
    threshold = 1.0 + (full_ratio - 1.0) * 0.5
    return start_ratio if observed_ratio >= threshold else 0.0


def _metric_backend(config: BenchmarkConfig, metric_name: str) -> str:
    backend = config.metric_backends.get(metric_name, "auto")
    if backend == "auto" and metric_name in {"fvmd", "jedi"}:
        return metric_name
    return backend


def _configured_metric_names(config: BenchmarkConfig, suites: set[str] | None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for suite_name, metric_names in config.metrics_by_suite.items():
        if suites is not None and suite_name not in suites:
            continue
        for metric_name in metric_names:
            if metric_name in seen:
                continue
            seen.add(metric_name)
            names.append(metric_name)
    return names


def _format_score(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        text = " ".join(str(value).split())
        return text[:80]


def _metric_status(output: MetricOutput) -> str:
    from worldatlas.benchmark.runtime_health import metric_output_failed
    if metric_output_failed(output.to_dict()):
        return "fail"
    if output.eligibility_status == "not_applicable":
        return "skip"
    if output.eligibility_status == "diagnostic" and output.normalized is not None:
        from worldatlas.benchmark.runtime_health import metric_output_failed
        if not metric_output_failed(output.to_dict()):
            return "ok"  # A zero-valued return gate may carry an explanatory reason.
    if output.error:
        if output.eligibility_status == "not_applicable":
            return "skip"
        return "fail"
    if output.normalized is None and output.raw is None:
        return "skip"
    return "ok"


def _metric_score_text(output: MetricOutput) -> str | None:
    return _format_score(output.normalized if output.normalized is not None else output.raw)


def _sample_status(metric_statuses: list[str], *, sample_error: str | None = None) -> str:
    if sample_error == "missing prediction":
        return "skip"
    if sample_error:
        return "fail"
    if not metric_statuses:
        return "ok"
    if any(status == "fail" for status in metric_statuses):
        return "fail"
    if all(status == "skip" for status in metric_statuses):
        return "skip"
    return "ok"


def _compact_scores(metrics: dict[str, MetricOutput]) -> str | None:
    parts: list[str] = []
    for name, output in metrics.items():
        if _metric_status(output) != "ok":
            continue
        score = _metric_score_text(output)
        if score is None:
            continue
        parts.append(f"{name}:{score}")
    if not parts:
        return None
    return ",".join(parts)


def _score_one_metric(
    metric: Any,
    *,
    sample: BenchmarkSample,
    prediction: Any,
    reference: Any,
    index_label: str,
) -> MetricOutput:
    started_at = time.perf_counter()
    try:
        with ProgressHeartbeat(
            sample_id=sample.sample_id,
            index=index_label,
            metric=metric.name,
            stage="score",
        ):
            output = metric.compute(sample=sample, prediction=prediction, reference=reference)
    except MetricLoadError as exc:
        log_exception(
            "metric",
            exc,
            sample_id=sample.sample_id,
            index=index_label,
            metric=metric.name,
            status="fail",
            elapsed_s=f"{time.perf_counter() - started_at:.1f}",
        )
        raise
    except Exception as exc:
        log_exception(
            "metric",
            exc,
            sample_id=sample.sample_id,
            index=index_label,
            metric=metric.name,
            status="fail",
            elapsed_s=f"{time.perf_counter() - started_at:.1f}",
        )
        raise_if_cuda_failure(exc)
        return MetricOutput(
            raw=None,
            normalized=None,
            backend=getattr(metric, "backend", "unknown"),
            error=f"{type(exc).__name__}: {exc}",
        )
    # Backends may encode exceptions in MetricOutput instead of raising them.
    # A CUDA context error is never a gate/support NA or a recoverable metric.
    raise_if_cuda_failure(output.error)
    elapsed_s = time.perf_counter() - started_at
    status = _metric_status(output)
    log_progress(
        "metric",
        sample_id=sample.sample_id,
        index=index_label,
        metric=metric.name,
        status=status,
        elapsed_s=f"{elapsed_s:.1f}",
        score=_metric_score_text(output) if status == "ok" else None,
        error=output.error,
    )
    return output


def _record_suite_metric_failure(
    aggregate_payload: dict[str, Any],
    *,
    suite: str,
    metric_name: str,
    error: str,
    sample_ids: list[str],
) -> None:
    aggregate_payload.setdefault("failure_summary", {}).setdefault(suite, {})[metric_name] = {
        "failed_samples": len(sample_ids),
        "eligibility_statuses": ["official"],
        "sample_ids": sample_ids[:5],
        "errors": [{"message": error, "count": 1}],
    }


def _record_suite_metric_not_applicable(
    aggregate_payload: dict[str, Any],
    *,
    suite: str,
    metric_name: str,
    reason: str,
    sample_ids: list[str],
) -> None:
    aggregate_payload.setdefault("not_applicable_summary", {}).setdefault(suite, {})[metric_name] = {
        "samples": len(sample_ids),
        "sample_ids": sample_ids[:5],
        "reasons": [{"message": reason, "count": 1}],
    }


def _estimated_clip_count(count: int, runtime: dict[str, Any]) -> int:
    per_video = int(runtime.get("max_clips_per_video", 1))
    return count * max(per_video, 1)


def _distribution_clip_floors(
    corpus: Any,
    runtime: dict[str, Any],
) -> tuple[int, int]:
    """Resolve clip-count floors, letting a metric override the corpus default.

    FVMD's Frechet covariance is singular at ``<= 1024`` windows, so production
    config raises that metric's floors without also starving JEDi.
    """
    prediction_floor = runtime.get("min_prediction_clips", corpus.min_prediction_clips)
    reference_floor = runtime.get("min_reference_clips", corpus.min_reference_clips)
    return int(prediction_floor), int(reference_floor)


def _run_distribution_backend(
    *,
    backend: str,
    task: DistributionTask,
    metric_name: str,
    config: BenchmarkConfig,
    output_dir: Path,
) -> dict[str, Any]:
    runtime = dict(config.metric_runtime.get(metric_name, {}))
    cache_dir = config.reference_corpus.cache_dir
    compute = (
        compute_frechet_video_motion_distance if backend == "fvmd" else compute_jedi_distance
    )
    return compute(
        prediction_entries=task.prediction_entries(),
        reference_entries=task.reference,
        normalization=config.normalization.get(metric_name, {}),
        runtime=runtime,
        output_dir=output_dir / "suite_metrics" / task.label / metric_name,
        reference_cache_dir=(None if cache_dir is None else Path(cache_dir) / backend),
    )


def apply_suite_level_metrics(
    *,
    aggregate_payload: dict[str, Any],
    suite_level_metric_names: dict[str, list[str]],
    suite_predictions: dict[str, list[PredictionEntry]],
    suite_sample_ids: dict[str, list[str]],
    config: BenchmarkConfig,
    output_dir: Path,
    reference_index: ReferenceIndex | None = None,
) -> None:
    """Score the distribution metrics against the fixed reference corpus.

    Each suite yields one headline score over its whole prediction set plus, for
    static suites, one score per ``(style, environment)`` stratum so a shift can
    be attributed to a look or a scene domain.
    """
    corpus = config.reference_corpus
    if reference_index is None:
        reference_index = resolve_reference_index(config)

    for suite, metric_names in suite_level_metric_names.items():
        if not metric_names:
            continue
        predictions = suite_predictions.get(suite, [])
        sample_ids = suite_sample_ids.get(suite, [])
        prediction_sample_ids = [entry.sample_id for entry in predictions]

        for metric_name in metric_names:
            backend = _metric_backend(config, metric_name)
            metric_label = "FVMD" if metric_name == "fvmd" else "JEDi"
            log_progress(
                "suite_metric_start",
                suite=suite,
                metric=metric_name,
                backend=backend,
                predictions=len(predictions),
                samples=len(sample_ids),
            )
            if backend not in {"fvmd", "jedi"}:
                error = f"unsupported suite-level backend '{backend}' for {metric_name}"
                log_progress(
                    "suite_metric", suite=suite, metric=metric_name, status="fail", error=error
                )
                _record_suite_metric_failure(
                    aggregate_payload,
                    suite=suite,
                    metric_name=metric_name,
                    error=error,
                    sample_ids=sample_ids,
                )
                continue

            if reference_index is None:
                reason = (
                    f"{metric_label} needs a reference corpus; enable "
                    "benchmark.reference_corpus"
                )
                log_progress(
                    "suite_metric", suite=suite, metric=metric_name, status="skip", error=reason
                )
                _record_suite_metric_not_applicable(
                    aggregate_payload,
                    suite=suite,
                    metric_name=metric_name,
                    reason=reason,
                    sample_ids=sample_ids,
                )
                continue

            if (len(prediction_sample_ids) != len(sample_ids)
                    or set(prediction_sample_ids) != set(sample_ids)
                    or len(set(prediction_sample_ids)) != len(prediction_sample_ids)):
                _record_suite_metric_failure(
                    aggregate_payload, suite=suite, metric_name=metric_name,
                    error='Distribution prediction population is incomplete or duplicated; refusing subset scoring',
                    sample_ids=sample_ids,
                )
                continue

            tasks = build_distribution_tasks(
                suite=suite,
                predictions=predictions,
                index=reference_index,
                stratified=corpus.stratified,
            )
            if not tasks:
                reason = f"{metric_label} found no prediction videos or no reference group for {suite}"
                log_progress(
                    "suite_metric", suite=suite, metric=metric_name, status="skip", error=reason
                )
                _record_suite_metric_not_applicable(
                    aggregate_payload,
                    suite=suite,
                    metric_name=metric_name,
                    reason=reason,
                    sample_ids=sample_ids,
                )
                continue

            runtime = dict(config.metric_runtime.get(metric_name, {}))
            reference_spec_runtime = reference_runtime(runtime)
            min_prediction_clips, min_reference_clips = _distribution_clip_floors(
                corpus, runtime
            )
            group_results: dict[str, Any] = {}
            headline: dict[str, Any] | None = None

            for task in tasks:
                estimated_predictions = _estimated_clip_count(len(task.predictions), runtime)
                estimated_reference = _estimated_clip_count(
                    len(task.reference), reference_spec_runtime
                )
                if (
                    estimated_predictions < min_prediction_clips
                    or estimated_reference < min_reference_clips
                ):
                    reason = (
                        f"{metric_label} needs at least {min_prediction_clips} prediction "
                        f"and {min_reference_clips} reference clips; "
                        f"{task.label} offers about {estimated_predictions} and {estimated_reference}"
                    )
                    log_progress(
                        "suite_metric",
                        suite=suite,
                        metric=metric_name,
                        group=task.label,
                        status="skip",
                        error=reason,
                    )
                    if task.is_overall:
                        _record_suite_metric_not_applicable(
                            aggregate_payload,
                            suite=suite,
                            metric_name=metric_name,
                            reason=reason,
                            sample_ids=sample_ids,
                        )
                    else:
                        group_results[task.label] = {"status": "not_applicable", "reason": reason}
                    continue

                started_at = time.perf_counter()
                try:
                    with ProgressHeartbeat(
                        suite=suite, metric=metric_name, stage=f"suite_score:{task.label}"
                    ):
                        result = _run_distribution_backend(
                            backend=backend,
                            task=task,
                            metric_name=metric_name,
                            config=config,
                            output_dir=output_dir,
                        )
                except Exception as exc:  # pragma: no cover - optional heavy runtime failures
                    log_exception(
                        "suite_metric",
                        exc,
                        suite=suite,
                        metric=metric_name,
                        group=task.label,
                        status="fail",
                        elapsed_s=f"{time.perf_counter() - started_at:.1f}",
                    )
                    error = f"{type(exc).__name__}: {exc}"
                    if task.is_overall:
                        _record_suite_metric_failure(
                            aggregate_payload,
                            suite=suite,
                            metric_name=metric_name,
                            error=error,
                            sample_ids=prediction_sample_ids or sample_ids,
                        )
                    else:
                        group_results[task.label] = {"status": "failed", "error": error}
                    continue

                normalized = result.get("normalized")
                elapsed_s = f"{time.perf_counter() - started_at:.1f}"
                if normalized is None:
                    error = f"{metric_name} score could not be normalized"
                    log_progress(
                        "suite_metric",
                        suite=suite,
                        metric=metric_name,
                        group=task.label,
                        status="fail",
                        elapsed_s=elapsed_s,
                        error=error,
                    )
                    if task.is_overall:
                        _record_suite_metric_failure(
                            aggregate_payload,
                            suite=suite,
                            metric_name=metric_name,
                            error=error,
                            sample_ids=prediction_sample_ids or sample_ids,
                        )
                    else:
                        group_results[task.label] = {"status": "failed", "error": error}
                    continue

                log_progress(
                    "suite_metric",
                    suite=suite,
                    metric=metric_name,
                    group=task.label,
                    status="ok",
                    elapsed_s=elapsed_s,
                    score=_format_score(normalized),
                )
                if task.is_overall:
                    headline = result
                else:
                    group_results[task.label] = {
                        "status": "ok",
                        "raw": result.get("raw"),
                        "normalized": round(float(normalized), 4),
                        "prediction_sample_count": len(task.predictions),
                        "reference_video_count": len(task.reference),
                        "details": result.get("details", {}),
                    }

            if headline is None:
                continue
            payload = dict(headline)
            payload["groups"] = group_results
            payload["reference_corpus_signature"] = reference_index.signature
            aggregate_payload.setdefault("per_metric_summary", {}).setdefault(suite, {})[
                metric_name
            ] = round(float(headline["normalized"]), 4)
            aggregate_payload.setdefault("distribution_metric_details", {}).setdefault(suite, {})[
                metric_name
            ] = payload

    refresh_aggregate_indexes(aggregate_payload)


@cache_file_digests()
def run_benchmark(
    *,
    config: BenchmarkConfig,
    manifest: list[BenchmarkSample],
    predictions_root: Path,
    output_dir: Path,
    model_name: str,
    suites: set[str] | None = None,
    limit: int | None = None,
    run_suite_level_metrics: bool = True,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    filtered_manifest = [
        sample
        for sample in manifest
        if not suites or sample.suite in suites
    ]
    if limit is not None:
        filtered_manifest = filtered_manifest[:limit]
    if len({s.sample_id for s in filtered_manifest}) != len(filtered_manifest):
        raise ValueError("Duplicate sample IDs in evaluation manifest")
    prediction_paths: dict[str, Path | None] = {}
    for sample in filtered_manifest:
        from worldatlas.benchmark.video_metric_identity import validate_metric_scope
        validate_metric_scope(sample, config.metrics_by_suite.get(sample.suite, []), config.require_verified_image_bindings)
        from worldatlas.benchmark.camera_video_identity import POLICY as CAMERA_POLICY
        if sample.sample_id_policy == CAMERA_POLICY and (
                config.metric_backends.get('camera_error') != 'megasam'
                or config.metric_runtime.get('camera_error', {}).get('camera_gt_policy') != 'native_intended'):
            raise ValueError('Recorded camera videos require MegaSAM native_intended scoring')
        _validate_image_identity_policy(sample, required=config.require_verified_image_bindings)
        prediction_paths[sample.sample_id] = _resolve_prediction_path(predictions_root, sample)

    per_sample_metric_names_by_suite: dict[str, list[str]] = {}
    suite_level_metric_names: dict[str, list[str]] = {}
    for suite_name, metric_names in config.metrics_by_suite.items():
        if suites is not None and suite_name not in suites:
            continue
        per_sample_metric_names, suite_metrics = split_suite_level_metrics(metric_names)
        per_sample_metric_names_by_suite[suite_name] = per_sample_metric_names
        suite_level_metric_names[suite_name] = suite_metrics
    metrics_by_suite = {
        suite_name: build_metrics(metric_names, config)
        for suite_name, metric_names in per_sample_metric_names_by_suite.items()
    }
    suite_predictions: dict[str, list[PredictionEntry]] = defaultdict(list)
    suite_sample_ids: dict[str, list[str]] = defaultdict(list)
    sample_scores: list[SampleScore] = []
    missing_predictions = 0
    status_counts = {"ok": 0, "skip": 0, "fail": 0}
    job_started_at = time.perf_counter()
    metric_names = _configured_metric_names(config, suites)
    shard_index = os.environ.get("WORLDATLAS_EVAL_SHARD_INDEX")
    num_shards = os.environ.get("WORLDATLAS_EVAL_NUM_SHARDS")
    shard_label = None
    if shard_index is not None and num_shards is not None:
        shard_label = f"{shard_index}/{num_shards}"
    selected_total = len(filtered_manifest)
    checkpoints = load_sample_checkpoint(checkpoint_path) if checkpoint_path is not None else {}
    from worldatlas.benchmark.integrity import sample_fingerprint, scoring_context
    integrity_context = (
        scoring_context(config, Path(__file__).resolve().parents[2], metric_names)
        if checkpoint_path is not None else None
    )
    log_progress(
        "job_start",
        job_id=os.environ.get("WORLDATLAS_JOB_ID"),
        model=model_name,
        suite=",".join(sorted(suites)) if suites else ",".join(sorted(config.metrics_by_suite)),
        metrics=",".join(metric_names) if metric_names else "config",
        samples=selected_total,
        shard=shard_label,
        gpu=os.environ.get("WORLDATLAS_EVAL_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES"),
        python=sys.executable,
        output_dir=str(output_dir),
        predictions_root=str(predictions_root),
    )

    for sample_index, sample in enumerate(filtered_manifest, start=1):
        index_label = f"{sample_index}/{selected_total}"
        suite_metrics = metrics_by_suite.get(sample.suite, [])
        metric_list = ",".join(metric.name for metric in suite_metrics) or ",".join(
            suite_level_metric_names.get(sample.suite, [])
        )
        log_progress(
            "sample_start",
            sample_id=sample.sample_id,
            index=index_label,
            suite=sample.suite,
            metrics=metric_list or None,
        )
        if run_suite_level_metrics and suite_level_metric_names.get(sample.suite):
            suite_sample_ids[sample.suite].append(sample.sample_id)
        prediction_path = prediction_paths[sample.sample_id]
        if prediction_path is None:
            sample_scores.append(
                SampleScore(
                    sample_id=sample.sample_id,
                    suite=sample.suite,
                    metrics={},
                    path=sample.path,
                    prediction_path=None,
                    style=sample.style,
                    environment=sample.environment,
                    scene=sample.scene,
                    motion_category=sample.motion_category,
                    error="missing prediction",
                )
            )
            missing_predictions += 1
            status_counts["skip"] += 1
            log_progress(
                "sample",
                sample_id=sample.sample_id,
                index=index_label,
                status="skip",
                metrics=metric_list or None,
                error="missing prediction",
                ok=status_counts["ok"],
                skipped=status_counts["skip"],
                failed=status_counts["fail"],
            )
            continue

        # Collected before the per-sample metrics run: the distribution metrics
        # score against the reference corpus, so they neither need this sample's
        # reference asset nor should they lose a prediction because an unrelated
        # per-sample metric failed. Undecodable predictions are skipped, and
        # reported, inside the backend.
        if (
            run_suite_level_metrics
            and suite_level_metric_names.get(sample.suite)
            and _prediction_modality(sample) == "video"
        ):
            suite_predictions[sample.suite].append(
                PredictionEntry(
                    path=prediction_path,
                    sample_id=sample.sample_id,
                    style=sample.style,
                    environment=sample.environment,
                )
            )

        sample_started_at = time.perf_counter()
        input_fingerprint = (
            sample_fingerprint(sample, prediction_path, integrity_context)
            if integrity_context is not None else None
        )
        cached = checkpoints.get(sample.sample_id)
        if (
            cached is not None
            and cached.get("prediction_path") == str(prediction_path)
            and cached.get("suite") == sample.suite
            and cached.get("input_fingerprint") == input_fingerprint
            and not sample_score_failed(cached, [metric.name for metric in suite_metrics])
        ):
            cached = dict(cached)
            cached["metrics"] = {name: MetricOutput(**value) for name, value in cached["metrics"].items()}
            sample_scores.append(SampleScore(**cached))
            status_counts["ok"] += 1
            log_progress("sample_resume", sample_id=sample.sample_id, index=index_label, status="ok")
            continue
        try:
            # Distribution-only shards collect input identities. Their full videos
            # are decoded once by the suite backend, where skipped clips are audited.
            if suite_metrics:
                reference_start_ratio = _evaluation_start_ratio(sample)
                prediction_start_ratio = _prediction_window_start_ratio(sample, prediction_path)
                with ProgressHeartbeat(
                    sample_id=sample.sample_id,
                    index=index_label,
                    stage="decode",
                ):
                    reference = None
                    if any(getattr(metric, "requires_reference", True) for metric in suite_metrics):
                        reference = load_media(
                            path=Path(sample.reference_path),
                            modality=sample.modality,
                            anchor_positions=config.anchor_positions,
                            frame_count=config.motion_frame_count,
                            sample_key=sample.sample_id,
                            protocol_name=config.protocol.official,
                            protocol_seed=config.protocol.seed,
                            segment_count=config.protocol.segment_count,
                            start_ratio=reference_start_ratio,
                            end_ratio=1.0,
                        )
                    prediction = load_media(
                        path=prediction_path,
                        modality=_prediction_modality(sample),
                        anchor_positions=config.anchor_positions,
                        frame_count=config.motion_frame_count,
                        sample_key=sample.sample_id,
                        protocol_name=config.protocol.official,
                        protocol_seed=config.protocol.seed,
                        segment_count=config.protocol.segment_count,
                        start_ratio=prediction_start_ratio,
                        end_ratio=1.0,
                    )
            metrics: dict[str, MetricOutput] = {}
            for metric in suite_metrics:
                output = _score_one_metric(
                    metric,
                    sample=sample,
                    prediction=prediction,
                    reference=reference,
                    index_label=index_label,
                )
                if output.protocol_name is None:
                    output.protocol_name = prediction.evaluation_protocol
                metrics[metric.name] = output
            metric_statuses = [_metric_status(output) for output in metrics.values()]
            sample_status = _sample_status(metric_statuses)
            status_counts[sample_status] += 1
            sample_scores.append(
                SampleScore(
                    sample_id=sample.sample_id,
                    suite=sample.suite,
                    metrics=metrics,
                    path=sample.path,
                    prediction_path=str(prediction_path),
                    input_fingerprint=input_fingerprint,
                    style=sample.style,
                    environment=sample.environment,
                    scene=sample.scene,
                    motion_category=sample.motion_category,
                )
            )
            if checkpoint_path is not None and not sample_score_failed(sample_scores[-1].to_dict()):
                append_sample_checkpoint(checkpoint_path, sample_scores[-1].to_dict())
            elapsed_s = time.perf_counter() - job_started_at
            done = status_counts["ok"] + status_counts["skip"] + status_counts["fail"]
            log_progress(
                "sample",
                sample_id=sample.sample_id,
                index=index_label,
                status=sample_status,
                metrics=metric_list or None,
                elapsed_s=f"{time.perf_counter() - sample_started_at:.1f}",
                total_elapsed_s=f"{elapsed_s:.1f}",
                ok=status_counts["ok"],
                skipped=status_counts["skip"],
                failed=status_counts["fail"],
                scores=_compact_scores(metrics),
                rate_samples_per_hour=rate_samples_per_hour(done, elapsed_s),
            )
        except (MetricLoadError, CudaContextError):
            raise
        except Exception as exc:  # pragma: no cover - runtime decoding failure
            status_counts["fail"] += 1
            log_exception(
                "sample",
                exc,
                sample_id=sample.sample_id,
                index=index_label,
                status="fail",
                elapsed_s=f"{time.perf_counter() - sample_started_at:.1f}",
            )
            sample_scores.append(
                SampleScore(
                    sample_id=sample.sample_id,
                    suite=sample.suite,
                    metrics={},
                    path=sample.path,
                    prediction_path=str(prediction_path),
                    style=sample.style,
                    environment=sample.environment,
                    scene=sample.scene,
                    motion_category=sample.motion_category,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    aggregate_payload = aggregate_results(sample_scores=sample_scores)
    reference_index = None
    if run_suite_level_metrics and any(suite_level_metric_names.values()):
        reference_index = resolve_reference_index(config)
        apply_suite_level_metrics(
            aggregate_payload=aggregate_payload,
            suite_level_metric_names=suite_level_metric_names,
            suite_predictions=suite_predictions,
            suite_sample_ids=suite_sample_ids,
            config=config,
            output_dir=output_dir,
            reference_index=reference_index,
        )
    run_manifest = {
        "integrity_version": "content_bound_v2",
        "scoring_context": integrity_context,
        "identity_policy": sorted({s.sample_id_policy or "legacy_unverified" for s in filtered_manifest}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "predictions_root": str(predictions_root),
        "output_dir": str(output_dir),
        "items": len(filtered_manifest),
        "missing_predictions": missing_predictions,
        "suites": sorted({sample.suite for sample in filtered_manifest}),
        "requested_metrics": per_sample_metric_names_by_suite,
        "reported_metrics": {
            suite: sorted(metric_map.keys())
            for suite, metric_map in aggregate_payload["per_metric_summary"].items()
        },
        "reported_dimensions": {
            suite: sorted(metric_map.keys())
            for suite, metric_map in aggregate_payload.get("per_dimension_summary", {}).items()
            if metric_map
        },
        "reported_diagnostics": {
            suite: sorted(metric_map.keys())
            for suite, metric_map in aggregate_payload.get("diagnostic_metric_summary", {}).items()
            if metric_map
        },
        "reported_failures": {
            suite: sorted(metric_map.keys())
            for suite, metric_map in aggregate_payload.get("failure_summary", {}).items()
            if metric_map
        },
        "reported_not_applicable": {
            suite: sorted(metric_map.keys())
            for suite, metric_map in aggregate_payload.get("not_applicable_summary", {}).items()
            if metric_map
        },
        "evaluation_protocol": config.protocol.official,
        "metric_implementation_revisions": aggregate_payload.get("metric_implementation_revisions", {}),
        # Distribution scores are only comparable against the same corpus, so the
        # signature travels with the run.
        "reference_corpus": (
            None if reference_index is None else reference_index.describe()
        ),
    }
    write_run_outputs(
        output_dir=output_dir,
        run_manifest=run_manifest,
        sample_scores=sample_scores,
        aggregate_payload=aggregate_payload,
        model_name=model_name,
    )
    elapsed_s = time.perf_counter() - job_started_at
    done = status_counts["ok"] + status_counts["skip"] + status_counts["fail"]
    log_progress(
        "job_done",
        job_id=os.environ.get("WORLDATLAS_JOB_ID"),
        model=model_name,
        shard=shard_label,
        ok=status_counts["ok"],
        skipped=status_counts["skip"],
        failed=status_counts["fail"],
        missing_predictions=missing_predictions,
        elapsed_s=f"{elapsed_s:.1f}",
        rate_samples_per_hour=rate_samples_per_hour(done, elapsed_s),
        output_dir=str(output_dir),
    )
    return {
        "run_manifest": run_manifest,
        "sample_scores": sample_scores,
        "aggregate_payload": aggregate_payload,
    }
