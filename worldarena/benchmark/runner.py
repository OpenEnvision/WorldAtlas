from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worldarena.benchmark.aggregates import aggregate_results, refresh_aggregate_indexes
from worldarena.benchmark.config import BenchmarkConfig
from worldarena.benchmark.fvmd_backend import compute_frechet_video_motion_distance
from worldarena.benchmark.jedi_backend import compute_jedi_distance
from worldarena.benchmark.media import load_media, probe_media_frame_count
from worldarena.benchmark.metrics.registry import build_metrics, split_suite_level_metrics
from worldarena.benchmark.report import write_run_outputs
from worldarena.benchmark.schemas import BenchmarkSample, SampleScore
from worldarena.benchmark.taxonomy import prediction_extensions, prediction_modality_for_task


def _prediction_modality(sample: BenchmarkSample) -> str:
    return prediction_modality_for_task(
        suite=sample.suite,
        source_modality=sample.modality,
        task_family=sample.task_family,
        artifact_type=sample.artifact_type,
    )


def _resolve_prediction_path(predictions_root: Path, sample: BenchmarkSample) -> Path | None:
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


def _apply_suite_level_metrics(
    *,
    aggregate_payload: dict[str, Any],
    suite_level_metric_names: dict[str, list[str]],
    suite_prediction_pairs: dict[str, list[tuple[Path, Path, str]]],
    suite_sample_ids: dict[str, list[str]],
    config: BenchmarkConfig,
    output_dir: Path,
) -> None:
    for suite, metric_names in suite_level_metric_names.items():
        if not metric_names:
            continue
        pairs = suite_prediction_pairs.get(suite, [])
        sample_ids = suite_sample_ids.get(suite, [])
        pair_sample_ids = [sample_id for _, _, sample_id in pairs]
        for metric_name in metric_names:
            backend = _metric_backend(config, metric_name)
            if backend not in {"fvmd", "jedi"}:
                _record_suite_metric_failure(
                    aggregate_payload,
                    suite=suite,
                    metric_name=metric_name,
                    error=f"unsupported suite-level backend '{backend}' for {metric_name}",
                    sample_ids=sample_ids,
                )
                continue
            if not pairs:
                metric_label = "FVMD" if metric_name == "fvmd" else "JEDi"
                _record_suite_metric_not_applicable(
                    aggregate_payload,
                    suite=suite,
                    metric_name=metric_name,
                    reason=f"{metric_label} requires video prediction/reference pairs",
                    sample_ids=sample_ids,
                )
                continue
            try:
                if backend == "fvmd":
                    result = compute_frechet_video_motion_distance(
                        prediction_reference_pairs=pairs,
                        normalization=config.normalization.get(metric_name, {}),
                        runtime=dict(config.metric_runtime.get(metric_name, {})),
                        output_dir=output_dir / "suite_metrics" / suite / metric_name,
                    )
                else:
                    result = compute_jedi_distance(
                        prediction_reference_pairs=pairs,
                        normalization=config.normalization.get(metric_name, {}),
                        runtime=dict(config.metric_runtime.get(metric_name, {})),
                        output_dir=output_dir / "suite_metrics" / suite / metric_name,
                    )
            except Exception as exc:  # pragma: no cover - optional heavy runtime failures
                _record_suite_metric_failure(
                    aggregate_payload,
                    suite=suite,
                    metric_name=metric_name,
                    error=str(exc),
                    sample_ids=pair_sample_ids or sample_ids,
                )
                continue

            normalized = result.get("normalized")
            if normalized is None:
                _record_suite_metric_failure(
                    aggregate_payload,
                    suite=suite,
                    metric_name=metric_name,
                    error=f"{metric_name} score could not be normalized",
                    sample_ids=pair_sample_ids or sample_ids,
                )
                continue
            aggregate_payload.setdefault("per_metric_summary", {}).setdefault(suite, {})[
                metric_name
            ] = round(float(normalized), 4)
            aggregate_payload.setdefault("distribution_metric_details", {}).setdefault(suite, {})[
                metric_name
            ] = result

    refresh_aggregate_indexes(aggregate_payload)


def run_benchmark(
    *,
    config: BenchmarkConfig,
    manifest: list[BenchmarkSample],
    predictions_root: Path,
    output_dir: Path,
    model_name: str,
    suites: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    filtered_manifest = [
        sample
        for sample in manifest
        if not suites or sample.suite in suites
    ]
    if limit is not None:
        filtered_manifest = filtered_manifest[:limit]

    per_sample_metric_names_by_suite: dict[str, list[str]] = {}
    suite_level_metric_names: dict[str, list[str]] = {}
    for suite_name, metric_names in config.metrics_by_suite.items():
        per_sample_metric_names, suite_metrics = split_suite_level_metrics(metric_names)
        per_sample_metric_names_by_suite[suite_name] = per_sample_metric_names
        suite_level_metric_names[suite_name] = suite_metrics
    metrics_by_suite = {
        suite_name: build_metrics(metric_names, config)
        for suite_name, metric_names in per_sample_metric_names_by_suite.items()
    }
    suite_prediction_pairs: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
    suite_sample_ids: dict[str, list[str]] = defaultdict(list)
    sample_scores: list[SampleScore] = []
    missing_predictions = 0

    for sample in filtered_manifest:
        if suite_level_metric_names.get(sample.suite):
            suite_sample_ids[sample.suite].append(sample.sample_id)
        prediction_path = _resolve_prediction_path(predictions_root, sample)
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
            continue

        try:
            reference_start_ratio = _evaluation_start_ratio(sample)
            prediction_start_ratio = _prediction_window_start_ratio(sample, prediction_path)
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
            if (
                suite_level_metric_names.get(sample.suite)
                and sample.modality == "video"
                and _prediction_modality(sample) == "video"
            ):
                suite_prediction_pairs[sample.suite].append(
                    (prediction_path, Path(sample.reference_path), sample.sample_id)
                )
            metrics = {
                metric.name: metric.compute(sample=sample, prediction=prediction, reference=reference)
                for metric in metrics_by_suite[sample.suite]
            }
            for output in metrics.values():
                output.protocol_name = prediction.evaluation_protocol
            sample_scores.append(
                SampleScore(
                    sample_id=sample.sample_id,
                    suite=sample.suite,
                    metrics=metrics,
                    path=sample.path,
                    prediction_path=str(prediction_path),
                    style=sample.style,
                    environment=sample.environment,
                    scene=sample.scene,
                    motion_category=sample.motion_category,
                )
            )
        except Exception as exc:  # pragma: no cover - runtime decoding failure
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
                    error=str(exc),
                )
            )

    aggregate_payload = aggregate_results(sample_scores=sample_scores)
    _apply_suite_level_metrics(
        aggregate_payload=aggregate_payload,
        suite_level_metric_names=suite_level_metric_names,
        suite_prediction_pairs=suite_prediction_pairs,
        suite_sample_ids=suite_sample_ids,
        config=config,
        output_dir=output_dir,
    )
    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "predictions_root": str(predictions_root),
        "output_dir": str(output_dir),
        "items": len(filtered_manifest),
        "missing_predictions": missing_predictions,
        "suites": sorted({sample.suite for sample in filtered_manifest}),
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
    }
    write_run_outputs(
        output_dir=output_dir,
        run_manifest=run_manifest,
        sample_scores=sample_scores,
        aggregate_payload=aggregate_payload,
        model_name=model_name,
    )
    return {
        "run_manifest": run_manifest,
        "sample_scores": sample_scores,
        "aggregate_payload": aggregate_payload,
    }
