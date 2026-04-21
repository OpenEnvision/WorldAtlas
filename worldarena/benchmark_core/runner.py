from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from worldarena.benchmark_core.aggregates import aggregate_results
from worldarena.benchmark_core.config import BenchmarkConfig
from worldarena.benchmark_core.media import load_media
from worldarena.benchmark_core.metrics.registry import build_metrics
from worldarena.benchmark_core.report import write_run_outputs
from worldarena.benchmark_core.schemas import BenchmarkSample, SampleScore
from worldarena.benchmark_core.taxonomy import prediction_extensions


def _resolve_prediction_path(predictions_root: Path, sample: BenchmarkSample) -> Path | None:
    candidates: list[Path] = []
    for extension in prediction_extensions(sample.modality):
        candidates.extend(
            [
                predictions_root / sample.suite / sample.split / f"{sample.prediction_stem}{extension}",
                predictions_root / sample.suite / f"{sample.prediction_stem}{extension}",
                predictions_root / sample.split / f"{sample.prediction_stem}{extension}",
                predictions_root / f"{sample.prediction_stem}{extension}",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_benchmark(
    *,
    config: BenchmarkConfig,
    manifest: list[BenchmarkSample],
    predictions_root: Path,
    output_dir: Path,
    model_name: str,
    suites: set[str] | None = None,
    splits: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    filtered_manifest = [
        sample
        for sample in manifest
        if (not suites or sample.suite in suites) and (not splits or sample.split in splits)
    ]
    if limit is not None:
        filtered_manifest = filtered_manifest[:limit]

    metrics_by_suite = {
        suite_name: build_metrics(metric_names, config)
        for suite_name, metric_names in config.metrics_by_suite.items()
    }
    sample_scores: list[SampleScore] = []
    missing_predictions = 0

    for sample in filtered_manifest:
        prediction_path = _resolve_prediction_path(predictions_root, sample)
        if prediction_path is None:
            sample_scores.append(
                SampleScore(
                    sample_id=sample.sample_id,
                    suite=sample.suite,
                    split=sample.split,
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
            reference = load_media(
                path=Path(sample.reference_path),
                modality=sample.modality,
                anchor_positions=config.anchor_positions,
                frame_count=config.motion_frame_count,
                sample_key=sample.sample_id,
                protocol_name=config.protocol.official,
                protocol_seed=config.protocol.seed,
                segment_count=config.protocol.segment_count,
            )
            prediction = load_media(
                path=prediction_path,
                modality=sample.modality,
                anchor_positions=config.anchor_positions,
                frame_count=config.motion_frame_count,
                sample_key=sample.sample_id,
                protocol_name=config.protocol.official,
                protocol_seed=config.protocol.seed,
                segment_count=config.protocol.segment_count,
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
                    split=sample.split,
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
                    split=sample.split,
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
    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "predictions_root": str(predictions_root),
        "output_dir": str(output_dir),
        "items": len(filtered_manifest),
        "missing_predictions": missing_predictions,
        "suites": sorted({sample.suite for sample in filtered_manifest}),
        "splits": sorted({sample.split for sample in filtered_manifest}),
        "reported_metrics": {
            suite: sorted(metric_map.keys())
            for suite, metric_map in aggregate_payload["per_metric_summary"].items()
        },
        "reported_diagnostics": {
            suite: sorted(metric_map.keys())
            for suite, metric_map in aggregate_payload.get("diagnostic_metric_summary", {}).items()
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
