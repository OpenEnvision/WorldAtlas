"""Prediction generation orchestrator for benchmark samples."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from worldatlas.common.video_io import extract_video_window, freeze_frame_to_video
from worldatlas.common.checkpoints import rehome_legacy_workspace_path
from worldatlas.common.progress import (
    ProgressHeartbeat,
    log_progress,
    rate_samples_per_hour,
)
from worldatlas.models.adapters.base import PreparedGenerationRequest
from worldatlas.models.adapters.common import first_frame_to_image
from worldatlas.models.config import ModelRuntimeConfig
from worldatlas.models.registry import build_model_adapter
from worldatlas.benchmark.schemas import BenchmarkSample
from worldatlas.benchmark.sample_identity import cache_file_digests


def _select_samples(
    manifest: list[BenchmarkSample],
    *,
    suites: set[str] | None,
    limit: int | None,
    supported_suites: set[str],
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[BenchmarkSample]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be within [0, {num_shards - 1}]")

    selected: list[BenchmarkSample] = []
    if limit == 0:
        return selected
    for sample in manifest:
        if sample.suite not in supported_suites:
            continue
        if suites and sample.suite not in suites:
            continue
        selected.append(sample)
        if limit is not None and len(selected) >= limit:
            break
    if num_shards == 1:
        return selected
    return selected[shard_index::num_shards]


def _conditioning_asset_for(
    sample: BenchmarkSample,
    *,
    reference_mode: str,
    cache_root: Path,
    min_video_frames: int,
) -> Path:
    if reference_mode == "none":
        return cache_root / f"{sample.prediction_stem}.unused"
    if sample.conditioning_path:
        remapped_conditioning = rehome_legacy_workspace_path(sample.conditioning_path)
        if remapped_conditioning is None:
            raise FileNotFoundError(f"conditioning asset not found: {sample.conditioning_path}")
        conditioning_path = Path(remapped_conditioning).expanduser().resolve()
        if not conditioning_path.exists():
            raise FileNotFoundError(f"conditioning asset not found: {conditioning_path}")
        return conditioning_path

    remapped_reference = rehome_legacy_workspace_path(sample.reference_path)
    if remapped_reference is None:
        raise FileNotFoundError(f"reference asset not found: {sample.reference_path}")
    reference_path = Path(remapped_reference).expanduser().resolve()
    image_modes = {"first_frame", "reference_image", "external_image"}
    video_modes = {"reference_video", "external_video"}
    if sample.sample_id_policy and reference_mode in video_modes:
        from worldatlas.benchmark.sample_identity import validate_source_identity, file_sha256
        cache_identity = json.dumps({
            "source": validate_source_identity(sample), "mode": reference_mode,
            "min_frames": min_video_frames, "frames": sample.conditioning_frame_count,
            "end_ratio": sample.conditioning_end_ratio,
            "conditioning_code": file_sha256(Path(__file__)),
            "video_io_code": file_sha256(Path(__file__).resolve().parents[1] / 'common/video_io.py'),
        }, sort_keys=True)
        cache_root = cache_root / hashlib.sha256(cache_identity.encode()).hexdigest()
        cache_root.mkdir(parents=True, exist_ok=True)

    if reference_mode in image_modes:
        if reference_mode == "reference_image" or sample.modality == "image":
            return reference_path
        conditioning_path = cache_root / f"{sample.prediction_stem}.png"
        if conditioning_path.exists():
            return conditioning_path
        return first_frame_to_image(reference_path, conditioning_path)

    if reference_mode in video_modes:
        conditioning_path = cache_root / f"{sample.prediction_stem}.mp4"
        if conditioning_path.exists():
            return conditioning_path
        if sample.modality == "image":
            return freeze_frame_to_video(
                reference_path,
                conditioning_path,
                frame_count=max(sample.conditioning_frame_count or 1, min_video_frames),
            )
        if sample.conditioning_end_ratio is not None and sample.conditioning_end_ratio > 0.0:
            return extract_video_window(
                reference_path,
                conditioning_path,
                start_ratio=0.0,
                end_ratio=sample.conditioning_end_ratio,
                min_frame_count=max(sample.conditioning_frame_count or 1, min_video_frames),
            )
        anchor_path = cache_root / f"{sample.prediction_stem}.anchor.png"
        if not anchor_path.exists():
            first_frame_to_image(reference_path, anchor_path)
        return freeze_frame_to_video(
            anchor_path,
            conditioning_path,
            frame_count=max(sample.conditioning_frame_count or 1, min_video_frames),
        )

    raise ValueError(f"unsupported reference mode: {reference_mode}")


def _apply_batch_generation_results(
    *,
    adapter,
    pending_requests: list[PreparedGenerationRequest],
    records_by_sample_id: dict[str, dict[str, Any]],
) -> None:
    started_at = time.perf_counter()
    try:
        batch_results = adapter.generate_batch(pending_requests)
    except Exception as exc:
        error_text = str(exc)
        elapsed = round(time.perf_counter() - started_at, 6)
        for request in pending_requests:
            record = records_by_sample_id[request.sample.sample_id]
            record["status"] = "failed"
            record["error"] = error_text
            record["generation_wall_time_seconds"] = elapsed
        raise
    elapsed = round(time.perf_counter() - started_at, 6)
    per_request_elapsed = round(elapsed / max(len(pending_requests), 1), 6)

    for request in pending_requests:
        record = records_by_sample_id[request.sample.sample_id]
        payload = dict(batch_results.get(request.sample.sample_id, {}))
        if not payload:
            payload = {
                "status": "failed",
                "error": "batch generation returned no result for this sample",
            }

        status = str(payload.pop("status", "generated" if not payload.get("error") else "failed"))
        error = payload.pop("error", None)
        if "prediction_path" not in payload and request.output_path.exists():
            payload["prediction_path"] = str(request.output_path)
        payload.setdefault("prompt", request.prompt)
        payload.setdefault("generation_wall_time_seconds", per_request_elapsed)
        payload.setdefault("generation_batch_wall_time_seconds", elapsed)
        payload.setdefault("generation_batch_size", len(pending_requests))
        record.update(payload)
        record["status"] = status
        record["error"] = error
        if status in {"generated", "skipped_existing"} and not _nonempty_file(request.output_path):
            record["status"] = "failed"
            record["error"] = f"generation did not write output: {request.output_path}"
            record["prediction_path"] = str(request.output_path)


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"generated": 0, "skipped_existing": 0, "failed": 0, "pending": 0}
    for record in records:
        status = str(record.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _log_sample_progress(
    *,
    sample_id: str,
    index: int,
    total: int,
    status: str,
    job_started_at: float,
    sample_elapsed_s: float | None = None,
    error: str | None = None,
    records: list[dict[str, Any]],
) -> None:
    counts = _status_counts(records)
    elapsed_s = time.perf_counter() - job_started_at
    done = counts["generated"] + counts["skipped_existing"] + counts["failed"]
    log_progress(
        "sample",
        sample_id=sample_id,
        index=f"{index}/{total}",
        status=status,
        elapsed_s=None if sample_elapsed_s is None else f"{sample_elapsed_s:.1f}",
        total_elapsed_s=f"{elapsed_s:.1f}",
        generated=counts["generated"],
        skipped=counts["skipped_existing"],
        failed=counts["failed"],
        rate_samples_per_hour=rate_samples_per_hour(done, elapsed_s),
        error=error,
    )


def _validate_record_outputs(records: list[dict[str, Any]]) -> None:
    for record in records:
        status = str(record.get("status") or "")
        if status not in {"generated", "skipped_existing"}:
            continue
        prediction_path = record.get("prediction_path")
        if not prediction_path:
            record["status"] = "failed"
            record["error"] = "generation did not report a prediction_path"
            continue
        path = Path(str(prediction_path)).expanduser()
        if not _nonempty_file(path):
            record["status"] = "failed"
            record["error"] = f"generation did not write output: {path}"
        elif record.get('sample_id_policy') and path.suffix.lower() in {'.mp4', '.mov', '.mkv', '.webm'}:
            import cv2
            capture = cv2.VideoCapture(str(path))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, _ = capture.read()
            capture.release()
            if not ok or count < 2:
                record['status'] = 'failed'
                record['error'] = f'Generated video is corrupt or has fewer than two reported frames: {path}'


@cache_file_digests()
def generate_predictions(
    *,
    model_config: ModelRuntimeConfig,
    manifest: list[BenchmarkSample],
    output_dir: Path,
    suites: set[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
    fail_fast: bool = False,
) -> dict[str, Any]:
    adapter = build_model_adapter(model_config)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (output_dir / "_conditioning").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    selected_samples = _select_samples(
        manifest,
        suites=suites,
        limit=limit,
        supported_suites=set(model_config.supported_suites),
        num_shards=num_shards,
        shard_index=shard_index,
    )
    records: list[dict[str, Any]] = []
    prompt_map: dict[str, str] = {}
    records_by_sample_id: dict[str, dict[str, Any]] = {}
    pending_requests: list[PreparedGenerationRequest] = []
    use_batch_generation = adapter.supports_batch_generation()
    checkpoint_load_policy = adapter.batch_checkpoint_load_policy()
    if bool(model_config.generation.get("require_persistent_batch", False)):
        if not use_batch_generation or checkpoint_load_policy != "load_once":
            raise RuntimeError(
                f"{model_config.name} requires persistent batch generation, but "
                f"{adapter.__class__.__name__} reports checkpoint policy "
                f"{checkpoint_load_policy!r}"
            )

    suites_label = ",".join(sorted(suites)) if suites else ",".join(model_config.supported_suites)
    job_started_at = time.perf_counter()
    selected_total = len(selected_samples)
    log_progress(
        "job_start",
        job_id=os.environ.get("WORLDATLAS_JOB_ID"),
        model=model_config.name,
        suite=suites_label,
        shard=f"{shard_index}/{num_shards}",
        worker=os.environ.get("WORLDATLAS_BATCH_WORKER_INDEX"),
        workers=os.environ.get("WORLDATLAS_BATCH_WORKER_COUNT"),
        gpus_per_worker=os.environ.get("WORLDATLAS_BATCH_GPUS_PER_WORKER"),
        samples=selected_total,
        live_gap=os.environ.get("WORLDATLAS_JOB_LIVE_GAP"),
        expected=os.environ.get("WORLDATLAS_JOB_EXPECTED"),
        output_dir=str(output_dir),
        checkpoint_policy=checkpoint_load_policy,
        batch=use_batch_generation,
    )

    for sample_index, sample in enumerate(selected_samples, start=1):
        if sample.source_name == 'worldatlas_formal_image' and not sample.sample_id_policy:
            raise ValueError('Formal WorldAtlas image generation requires the filename-ID manifest; legacy ordinal generation is disabled')
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        source_hash = None
        if sample.sample_id_policy:
            from worldatlas.benchmark.sample_identity import validate_source_identity, resolve_bound_prediction
            source_hash = validate_source_identity(sample)
            if prediction_path.exists() and not overwrite:
                bound_path = resolve_bound_prediction(sample, output_dir)
                if bound_path.resolve() != prediction_path.resolve():
                    raise ValueError(f"Existing output is not the verified bound video: {prediction_path}")
        prompt = adapter.prompt_for(sample)
        record = {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "prediction_path": str(prediction_path),
            "conditioning_image": None,
            "conditioning_input": None,
            "prompt": prompt,
            "status": "pending",
            "error": None,
        }
        if sample.sample_id_policy:
            record.update(sample_id_policy=sample.sample_id_policy, source_sha256=source_hash, source_path=sample.path)
        try:
            conditioning_image = _conditioning_asset_for(
                sample,
                reference_mode=model_config.reference_mode,
                cache_root=cache_root,
                min_video_frames=int(model_config.generation.get("conditioning_video_frames", 8)),
            )
            record["conditioning_image"] = str(conditioning_image)
            record["conditioning_input"] = str(conditioning_image)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            records.append(record)
            records_by_sample_id[sample.sample_id] = record
            prompt_map[str(prediction_path)] = prompt
            _log_sample_progress(
                sample_id=sample.sample_id,
                index=sample_index,
                total=selected_total,
                status="failed",
                job_started_at=job_started_at,
                error=record["error"],
                records=records,
            )
            if fail_fast:
                _write_generation_outputs(
                    output_dir=output_dir,
                    records=records,
                    prompt_map=prompt_map,
                    model_config=model_config,
                    shard_index=shard_index,
                    num_shards=num_shards,
                )
                raise RuntimeError(
                    f"Failed to prepare conditioning input for {sample.sample_id}: {record['error']}"
                ) from exc
            continue
        generation_started_at: float | None = None
        try:
            if overwrite or not prediction_path.exists():
                if use_batch_generation:
                    pending_requests.append(
                        PreparedGenerationRequest(
                            sample=sample,
                            conditioning_image=conditioning_image,
                            output_path=prediction_path,
                            prompt=prompt,
                        )
                    )
                else:
                    log_progress(
                        "sample_start",
                        sample_id=sample.sample_id,
                        index=f"{sample_index}/{selected_total}",
                    )
                    generation_started_at = time.perf_counter()
                    with ProgressHeartbeat(
                        sample_id=sample.sample_id,
                        index=f"{sample_index}/{selected_total}",
                    ):
                        payload = adapter.generate(
                            sample=sample,
                            conditioning_image=conditioning_image,
                            output_path=prediction_path,
                            prompt=prompt,
                        )
                    payload.setdefault(
                        "generation_wall_time_seconds",
                        round(time.perf_counter() - generation_started_at, 6),
                    )
                    record.update(payload)
                    record["status"] = "generated"
            else:
                record["status"] = "skipped_existing"
                record["prediction_path"] = str(prediction_path)
                # This is a reuse request, not evidence of how the existing
                # video was generated. Never relabel old pixels as new input.
                record["reuse_request"] = {key: record[key] for key in
                                           ("conditioning_image", "conditioning_input", "prompt")}
                record["conditioning_image"] = None
                record["conditioning_input"] = None
                record["prompt"] = None
                record["provenance_status"] = "verified_binding" if sample.prediction_binding else "unverified_existing"
                if sample.prediction_binding:
                    record["prediction_binding"] = dict(sample.prediction_binding)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            if generation_started_at is not None:
                record["generation_wall_time_seconds"] = round(
                    time.perf_counter() - generation_started_at,
                    6,
                )
            records.append(record)
            records_by_sample_id[sample.sample_id] = record
            prompt_map[str(prediction_path)] = prompt
            sample_elapsed = record.get("generation_wall_time_seconds")
            _log_sample_progress(
                sample_id=sample.sample_id,
                index=sample_index,
                total=selected_total,
                status="failed",
                job_started_at=job_started_at,
                sample_elapsed_s=float(sample_elapsed) if sample_elapsed is not None else None,
                error=record["error"],
                records=records,
            )
            if fail_fast:
                _write_generation_outputs(
                    output_dir=output_dir,
                    records=records,
                    prompt_map=prompt_map,
                    model_config=model_config,
                    shard_index=shard_index,
                    num_shards=num_shards,
                )
                raise RuntimeError(
                    f"Generation failed for {sample.sample_id}: {record['error']}"
                ) from exc
            continue
        records.append(record)
        records_by_sample_id[sample.sample_id] = record
        prompt_map[str(prediction_path)] = prompt
        if record["status"] != "pending":
            sample_elapsed = record.get("generation_wall_time_seconds")
            _log_sample_progress(
                sample_id=sample.sample_id,
                index=sample_index,
                total=selected_total,
                status=str(record["status"]),
                job_started_at=job_started_at,
                sample_elapsed_s=float(sample_elapsed) if sample_elapsed is not None else None,
                records=records,
            )

    if use_batch_generation and pending_requests:
        counts = _status_counts(records)
        log_progress(
            "batch_start",
            samples=len(pending_requests),
            skipped=counts["skipped_existing"],
            failed=counts["failed"],
            checkpoint_policy=checkpoint_load_policy,
        )
        batch_started_at = time.perf_counter()
        with ProgressHeartbeat(samples=len(pending_requests), model=model_config.name):
            _apply_batch_generation_results(
                adapter=adapter,
                pending_requests=pending_requests,
                records_by_sample_id=records_by_sample_id,
            )
        _validate_record_outputs(records)
        counts = _status_counts(records)
        log_progress(
            "batch_done",
            elapsed_s=f"{time.perf_counter() - batch_started_at:.1f}",
            generated=counts["generated"],
            skipped=counts["skipped_existing"],
            failed=counts["failed"],
        )
        if fail_fast:
            first_failed = next((item for item in records if item["status"] == "failed"), None)
            if first_failed is not None:
                _write_generation_outputs(
                    output_dir=output_dir,
                    records=records,
                    prompt_map=prompt_map,
                    model_config=model_config,
                    shard_index=shard_index,
                    num_shards=num_shards,
                )
                raise RuntimeError(
                    f"Generation failed for {first_failed['sample_id']}: {first_failed['error']}"
                )
    else:
        _validate_record_outputs(records)

    result = _write_generation_outputs(
        output_dir=output_dir,
        records=records,
        prompt_map=prompt_map,
        model_config=model_config,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    from worldatlas.benchmark.prediction_manifest import write_prediction_manifest

    evaluation_manifest = write_prediction_manifest(
        selected_samples, records, output_dir,
        shard_index=shard_index, num_shards=num_shards,
    )
    result["manifest"]["evaluation_manifest"] = str(evaluation_manifest)
    shard_suffix = f".shard{shard_index}" if num_shards > 1 else ""
    (output_dir / f"generation_manifest{shard_suffix}.json").write_text(
        json.dumps(result["manifest"], indent=2, ensure_ascii=False), encoding="utf-8",
    )
    counts = _status_counts(records)
    elapsed_s = time.perf_counter() - job_started_at
    done = counts["generated"] + counts["skipped_existing"] + counts["failed"]
    log_progress(
        "job_done",
        job_id=os.environ.get("WORLDATLAS_JOB_ID"),
        model=model_config.name,
        shard=f"{shard_index}/{num_shards}",
        generated=counts["generated"],
        skipped=counts["skipped_existing"],
        failed=counts["failed"],
        elapsed_s=f"{elapsed_s:.1f}",
        rate_samples_per_hour=rate_samples_per_hour(done, elapsed_s),
        output_dir=str(output_dir),
    )
    return result


def _write_generation_outputs(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    prompt_map: dict[str, str],
    model_config: ModelRuntimeConfig,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    from worldatlas.benchmark.sample_identity import file_sha256
    for record in records:
        if record['status'] == 'skipped_existing':
            if 'reuse_request' not in record:
                record['reuse_request'] = {key: record.get(key) for key in ('conditioning_image', 'conditioning_input', 'prompt')}
            for key in ('conditioning_image', 'conditioning_input', 'prompt'):
                record[key] = None
            record.setdefault('provenance_status', 'unverified_existing')
        if record.get("sample_id_policy") and record["status"] == "generated":
            if file_sha256(Path(record['source_path'])) != record['source_sha256']:
                raise RuntimeError(f"Source image changed during generation: {record['sample_id']}")
            record["prediction_sha256"] = file_sha256(Path(record["prediction_path"]))
    manifest_payload = {
        "model_name": model_config.name,
        "family": model_config.family,
        "output_dir": str(output_dir),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "records": len(records),
        "generated": sum(1 for item in records if item["status"] == "generated"),
        "skipped_existing": sum(1 for item in records if item["status"] == "skipped_existing"),
        "failed": sum(1 for item in records if item["status"] == "failed"),
        "supported_suites": model_config.supported_suites,
    }
    shard_suffix = f".shard{shard_index}" if num_shards > 1 else ""
    # Keep one record per sample, preserving the original receipt on reuse.
    records_path = output_dir / f"generation_records{shard_suffix}.jsonl"
    saved_records = {}
    if records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                previous = json.loads(line)
                saved_records[previous["sample_id"]] = previous
    previous_prompt_file = output_dir / f'prompt_map{shard_suffix}.json'
    previous_prompts = json.loads(previous_prompt_file.read_text()) if previous_prompt_file.is_file() else {}
    prompt_map = {**previous_prompts, **prompt_map}
    for record in records:
        saved = record
        if record['status'] == 'skipped_existing':
            previous = saved_records.get(record['sample_id'], {})
            if previous.get('status') == 'generated':
                saved = previous
            else:
                saved = (record.get('prediction_binding') or {}).get('generation_record') or record
            key = record.get('prediction_path')
            if saved.get('prompt') is not None:
                prompt_map[key] = saved['prompt']
            elif key in previous_prompts:
                prompt_map[key] = previous_prompts[key]
            else:
                prompt_map.pop(key, None)
        saved_records[record['sample_id']] = saved
    (output_dir / f"generation_manifest{shard_suffix}.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary = records_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in saved_records.values():
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(records_path)
    (output_dir / f"prompt_map{shard_suffix}.json").write_text(
        json.dumps(prompt_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_payload,
        "records": records,
    }
