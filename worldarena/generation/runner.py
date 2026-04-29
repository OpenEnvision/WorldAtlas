from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldarena.common.video_io import extract_video_window, freeze_frame_to_video
from worldarena.common.checkpoints import rehome_legacy_workspace_path
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.common import first_frame_to_image
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter
from worldarena.benchmark.schemas import BenchmarkSample


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
    try:
        batch_results = adapter.generate_batch(pending_requests)
    except Exception as exc:
        error_text = str(exc)
        for request in pending_requests:
            record = records_by_sample_id[request.sample.sample_id]
            record["status"] = "failed"
            record["error"] = error_text
        return

    for request in pending_requests:
        record = records_by_sample_id[request.sample.sample_id]
        payload = dict(batch_results.get(request.sample.sample_id, {}))
        if not payload:
            if request.output_path.exists():
                payload = {
                    "status": "generated",
                    "prediction_path": str(request.output_path),
                    "prompt": request.prompt,
                }
            else:
                payload = {
                    "status": "failed",
                    "error": "batch generation returned no result for this sample",
                }

        status = str(payload.pop("status", "generated" if not payload.get("error") else "failed"))
        error = payload.pop("error", None)
        if "prediction_path" not in payload and request.output_path.exists():
            payload["prediction_path"] = str(request.output_path)
        payload.setdefault("prompt", request.prompt)
        record.update(payload)
        record["status"] = status
        record["error"] = error


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

    for sample in selected_samples:
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
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
            continue
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
                    payload = adapter.generate(
                        sample=sample,
                        conditioning_image=conditioning_image,
                        output_path=prediction_path,
                        prompt=prompt,
                    )
                    record.update(payload)
                    record["status"] = "generated"
            else:
                record["status"] = "skipped_existing"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
        records.append(record)
        records_by_sample_id[sample.sample_id] = record
        prompt_map[str(prediction_path)] = prompt

    if use_batch_generation and pending_requests:
        _apply_batch_generation_results(
            adapter=adapter,
            pending_requests=pending_requests,
            records_by_sample_id=records_by_sample_id,
        )

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
    (output_dir / f"generation_manifest{shard_suffix}.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / f"generation_records{shard_suffix}.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / f"prompt_map{shard_suffix}.json").write_text(
        json.dumps(prompt_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_payload,
        "records": records,
    }
