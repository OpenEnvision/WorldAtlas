"""Bind newly generated videos to the exact image inputs recorded by generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from worldatlas.benchmark.integrity import digest_json
from worldatlas.benchmark.sample_identity import (
    SOURCE_FILENAME_POLICY,
    file_sha256,
    resolve_bound_prediction,
    validate_source_identity,
)
from worldatlas.benchmark.schemas import BenchmarkSample


def write_prediction_manifest(
    samples: list[BenchmarkSample],
    records: list[dict],
    output_dir: Path,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    """Publish successful samples; never promote an unverified existing video."""
    root = output_dir.resolve()
    by_id = {sample.sample_id: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("Duplicate sample IDs in generation manifest")
    bound = []
    seen = set()
    for record in records:
        if record.get("status") not in {"generated", "skipped_existing"}:
            continue
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate generation record: {sample_id}")
        seen.add(sample_id)
        sample = by_id[sample_id]
        if sample.sample_id_policy == SOURCE_FILENAME_POLICY:
            if record["status"] == "skipped_existing":
                resolve_bound_prediction(sample, root)
            else:
                source_hash = validate_source_identity(sample)
                if (record.get("sample_id_policy") != SOURCE_FILENAME_POLICY
                        or record.get("source_sha256") != source_hash
                        or Path(record.get("source_path", "")).resolve() != Path(sample.path).resolve()):
                    raise ValueError(f"Generation record does not identify the source image: {sample_id}")
                prediction = Path(record["prediction_path"]).resolve()
                if not prediction.is_relative_to(root) or not prediction.is_file() or not prediction.stat().st_size:
                    raise ValueError(f"Generated video is missing or outside the output directory: {sample_id}")
                if file_sha256(prediction) != record.get("prediction_sha256"):
                    raise ValueError(f"Generated video changed: {sample_id}")
                receipt = deepcopy(record)
                sample = replace(sample, source_sha256=source_hash, prediction_binding={
                    "sample_id": sample_id,
                    "source_sha256": source_hash,
                    "predictions_root": str(root),
                    "path": str(prediction.relative_to(root)),
                    "prediction_sha256": record["prediction_sha256"],
                    "evidence": "worldatlas_generation_receipt_v1",
                    "generation_record": receipt,
                    "generation_record_sha256": digest_json(receipt),
                })
                resolve_bound_prediction(sample, root)
        bound.append(sample)
    suffix = f".shard{shard_index}" if num_shards > 1 else ""
    path = root / f"evaluation_manifest{suffix}.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n" for sample in bound), encoding="utf-8")
    temporary.replace(path)
    return path
