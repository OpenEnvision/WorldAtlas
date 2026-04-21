from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldarena.models.adapters.common import first_frame_to_image
from worldarena.models.config import ModelRuntimeConfig
from worldarena.models.registry import build_model_adapter
from worldarena.benchmark.schemas import BenchmarkSample


def _select_samples(
    manifest: list[BenchmarkSample],
    *,
    suites: set[str] | None,
    splits: set[str] | None,
    limit: int | None,
    supported_suites: set[str],
) -> list[BenchmarkSample]:
    selected: list[BenchmarkSample] = []
    for sample in manifest:
        if sample.suite not in supported_suites:
            continue
        if suites and sample.suite not in suites:
            continue
        if splits and sample.split not in splits:
            continue
        selected.append(sample)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _conditioning_image_for(
    sample: BenchmarkSample,
    *,
    reference_mode: str,
    cache_root: Path,
) -> Path:
    if reference_mode == "none":
        return cache_root / f"{sample.prediction_stem}.unused.png"
    if sample.conditioning_path:
        conditioning_path = Path(sample.conditioning_path).expanduser().resolve()
        if not conditioning_path.exists():
            raise FileNotFoundError(f"conditioning image not found: {conditioning_path}")
        return conditioning_path
    reference_path = Path(sample.reference_path)
    if reference_mode == "reference_image" or sample.modality == "image":
        return reference_path
    conditioning_path = cache_root / f"{sample.prediction_stem}.png"
    if conditioning_path.exists():
        return conditioning_path
    return first_frame_to_image(reference_path, conditioning_path)


def generate_predictions(
    *,
    model_config: ModelRuntimeConfig,
    manifest: list[BenchmarkSample],
    output_dir: Path,
    suites: set[str] | None = None,
    splits: set[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    adapter = build_model_adapter(model_config)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (output_dir / "_conditioning").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    selected_samples = _select_samples(
        manifest,
        suites=suites,
        splits=splits,
        limit=limit,
        supported_suites=set(model_config.supported_suites),
    )
    records: list[dict[str, Any]] = []
    prompt_map: dict[str, str] = {}

    for sample in selected_samples:
        prediction_path = output_dir / f"{sample.prediction_stem}{model_config.output_ext}"
        prompt = adapter.prompt_for(sample)
        conditioning_image = _conditioning_image_for(
            sample,
            reference_mode=model_config.reference_mode,
            cache_root=cache_root,
        )

        record = {
            "sample_id": sample.sample_id,
            "suite": sample.suite,
            "split": sample.split,
            "prediction_path": str(prediction_path),
            "conditioning_image": str(conditioning_image),
            "prompt": prompt,
            "status": "pending",
            "error": None,
        }
        try:
            if overwrite or not prediction_path.exists():
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
        prompt_map[str(prediction_path)] = prompt

    manifest_payload = {
        "model_name": model_config.name,
        "family": model_config.family,
        "output_dir": str(output_dir),
        "records": len(records),
        "generated": sum(1 for item in records if item["status"] == "generated"),
        "skipped_existing": sum(1 for item in records if item["status"] == "skipped_existing"),
        "failed": sum(1 for item in records if item["status"] == "failed"),
        "supported_suites": model_config.supported_suites,
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "generation_records.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "prompt_map.json").write_text(
        json.dumps(prompt_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_payload,
        "records": records,
    }
