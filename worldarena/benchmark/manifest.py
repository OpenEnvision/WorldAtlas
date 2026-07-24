"""Build benchmark manifests from discovered dataset assets."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from worldarena.benchmark.annotations import resolve_prompt_contract
from worldarena.common.serialization import ensure_dir, write_json
from worldarena.benchmark.config import BenchmarkConfig
from worldarena.benchmark.schemas import BenchmarkSample, DiscoveredAsset
from worldarena.benchmark.taxonomy import (
    artifact_type_for_modality,
    benchmark_split,
    build_sample_id,
    control_signals_for_sample,
    prediction_modality_for_task,
    suite_for_asset,
    task_family_for_suite,
)


def _conditioning_protocol(
    asset: DiscoveredAsset,
    *,
    config: BenchmarkConfig,
) -> tuple[str, int | None, float | None, float | None, str | None]:
    if asset.modality == "image":
        return "reference_image", 1, None, 0.0, None
    if asset.asset_level == "video_static":
        return "first_frame", 1, None, 0.0, "camera_conditioned_scene_exploration"
    if asset.asset_level == "video_dynamic":
        prefix_ratio = min(max(config.dynamic_prefix_ratio, 0.0), 0.95)
        return (
            "reference_video",
            max(int(config.dynamic_prefix_min_frames), 1),
            prefix_ratio,
            prefix_ratio,
            "prefix_rollout_world_continuation",
        )
    return "first_frame", 1, None, 0.0, None


def build_manifest(assets: list[DiscoveredAsset], config: BenchmarkConfig) -> list[BenchmarkSample]:
    counters: dict[tuple[str, str, str], int] = defaultdict(int)
    manifest: list[BenchmarkSample] = []

    ordered_assets = sorted(
        assets,
        key=lambda item: (
            item.asset_level,
            item.style or "",
            item.fine_class or "",
            item.relative_path,
        ),
    )
    for asset in ordered_assets:
        counter_key = (asset.asset_level, asset.style or "unknown", asset.fine_class or "unknown")
        counters[counter_key] += 1
        sample_id = build_sample_id(
            asset.asset_level,
            asset.style,
            asset.fine_class,
            counters[counter_key],
        )
        split = benchmark_split()
        suite = suite_for_asset(
            asset_level=asset.asset_level if asset.is_formal else asset.asset_level,
            is_formal=asset.is_formal,
            has_annotation=asset.has_annotation,
            has_instruction=asset.has_instruction,
            has_pose=asset.has_pose,
            has_mask=asset.has_mask,
        )
        prompt_contract = resolve_prompt_contract(asset.annotation_path)
        (
            conditioning_strategy,
            conditioning_frame_count,
            conditioning_end_ratio,
            evaluation_start_ratio,
            track,
        ) = _conditioning_protocol(asset, config=config)
        task_family = task_family_for_suite(suite, track=track)
        prediction_modality = prediction_modality_for_task(
            suite=suite,
            source_modality=asset.modality,
            task_family=task_family,
        )
        manifest.append(
            BenchmarkSample(
                sample_id=sample_id,
                suite=suite,
                split=split,
                asset_level=asset.asset_level,
                modality=asset.modality,
                is_formal=asset.is_formal,
                eligible_for_official=asset.eligible_for_official,
                category_path=asset.category_path,
                path=asset.path,
                relative_path=asset.relative_path,
                reference_path=asset.path,
                prediction_stem=sample_id,
                conditioning_strategy=conditioning_strategy,
                prompt_current=str(prompt_contract.get("prompt_current", "")),
                prompt_target=str(prompt_contract.get("prompt_target", "")),
                style=asset.style,
                environment=asset.environment,
                scene=asset.scene,
                motion_category=asset.motion_category,
                source_name=asset.source_name,
                source_type=asset.source_type,
                source_group_id=asset.source_group_id,
                license_bucket=asset.license_bucket,
                prompt_sequence=list(prompt_contract.get("prompt_sequence", [])),
                camera_path=list(prompt_contract.get("camera_path", [])),
                generation_mode=prompt_contract.get("mode"),
                benchmark_family="worldarena",
                task_family=task_family,
                artifact_type=artifact_type_for_modality(prediction_modality),
                control_signals=control_signals_for_sample(
                    modality=asset.modality,
                    conditioning_strategy=conditioning_strategy,
                    has_pose=asset.has_pose,
                ),
                annotation_path=asset.annotation_path,
                mask_path=asset.mask_path,
                has_annotation=asset.has_annotation,
                has_instruction=asset.has_instruction,
                has_pose=asset.has_pose,
                has_mask=asset.has_mask,
                mask_count=asset.mask_count,
                width=asset.width,
                height=asset.height,
                fps=asset.fps,
                duration_seconds=asset.duration_seconds,
                duration_bucket=asset.duration_bucket,
                conditioning_frame_count=conditioning_frame_count,
                conditioning_end_ratio=conditioning_end_ratio,
                evaluation_start_ratio=evaluation_start_ratio,
                track=track,
            )
        )

    return manifest


def load_manifest(path: Path) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        samples.append(BenchmarkSample(**payload))
    return samples


def write_manifest_artifacts(manifest: list[BenchmarkSample], output_dir: Path, snapshot_name: str) -> Path:
    import pandas as pd

    ensure_dir(output_dir)
    manifest_path = output_dir / "worldarena_manifest.jsonl"
    lines = [json.dumps(sample.to_dict(), ensure_ascii=False) for sample in manifest]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    frame = pd.DataFrame.from_records([sample.to_dict() for sample in manifest])
    if not frame.empty:
        summary = (
            frame.groupby(["suite", "asset_level"], dropna=False)
            .size()
            .reset_index(name="items")
            .sort_values(["suite", "asset_level"])
        )
    else:
        summary = pd.DataFrame(columns=["suite", "asset_level", "items"])
    summary.to_csv(output_dir / "worldarena_suite_summary.csv", index=False)
    write_json(
        output_dir / "worldarena_manifest_summary.json",
        {
            "snapshot_name": snapshot_name,
            "items": int(frame.shape[0]),
            "suites": summary.to_dict(orient="records"),
        },
    )
    return manifest_path
