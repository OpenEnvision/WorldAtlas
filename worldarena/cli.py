from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from worldarena.benchmark import (
    build_inventory,
    build_manifest,
    load_config,
    load_manifest,
    run_benchmark,
    write_inventory_artifacts,
    write_manifest_artifacts,
)
from worldarena.benchmark.config import apply_model_metric_overrides
from worldarena.benchmark_core.physics_manifest import build_worldarena_physics_manifest
from worldarena.common.cleanup import clean_project
from worldarena.generation import generate_predictions
from worldarena.models import load_model_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldarena")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_docs = subparsers.add_parser(
        "build-docs",
        help="Scan dataset roots, refresh artifacts, and regenerate MkDocs pages.",
    )
    build_docs.add_argument("--config", default="config/datasets.yaml")
    build_docs.add_argument("--max-workers", type=int, default=8)

    scan = subparsers.add_parser(
        "scan",
        help="Scan the configured datasets and print a compact JSON summary.",
    )
    scan.add_argument("--config", default="config/datasets.yaml")
    scan.add_argument("--max-workers", type=int, default=8)

    build_inventory_benchmark = subparsers.add_parser(
        "build-inventory",
        help="Scan raw WorldArena assets and write benchmark inventory artifacts.",
    )
    build_inventory_benchmark.add_argument("--config", default="config/benchmark.yaml")

    build_manifest_benchmark = subparsers.add_parser(
        "build-manifest",
        help="Build the WorldArena benchmark manifest from the scanned raw assets.",
    )
    build_manifest_benchmark.add_argument("--config", default="config/benchmark.yaml")

    build_physics_manifest = subparsers.add_parser(
        "build-physics-manifest",
        help="Build a simulator-backed physics manifest from physics-pipeline assets.",
    )
    build_physics_manifest.add_argument("--physics-pipeline-root", required=True)
    build_physics_manifest.add_argument(
        "--output-root",
        default="artifacts/benchmark/physics/manifest",
    )
    build_physics_manifest.add_argument("--sample-ids", default=None)

    generate = subparsers.add_parser(
        "generate",
        help="Generate benchmark predictions with a configured model adapter.",
    )
    generate.add_argument("--config", default="config/benchmark.yaml")
    generate.add_argument("--model-config", required=True)
    generate.add_argument("--manifest", default=None)
    generate.add_argument("--output-dir", default=None)
    generate.add_argument("--suites", default=None)
    generate.add_argument("--splits", default=None)
    generate.add_argument("--limit", type=int, default=None)
    generate.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a prediction folder against the benchmark manifest.",
    )
    evaluate.add_argument("--config", default="config/benchmark.yaml")
    evaluate.add_argument("--model-config", default=None)
    evaluate.add_argument("--manifest", default=None)
    evaluate.add_argument("--predictions-root", required=True)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--model-name", default=None)
    evaluate.add_argument("--suites", default=None)
    evaluate.add_argument("--splits", default=None)
    evaluate.add_argument("--limit", type=int, default=None)

    run_model = subparsers.add_parser(
        "run-model",
        help="Generate predictions with a configured model and immediately evaluate them.",
    )
    run_model.add_argument("--config", default="config/benchmark.yaml")
    run_model.add_argument("--model-config", required=True)
    run_model.add_argument("--manifest", default=None)
    run_model.add_argument("--predictions-root", default=None)
    run_model.add_argument("--output-dir", default=None)
    run_model.add_argument("--suites", default=None)
    run_model.add_argument("--splits", default=None)
    run_model.add_argument("--limit", type=int, default=None)
    run_model.add_argument("--overwrite", action="store_true")

    clean = subparsers.add_parser(
        "clean",
        help="Remove transient caches and build outputs to keep the repo tidy.",
    )
    clean.add_argument("--project-root", default=".")
    clean.add_argument(
        "--generated",
        action="store_true",
        help="Also remove generated docs pages and dataset artifacts.",
    )

    return parser


def _ensure_manifest(config, manifest_arg: str | None) -> Path:
    manifest_path = (
        Path(manifest_arg).resolve()
        if manifest_arg
        else config.paths.manifest_dir / "worldarena_manifest.jsonl"
    )
    if manifest_path.exists():
        return manifest_path

    inventory = build_inventory(config)
    write_inventory_artifacts(inventory, config.paths.inventory_dir)
    manifest = build_manifest(inventory["assets"], config)
    return write_manifest_artifacts(
        manifest,
        output_dir=config.paths.manifest_dir,
        snapshot_name=config.snapshot_name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "clean":
        removed = clean_project(
            project_root=Path(args.project_root).resolve(),
            include_generated=args.generated,
        )
        print(json.dumps({"removed": removed}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "build-inventory":
        config = load_config(Path(args.config))
        result = build_inventory(config)
        write_inventory_artifacts(result, config.paths.inventory_dir)
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
        return 0

    if args.command == "build-manifest":
        config = load_config(Path(args.config))
        inventory = build_inventory(config)
        write_inventory_artifacts(inventory, config.paths.inventory_dir)
        manifest = build_manifest(inventory["assets"], config)
        manifest_path = write_manifest_artifacts(
            manifest,
            output_dir=config.paths.manifest_dir,
            snapshot_name=config.snapshot_name,
        )
        print(
            json.dumps(
                {
                    "snapshot_name": config.snapshot_name,
                    "manifest_path": str(manifest_path),
                    "items": len(manifest),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "build-physics-manifest":
        sample_ids = (
            [item.strip() for item in args.sample_ids.split(",") if item.strip()]
            if args.sample_ids
            else None
        )
        result = build_worldarena_physics_manifest(
            physics_pipeline_root=args.physics_pipeline_root,
            output_root=args.output_root,
            sample_ids=sample_ids,
        )
        print(
            json.dumps(
                {
                    "manifest_path": result["manifest_path"],
                    "items": result["items"],
                    "physics_pipeline_root": result["physics_pipeline_root"],
                    "cases_root": result["cases_root"],
                    "conditioning_root": result["conditioning_root"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "generate":
        config = load_config(Path(args.config))
        model_config = load_model_config(Path(args.model_config))
        manifest = load_manifest(_ensure_manifest(config, args.manifest))
        suites = {item.strip() for item in args.suites.split(",")} if args.suites else None
        splits = {item.strip() for item in args.splits.split(",")} if args.splits else None
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (config.paths.reports_dir.parent.parent / "predictions" / model_config.name).resolve()
        )
        result = generate_predictions(
            model_config=model_config,
            manifest=manifest,
            output_dir=output_dir,
            suites=suites,
            splits=splits,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        print(json.dumps(result["manifest"], indent=2, ensure_ascii=False))
        return 0

    if args.command == "evaluate":
        config = load_config(Path(args.config))
        model_config = load_model_config(Path(args.model_config)) if getattr(args, "model_config", None) else None
        config = apply_model_metric_overrides(config, model_config)
        manifest = load_manifest(_ensure_manifest(config, args.manifest))
        predictions_root = Path(args.predictions_root).resolve()
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (config.paths.reports_dir / (args.model_name or (model_config.name if model_config else predictions_root.name))).resolve()
        )
        suites = {item.strip() for item in args.suites.split(",")} if args.suites else None
        splits = {item.strip() for item in args.splits.split(",")} if args.splits else None
        result = run_benchmark(
            config=config,
            manifest=manifest,
            predictions_root=predictions_root,
            output_dir=output_dir,
            model_name=args.model_name or (model_config.name if model_config else predictions_root.name),
            suites=suites,
            splits=splits,
            limit=args.limit,
        )
        print(json.dumps(result["run_manifest"], indent=2, ensure_ascii=False))
        return 0

    if args.command == "run-model":
        config = load_config(Path(args.config))
        model_config = load_model_config(Path(args.model_config))
        manifest = load_manifest(_ensure_manifest(config, args.manifest))
        suites = {item.strip() for item in args.suites.split(",")} if args.suites else None
        splits = {item.strip() for item in args.splits.split(",")} if args.splits else None
        predictions_root = (
            Path(args.predictions_root).resolve()
            if args.predictions_root
            else (config.paths.reports_dir.parent.parent / "predictions" / model_config.name).resolve()
        )
        generation_result = generate_predictions(
            model_config=model_config,
            manifest=manifest,
            output_dir=predictions_root,
            suites=suites,
            splits=splits,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        eval_config = apply_model_metric_overrides(config, model_config)
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (config.paths.reports_dir / model_config.name).resolve()
        )
        evaluation_result = run_benchmark(
            config=eval_config,
            manifest=manifest,
            predictions_root=predictions_root,
            output_dir=output_dir,
            model_name=model_config.name,
            suites=suites,
            splits=splits,
            limit=args.limit,
        )
        print(
            json.dumps(
                {
                    "generation": generation_result["manifest"],
                    "evaluation": evaluation_result["run_manifest"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    from worldarena.docs.builder import build_docs_site

    result = build_docs_site(
        config_path=Path(args.config),
        max_workers=args.max_workers,
        write_docs=args.command == "build-docs",
    )

    if args.command == "scan":
        summary = {
            "generated_at": result["generated_at"],
            "records": int(result["frame"].shape[0]),
            "images": int(result["images"].shape[0]),
            "videos": int(result["videos"].shape[0]),
            "collections": result["collection_summary"].to_dict(orient="records"),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
