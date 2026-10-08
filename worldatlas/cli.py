"""WorldAtlas command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from worldatlas.benchmark import (
    build_inventory,
    build_manifest,
    load_config,
    load_manifest,
    run_benchmark,
    write_inventory_artifacts,
    write_manifest_artifacts,
)
from worldatlas.benchmark.config import apply_model_metric_overrides
from worldatlas.benchmark.memory_manifest import (
    DEFAULT_MEMORY_LOOPS,
    build_memory_manifest,
    memory_manifest_summary,
)
from worldatlas.benchmark.memory_pool import build_memory_pool_manifest
from worldatlas.common.cleanup import clean_project
from worldatlas.generation import generate_predictions
from worldatlas.models import load_model_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldatlas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_inventory_benchmark = subparsers.add_parser(
        "build-inventory",
        help="Scan raw WorldAtlas assets and write benchmark inventory artifacts.",
    )
    build_inventory_benchmark.add_argument("--config", default="config/benchmark.yaml")

    build_reference_corpus = subparsers.add_parser(
        "build-reference-corpus",
        help=(
            "Index the collected video corpus that the distribution metrics "
            "(JEDi, FVMD) score against."
        ),
    )
    build_reference_corpus.add_argument("--config", default="config/benchmark.yaml")
    build_reference_corpus.add_argument(
        "--output",
        default=None,
        help="Where to write the index (defaults to benchmark.reference_corpus.index_path).",
    )

    build_manifest_benchmark = subparsers.add_parser(
        "build-manifest",
        help="Build the WorldAtlas benchmark manifest from the scanned raw assets.",
    )
    build_manifest_benchmark.add_argument("--config", default="config/benchmark.yaml")
    build_manifest_benchmark.add_argument(
        "--memory-loops",
        default="default",
        help=(
            "Closed-loop itineraries to pair with static images for the memory track: "
            "'default', 'none', or a comma-separated list of loop ids."
        ),
    )
    build_manifest_benchmark.add_argument(
        "--memory-limit-per-loop",
        type=int,
        default=None,
        help=(
            "Only for --memory-source image_static: cap how many source images each "
            "memory loop expands over."
        ),
    )
    build_manifest_benchmark.add_argument(
        "--memory-source",
        choices=("pool", "image_static"),
        default="pool",
        help=(
            "Where memory-track stills come from. 'pool' reads the curated set under "
            "<image_root>/memory and gives each image one loop. 'image_static' expands "
            "every requested loop over the static image suite instead."
        ),
    )
    build_manifest_benchmark.add_argument(
        "--memory-pool-limit",
        type=int,
        default=None,
        help="Only for --memory-source pool: cap how many pool images are used.",
    )
    build_manifest_benchmark.add_argument(
        "--memory-duration-seconds", type=int, choices=(30, 60), default=None,
        help="Read memory/30s or memory/60s; omitted preserves the legacy pool.",
    )

    generate = subparsers.add_parser(
        "generate",
        help="Generate benchmark predictions with a configured model adapter.",
    )
    generate.add_argument("--config", default="config/benchmark.yaml")
    generate.add_argument("--model-config", required=True)
    generate.add_argument("--manifest", default=None)
    generate.add_argument("--output-dir", default=None)
    generate.add_argument("--suites", default=None)
    generate.add_argument("--limit", type=int, default=None)
    generate.add_argument("--num-shards", type=int, default=1)
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop generation on the first failed sample instead of continuing.",
    )

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
    run_model.add_argument("--limit", type=int, default=None)
    run_model.add_argument("--num-shards", type=int, default=1)
    run_model.add_argument("--shard-index", type=int, default=0)
    run_model.add_argument("--overwrite", action="store_true")
    run_model.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop generation on the first failed sample instead of continuing.",
    )

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



def _resolve_memory_loops(value: str | None) -> tuple[str, ...]:
    """Interpret the ``--memory-loops`` selector into concrete loop identifiers."""
    token = str(value or "default").strip().lower()
    if token in {"none", "off", ""}:
        return ()
    if token == "default":
        return DEFAULT_MEMORY_LOOPS
    return tuple(item.strip() for item in token.split(",") if item.strip())


def _ensure_manifest(config, manifest_arg: str | None) -> Path:
    manifest_path = (
        Path(manifest_arg).resolve()
        if manifest_arg
        else config.paths.manifest_dir / "worldatlas_manifest.jsonl"
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

    if args.command == "build-reference-corpus":
        from worldatlas.benchmark.reference_corpus import (
            build_reference_index,
            write_reference_index,
        )

        config = load_config(Path(args.config))
        corpus = config.reference_corpus
        root = corpus.root or config.paths.video_root
        index = build_reference_index(
            Path(root),
            max_videos_per_group=corpus.max_videos_per_group,
            seed=corpus.seed,
        )
        destination = Path(args.output) if args.output else corpus.index_path
        if destination is None:
            destination = config.paths.inventory_dir / "reference_corpus_index.json"
        write_reference_index(index, Path(destination))
        print(
            json.dumps(
                {**index.describe(), "index_path": str(destination)},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "build-manifest":
        config = load_config(Path(args.config))
        inventory = build_inventory(config)
        write_inventory_artifacts(inventory, config.paths.inventory_dir)
        manifest = build_manifest(inventory["assets"], config)
        memory_loops = _resolve_memory_loops(args.memory_loops)
        if not memory_loops:
            memory_samples: list = []
        elif args.memory_source == "pool":
            memory_samples = build_memory_pool_manifest(
                config.paths.image_root,
                loops=memory_loops,
                limit=args.memory_pool_limit,
                duration_seconds=args.memory_duration_seconds,
            )
        else:
            memory_samples = build_memory_manifest(
                manifest,
                loops=memory_loops,
                limit_per_loop=args.memory_limit_per_loop,
            )
        manifest.extend(memory_samples)
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
                    "memory_source": args.memory_source,
                    "memory": memory_manifest_summary(memory_samples),
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
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (config.paths.predictions_dir / model_config.name).resolve()
        )
        result = generate_predictions(
            model_config=model_config,
            manifest=manifest,
            output_dir=output_dir,
            suites=suites,
            limit=args.limit,
            overwrite=args.overwrite,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            fail_fast=args.fail_fast,
        )
        print(json.dumps(result["manifest"], indent=2, ensure_ascii=False))
        # Continuing after a failed sample must not make the shard successful.
        return 1 if result["manifest"]["failed"] else 0

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
        result = run_benchmark(
            config=config,
            manifest=manifest,
            predictions_root=predictions_root,
            output_dir=output_dir,
            model_name=args.model_name or (model_config.name if model_config else predictions_root.name),
            suites=suites,
            limit=args.limit,
        )
        print(json.dumps(result["run_manifest"], indent=2, ensure_ascii=False, default=str))
        return 0

    if args.command == "run-model":
        config = load_config(Path(args.config))
        model_config = load_model_config(Path(args.model_config))
        manifest = load_manifest(_ensure_manifest(config, args.manifest))
        suites = {item.strip() for item in args.suites.split(",")} if args.suites else None
        predictions_root = (
            Path(args.predictions_root).resolve()
            if args.predictions_root
            else (config.paths.predictions_dir / model_config.name).resolve()
        )
        generation_result = generate_predictions(
            model_config=model_config,
            manifest=manifest,
            output_dir=predictions_root,
            suites=suites,
            limit=args.limit,
            overwrite=args.overwrite,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            fail_fast=args.fail_fast,
        )
        if generation_result["manifest"]["failed"]:
            print(json.dumps({"generation": generation_result["manifest"],
                              "evaluation": "not run because generation failed"}, indent=2))
            return 1
        evaluation_manifest = load_manifest(Path(generation_result["manifest"]["evaluation_manifest"]))
        if not evaluation_manifest:
            raise ValueError("No samples were generated for the selected suites/shard")
        eval_config = apply_model_metric_overrides(config, model_config)
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (config.paths.reports_dir / model_config.name).resolve()
        )
        evaluation_result = run_benchmark(
            config=eval_config,
            manifest=evaluation_manifest,
            predictions_root=predictions_root,
            output_dir=output_dir,
            model_name=model_config.name,
            suites=None,
            limit=None,
        )
        print(
            json.dumps(
                {
                    "generation": generation_result["manifest"],
                    "evaluation": evaluation_result["run_manifest"],
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
