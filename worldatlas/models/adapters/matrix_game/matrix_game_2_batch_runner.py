"""Load Matrix-Game 2 once, then generate every sample in the shard in-process."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from worldatlas.common.checkpoints import resolve_checkpoint_path
from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    begin_sample,
    load_batch_spec,
    load_json,
    log_pipeline,
    print_status,
)
from worldatlas.models.adapters.matrix_game.matrix_game_2_runner import (
    generate_matrix_game_2,
    load_matrix_game_2,
    prepare_matrix_game_2_repo,
    settings_from_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Matrix-Game-2 batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = prepare_matrix_game_2_repo(Path(args.repo_root).expanduser().resolve())
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("Matrix-Game-2 batch runner requires checkpoint_dir")
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    settings = settings_from_generation(generation)

    log_pipeline(
        "checkpoint_load",
        label="MatrixGame2",
        samples=len(rows),
        compile_vae=bool(settings.compile_vae),
        repo=str(repo_root),
    )
    runtime = load_matrix_game_2(repo_root, checkpoint_dir, settings)
    log_pipeline("checkpoint_ready", label="MatrixGame2", samples=len(rows))

    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "unknown")
        begin_sample(sample_id, index=row_index + 1, total=len(rows), label="MatrixGame2")
        try:
            action_spec = row.get("action_payload")
            if not isinstance(action_spec, dict):
                raise ValueError(f"{sample_id}: batch spec is missing action_payload")
            generate_matrix_game_2(
                runtime,
                action_spec=action_spec,
                image_path=Path(str(row["conditioning_image"])).expanduser().resolve(),
                output_path=Path(str(row["output_path"])).expanduser().resolve(),
            )
            print_status(
                sample_id,
                "generated",
                prediction_path=str(row["output_path"]),
                num_output_frames=action_spec.get("num_output_frames"),
            )
        except Exception as exc:
            failed += 1
            print_status(
                sample_id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
    if failed:
        raise RuntimeError(f"Matrix-Game 2 failed for {failed} sample(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
