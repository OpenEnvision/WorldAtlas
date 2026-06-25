from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    latest_mp4,
    load_batch_spec,
    load_json,
    print_status,
)

INFERENCE_MODULE = "cosmos_framework.scripts.inference"
DEFAULT_CHECKPOINT = "Cosmos3-Nano"
VISION_CONDITIONED_MODES = {"image2video", "video2video", "image2image", "reasoner"}

# Per-sample fields written into the cosmos-framework sample JSONL when present
# in the generation config. Keys must match OmniSampleOverrides fields exactly,
# because the framework validates samples with ``extra="forbid"``.
SAMPLE_VALUE_FIELDS: dict[str, type] = {
    "resolution": str,
    "aspect_ratio": str,
    "num_frames": int,
    "fps": int,
    "negative_prompt": str,
    "num_steps": int,
    "guidance": float,
}

# Inference CLI value flags forwarded from the generation config. The parallelism
# axes let large checkpoints shard (FSDP) across GPUs when launched with torchrun.
CLI_VALUE_FLAGS: dict[str, tuple[str, type]] = {
    "parallelism_preset": ("--parallelism-preset", str),
    "sampler": ("--sampler", str),
    "dp_shard_size": ("--dp-shard-size", int),
    "dp_replicate_size": ("--dp-replicate-size", int),
    "cp_size": ("--cp-size", int),
    "cfgp_size": ("--cfgp-size", int),
    "max_num_seqs": ("--max-num-seqs", int),
    "max_model_len": ("--max-model-len", int),
    "device_memory_utilization": ("--device-memory-utilization", float),
}

# Boolean inference switches forwarded as bare flags when truthy.
CLI_FLAG_SWITCHES: dict[str, str] = {
    "no_guardrails": "--no-guardrails",
    "offload_guardrail_models": "--offload-guardrail-models",
    "benchmark": "--benchmark",
}

DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Cosmos Framework batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def resolve_checkpoint(generation: dict, checkpoint_dir: str | None) -> str:
    raw = (
        generation.get("checkpoint_path")
        or checkpoint_dir
        or generation.get("checkpoint_name")
        or DEFAULT_CHECKPOINT
    )
    return str(raw)


def build_sample_rows(rows: list[dict], generation: dict) -> tuple[str, list[dict]]:
    model_mode = str(generation.get("model_mode", "image2video"))
    sample_extra = generation.get("sample_extra")
    samples: list[dict] = []
    for row in rows:
        sample: dict = {
            "name": str(row["prediction_stem"]),
            "model_mode": model_mode,
            "prompt": str(row["prompt"]),
        }
        for field, cast in SAMPLE_VALUE_FIELDS.items():
            value = generation.get(field)
            if value is not None:
                sample[field] = cast(value)
        if model_mode in VISION_CONDITIONED_MODES:
            sample["vision_path"] = str(row["conditioning_image"])
        if isinstance(sample_extra, dict):
            sample.update(sample_extra)
        samples.append(sample)
    return model_mode, samples


def build_command(
    generation: dict,
    checkpoint_path: str,
    cosmos_input: Path,
    cosmos_output: Path,
) -> list[str]:
    num_gpus = int(generation.get("num_gpus") or generation.get("nproc_per_node") or 1)
    if num_gpus > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            f"--nproc-per-node={num_gpus}",
            "-m",
            INFERENCE_MODULE,
        ]
    else:
        command = [sys.executable, "-m", INFERENCE_MODULE]

    command += [
        "-i",
        str(cosmos_input),
        "-o",
        str(cosmos_output),
        "--checkpoint-path",
        checkpoint_path,
        "--seed",
        str(int(generation.get("seed", 0))),
    ]

    for key, (flag, cast) in CLI_VALUE_FLAGS.items():
        value = generation.get(key)
        if value is not None:
            command += [flag, str(cast(value))]

    for key, flag in CLI_FLAG_SWITCHES.items():
        if generation.get(key):
            command.append(flag)

    extra_cli_args = generation.get("extra_cli_args")
    if isinstance(extra_cli_args, (list, tuple)):
        command += [str(arg) for arg in extra_cli_args]

    return command


def build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in DISTRIBUTED_ENV_KEYS:
        env.pop(key, None)
    env["MASTER_ADDR"] = "127.0.0.1"
    return env


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # cosmos-framework fetches VLM/tokenizer weights by shelling out to ``uvx hf
    # download``; make sure the interpreter's own bin dir (which ships uvx and
    # torchrun) is reachable regardless of how WorldArena was launched.
    bin_dir = os.path.dirname(sys.executable)
    if bin_dir and bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    # WorldArena pins HF_HUB_OFFLINE=1 globally, but cosmos-framework still
    # shells out to ``uvx hf download`` for Wan VAE / tokenizer files on cache miss.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    spec_path = Path(args.batch_spec_path).expanduser().resolve()
    rows = load_batch_spec(spec_path)
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    cosmos_input = spec_path.with_suffix(".cosmos.jsonl")
    cosmos_output = spec_path.parent / f"{spec_path.stem}_cosmos_outputs"

    _, samples = build_sample_rows(rows, generation)
    with cosmos_input.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    checkpoint_path = resolve_checkpoint(generation, args.checkpoint_dir)
    command = build_command(generation, checkpoint_path, cosmos_input, cosmos_output)

    if os.environ.get("WORLDARENA_COSMOS_DRY_RUN"):
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "command": command,
                    "checkpoint_path": checkpoint_path,
                    "cosmos_input": str(cosmos_input),
                    "cosmos_output": str(cosmos_output),
                    "num_samples": len(samples),
                    "sample_preview": samples[0] if samples else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    subprocess.run(command, check=True, cwd=str(repo_root), env=build_child_env())

    for row in rows:
        sample_id = str(row["sample_id"])
        stem_dir = cosmos_output / str(row["prediction_stem"])
        try:
            source = stem_dir / "vision.mp4"
            if not source.is_file():
                source = latest_mp4(stem_dir)
            copy_output(source, Path(str(row["output_path"])))
            print_status(sample_id, "generated", output_path=str(row["output_path"]), source=str(source))
        except Exception as exc:
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
