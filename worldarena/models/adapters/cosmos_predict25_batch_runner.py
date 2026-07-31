"""Batch subprocess runner for Cosmos-Predict2.5 inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)

INFERENCE_MODULE = "worldarena.models.adapters.cosmos_predict25_inference_entry"
DEFAULT_MODEL = "14B/post-trained"

# Per-sample fields written into the Predict2.5 input JSONL when present in the
# generation config. Keys must match InferenceArguments fields in cosmos_predict2.
SAMPLE_VALUE_FIELDS: dict[str, type] = {
    "resolution": str,
    "num_output_frames": int,
    "num_steps": int,
    "guidance": int,
    "seed": int,
    "negative_prompt": str,
    "enable_autoregressive": bool,
    "chunk_size": int,
    "chunk_overlap": int,
}

# Setup CLI flags forwarded from the generation config (except model, handled below).
SETUP_VALUE_FLAGS: dict[str, tuple[str, type]] = {
    "experiment": ("--experiment", str),
    "config_file": ("--config-file", str),
    "context_parallel_size": ("--context-parallel-size", int),
}

SETUP_FLAG_SWITCHES: dict[str, str] = {
    "disable_guardrails": "--disable-guardrails",
    "offload_diffusion_model": "--offload-diffusion-model",
    "offload_tokenizer": "--offload-tokenizer",
    "offload_text_encoder": "--offload-text-encoder",
    "offload_guardrail_models": "--offload-guardrail-models",
    "keep_going": "--keep-going",
    "profile": "--profile",
    "skip_existing_output": "--skip-existing-output",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _stage_rgb_conditioning_image(source: Path, staging_dir: Path, prediction_stem: str) -> Path:
    suffix = source.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS or suffix in {".jpg", ".jpeg"}:
        return source

    from PIL import Image

    destination = staging_dir / f"{prediction_stem}.rgb.png"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination.resolve()

    with Image.open(source) as image:
        has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
        if image.mode == "RGB" and not has_alpha:
            return source
        if has_alpha:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            rgb = background.convert("RGB")
        else:
            rgb = image.convert("RGB")
        staging_dir.mkdir(parents=True, exist_ok=True)
        rgb.save(destination)
        return destination.resolve()


def _resolve_checkpoint_root(generation: dict, checkpoint_dir: Path | None) -> Path | None:
    raw = generation.get("checkpoint_root") or generation.get("local_aux_root")
    if raw:
        root = Path(str(raw)).expanduser().resolve()
        if root.is_dir():
            return root
    env_root = os.environ.get("WORLDARENA_CHECKPOINT_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if root.is_dir():
            return root
    if checkpoint_dir is not None:
        parent = Path(checkpoint_dir).expanduser().resolve().parent
        if parent.is_dir():
            return parent
    return None


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate.resolve()
    return None


def _resolve_local_aux_paths(checkpoint_root: Path | None, generation: dict) -> dict[str, Path]:
    local_aux = generation.get("local_aux")
    if not isinstance(local_aux, dict):
        local_aux = {}

    def pick(key: str, candidates: list[Path]) -> Path | None:
        explicit = local_aux.get(key) or generation.get(key)
        if explicit:
            path = Path(str(explicit)).expanduser().resolve()
            if path.exists():
                return path
        return _first_existing_path(candidates)

    root_candidates: list[Path] = []
    if checkpoint_root is not None:
        root_candidates = [
            checkpoint_root / "Cosmos-Reason1-7B",
            checkpoint_root / "nvidia/Cosmos-Reason1-7B",
        ]
    reason1 = pick("reason1_dir", root_candidates)

    tokenizer_candidates: list[Path] = []
    if checkpoint_root is not None:
        tokenizer_candidates = [
            checkpoint_root / "Cosmos-Predict2.5-2B/tokenizer.pth",
            checkpoint_root / "Cosmos-Predict2.5-2B/Wan2.1_VAE.pth",
            checkpoint_root / "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
            checkpoint_root / "Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
        ]
    tokenizer = pick("tokenizer_vae", tokenizer_candidates)
    resolved: dict[str, Path] = {}
    if reason1 is not None:
        resolved["reason1_dir"] = reason1
    if tokenizer is not None:
        resolved["tokenizer_vae"] = tokenizer
    return resolved



def _build_local_checkpoint_aliases(local_aux: dict[str, Path]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    reason1 = local_aux.get("reason1_dir")
    if reason1 is not None:
        reason1_s3 = (
            "s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model"
        )
        qwen_s3 = "s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/Qwen/Qwen2.5-VL-7B-Instruct"
        aliases[reason1_s3] = str(reason1)
        aliases[qwen_s3] = str(reason1)
    tokenizer = local_aux.get("tokenizer_vae")
    if tokenizer is not None:
        aliases["s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/Wan2.1_VAE.pth"] = str(tokenizer)
    return aliases

def _build_local_experiment_opts(local_aux: dict[str, Path]) -> list[str]:
    opts: list[str] = []
    reason1 = local_aux.get("reason1_dir")
    if reason1 is not None:
        opts.append(f"model.config.text_encoder_config.ckpt_path={reason1}")
    tokenizer = local_aux.get("tokenizer_vae")
    if tokenizer is not None:
        opts.append(f"+model.config.tokenizer.vae_pth={tokenizer}")
    return opts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Arena Cosmos-Predict2.5 batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _resolve_checkpoint_pt(checkpoint_dir: Path | None, generation: dict) -> str:
    explicit = generation.get("checkpoint_path")
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        if path.is_file():
            return str(path)
        if path.is_dir():
            return str(_pick_pt_in_dir(path, generation))
    if checkpoint_dir is None:
        raise ValueError("Cosmos-Predict2.5 requires checkpoint_dir or generation.checkpoint_path")
    base = Path(checkpoint_dir).expanduser().resolve()
    if base.is_file():
        return str(base)
    return str(_pick_pt_in_dir(base, generation))


def _pick_pt_in_dir(base: Path, generation: dict) -> Path:
    variant = str(generation.get("checkpoint_variant", "post-trained"))
    model_size = str(generation.get("model_size", "14B"))
    candidates = [
        base / "base" / variant,
        base / variant,
    ]
    for directory in candidates:
        if not directory.is_dir():
            continue
        pt_files = sorted(directory.glob("*_ema_bf16.pt"))
        if pt_files:
            return pt_files[0].resolve()
    raise FileNotFoundError(
        f"no Predict2.5 .pt checkpoint under {base} (model_size={model_size}, variant={variant})"
    )


def build_sample_rows(
    rows: list[dict],
    generation: dict,
    *,
    conditioning_cache_dir: Path | None = None,
) -> list[dict]:
    inference_type = str(generation.get("inference_type", "image2world"))
    sample_extra = generation.get("sample_extra")
    stage_rgb_inputs = bool(generation.get("stage_rgb_inputs", True))
    samples: list[dict] = []
    for row in rows:
        prediction_stem = str(row["prediction_stem"])
        sample: dict = {
            "name": prediction_stem,
            "inference_type": inference_type,
            "prompt": str(row["prompt"]),
        }
        if inference_type in {"image2world", "video2world"}:
            input_path = Path(str(row["conditioning_image"])).expanduser().resolve()
            if stage_rgb_inputs and conditioning_cache_dir is not None:
                input_path = _stage_rgb_conditioning_image(input_path, conditioning_cache_dir, prediction_stem)
            sample["input_path"] = str(input_path)
        for field, cast in SAMPLE_VALUE_FIELDS.items():
            value = generation.get(field)
            if value is None and field == "num_output_frames" and generation.get("num_frames") is not None:
                value = generation.get("num_frames")
            if value is not None:
                sample[field] = cast(value) if cast is not bool else bool(value)
        if isinstance(sample_extra, dict):
            sample.update(sample_extra)
        samples.append(sample)
    return samples


def build_command(
    *,
    python_executable: str,
    repo_root: Path,
    generation: dict,
    checkpoint_path: str,
    predict_input: Path,
    predict_output: Path,
) -> list[str]:
    num_gpus = int(generation.get("num_gpus") or generation.get("nproc_per_node") or 1)
    if num_gpus > 1:
        command = [
            python_executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={num_gpus}",
            "-m",
            INFERENCE_MODULE,
        ]
    else:
        command = [python_executable, "-m", INFERENCE_MODULE]

    command += [
        "-i",
        str(predict_input),
        "-o",
        str(predict_output),
        "--checkpoint-path",
        checkpoint_path,
    ]

    model = generation.get("model")
    if model is None:
        model_size = str(generation.get("model_size", "14B"))
        variant = str(generation.get("checkpoint_variant", "post-trained"))
        model = f"{model_size}/{variant}"
    command += ["--model", str(model)]

    for key, (flag, cast) in SETUP_VALUE_FLAGS.items():
        value = generation.get(key)
        if value is not None:
            command += [flag, str(cast(value))]

    for key, flag in SETUP_FLAG_SWITCHES.items():
        if generation.get(key):
            command.append(flag)

    extra_cli_args = generation.get("extra_cli_args")
    if isinstance(extra_cli_args, (list, tuple)):
        command += [str(arg) for arg in extra_cli_args]

    return command


def _copy_completed_outputs(
    rows: list[dict],
    predict_output: Path,
    copied_stems: set[str],
    observed_sizes: dict[str, int],
    *,
    final: bool = False,
    min_source_age_seconds: float = 8.0,
) -> None:
    now = time.time()
    for row in rows:
        sample_id = str(row["sample_id"])
        stem = str(row["prediction_stem"])
        if stem in copied_stems:
            continue
        source = predict_output / f"{stem}.mp4"
        try:
            stat = source.stat()
        except FileNotFoundError:
            if final:
                print_status(sample_id, "failed", error=f"missing generated output: {source}")
            continue
        if stat.st_size <= 0:
            if final:
                print_status(sample_id, "failed", error=f"empty generated output: {source}")
            continue
        previous_size = observed_sizes.get(stem)
        observed_sizes[stem] = stat.st_size
        if not final and previous_size != stat.st_size and now - stat.st_mtime < min_source_age_seconds:
            continue
        try:
            copy_output(source, Path(str(row["output_path"])))
            copied_stems.add(stem)
            print_status(sample_id, "generated", output_path=str(row["output_path"]), source=str(source))
        except Exception as exc:
            if final:
                print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())


def _run_with_live_output_copy(command: list[str], *, cwd: Path, rows: list[dict], predict_output: Path, generation: dict) -> None:
    copied_stems: set[str] = set()
    observed_sizes: dict[str, int] = {}
    interval = float(generation.get("live_copy_interval_seconds", 8.0))
    min_age = float(generation.get("live_copy_min_source_age_seconds", 8.0))
    process = subprocess.Popen(command, cwd=str(cwd))
    try:
        while process.poll() is None:
            _copy_completed_outputs(
                rows,
                predict_output,
                copied_stems,
                observed_sizes,
                min_source_age_seconds=min_age,
            )
            time.sleep(interval)
        _copy_completed_outputs(rows, predict_output, copied_stems, observed_sizes, final=True)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            process.wait()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    bin_dir = os.path.dirname(sys.executable)
    if bin_dir and bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    cuda_lib_dirs = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12.9/targets/x86_64-linux/lib",
    ]
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in existing_ld.split(os.pathsep) if part]
    for lib_dir in cuda_lib_dirs:
        if os.path.isdir(lib_dir) and lib_dir not in ld_parts:
            ld_parts.insert(0, lib_dir)
    if ld_parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        for candidate in ("/usr/local/cuda-12.9", "/usr/local/cuda"):
            if os.path.isdir(candidate):
                cuda_home = candidate
                break
    if cuda_home:
        os.environ.setdefault("CUDA_HOME", cuda_home)
        os.environ.setdefault("CUDA_PATH", cuda_home)

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    spec_path = Path(args.batch_spec_path).expanduser().resolve()
    rows = load_batch_spec(spec_path)
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else None

    predict_input = spec_path.with_suffix(".predict25.jsonl")
    predict_output = spec_path.parent / f"{spec_path.stem}_predict25_outputs"
    conditioning_cache_dir = spec_path.parent / f"{spec_path.stem}_conditioning_rgb"
    samples = build_sample_rows(rows, generation, conditioning_cache_dir=conditioning_cache_dir)
    with predict_input.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    checkpoint_path = _resolve_checkpoint_pt(checkpoint_dir, generation)
    checkpoint_root = _resolve_checkpoint_root(generation, checkpoint_dir)
    local_aux = _resolve_local_aux_paths(checkpoint_root, generation)
    experiment_opts = _build_local_experiment_opts(local_aux)
    extra_opts = generation.get("experiment_opts")
    if isinstance(extra_opts, (list, tuple)):
        experiment_opts.extend(str(item) for item in extra_opts)
    if experiment_opts:
        os.environ["WORLDARENA_COSMOS_PREDICT25_EXPERIMENT_OPTS"] = json.dumps(experiment_opts, ensure_ascii=False)
    local_aliases = _build_local_checkpoint_aliases(local_aux)
    if local_aliases:
        os.environ["WORLDARENA_COSMOS_PREDICT25_LOCAL_ALIASES"] = json.dumps(local_aliases, ensure_ascii=False)
    command = build_command(
        python_executable=sys.executable,
        repo_root=repo_root,
        generation=generation,
        checkpoint_path=checkpoint_path,
        predict_input=predict_input,
        predict_output=predict_output,
    )

    if os.environ.get("WORLDARENA_COSMOS_PREDICT25_DRY_RUN"):
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "command": command,
                    "checkpoint_path": checkpoint_path,
                    "local_aux": {key: str(value) for key, value in local_aux.items()},
                    "experiment_opts": experiment_opts,
                    "predict_input": str(predict_input),
                    "predict_output": str(predict_output),
                    "num_samples": len(samples),
                    "sample_preview": samples[0] if samples else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    _run_with_live_output_copy(
        command,
        cwd=repo_root,
        rows=rows,
        predict_output=predict_output,
        generation=generation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
