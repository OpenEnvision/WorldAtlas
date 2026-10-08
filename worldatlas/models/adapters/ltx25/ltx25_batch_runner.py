"""Persistent LTX-2.5 I2V worker; sampling and conditioning stay in upstream code."""
from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from uuid import uuid4

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args, begin_sample, load_batch_spec, load_json, print_status,
)
from worldatlas.models.adapters.ltx25.ltx25 import runtime_options

COMPONENTS = ("transformer_path", "text_encoder_path", "video_vae_path", "audio_vae_path", "spatial_upsampler_path")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def add_repo(repo_root: Path) -> None:
    for package in ("ltx-core", "ltx-pipelines"):
        source = repo_root / "packages" / package / "src"
        if not source.is_dir():
            raise FileNotFoundError(source)
        sys.path.insert(0, str(source))


def component_paths(generation: dict, checkpoint_dir: Path) -> dict[str, str]:
    paths = {}
    for key in COMPONENTS:
        if not generation.get(key):
            raise ValueError(f"Missing LTX-2.5 split component: {key}")
        path = Path(generation[key]).expanduser()
        path = (checkpoint_dir / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        paths[key] = str(path)
    return paths


def make_cpu_registry():
    """Use upstream's registry interface without keeping every component in VRAM.

    Builders accept CPU-retained StateDicts and move only the active model to GPU.
    No upstream monkey patches or sampling changes are needed.
    """
    import torch
    from ltx_core.loader.registry import ModelRegistry

    class CPURegistry(ModelRegistry):
        def add(self, paths, sd_ops, state_dict):
            cpu_sd = replace(state_dict, sd={k: v.to("cpu") for k, v in state_dict.sd.items()},
                             device=torch.device("cpu"))
            return super().add(paths, sd_ops, cpu_sd)

    return CPURegistry(cache_weights=True, cache_models=True)


def preflight(repo_root: Path, checkpoint_dir: Path, generation: dict) -> dict:
    options = runtime_options(generation)
    paths = component_paths(options, checkpoint_dir)
    add_repo(repo_root)
    from safetensors import safe_open
    from ltx_pipelines.distilled import DistilledPipeline, should_use_ancestral_sampler
    from ltx_pipelines.utils.constants import detect_params

    assets = {}
    for key, name in paths.items():
        path = Path(name)
        with safe_open(name, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            tensors = len(handle.keys())
        if key in ("transformer_path", "video_vae_path", "audio_vae_path"):
            if not metadata.get("model_version", "").startswith("2.5."):
                raise ValueError(f"Expected LTX-2.5 component: {path}")
        assets[key] = dict(path=name, bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                           metadata_sha256=json_sha(metadata), tensors=tensors)
    if not should_use_ancestral_sampler(paths["transformer_path"]):
        raise ValueError("Official LTX-2.5 ancestral sampler was not selected")
    crf = options["image_crf"]
    if crf is None:
        crf = detect_params(paths["video_vae_path"]).default_image_crf
    return dict(assets=assets, image_crf=crf, ancestral_sampler=True,
                pipeline_module=DistilledPipeline.__module__, generation_sha256=json_sha(generation))


def load_pipeline(paths: dict, *, device=None):
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import OffloadMode

    return DistilledPipeline(
        model_paths=ModelPaths.from_split(**{k: paths[k] for k in COMPONENTS if k != "spatial_upsampler_path"}),
        spatial_upsampler_path=paths["spatial_upsampler_path"],
        loras=[], registry=make_cpu_registry(), offload_mode=OffloadMode.NONE, device=device,
    )


def inspect_output(path: Path, options: dict) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    stream = json.loads(result.stdout)["streams"][0]
    actual = (int(stream["width"]), int(stream["height"]), int(stream["nb_frames"]), Fraction(stream["r_frame_rate"]))
    expected = (options["width"], options["height"], options["num_frames"], Fraction(str(options["frame_rate"])))
    if actual != expected or abs(float(stream["duration"]) - options["num_frames"] / options["frame_rate"]) > 0.02:
        raise ValueError(f"LTX-2.5 output geometry/timing mismatch: {stream}")
    return stream


def generate_one(pipeline, row: dict, options: dict, runtime: dict) -> Path:
    import torch
    from PIL import Image
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    if row["suite"] not in ("image_static", "image_dynamic"):
        raise ValueError(f"Unsupported suite: {row['suite']}")
    source = Path(row["conditioning_image"]).resolve()
    source_sha = sha_file(source)
    output = Path(row["output_path"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid4().hex}.pending.mp4")
    images = [ImageConditioningInput(path=str(source), frame_idx=0,
              strength=options["image_conditioning_strength"], crf=options["image_crf"])]
    try:
        with torch.inference_mode():
            video, audio, frames, tiling = pipeline(
                prompt=row["prompt"], seed=options["seed"], height=options["height"], width=options["width"],
                num_frames=options["num_frames"], frame_rate=options["frame_rate"], images=images,
                tiling_config=AUTO_TILING, enhance_prompt=False,
            )
            if int(frames) != options["num_frames"]:
                raise ValueError(f"Upstream changed explicit frame count: {frames}")
            chunks = get_video_chunks_number(frames, tiling) if tiling is not None else 1
            encode_video(video=video, fps=options["frame_rate"], audio=audio,
                         output_path=str(temporary), video_chunks_number=chunks, thread_count=4)
        stream = inspect_output(temporary, options)
        if sha_file(source) != source_sha:
            raise ValueError(f"Conditioning image changed during generation: {source}")
        with Image.open(source) as image:
            source_size = list(image.size)
        receipt = dict(model="ltx25", sample_id=row["sample_id"], suite=row["suite"],
                       conditioning_image=str(source), conditioning_sha256=source_sha, prompt=row["prompt"],
                       seed=options["seed"], generation_sha256=runtime["generation_sha256"],
                       runtime_sha256=json_sha(runtime), output_path=str(output), output_sha256=sha_file(temporary),
                       stream=stream, conditioning=dict(source_size=source_size, image_crf=runtime["image_crf"],
                       stage1_size=[options["width"] // 2, options["height"] // 2],
                       stage2_size=[options["width"], options["height"]],
                       protocol="official LTX preprocess and ceil/floor center crop separately at each stage"))
        temporary.replace(output)
        receipt_path = output.with_suffix(".ltx25.json")
        receipt_tmp = receipt_path.with_suffix(".json.tmp")
        receipt_tmp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        receipt_tmp.replace(receipt_path)
        return output
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_batch_args(parser)
    args = parser.parse_args()
    if not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required")
    generation = load_json(Path(args.generation_config_path))
    options = runtime_options(generation)
    rows = load_batch_spec(Path(args.batch_spec_path))
    runtime = preflight(Path(args.repo_root).resolve(), Path(args.checkpoint_dir).resolve(), generation)
    paths = component_paths(options, Path(args.checkpoint_dir).resolve())
    pipeline = load_pipeline(paths)
    print_status("ltx25", "pipeline_ready", samples=len(rows), image_crf=runtime["image_crf"])
    failed = 0
    for index, row in enumerate(rows):
        begin_sample(row["sample_id"], index=index + 1, total=len(rows), label="LTX-2.5")
        try:
            output = generate_one(pipeline, row, options, runtime)
            print_status(row["sample_id"], "generated", output_path=str(output))
        except Exception as exc:
            failed += 1
            print_status(row["sample_id"], "failed", error=str(exc), traceback=traceback.format_exc())
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    # Let the parent collect completed videos and mark individual missing ones.
    return int(failed == len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
