"""WorldAtlas Arena entry for Cosmos-Predict2.5 inference with local aux checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import pydantic
import torch
import tyro

from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_predict2._src.imaginaire.flags import SMOKE
from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
    path_to_str,
)




def _install_local_checkpoint_aliases() -> None:
    """Install local checkpoint aliases."""
    raw = os.environ.get("WORLDARENA_COSMOS_PREDICT25_LOCAL_ALIASES", "").strip()
    if not raw:
        return
    aliases = json.loads(raw)
    if not isinstance(aliases, dict):
        raise ValueError("WORLDARENA_COSMOS_PREDICT25_LOCAL_ALIASES must be a JSON object")

    import cosmos_predict2._src.imaginaire.utils.checkpoint_db as checkpoint_db

    if getattr(checkpoint_db, "_worldarena_aliases_installed", False):
        return

    original_download = checkpoint_db.download_checkpoint
    original_get_uri = checkpoint_db.get_checkpoint_uri

    normalized_aliases = {
        checkpoint_db.normalize_uri(key): str(value)
        for key, value in aliases.items()
    }

    def _resolve_local(uri: str) -> str | None:
        normalized = checkpoint_db.normalize_uri(uri)
        if normalized in normalized_aliases:
            return normalized_aliases[normalized]
        cfg = checkpoint_db.CheckpointConfig.maybe_from_uri(normalized)
        if cfg is not None:
            mapped = normalized_aliases.get(checkpoint_db.normalize_uri(cfg.s3.uri))
            if mapped:
                return mapped
        return None

    def download_checkpoint(checkpoint_uri: str, *, check_exists: bool = True) -> str:
        local = _resolve_local(checkpoint_uri)
        if local is not None:
            if check_exists and not os.path.exists(local):
                raise ValueError(f"Local checkpoint alias missing: {local}")
            return local
        return original_download(checkpoint_uri, check_exists=check_exists)

    def get_checkpoint_uri(checkpoint_uri: str, *, check_exists: bool = False) -> str:
        local = _resolve_local(checkpoint_uri)
        if local is not None:
            if check_exists and not os.path.exists(local):
                raise ValueError(f"Local checkpoint alias missing: {local}")
            return local
        return original_get_uri(checkpoint_uri, check_exists=check_exists)

    checkpoint_db.download_checkpoint = download_checkpoint
    checkpoint_db.get_checkpoint_path = download_checkpoint
    checkpoint_db.get_checkpoint_uri = get_checkpoint_uri
    checkpoint_db._worldarena_aliases_installed = True


def _load_experiment_opts() -> list[str]:
    """Load experiment opts -> list[str]."""
    raw = os.environ.get("WORLDARENA_COSMOS_PREDICT25_EXPERIMENT_OPTS", "").strip()
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("WORLDARENA_COSMOS_PREDICT25_EXPERIMENT_OPTS must be a JSON list")
    return [str(item) for item in parsed]


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    setup: SetupArguments
    overrides: InferenceOverrides


class Inference:
    def __init__(self, args: SetupArguments):
        log.debug(f"{args.__class__.__name__}({args})")

        torch.enable_grad(False)
        _install_local_checkpoint_aliases()

        self.rank0 = distributed.is_rank0()
        self.setup_args = args
        experiment_opts = _load_experiment_opts()
        if args.model_key.distilled:
            experiment_opts.append("model.config.init_student_with_teacher=False")
        self.pipe = Video2WorldInference(
            experiment_name=args.experiment,
            ckpt_path=args.checkpoint_path,
            s3_credential_path="",
            context_parallel_size=args.context_parallel_size,
            config_file=args.config_file,
            experiment_opts=experiment_opts,
            offload_diffusion_model=args.offload_diffusion_model,
            offload_text_encoder=args.offload_text_encoder,
            offload_tokenizer=args.offload_tokenizer,
        )
        if self.rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_path = args.output_dir / "config.yaml"
            LazyConfig.save_yaml(self.pipe.config, config_path)
            log.info(f"Saved config to {config_path}")

        self.guardrail_enabled = not args.disable_guardrails

        if self.rank0 and self.guardrail_enabled:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            self.text_guardrail_runner = None
            self.video_guardrail_runner = None

    def generate(self, samples: list[InferenceArguments], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]

        sample_names = [sample.name for sample in samples]
        log.info(f"Generating {len(samples)} samples: {sample_names}")

        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
            output_path = self._generate_sample(sample, output_dir)
            if output_path is not None:
                output_paths.append(output_path)
        return output_paths

    def _generate_sample(self, sample: InferenceArguments, output_dir: Path) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        output_path = output_dir / sample.name

        if self.rank0:
            output_dir.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}.json")

            if self.text_guardrail_runner is not None:
                if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):
                    message = f"Guardrail blocked text2world generation. Prompt: {sample.prompt}"
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    raise Exception(message)
                log.success("Passed guardrail on prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on prompt are disabled")

        if sample.enable_autoregressive:
            log.info("Generating video with autoregressive mode...")
            video = self.pipe.generate_autoregressive_from_batch(
                prompt=sample.prompt,
                input_path=path_to_str(sample.input_path),
                num_output_frames=sample.num_output_frames,
                chunk_size=sample.chunk_size,
                chunk_overlap=sample.chunk_overlap,
                guidance=sample.guidance,
                num_latent_conditional_frames=sample.num_input_frames,
                resolution=sample.resolution,
                seed=sample.seed,
                negative_prompt=sample.negative_prompt,
                num_steps=sample.num_steps,
            )
        else:
            log.info("Generating video with standard mode...")
            video = self.pipe.generate_vid2world(
                prompt=sample.prompt,
                input_path=path_to_str(sample.input_path),
                guidance=sample.guidance,
                num_video_frames=sample.num_output_frames,
                num_latent_conditional_frames=sample.num_input_frames,
                resolution=sample.resolution,
                seed=sample.seed,
                negative_prompt=sample.negative_prompt,
                num_steps=sample.num_steps,
            )

        if self.rank0:
            video = (1.0 + video[0]) / 2

            if self.video_guardrail_runner is not None:
                log.info("Running guardrail check on video...")
                frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
                processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                if processed_frames is None:
                    message = "Guardrail blocked video2world generation."
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    raise Exception(message)
                log.success("Passed guardrail on generated video")
                processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                video = processed_video.to(video.device, dtype=video.dtype)
            else:
                log.warning("Guardrail checks on video are disabled")

            save_img_or_video(video, str(output_path), fps=16)
            log.success(f"Saved video to {output_path}.mp4")
        return f"{output_path}.mp4"


def main(args: Args) -> None:
    """Entry point for the module."""
    inference_samples = InferenceArguments.from_files(
        args.input_files, overrides=args.overrides, setup_args=args.setup
    )
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    inference = Inference(args.setup)
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()
    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as exc:
        handle_tyro_exception(exc)
    main(args)
    cleanup_environment()
