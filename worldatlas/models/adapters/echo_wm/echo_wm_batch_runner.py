"""Load Echo-WM Base once and process a WorldAtlas batch spec."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any

from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    begin_sample,
    copy_output,
    load_batch_spec,
    load_json,
    log_pipeline,
    print_status,
)


DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, game UI, "
    "video game interface, HUD, heads-up display, menu, status bar, health bar, "
    "score, minimap, crosshair, reticle, buttons, icons, subtitles, captions, "
    "watermark, logo, text overlay, user interface"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Echo-WM batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def _checkpoint_path(root: Path, generation: dict[str, Any], key: str, default: str) -> Path:
    raw = str(generation.get(key) or default)
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def resolve_runtime_paths(
    repo_root: Path,
    checkpoint_root: Path,
    generation: dict[str, Any],
) -> tuple[Path, Path, Path]:
    echo_root = repo_root / "echo_wm"
    checkpoint = _checkpoint_path(
        checkpoint_root,
        generation,
        "checkpoint_file",
        "echo-wm-base.safetensors",
    )
    gemma_root = _checkpoint_path(checkpoint_root, generation, "gemma_dir", "gemma-3")
    required_sources = (
        echo_root / "inference_wm.py",
        echo_root / "helpers" / "action_condition.py",
        echo_root / "ltx-core" / "src" / "ltx_core",
        echo_root / "ltx-pipelines" / "src" / "ltx_pipelines",
    )
    missing_sources = [path for path in required_sources if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Echo-WM source tree is incomplete: "
            + ", ".join(str(path) for path in missing_sources)
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Echo-WM checkpoint not found: {checkpoint}")
    if not gemma_root.is_dir():
        raise FileNotFoundError(f"Echo-WM Gemma directory not found: {gemma_root}")
    if not (gemma_root / "config.json").is_file():
        raise FileNotFoundError(f"Echo-WM Gemma config not found: {gemma_root / 'config.json'}")
    return echo_root, checkpoint, gemma_root


def _dtype_device(torch: Any) -> Any:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def install_checkpoint_registry(pipeline: Any, registry: Any) -> None:
    """Make all ledger builders reuse CPU checkpoint tensors across samples.

    Echo's upstream pipeline deliberately destroys GPU component instances between
    stages to fit on one accelerator. Its default ``DummyRegistry`` also drops the
    corresponding state dicts, which causes every subsequent sample to reread the
    large safetensors files. A real registry preserves the state dicts on CPU while
    retaining the upstream GPU component swapping behavior.
    """
    ledger = pipeline.model_ledger
    ledger.registry = registry
    ledger.build_model_builders()


def generate_and_encode_window(
    pipeline: Any,
    *,
    pipeline_kwargs: dict[str, Any],
    encoder: Any,
    output_path: Path,
    fps: int,
    with_audio: bool,
    video_chunks_number: int,
) -> None:
    """Consume lazy VAE decoding under inference mode, then commit the MP4.

    The pipeline returns a generator: the decoder actually runs when the encoder
    iterates it. Leaving inference mode before encoding enables autograd on
    inference tensors. Keep both stages in the same scope, as the upstream CLI
    does, without rebuilding the pipeline or rereading checkpoints.
    """
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-", dir=output_path.parent) as temporary:
        pending = Path(temporary) / output_path.name
        with torch.inference_mode():
            video, audio = pipeline(**pipeline_kwargs)
            encoder(
                video=video,
                fps=fps,
                audio=audio if with_audio else None,
                output_path=str(pending),
                video_chunks_number=video_chunks_number,
            )
        if not pending.is_file() or pending.stat().st_size <= 0:
            raise FileNotFoundError(f"Echo encoder did not write a complete window: {pending}")
        pending.replace(output_path)


def split_action_windows(
    action_str: str,
    *,
    num_frames: int,
    chunk_num_frames: int,
) -> list[tuple[str, int]]:
    """Partition one full action timeline into legal overlapping Echo windows.

    Every window has one conditioning frame followed by ``chunk_num_frames - 1``
    generated intervals. The original action sequence is expanded only in memory
    (at most 1440 tokens for the minute track) and run-length encoded again after
    slicing, so a closed itinerary is executed once rather than repeated per chunk.
    """
    if num_frames < 2 or chunk_num_frames < 9:
        raise ValueError("Echo action windows require num_frames >= 2 and chunk_num_frames >= 9")
    if (chunk_num_frames - 1) % 8:
        raise ValueError(f"Echo chunk_num_frames must lie on the 1+8m lattice, got {chunk_num_frames}")

    actions: list[str] = []
    for raw_segment in str(action_str).split(","):
        token, separator, raw_count = raw_segment.strip().rpartition("-")
        if not separator or not token:
            raise ValueError(f"invalid Echo action segment: {raw_segment!r}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"Echo action segment count must be positive: {raw_segment!r}")
        actions.extend([token] * count)
    if len(actions) != num_frames - 1:
        raise ValueError(
            f"Echo action intervals mismatch: action={len(actions)}, frames={num_frames}"
        )

    stride = chunk_num_frames - 1
    if len(actions) % stride:
        raise ValueError(
            f"Echo long rollout must divide into complete {stride}-interval windows; "
            f"got {len(actions)} intervals"
        )

    windows: list[tuple[str, int]] = []
    for start in range(0, len(actions), stride):
        chunk = actions[start : start + stride]
        encoded: list[str] = []
        current = chunk[0]
        count = 0
        for token in chunk:
            if token == current:
                count += 1
                continue
            encoded.append(f"{current}-{count}")
            current, count = token, 1
        encoded.append(f"{current}-{count}")
        windows.append((",".join(encoded), len(chunk) + 1))
    return windows


def _extract_last_frame(ffmpeg_bin: str, video: Path, output: Path, frame_index: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise FileNotFoundError(f"failed to extract Echo continuation frame: {output}")


def _concatenate_windows(ffmpeg_bin: str, chunks: list[Path], output: Path, fps: float) -> None:
    if not chunks:
        raise ValueError("Echo window concatenation requires at least one chunk")
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for index, chunk in enumerate(chunks):
        inputs.extend(["-i", str(chunk)])
        trim = "" if index == 0 else "trim=start_frame=1,"
        filters.append(f"[{index}:v:0]{trim}setpts=PTS-STARTPTS[v{index}]")
        labels.append(f"[v{index}]")
    filters.append(f"{''.join(labels)}concat=n={len(chunks)}:v=1:a=0[outv]")
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(temporary),
        ],
        check=True,
    )
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise FileNotFoundError(f"Echo window stitcher did not write output: {temporary}")
    temporary.replace(output)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_root = Path(str(args.checkpoint_dir or "")).expanduser().resolve()
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())
    echo_root, checkpoint, gemma_root = resolve_runtime_paths(
        repo_root,
        checkpoint_root,
        generation,
    )
    for source in (echo_root, echo_root / "ltx-core" / "src", echo_root / "ltx-pipelines" / "src"):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

    import torch
    from helpers.action_camera import (
        DEFAULT_PITCH_LIMIT_DEG,
        DEFAULT_ROTATION_SPEED_DEG,
        DEFAULT_TRANSLATION_SPEED,
    )
    from helpers.action_condition import action_config, build_action_condition
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader.registry import StateDictRegistry
    from ltx_core.model.video_vae.tiling import TilingConfig
    from ltx_core.model.video_vae.video_vae import get_video_chunks_number
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    width = int(generation.get("width", 1280))
    height = int(generation.get("height", 704))
    fps = float(generation.get("fps", 24.0))
    steps = int(generation.get("steps", 30))
    base_seed = int(generation.get("seed", 42))
    video_cfg = float(generation.get("video_cfg", 4.0))
    audio_cfg = float(generation.get("audio_cfg", 2.0))
    stg_scale = float(generation.get("stg_scale", 1.0))
    stg_blocks = [int(value) for value in generation.get("stg_blocks", [29])]
    fov = float(generation.get("fov_deg", 70.0))
    translation_speed = float(generation.get("translation_speed", DEFAULT_TRANSLATION_SPEED))
    rotation_speed = float(generation.get("rotation_speed_deg", DEFAULT_ROTATION_SPEED_DEG))
    pitch_limit = float(generation.get("pitch_limit_deg", DEFAULT_PITCH_LIMIT_DEG))
    negative_prompt = str(generation.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
    with_audio = not bool(generation.get("no_audio", False))
    chunk_num_frames = int(generation.get("chunk_num_frames", 241))
    ffmpeg_bin = str(generation.get("ffmpeg_bin", "ffmpeg"))
    max_consecutive_failures = int(generation.get("max_consecutive_failures", 3))
    if max_consecutive_failures < 1:
        raise ValueError("Echo max_consecutive_failures must be positive")
    device = _dtype_device(torch)

    log_pipeline(
        "pipeline_loading",
        label="Echo-WM",
        samples=len(rows),
        checkpoint=str(checkpoint),
        gemma_root=str(gemma_root),
    )
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=str(checkpoint),
        gemma_root=str(gemma_root),
        loras=(),
        device=device,
        action_config=action_config(width, height),
    )
    install_checkpoint_registry(pipeline, StateDictRegistry())
    log_pipeline(
        "pipeline_loaded",
        label="Echo-WM",
        samples=len(rows),
        checkpoint_cache="cpu_state_dict",
        checkpoint_load_policy="load_once",
    )

    failed = 0
    consecutive_failures = 0
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", index))
        begin_sample(sample_id, index=index + 1, total=len(rows), label="Echo-WM")
        try:
            image_path = Path(str(row["conditioning_image"])).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Echo-WM conditioning image not found: {image_path}")
            output_path = Path(str(row["output_path"])).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            num_frames = int(row.get("num_frames", generation.get("num_frames", 241)))
            action_windows = split_action_windows(
                str(row["action_str"]),
                num_frames=num_frames,
                chunk_num_frames=chunk_num_frames if num_frames > chunk_num_frames else num_frames,
            )
            if len(action_windows) > 1 and with_audio:
                raise ValueError("Echo windowed memory rollout requires no_audio=true")

            window_root = output_path.parent / "_echo_windows" / output_path.stem
            window_root.mkdir(parents=True, exist_ok=True)
            chunks: list[Path] = []
            conditioning_path = image_path
            for window_index, (window_action, window_frames) in enumerate(action_windows):
                chunk_path = window_root / f"window-{window_index:03d}.mp4"
                if not chunk_path.is_file() or chunk_path.stat().st_size <= 0:
                    action = build_action_condition(
                        window_action,
                        num_frames=window_frames,
                        width=width,
                        height=height,
                        translation_speed=translation_speed,
                        rotation_speed_deg=rotation_speed,
                        pitch_limit_deg=pitch_limit,
                        fov_deg=fov,
                        device=device,
                        fps=fps,
                    )
                    generate_and_encode_window(
                        pipeline,
                        pipeline_kwargs=dict(
                            prompt=str(row["prompt"]),
                            negative_prompt=negative_prompt,
                            seed=base_seed + index * 1000 + window_index,
                            height=height,
                            width=width,
                            num_frames=window_frames,
                            frame_rate=fps,
                            num_inference_steps=steps,
                            video_guider_params=MultiModalGuiderParams(
                                cfg_scale=video_cfg,
                                stg_scale=stg_scale,
                                stg_blocks=stg_blocks,
                            ),
                            audio_guider_params=MultiModalGuiderParams(
                                cfg_scale=audio_cfg,
                                stg_scale=stg_scale,
                                stg_blocks=stg_blocks,
                            ),
                            images=[ImageConditioningInput(str(conditioning_path), 0, 1.0)],
                            action_cond=action,
                            video_tiling_config=TilingConfig.default(),
                        ),
                        encoder=encode_video,
                        output_path=chunk_path,
                        fps=int(fps),
                        with_audio=with_audio,
                        video_chunks_number=get_video_chunks_number(
                            window_frames,
                            TilingConfig.default(),
                        ),
                    )
                chunks.append(chunk_path)
                if window_index + 1 < len(action_windows):
                    continuation = window_root / f"condition-{window_index + 1:03d}.png"
                    if not continuation.is_file() or continuation.stat().st_size <= 0:
                        _extract_last_frame(
                            ffmpeg_bin,
                            chunk_path,
                            continuation,
                            window_frames - 1,
                        )
                    conditioning_path = continuation

            if len(chunks) == 1:
                copy_output(chunks[0], output_path)
            else:
                _concatenate_windows(ffmpeg_bin, chunks, output_path, fps)
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise FileNotFoundError(f"Echo-WM did not write output: {output_path}")
            print_status(
                sample_id,
                "generated",
                output_path=str(output_path),
                action_str=row["action_str"],
                action_approximations=row.get("action_approximations", []),
                num_frames=num_frames,
                audio=with_audio,
                windows=len(action_windows),
                checkpoint_cache="cpu_state_dict",
            )
            consecutive_failures = 0
            if bool(generation.get("cleanup_window_files", True)):
                shutil.rmtree(window_root)
        except Exception as exc:
            failed += 1
            consecutive_failures += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if consecutive_failures >= max_consecutive_failures:
                log_pipeline(
                    "batch_aborted", label="Echo-WM",
                    consecutive_failures=consecutive_failures,
                    reason="consecutive sample failures; fix the error before resubmitting",
                )
                return 1
    return 1 if failed == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["install_checkpoint_registry", "resolve_runtime_paths", "split_action_windows"]
