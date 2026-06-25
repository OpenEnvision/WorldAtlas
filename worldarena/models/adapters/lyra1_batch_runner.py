from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Any

from worldarena.common.checkpoints import apply_checkpoint_env, resolve_checkpoint_path
from worldarena.models.adapters.lyra1_runner import (
    DEFAULT_STATIC_VIEW_INDICES,
    _checkpoint_layout,
    _copy_validated_video,
    _dataset_root,
    _parse_static_view_indices,
    _prepare_apex_amp_shim,
    _prepare_checkpoint_root,
    _run_command,
    _run_stage2,
    _temporary_dataset_registration,
)
from worldarena.models.adapters.lyra2_runner import _prepare_transformer_engine_shim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Lyra-1 batch subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--batch_spec_path", required=True, type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--stage1_num_gpus", default=None, type=int)
    parser.add_argument("--num_video_frames", default=121, type=int)
    parser.add_argument("--height", default=704, type=int)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--fps", default=24, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--guidance", default=1.0, type=float)
    parser.add_argument("--num_steps", default=35, type=int)
    parser.add_argument("--trajectory", default="left", type=str)
    parser.add_argument("--camera_rotation", default="center_facing", type=str)
    parser.add_argument("--movement_distance", default=0.3, type=float)
    parser.add_argument("--total_movement_distance_factor", default=1.0, type=float)
    parser.add_argument("--filter_points_threshold", default=0.05, type=float)
    parser.add_argument("--center_depth_quantile_value", default=0.5, type=float)
    parser.add_argument("--target_index_subsample", default=4, type=int)
    parser.add_argument("--output_view_index", default=0, type=int)
    parser.add_argument("--negative_prompt", default=None, type=str)
    parser.add_argument("--static_view_indices_fixed", default=",".join(DEFAULT_STATIC_VIEW_INDICES), type=str)
    parser.add_argument("--foreground_masking", action="store_true")
    parser.add_argument("--center_depth_quantile", action="store_true")
    parser.add_argument("--multi_trajectory", action="store_true")
    parser.add_argument("--offload_diffusion_transformer", action="store_true")
    parser.add_argument("--offload_tokenizer", action="store_true")
    parser.add_argument("--offload_text_encoder_model", action="store_true")
    parser.add_argument("--offload_prompt_upsampler", action="store_true")
    parser.add_argument("--offload_guardrail_models", action="store_true")
    parser.add_argument("--disable_guardrail", action="store_true")
    parser.add_argument("--disable_prompt_encoder", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument(
        "--batch_chunk_size",
        default=int(os.environ.get("WORLDARENA_LYRA1_BATCH_CHUNK_SIZE", "0")),
        type=int,
    )
    return parser.parse_args()


def _stage1_num_gpus(args: argparse.Namespace) -> int:
    return max(1, int(args.stage1_num_gpus if args.stage1_num_gpus is not None else args.num_gpus))


def _load_batch_spec(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty Lyra-1 batch spec: {path}")
    return rows


def _stage_input_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _write_stage1_batch_inputs(
    *,
    rows: list[dict[str, Any]],
    input_root: Path,
    batch_input_path: Path,
) -> None:
    del input_root
    per_sample_keys = (
        "trajectory",
        "camera_rotation",
        "movement_distance",
        "total_movement_distance_factor",
        "negative_prompt",
        "camera_path_primary",
    )
    with batch_input_path.open("w", encoding="utf-8") as file:
        for row in rows:
            stem = str(row["prediction_stem"])
            conditioning_source = Path(row["conditioning_image"]).expanduser().resolve()
            payload: dict[str, Any] = {
                "prompt": str(row["prompt"]),
                "visual_input": str(conditioning_source),
                "output_name": stem,
            }
            generation = row.get("generation")
            if isinstance(generation, dict):
                for key in per_sample_keys:
                    if generation.get(key) is not None:
                        payload[key] = generation[key]
            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )


def _stage1_batch_command(
    *,
    batch_input_path: Path,
    generated_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    stage1_num_gpus = _stage1_num_gpus(args)
    script_and_args = [
        "cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py",
        "--checkpoint_dir",
        "checkpoints",
        "--num_gpus",
        str(stage1_num_gpus),
        "--batch_input_path",
        str(batch_input_path),
        "--video_save_folder",
        str(generated_root),
        "--num_video_frames",
        str(args.num_video_frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
        "--guidance",
        str(args.guidance),
        "--num_steps",
        str(args.num_steps),
        "--trajectory",
        args.trajectory,
        "--camera_rotation",
        args.camera_rotation,
        "--movement_distance",
        str(args.movement_distance),
        "--total_movement_distance_factor",
        str(args.total_movement_distance_factor),
        "--filter_points_threshold",
        str(args.filter_points_threshold),
        "--center_depth_quantile_value",
        str(args.center_depth_quantile_value),
    ]
    command = [sys.executable]
    if stage1_num_gpus > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes",
                "1",
                "--nproc_per_node",
                str(stage1_num_gpus),
            ]
        )
    command.extend(script_and_args)
    if args.negative_prompt:
        command.extend(["--negative_prompt", args.negative_prompt])
    for flag_name in (
        "foreground_masking",
        "center_depth_quantile",
        "multi_trajectory",
        "offload_diffusion_transformer",
        "offload_tokenizer",
        "offload_text_encoder_model",
        "offload_prompt_upsampler",
        "offload_guardrail_models",
        "disable_guardrail",
        "disable_prompt_encoder",
    ):
        if getattr(args, flag_name):
            command.append(f"--{flag_name}")
    return command


def _find_stage1_video(generated_root: Path, stem: str, output_view_index: int) -> Path | None:
    direct = generated_root / "rgb" / f"{stem}.mp4"
    if direct.exists():
        return direct
    view_candidate = generated_root / str(output_view_index) / "rgb" / f"{stem}.mp4"
    if view_candidate.exists():
        return view_candidate
    candidates = sorted(generated_root.rglob(f"{stem}.mp4"))
    return candidates[0] if candidates else None


def _find_reconstruction_video(output_root: Path, stem: str, output_view_index: int) -> Path | None:
    candidates = sorted(output_root.rglob(f"{stem}.mp4"))
    for candidate in candidates:
        if candidate.parent.name == str(output_view_index):
            return candidate
    return candidates[0] if candidates else None


def _record_copy_error(row: dict[str, Any], error: Exception) -> None:
    output_path = Path(row["output_path"]).expanduser().resolve()
    error_path = output_path.with_suffix(".lyra1.error.txt")
    error_path.write_text(str(error), encoding="utf-8")


def _copy_batch_stage1_outputs(
    *,
    rows: list[dict[str, Any]],
    generated_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for row in rows:
        stem = str(row["prediction_stem"])
        selected_video = _find_stage1_video(generated_root, stem, args.output_view_index)
        try:
            if selected_video is None:
                raise FileNotFoundError(f"Lyra-1 stage1 output video was not written for {stem}")
            _copy_validated_video(
                selected_video,
                Path(row["output_path"]).expanduser().resolve(),
                expected_frames=args.num_video_frames,
                expected_fps=args.fps,
                expected_width=args.width,
                expected_height=args.height,
            )
        except Exception as exc:
            _record_copy_error(row, exc)
            errors.append({"sample_id": str(row["sample_id"]), "error": str(exc)})
    return errors


def _append_progressive_sync_log(log_path: Path, payload: dict[str, Any]) -> None:
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _find_progressive_video(
    source_root: Path,
    stem: str,
    output_view_index: int,
    source_kind: str,
) -> Path | None:
    if source_kind == "stage2":
        return _find_reconstruction_video(source_root, stem, output_view_index)
    return _find_stage1_video(source_root, stem, output_view_index)


def _sync_progressive_outputs_once(
    *,
    rows: list[dict[str, Any]],
    source_root: Path,
    source_kind: str,
    args: argparse.Namespace,
    stable_state: dict[Path, tuple[int, float]],
    log_path: Path,
    require_stable: bool,
    stable_seconds: float = 8.0,
) -> int:
    copied = 0
    now = time.monotonic()
    for row in rows:
        output_path = Path(row["output_path"]).expanduser().resolve()
        if output_path.exists():
            continue
        stem = str(row["prediction_stem"])
        selected_video = _find_progressive_video(
            source_root,
            stem,
            args.output_view_index,
            source_kind,
        )
        if selected_video is None:
            continue
        try:
            size = selected_video.stat().st_size
        except FileNotFoundError:
            continue
        old_size, first_seen = stable_state.get(selected_video, (-1, now))
        if old_size != size:
            stable_state[selected_video] = (size, now)
            continue
        if require_stable and now - first_seen < stable_seconds:
            continue
        try:
            _copy_validated_video(
                selected_video,
                output_path,
                expected_frames=args.num_video_frames,
                expected_fps=args.fps,
                expected_width=args.width,
                expected_height=args.height,
            )
            copied += 1
            _append_progressive_sync_log(
                log_path,
                {
                    "event": "copied",
                    "source_kind": source_kind,
                    "source": str(selected_video),
                    "dest": str(output_path),
                    "size": size,
                },
            )
        except Exception as exc:
            _append_progressive_sync_log(
                log_path,
                {
                    "event": "copy_failed",
                    "source_kind": source_kind,
                    "source": str(selected_video),
                    "dest": str(output_path),
                    "error": str(exc),
                },
            )
    return copied


def _progressive_output_sync_loop(
    *,
    rows: list[dict[str, Any]],
    source_root: Path,
    source_kind: str,
    args: argparse.Namespace,
    stop_event: threading.Event,
    log_path: Path,
) -> None:
    stable_state: dict[Path, tuple[int, float]] = {}
    _append_progressive_sync_log(
        log_path,
        {"event": "start", "source_root": str(source_root), "source_kind": source_kind},
    )
    while not stop_event.is_set():
        copied = _sync_progressive_outputs_once(
            rows=rows,
            source_root=source_root,
            source_kind=source_kind,
            args=args,
            stable_state=stable_state,
            log_path=log_path,
            require_stable=True,
        )
        _append_progressive_sync_log(log_path, {"event": "heartbeat", "copied_this_round": copied})
        stop_event.wait(5.0)
    copied = _sync_progressive_outputs_once(
        rows=rows,
        source_root=source_root,
        source_kind=source_kind,
        args=args,
        stable_state=stable_state,
        log_path=log_path,
        require_stable=False,
    )
    _append_progressive_sync_log(log_path, {"event": "stop", "copied_this_round": copied})


def _copy_batch_stage2_outputs(
    *,
    rows: list[dict[str, Any]],
    output_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for row in rows:
        stem = str(row["prediction_stem"])
        selected_video = _find_reconstruction_video(output_root, stem, args.output_view_index)
        try:
            if selected_video is None:
                raise FileNotFoundError(f"Lyra-1 output video was not written for {stem}")
            _copy_validated_video(
                selected_video,
                Path(row["output_path"]).expanduser().resolve(),
                expected_frames=args.num_video_frames,
                expected_fps=args.fps,
                expected_width=args.width,
                expected_height=args.height,
            )
        except Exception as exc:
            _record_copy_error(row, exc)
            errors.append({"sample_id": str(row["sample_id"]), "error": str(exc)})
    return errors


def _iter_chunks(rows: list[dict[str, Any]], chunk_size: int) -> list[tuple[int, list[dict[str, Any]]]]:
    if int(chunk_size) <= 0:
        return [(0, rows)]
    safe_chunk_size = int(chunk_size)
    return [
        (chunk_index, rows[start : start + safe_chunk_size])
        for chunk_index, start in enumerate(range(0, len(rows), safe_chunk_size))
    ]


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("Lyra-1 batch runner requires checkpoint_dir")
    batch_spec_path = Path(args.batch_spec_path).expanduser().resolve()
    rows = _load_batch_spec(batch_spec_path)
    output_dirs = {Path(row["output_path"]).expanduser().resolve().parent for row in rows}
    if len(output_dirs) != 1:
        raise ValueError("Lyra-1 batch runner requires all outputs to share one directory")
    output_dir = next(iter(output_dirs))
    output_dir.mkdir(parents=True, exist_ok=True)
    static_view_indices = _parse_static_view_indices(args.static_view_indices_fixed)
    temp_parent = output_dir / "_tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="worldarena_lyra1_batch_", dir=str(temp_parent)) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        workspace_tmp_parent_raw = os.environ.get("WORLDARENA_LYRA1_WORKSPACE_TMPDIR")
        workspace_tmp_parent = (
            Path(workspace_tmp_parent_raw).expanduser().resolve()
            if workspace_tmp_parent_raw
            else None
        )
        if workspace_tmp_parent is not None:
            workspace_tmp_parent.mkdir(parents=True, exist_ok=True)
        workspace_root = Path(
            tempfile.mkdtemp(
                prefix="worldarena_lyra1_workspace_",
                dir=str(workspace_tmp_parent) if workspace_tmp_parent is not None else None,
            )
        )
        atexit.register(shutil.rmtree, workspace_root, ignore_errors=True)
        workspace_root.mkdir(parents=True, exist_ok=True)
        _prepare_transformer_engine_shim(workspace_root)
        _prepare_apex_amp_shim(workspace_root)
        (workspace_root / "cosmos_predict1").symlink_to(repo_root / "cosmos_predict1", target_is_directory=True)
        layout = _checkpoint_layout(checkpoint_dir)
        checkpoints_root = _prepare_checkpoint_root(workspace_root, layout)
        (workspace_root / "Ruicheng").mkdir(parents=True, exist_ok=True)
        (workspace_root / "Ruicheng" / "moge-vitl").symlink_to(
            layout["moge_dir"] / "model.pt",
            target_is_directory=False,
        )

        env = apply_checkpoint_env()
        pythonpath = [str(workspace_root), str(repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        errors: list[dict[str, str]] = []
        sync_log_path = output_dir / "final_recon_sync.log"
        for chunk_index, chunk_rows in _iter_chunks(rows, args.batch_chunk_size):
            chunk_label = f"chunk_{chunk_index:06d}"
            input_root = temp_dir / f"input_{chunk_label}"
            generated_root = temp_dir / f"generated_{chunk_label}"
            batch_input_path = temp_dir / f"stage1_batch_inputs_{chunk_label}.jsonl"
            output_root = temp_dir / f"recon_{chunk_label}"
            input_root.mkdir(parents=True, exist_ok=True)
            _append_progressive_sync_log(
                sync_log_path,
                {
                    "event": "chunk_start",
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_rows),
                    "first_sample_id": str(chunk_rows[0]["sample_id"]),
                    "stage1_num_gpus": _stage1_num_gpus(args),
                    "num_gpus": args.num_gpus,
                },
            )
            try:
                _write_stage1_batch_inputs(
                    rows=chunk_rows,
                    input_root=input_root,
                    batch_input_path=batch_input_path,
                )
                stage1_sync_event: threading.Event | None = None
                stage1_sync_thread: threading.Thread | None = None
                if args.skip_stage2:
                    stage1_sync_event = threading.Event()
                    stage1_sync_thread = threading.Thread(
                        target=_progressive_output_sync_loop,
                        kwargs={
                            "rows": chunk_rows,
                            "source_root": generated_root,
                            "source_kind": "stage1",
                            "args": args,
                            "stop_event": stage1_sync_event,
                            "log_path": sync_log_path,
                        },
                        daemon=True,
                    )
                    stage1_sync_thread.start()
                try:
                    _run_command(
                        _stage1_batch_command(
                            batch_input_path=batch_input_path,
                            generated_root=generated_root,
                            args=args,
                        ),
                        cwd=workspace_root,
                        env=env,
                    )
                finally:
                    if stage1_sync_event is not None:
                        stage1_sync_event.set()
                    if stage1_sync_thread is not None:
                        stage1_sync_thread.join(timeout=120.0)
                time.sleep(float(os.environ.get("WORLDARENA_LYRA1_POST_RUN_SLEEP_SECONDS", "5")))

                if args.skip_stage2:
                    errors.extend(
                        _copy_batch_stage1_outputs(
                            rows=chunk_rows,
                            generated_root=generated_root,
                            args=args,
                        )
                    )
                else:
                    dataset_name = f"worldarena_lyra1_batch_{batch_spec_path.stem}_{chunk_label}"
                    effective_view_indices = static_view_indices if args.multi_trajectory else ["0"]
                    with _temporary_dataset_registration(
                        repo_root=repo_root,
                        dataset_name=dataset_name,
                        root_path=_dataset_root(generated_root, multi_trajectory=args.multi_trajectory),
                        view_indices=effective_view_indices,
                        sample_stems=[str(row["prediction_stem"]) for row in chunk_rows],
                    ):
                        stop_sync_event = threading.Event()
                        sync_thread = threading.Thread(
                            target=_progressive_output_sync_loop,
                            kwargs={
                                "rows": chunk_rows,
                                "source_root": output_root,
                                "source_kind": "stage2",
                                "args": args,
                                "stop_event": stop_sync_event,
                                "log_path": sync_log_path,
                            },
                            daemon=True,
                        )
                        sync_thread.start()
                        try:
                            _run_stage2(
                                repo_root=repo_root,
                                workspace_root=workspace_root,
                                checkpoints_root=checkpoints_root,
                                dataset_name=dataset_name,
                                output_root=output_root,
                                target_index_subsample=args.target_index_subsample,
                                output_fps=args.fps,
                                static_view_indices_fixed=effective_view_indices,
                                num_test_images=len(chunk_rows),
                            )
                        finally:
                            stop_sync_event.set()
                            sync_thread.join(timeout=120.0)
                    errors.extend(
                        _copy_batch_stage2_outputs(
                            rows=chunk_rows,
                            output_root=output_root,
                            args=args,
                        )
                    )
            finally:
                _append_progressive_sync_log(
                    sync_log_path,
                    {
                        "event": "chunk_done",
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_rows),
                        "errors_so_far": len(errors),
                    },
                )
                shutil.rmtree(input_root, ignore_errors=True)
                shutil.rmtree(generated_root, ignore_errors=True)
                shutil.rmtree(output_root, ignore_errors=True)

    if errors:
        error_path = batch_spec_path.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
