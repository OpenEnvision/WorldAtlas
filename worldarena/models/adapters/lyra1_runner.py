from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Iterator

import cv2

from worldarena.common.checkpoints import apply_checkpoint_env, hf_local_dir, resolve_checkpoint_path
from worldarena.models.adapters.lyra2_runner import _prepare_transformer_engine_shim


DEFAULT_STATIC_VIEW_INDICES = ("5", "0", "1", "2", "3", "4")
APEX_INIT_SHIM = "from __future__ import annotations\n"
APEX_MULTI_TENSOR_APPLY_SHIM = textwrap.dedent(
    """
    from __future__ import annotations


    class MultiTensorApply:
        available = True

        def __init__(self, chunk_size: int = 2048 * 32) -> None:
            self.chunk_size = int(chunk_size)

        def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
            return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


    multi_tensor_applier = MultiTensorApply()
    """
)
AMP_C_SHIM = textwrap.dedent(
    """
    from __future__ import annotations

    import torch


    def _default_device(noop_flag):
        if isinstance(noop_flag, torch.Tensor):
            return noop_flag.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor=False, *args):
        del chunk_size, args
        tensors = tensor_lists[0] if tensor_lists else []
        device = _default_device(noop_flag)
        total_sq = torch.zeros((), dtype=torch.float32, device=device)
        per_tensor_norms = []
        for tensor in tensors:
            if tensor is None:
                continue
            current = torch.linalg.vector_norm(tensor.detach().float())
            current = current.to(device=device, dtype=torch.float32)
            if per_tensor:
                per_tensor_norms.append(current)
            total_sq = total_sq + current * current
        total_norm = torch.sqrt(total_sq)
        if per_tensor:
            values = torch.stack(per_tensor_norms) if per_tensor_norms else torch.empty(0, device=device)
            return total_norm, values
        return total_norm, None


    def multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
        del chunk_size, noop_flag
        sources = tensor_lists[0] if tensor_lists else []
        targets = tensor_lists[1] if len(tensor_lists) > 1 else sources
        for source, target in zip(sources, targets):
            target.copy_(source * scale)
        return None


    def _unavailable_adam(*args, **kwargs):
        raise RuntimeError("WorldArena Lyra-1 inference shim does not implement Apex fused Adam.")


    multi_tensor_adam = _unavailable_adam
    multi_tensor_adam_capturable = _unavailable_adam
    multi_tensor_adam_capturable_master = _unavailable_adam
    """
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Lyra-1 subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--conditioning_image", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--sample_name", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
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
    parser.add_argument("--target_index_subsample", default=4, type=int)
    parser.add_argument("--output_view_index", default=0, type=int)
    parser.add_argument("--negative_prompt", default=None, type=str)
    parser.add_argument("--static_view_indices_fixed", default=",".join(DEFAULT_STATIC_VIEW_INDICES), type=str)
    parser.add_argument("--foreground_masking", action="store_true")
    parser.add_argument("--multi_trajectory", action="store_true")
    parser.add_argument("--offload_diffusion_transformer", action="store_true")
    parser.add_argument("--offload_tokenizer", action="store_true")
    parser.add_argument("--offload_text_encoder_model", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    return parser.parse_args()


def _checkpoint_layout(checkpoint_dir: Path) -> dict[str, Path]:
    resolved = checkpoint_dir.expanduser().resolve()
    lyra_dir = resolved
    gen3c_dir = hf_local_dir("nvidia/GEN3C-Cosmos-7B", required=True)
    tokenizer_dir = hf_local_dir("nvidia/Cosmos-Tokenize1-CV8x8x8-720p", required=True)
    t5_dir = hf_local_dir("google-t5/t5-11b", required=True)
    moge_dir = hf_local_dir("Ruicheng/moge-vitl", required=True)

    missing: list[str] = []
    if not (lyra_dir / "lyra_static.pt").is_file():
        missing.append("Lyra/lyra_static.pt")
    if not (gen3c_dir / "model.pt").is_file():
        missing.append("Gen3C-Cosmos-7B/model.pt")
    if not all((tokenizer_dir / name).exists() for name in ("encoder.jit", "decoder.jit", "mean_std.pt", "image_mean_std.pt")):
        missing.append(
            "Cosmos-Tokenize1-CV8x8x8-720p/{encoder.jit,decoder.jit,mean_std.pt,image_mean_std.pt}"
        )
    if not all((t5_dir / name).exists() for name in ("config.json", "spiece.model")):
        missing.append("google-t5/t5-11b/{config.json,spiece.model}")
    if not (moge_dir / "model.pt").is_file():
        missing.append("Ruicheng/moge-vitl/model.pt")
    if missing:
        raise FileNotFoundError(
            "Lyra-1 checkpoint dependencies are incomplete. Missing: " + ", ".join(missing)
        )

    return {
        "lyra_dir": lyra_dir,
        "gen3c_dir": gen3c_dir,
        "tokenizer_dir": tokenizer_dir,
        "t5_dir": t5_dir,
        "moge_dir": moge_dir,
    }


def _prepare_checkpoint_root(target_root: Path, layout: dict[str, Path]) -> Path:
    checkpoints_root = target_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    (checkpoints_root / "Lyra").symlink_to(layout["lyra_dir"], target_is_directory=True)
    (checkpoints_root / "Gen3C-Cosmos-7B").symlink_to(layout["gen3c_dir"], target_is_directory=True)
    (checkpoints_root / "Cosmos-Tokenize1-CV8x8x8-720p").symlink_to(
        layout["tokenizer_dir"],
        target_is_directory=True,
    )
    cosmos_predict1_root = checkpoints_root / "cosmos_predict1"
    cosmos_predict1_root.mkdir(parents=True, exist_ok=True)
    (cosmos_predict1_root / "Cosmos-Tokenize1-CV8x8x8-720p").symlink_to(
        layout["tokenizer_dir"],
        target_is_directory=True,
    )
    google_t5_root = checkpoints_root / "google-t5"
    google_t5_root.mkdir(parents=True, exist_ok=True)
    (google_t5_root / "t5-11b").symlink_to(layout["t5_dir"], target_is_directory=True)
    return checkpoints_root


def _prepare_apex_amp_shim(workspace_root: Path) -> None:
    apex_root = workspace_root / "apex"
    apex_root.mkdir(parents=True, exist_ok=True)
    (apex_root / "__init__.py").write_text(APEX_INIT_SHIM, encoding="utf-8")
    (apex_root / "multi_tensor_apply.py").write_text(
        APEX_MULTI_TENSOR_APPLY_SHIM,
        encoding="utf-8",
    )
    (workspace_root / "amp_C.py").write_text(AMP_C_SHIM, encoding="utf-8")


def _parse_static_view_indices(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(DEFAULT_STATIC_VIEW_INDICES)


def _static_dataset_entry(*, root_path: Path, view_indices: list[str]) -> dict[str, object]:
    return {
        "cls": None,
        "kwargs": {
            "root_path": str(root_path),
            "is_static": True,
            "is_multi_view": True,
            "has_latents": True,
            "is_generated_cosmos_latent": True,
            "sampling_buckets": [[value] for value in view_indices],
            "start_view_idx": 0,
        },
        "scene_scale": 1.0,
        "max_gap": 121,
        "min_gap": 45,
    }


@contextlib.contextmanager
def _temporary_dataset_registration(
    *,
    repo_root: Path,
    dataset_name: str,
    root_path: Path,
    view_indices: list[str],
) -> Iterator[None]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.models.data.radym_wrapper import RadymWrapper
    from src.models.data.registry import dataset_registry

    existing = dataset_registry.get(dataset_name)
    dataset_entry = _static_dataset_entry(root_path=root_path, view_indices=view_indices)
    dataset_entry["cls"] = RadymWrapper
    dataset_registry[dataset_name] = dataset_entry
    try:
        yield
    finally:
        if existing is None:
            dataset_registry.pop(dataset_name, None)
        else:
            dataset_registry[dataset_name] = existing


def _stage1_command(
    *,
    input_image_path: Path,
    generated_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    script_and_args = [
        "cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py",
        "--checkpoint_dir",
        "checkpoints",
        "--num_gpus",
        str(args.num_gpus),
        "--input_image_path",
        str(input_image_path),
        "--video_save_folder",
        str(generated_root),
        "--prompt",
        args.prompt,
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
    ]
    command = [sys.executable]
    if args.num_gpus > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes",
                "1",
                "--nproc_per_node",
                str(args.num_gpus),
            ]
        )
    command.extend(script_and_args)
    if args.negative_prompt:
        command.extend(["--negative_prompt", args.negative_prompt])
    for flag_name in (
        "foreground_masking",
        "multi_trajectory",
        "offload_diffusion_transformer",
        "offload_tokenizer",
        "offload_text_encoder_model",
    ):
        if getattr(args, flag_name):
            command.append(f"--{flag_name}")
    return command


def _run_stage2(
    *,
    repo_root: Path,
    workspace_root: Path,
    checkpoints_root: Path,
    dataset_name: str,
    output_root: Path,
    target_index_subsample: int,
    output_fps: int,
    static_view_indices_fixed: list[str],
) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from sample import main as sample_main
    from src.models.utils.misc import load_and_merge_configs

    demo_config = load_and_merge_configs(
        [
            str((repo_root / "configs" / "inference" / "default.yaml").resolve()),
            str((repo_root / "configs" / "demo" / "lyra_static.yaml").resolve()),
        ]
    )
    demo_config.dataset_name = dataset_name
    demo_config.out_dir_inference = str(output_root)
    demo_config.config_path = [
        str((repo_root / "configs" / "training" / "default.yaml").resolve()),
        str((repo_root / "configs" / "training" / "3dgs_res_704_1280_views_121_multi_6_prune.yaml").resolve()),
    ]
    demo_config.ckpt_path = str((checkpoints_root / "Lyra" / "lyra_static.pt").resolve())
    demo_config.do_eval = True
    demo_config.skip_existing = False
    demo_config.num_test_images = 1
    demo_config.use_depth = False
    demo_config.out_fps = output_fps
    demo_config.target_index_subsample = target_index_subsample
    demo_config.static_view_indices_fixed = static_view_indices_fixed
    demo_config.save_grid = False
    demo_config.save_gt_input = False
    demo_config.save_video_input = False
    demo_config.save_gt_depth = False
    demo_config.save_rgb_decoding = False
    demo_config.save_gaussians = False
    demo_config.save_gaussians_orig = False

    previous_cwd = Path.cwd()
    os.chdir(workspace_root)
    try:
        sample_main(demo_config)
    finally:
        os.chdir(previous_cwd)


def _dataset_root(generated_root: Path, *, multi_trajectory: bool) -> Path:
    if multi_trajectory:
        return generated_root

    wrapped_root = generated_root.parent / f"{generated_root.name}_wrapped"
    wrapped_camera_root = wrapped_root / "0"
    wrapped_camera_root.mkdir(parents=True, exist_ok=True)
    for name in ("rgb", "intrinsics", "latent", "pose"):
        source = generated_root / name
        if source.exists():
            (wrapped_camera_root / name).symlink_to(source, target_is_directory=True)
    return wrapped_root


def _terminate_process_group(process_group_id: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process_group_id, sig)
        except ProcessLookupError:
            return
        time.sleep(2.0)


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        start_new_session=True,
    )
    try:
        return_code = process.wait()
    except BaseException:
        _terminate_process_group(process.pid)
        raise
    if return_code != 0:
        _terminate_process_group(process.pid)
        raise subprocess.CalledProcessError(return_code, command)


def _validate_video(
    video_path: Path,
    *,
    expected_frames: int,
    expected_fps: int,
    expected_width: int,
    expected_height: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Lyra-1 output video cannot be opened: {video_path}")
    declared_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    read_frames = 0
    decoded_width: int | None = None
    decoded_height: int | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            read_frames += 1
            frame_height, frame_width = frame.shape[:2]
            if decoded_width is None:
                decoded_width = int(frame_width)
                decoded_height = int(frame_height)
            elif decoded_width != frame_width or decoded_height != frame_height:
                raise RuntimeError(
                    f"Lyra-1 output video has inconsistent frame sizes: {video_path}"
                )
    finally:
        capture.release()

    if read_frames != expected_frames:
        raise RuntimeError(
            f"Lyra-1 output video decoded {read_frames} frames, expected {expected_frames}: {video_path}"
        )
    if decoded_height != expected_height:
        raise RuntimeError(
            f"Lyra-1 output video height is {decoded_height}, expected {expected_height}: {video_path}"
        )
    if decoded_width is None or decoded_width < expected_width:
        raise RuntimeError(
            f"Lyra-1 output video width is {decoded_width}, expected at least {expected_width}: {video_path}"
        )
    if declared_fps > 0 and abs(declared_fps - float(expected_fps)) > 0.5:
        raise RuntimeError(
            f"Lyra-1 output video fps is {declared_fps:.3f}, expected {expected_fps}: {video_path}"
        )


def _copy_validated_video(
    selected_video: Path,
    output_path: Path,
    *,
    expected_frames: int,
    expected_fps: int,
    expected_width: int,
    expected_height: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    if temp_output.exists():
        temp_output.unlink()
    try:
        shutil.copy2(selected_video, temp_output)
        _validate_video(
            temp_output,
            expected_frames=expected_frames,
            expected_fps=expected_fps,
            expected_width=expected_width,
            expected_height=expected_height,
        )
        temp_output.replace(output_path)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise


def _copy_stage1_video(
    generated_root: Path,
    output_path: Path,
    *,
    output_view_index: int,
    expected_frames: int,
    expected_fps: int,
    expected_width: int,
    expected_height: int,
) -> None:
    direct_candidates = sorted((generated_root / "rgb").glob("*.mp4"))
    selected_video: Path | None = direct_candidates[0] if direct_candidates else None
    if selected_video is None:
        for candidate in sorted(generated_root.rglob("*.mp4")):
            if candidate.parent.name == "rgb" and candidate.parent.parent.name == str(output_view_index):
                selected_video = candidate
                break
    if selected_video is None:
        video_candidates = sorted(generated_root.rglob("*.mp4"))
        if video_candidates:
            selected_video = video_candidates[0]
    if selected_video is None:
        raise FileNotFoundError(f"Lyra-1 stage1 output video was not written under: {generated_root}")
    _copy_validated_video(
        selected_video,
        output_path,
        expected_frames=expected_frames,
        expected_fps=expected_fps,
        expected_width=expected_width,
        expected_height=expected_height,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir, kind="dir", required=True)
    if checkpoint_dir is None:
        raise ValueError("Lyra-1 runner requires checkpoint_dir")
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    static_view_indices = _parse_static_view_indices(args.static_view_indices_fixed)

    with tempfile.TemporaryDirectory(prefix="worldarena_lyra1_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        workspace_root = temp_dir / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)
        _prepare_transformer_engine_shim(workspace_root)
        _prepare_apex_amp_shim(workspace_root)
        (workspace_root / "cosmos_predict1").symlink_to(repo_root / "cosmos_predict1", target_is_directory=True)
        layout = _checkpoint_layout(checkpoint_dir)
        checkpoints_root = _prepare_checkpoint_root(
            workspace_root,
            layout,
        )
        (workspace_root / "Ruicheng").mkdir(parents=True, exist_ok=True)
        (workspace_root / "Ruicheng" / "moge-vitl").symlink_to(
            layout["moge_dir"] / "model.pt",
            target_is_directory=False,
        )

        input_root = temp_dir / "input"
        input_root.mkdir(parents=True, exist_ok=True)
        conditioning_source = Path(args.conditioning_image).expanduser().resolve()
        conditioning_copy = input_root / f"{args.sample_name}{conditioning_source.suffix or '.png'}"
        shutil.copy2(conditioning_source, conditioning_copy)

        generated_root = temp_dir / "generated"
        env = apply_checkpoint_env()
        pythonpath = [str(workspace_root), str(repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        _run_command(
            _stage1_command(
                input_image_path=conditioning_copy,
                generated_root=generated_root,
                args=args,
            ),
            cwd=workspace_root,
            env=env,
        )
        time.sleep(float(os.environ.get("WORLDARENA_LYRA1_POST_RUN_SLEEP_SECONDS", "5")))
        if args.skip_stage2:
            _copy_stage1_video(
                generated_root,
                output_path,
                output_view_index=args.output_view_index,
                expected_frames=args.num_video_frames,
                expected_fps=args.fps,
                expected_width=args.width,
                expected_height=args.height,
            )
            return

        dataset_name = f"worldarena_lyra1_{args.sample_name}"
        output_root = temp_dir / "recon"
        effective_view_indices = static_view_indices if args.multi_trajectory else ["0"]
        with _temporary_dataset_registration(
            repo_root=repo_root,
            dataset_name=dataset_name,
            root_path=_dataset_root(generated_root, multi_trajectory=args.multi_trajectory),
            view_indices=effective_view_indices,
        ):
            _run_stage2(
                repo_root=repo_root,
                workspace_root=workspace_root,
                checkpoints_root=checkpoints_root,
                dataset_name=dataset_name,
                output_root=output_root,
                target_index_subsample=args.target_index_subsample,
                output_fps=args.fps,
                static_view_indices_fixed=effective_view_indices,
            )

        video_candidates = sorted(output_root.rglob(f"{args.sample_name}.mp4"))
        selected_video: Path | None = None
        for candidate in video_candidates:
            if candidate.parent.name == str(args.output_view_index):
                selected_video = candidate
                break
        if selected_video is None and video_candidates:
            selected_video = video_candidates[0]
        if selected_video is None:
            raise FileNotFoundError(f"Lyra-1 output video was not written under: {output_root}")
        _copy_validated_video(
            selected_video,
            output_path,
            expected_frames=args.num_video_frames,
            expected_fps=args.fps,
            expected_width=args.width,
            expected_height=args.height,
        )


if __name__ == "__main__":
    main()
