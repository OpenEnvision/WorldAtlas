from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator


DEFAULT_STATIC_VIEW_INDICES = ("5", "0", "1", "2", "3", "4")


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
    return parser.parse_args()


def _existing_dir(candidates: list[Path], *, required_files: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir() and all((resolved / name).exists() for name in required_files):
            return resolved
    return None


def _checkpoint_layout(checkpoint_dir: Path) -> dict[str, Path]:
    resolved = checkpoint_dir.expanduser().resolve()
    cache_root = resolved.parent

    lyra_dir = _existing_dir(
        [resolved / "Lyra", resolved, cache_root / "Lyra"],
        required_files=("lyra_static.pt",),
    )
    gen3c_dir = _existing_dir(
        [resolved / "Gen3C-Cosmos-7B", cache_root / "nvidia--GEN3C-Cosmos-7B", cache_root / "Gen3C-Cosmos-7B"],
        required_files=("model.pt",),
    )
    tokenizer_dir = _existing_dir(
        [
            resolved / "Cosmos-Tokenize1-CV8x8x8-720p",
            cache_root / "nvidia--Cosmos-Tokenize1-CV8x8x8-720p",
            cache_root / "Cosmos-Tokenize1-CV8x8x8-720p",
        ],
        required_files=("encoder.jit", "decoder.jit", "mean_std.pt"),
    )
    t5_dir = _existing_dir(
        [
            resolved / "google-t5" / "t5-11b",
            cache_root / "google-t5--t5-11b",
            cache_root / "google-t5" / "t5-11b",
        ],
        required_files=("config.json", "spiece.model"),
    )

    missing: list[str] = []
    if lyra_dir is None:
        missing.append("Lyra/lyra_static.pt")
    if gen3c_dir is None:
        missing.append("Gen3C-Cosmos-7B/model.pt")
    if tokenizer_dir is None:
        missing.append("Cosmos-Tokenize1-CV8x8x8-720p/{encoder.jit,decoder.jit,mean_std.pt}")
    if t5_dir is None:
        missing.append("google-t5/t5-11b/{config.json,spiece.model}")
    if missing:
        raise FileNotFoundError(
            "Lyra-1 checkpoint dependencies are incomplete. Missing: " + ", ".join(missing)
        )

    return {
        "lyra_dir": lyra_dir,
        "gen3c_dir": gen3c_dir,
        "tokenizer_dir": tokenizer_dir,
        "t5_dir": t5_dir,
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
    google_t5_root = checkpoints_root / "google-t5"
    google_t5_root.mkdir(parents=True, exist_ok=True)
    (google_t5_root / "t5-11b").symlink_to(layout["t5_dir"], target_is_directory=True)
    return checkpoints_root


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
    command = [
        sys.executable,
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


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    static_view_indices = _parse_static_view_indices(args.static_view_indices_fixed)

    with tempfile.TemporaryDirectory(prefix="worldarena_lyra1_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        workspace_root = temp_dir / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)
        (workspace_root / "cosmos_predict1").symlink_to(repo_root / "cosmos_predict1", target_is_directory=True)
        checkpoints_root = _prepare_checkpoint_root(
            workspace_root,
            _checkpoint_layout(checkpoint_dir),
        )

        input_root = temp_dir / "input"
        input_root.mkdir(parents=True, exist_ok=True)
        conditioning_source = Path(args.conditioning_image).expanduser().resolve()
        conditioning_copy = input_root / f"{args.sample_name}{conditioning_source.suffix or '.png'}"
        shutil.copy2(conditioning_source, conditioning_copy)

        generated_root = temp_dir / "generated"
        env = os.environ.copy()
        pythonpath = [str(workspace_root), str(repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        subprocess.run(
            _stage1_command(
                input_image_path=conditioning_copy,
                generated_root=generated_root,
                args=args,
            ),
            check=True,
            cwd=str(workspace_root),
            env=env,
        )

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
        shutil.copy2(selected_video, output_path)


if __name__ == "__main__":
    main()
