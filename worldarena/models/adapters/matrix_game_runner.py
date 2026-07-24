"""Subprocess router for Matrix-Game inference (delegates to variant runners)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _install_config_aliases(config_module, alias_map: dict[str, str]) -> None:
    config_class = type(config_module)

    if getattr(config_class, "_worldarena_alias_patch_installed", False):
        return

    original_setattr = getattr(config_class, "__setattr__", object.__setattr__)
    original_getattr = getattr(config_class, "__getattr__", None)

    def compat_setattr(self, name, value):
        target = alias_map.get(name)
        if target and target in self._config:
            allowed_keys = getattr(self, "_allowed_keys", None)
            if allowed_keys is not None:
                allowed_keys.add(name)
            self._config[name] = value
            self._config[target] = value
            return
        return original_setattr(self, name, value)

    def compat_getattr(self, name):
        target = alias_map.get(name)
        if target and target in self._config:
            return self._config.get(name, self._config[target])
        if original_getattr is not None:
            return original_getattr(self, name)
        raise AttributeError(f"{type(self).__name__}.{name} does not exist")

    config_class.__setattr__ = compat_setattr
    config_class.__getattr__ = compat_getattr
    config_class._worldarena_alias_patch_installed = True

    allowed_keys = getattr(config_module, "_allowed_keys", None)
    for alias, target in alias_map.items():
        if target in config_module._config:
            if allowed_keys is not None:
                allowed_keys.add(alias)
            config_module._config.setdefault(alias, config_module._config[target])


def patch_torch_dynamo_config_aliases() -> None:
    """Patch torch dynamo config aliases."""
    import torch._dynamo

    alias_map = {
        "recompile_limit": "cache_size_limit",
        "accumulated_recompile_limit": "accumulated_cache_size_limit",
    }
    config_module = torch._dynamo.config
    if all(alias in getattr(config_module, "_config", {}) for alias in alias_map):
        return
    _install_config_aliases(config_module, alias_map)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matrix-Game subprocess runner with explicit actions.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--actions_json", required=True, type=str)
    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--save_name", required=True, type=str)
    parser.add_argument("--size", default="704*1280", type=str)
    parser.add_argument("--fps", default=17.0, type=float)
    parser.add_argument("--save_num_frames", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_iterations", default=1, type=int)
    parser.add_argument("--num_inference_steps", default=3, type=int)
    parser.add_argument("--sample_shift", default=None, type=float)
    parser.add_argument("--sample_guide_scale", default=5.0, type=float)
    parser.add_argument("--ulysses_size", default=1, type=int)
    parser.add_argument("--vae_type", default="mg_lightvae_v2", type=str)
    parser.add_argument("--lightvae_pruning_rate", default=None, type=float)
    parser.add_argument("--fa_version", default=None, type=str, choices=["0", "2", "3"])
    parser.add_argument("--visualize_ops", action="store_true")
    parser.add_argument("--use_base_model", action="store_true")
    parser.add_argument("--use_int8", action="store_true")
    parser.add_argument("--verify_quant", action="store_true")
    parser.add_argument("--use_async_vae", action="store_true")
    parser.add_argument("--async_vae_warmup_iters", default=0, type=int)
    parser.add_argument("--compile_vae", action="store_true")
    parser.add_argument("--t5_fsdp", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--dit_fsdp", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true")
    return parser.parse_args()


def normalize_to_neg_one_to_one(x):
    return 2.0 * x - 1.0


def build_runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ckpt_dir=args.ckpt_dir,
        size=args.size,
        save_name=args.save_name,
        seed=args.seed,
        num_iterations=args.num_iterations,
        output_dir=args.output_dir,
        save_num_frames=args.save_num_frames,
        sample_shift=args.sample_shift,
        sample_guide_scale=args.sample_guide_scale,
        num_inference_steps=args.num_inference_steps,
        ulysses_size=args.ulysses_size,
        t5_fsdp=args.t5_fsdp,
        t5_cpu=args.t5_cpu,
        dit_fsdp=args.dit_fsdp,
        convert_model_dtype=args.convert_model_dtype,
        use_base_model=args.use_base_model,
        use_int8=args.use_int8,
        verify_quant=args.verify_quant,
        vae_type=args.vae_type,
        lightvae_pruning_rate=args.lightvae_pruning_rate,
        use_async_vae=args.use_async_vae,
        async_vae_warmup_iters=args.async_vae_warmup_iters,
        compile_vae=args.compile_vae,
        fa_version=args.fa_version,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    patch_torch_dynamo_config_aliases()
    import imageio
    import torch

    with open(args.actions_json, "r", encoding="utf-8") as file:
        action_spec = json.load(file)
    keyboard_condition_np = np.asarray(action_spec["keyboard_condition"], dtype=np.float32)
    mouse_condition_np = np.asarray(action_spec["mouse_condition"], dtype=np.float32)

    from PIL import Image

    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    import pipeline.inference_pipeline as mg_inference
    from utils.cam_utils import get_extrinsics
    from utils.transform import get_video_transform
    from utils.utils import compute_all_poses_from_actions
    import utils.visualize as visualize_utils

    def custom_get_data(num_frames, height, width, pil_image, device=None, dtype=None):
        if keyboard_condition_np.shape[0] != num_frames or mouse_condition_np.shape[0] != num_frames:
            raise ValueError(
                "Custom action sequence length mismatch: "
                f"expected {num_frames}, got keyboard={keyboard_condition_np.shape[0]}, "
                f"mouse={mouse_condition_np.shape[0]}"
            )

        input_image = torch.from_numpy(np.array(pil_image)).unsqueeze(0)
        input_image = input_image.permute(0, 3, 1, 2)
        transform = get_video_transform(height, width, normalize_to_neg_one_to_one)
        input_image = transform(input_image)
        input_image = input_image.transpose(0, 1).unsqueeze(0)

        first_pose = np.zeros(5, dtype=np.float32)
        all_poses = compute_all_poses_from_actions(
            keyboard_condition_np,
            mouse_condition_np,
            first_pose=first_pose,
        )
        positions = all_poses[:, :3].tolist()
        rotations = np.concatenate(
            [
                np.zeros((all_poses.shape[0], 1), dtype=np.float32),
                all_poses[:, 3:5],
            ],
            axis=1,
        ).tolist()
        extrinsics_all = get_extrinsics(rotations, positions)

        keyboard_tensor = torch.tensor(keyboard_condition_np, dtype=dtype, device=device).unsqueeze(0)
        mouse_tensor = torch.tensor(mouse_condition_np, dtype=dtype, device=device).unsqueeze(0)
        return (
            input_image.to(device, dtype),
            extrinsics_all,
            keyboard_tensor,
            mouse_tensor,
        )

    def patched_process_video(
        input_video,
        output_video,
        config,
        mouse_icon_path,
        mouse_scale=1.0,
        mouse_rotation=0,
        default_frame_res=(704, 1280),
    ):
        output_path = Path(output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.visualize_ops:
            return visualize_utils.process_video(
                input_video,
                output_video,
                config,
                mouse_icon_path,
                mouse_scale=mouse_scale,
                mouse_rotation=mouse_rotation,
                default_frame_res=default_frame_res,
            )
        frames = input_video
        if args.save_num_frames:
            if args.save_num_frames > len(frames):
                raise ValueError(
                    f"save_num_frames={args.save_num_frames} exceeds generated frame count {len(frames)}"
                )
            frames = frames[: args.save_num_frames]
        imageio.mimsave(str(output_path), np.ascontiguousarray(frames), fps=args.fps)

    mg_inference.get_data = custom_get_data
    mg_inference.process_video = patched_process_video

    runtime_args = build_runtime_args(args)
    cfg = WAN_CONFIGS["matrix_game3"]
    if runtime_args.sample_shift is None:
        runtime_args.sample_shift = cfg.sample_shift
    if runtime_args.sample_guide_scale is None:
        runtime_args.sample_guide_scale = cfg.sample_guide_scale

    pipeline = mg_inference.MatrixGame3Pipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=runtime_args.t5_fsdp,
        dit_fsdp=runtime_args.dit_fsdp,
        use_sp=(runtime_args.ulysses_size > 1),
        t5_cpu=runtime_args.t5_cpu,
        convert_model_dtype=runtime_args.convert_model_dtype,
        args=runtime_args,
        fa_version=runtime_args.fa_version,
        use_base_model=runtime_args.use_base_model,
    )

    pil_image = Image.open(args.image_path).convert("RGB")
    pipeline.generate(
        args.prompt,
        pil_image,
        max_area=MAX_AREA_CONFIGS[args.size],
        shift=runtime_args.sample_shift,
        num_inference_steps=runtime_args.num_inference_steps,
        guide_scale=runtime_args.sample_guide_scale,
        seed=runtime_args.seed,
        use_base_model=runtime_args.use_base_model,
        args=runtime_args,
    )


if __name__ == "__main__":
    main()
