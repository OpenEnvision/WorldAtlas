from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from worldarena.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena Sana-WM batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())

    import pyrallis
    import torch
    from PIL import Image
    from diffusion.utils.logger import get_root_logger
    from inference_video_scripts import inference_sana_wm as sana_wm
    from sana.tools import resolve_hf_path

    logger = get_root_logger()
    device = torch.device(str(generation.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    config = pyrallis.parse(
        config_class=sana_wm.InferenceConfig,
        config_path=resolve_hf_path(str(generation.get("config", sana_wm.HF_DEFAULTS["config"]))),
        args=[],
    )
    no_refiner = bool(generation.get("no_refiner", False))
    refiner = None
    if not no_refiner:
        refiner = sana_wm.RefinerSettings(
            root=str(generation.get("refiner_root", sana_wm.HF_DEFAULTS["refiner_root"])),
            gemma_root=str(generation.get("refiner_gemma_root", sana_wm.HF_DEFAULTS["refiner_gemma_root"])),
            sink_size=int(generation.get("sink_size", 1)),
            seed=int(generation.get("refiner_seed", 42)),
        )
    pipeline = sana_wm.SanaWMPipeline(
        config=config,
        model_path=resolve_hf_path(str(generation.get("model_path", sana_wm.HF_DEFAULTS["model_path"]))),
        device=device,
        refiner=refiner,
        offload_vae=bool(generation.get("offload_vae", False)),
        offload_refiner=bool(generation.get("offload_refiner", False)),
        logger=logger,
    )

    requested_frames = int(generation.get("num_frames", 161))
    fps = int(generation.get("fps", 16))
    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            annotation_path = Path(str(row["annotation_path"])).expanduser().resolve()
            c2w_full = np.load(annotation_path / "poses.npy").astype(np.float32)
            num_frames = min(requested_frames, int(c2w_full.shape[0]))
            num_frames = sana_wm._snap_num_frames(num_frames, stride=8, upper_bound=int(c2w_full.shape[0]))
            c2w = c2w_full[:num_frames]
            intr_src = sana_wm.load_intrinsics(annotation_path / "intrinsics.npy", num_frames)
            image = Image.open(str(row["conditioning_image"])).convert("RGB")
            cropped, src_size, resized_size, crop_offset = sana_wm.resize_and_center_crop(image)
            intrinsics_vec4 = sana_wm.transform_intrinsics_for_crop(
                intr_src,
                src_size,
                resized_size,
                crop_offset,
            )
            params = sana_wm.GenerationParams(
                num_frames=num_frames,
                fps=fps,
                step=int(generation.get("step", 60)),
                cfg_scale=float(generation.get("cfg_scale", 5.0)),
                flow_shift=generation.get("flow_shift"),
                seed=int(generation.get("seed", 42)) + row_index,
                negative_prompt=str(generation.get("negative_prompt", "")),
                sampling_algo=str(generation.get("sampling_algo", "flow_euler_ltx")),
            )
            out = pipeline.generate(cropped, str(row["prompt"]), c2w, intrinsics_vec4, params)
            video_hwc = out["video"]
            if not bool(generation.get("no_action_overlay", True)):
                video_hwc = sana_wm.apply_overlay(video_hwc, out["c2w"])
            temp_dir = Path(str(row["output_path"])).expanduser().resolve().parent / "_sana_wm_outputs"
            temp_path = sana_wm.write_video(temp_dir, str(row["prediction_stem"]), video_hwc, fps, logger)
            copy_output(temp_path, Path(str(row["output_path"])))
            print_status(sample_id, "generated", output_path=str(row["output_path"]), num_frames=num_frames, fps=fps)
        except Exception as exc:
            failed += 1
            print_status(sample_id, "failed", error=str(exc), traceback=traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
