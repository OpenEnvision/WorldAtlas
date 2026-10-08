from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from worldatlas.common.checkpoints import project_root as worldatlas_project_root
from worldatlas.models.adapters.batch_runner_common import (
    add_common_batch_args,
    copy_output,
    load_batch_spec,
    load_json,
    print_status,
)
from worldatlas.models.adapters.windowed_rollout import plan_sana_wm_windows, stitch_windowed_frames

# Stage-1 Sana loads Gemma by this Hub id. The weights already live in arena/ckpts
# as a plain snapshot, not under huggingface/hub, so from_pretrained would miss
# them and (with HF_HUB_OFFLINE=1) fail instead of using the local files.
SANA_STAGE1_TEXT_ENCODER_REPO = "Efficient-Large-Model/gemma-2-2b-it"
DEFAULT_TEXT_ENCODER_PATH = "../ckpts/Efficient-Large-Model--gemma-2-2b-it"

# Official inference_sana_wm.py setdefaults this before importing Sana nets.
# Our runner used to import diffusion.utils.logger first; diffusion/__init__
# pulls LongLiveFlowEuler → sana_blocks, which then enables xformers and dies
# on from_seqlens(mask: Tensor). Force the flag before any Sana import.
os.environ["DISABLE_XFORMERS"] = "1"


def disable_sana_xformers() -> None:
    """Keep Sana on PyTorch SDPA. Newer xformers rejects a Tensor caption mask."""
    os.environ["DISABLE_XFORMERS"] = "1"


def force_disable_loaded_sana_xformers() -> int:
    """Flip ``_xformers_available`` on Sana modules already imported too early."""
    patched = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith("diffusion.model.nets"):
            continue
        if getattr(module, "_xformers_available", False):
            module._xformers_available = False
            patched += 1
    return patched


def pin_hf_from_pretrained(cls: Any, repo_id: str, local_dir: Path) -> None:
    """Rewrite ``cls.from_pretrained(repo_id)`` to a local directory, offline-only."""
    original = cls.from_pretrained

    def _from_pretrained(inner_cls, pretrained_model_name_or_path, *args, **kwargs):
        del inner_cls
        if str(pretrained_model_name_or_path) == repo_id:
            pretrained_model_name_or_path = str(local_dir)
            kwargs.setdefault("local_files_only", True)
        return original(pretrained_model_name_or_path, *args, **kwargs)

    cls.from_pretrained = classmethod(_from_pretrained)


def _resolve_existing(value: str | None, bases: list[Path]) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Sana-WM batch runner.")
    add_common_batch_args(parser)
    return parser.parse_args()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_camera_receipt(row, generation, *, model_path, seed, windows, video_shape, camera_hashes):
    annotation = Path(row["annotation_path"])
    for name, digest in camera_hashes.items():
        if file_sha(annotation / name) != digest:
            raise ValueError(f"Sana camera changed during inference: {annotation / name}")
    output = Path(row["output_path"]).resolve()
    receipt = dict(sample_id=row["sample_id"], output_path=str(output), output_sha256=file_sha(output),
        prompt=row["prompt"], model_path=str(Path(model_path).resolve()), generation=generation,
        seed=seed, window_seeds=[seed + i for i in range(len(windows))], windows=windows,
        output_shape=list(video_shape), fps=int(generation.get("fps", 16)),
        annotation_path=str(annotation), camera_sha256=camera_hashes,
        camera_control_recipe=row.get("camera_control_recipe"), pose_source=row["pose_source"],
        output_pose_offset=0 if generation.get("no_refiner", False) else 1)
    destination = output.with_suffix(".sana_wm.json")
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex)
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(destination)
    return receipt


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Current Sana checkout keeps the WM entrypoint under inference_video_scripts/wm/,
    # not at inference_video_scripts/inference_sana_wm.py.
    wm_dir = repo_root / "inference_video_scripts" / "wm"
    if str(wm_dir) not in sys.path:
        sys.path.insert(0, str(wm_dir))
    rows = load_batch_spec(Path(args.batch_spec_path).expanduser().resolve())
    generation = load_json(Path(args.generation_config_path).expanduser().resolve())

    disable_sana_xformers()
    import inference_sana_wm as sana_wm
    force_disable_loaded_sana_xformers()
    import pyrallis
    import torch
    from PIL import Image
    from diffusion.utils.logger import get_root_logger

    arena_root = worldatlas_project_root()
    search_bases = [arena_root, arena_root.parent, repo_root]
    config_path = _resolve_existing(str(generation.get("config") or ""), search_bases)
    model_path = _resolve_existing(str(generation.get("model_path") or ""), search_bases)
    refiner_root = _resolve_existing(str(generation.get("refiner_root") or ""), search_bases)
    refiner_gemma_root = _resolve_existing(
        str(generation.get("refiner_gemma_root") or ""), search_bases
    )
    text_encoder_path = _resolve_existing(
        str(generation.get("text_encoder_path") or DEFAULT_TEXT_ENCODER_PATH),
        search_bases,
    )
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"SANA-WM config.yaml not found: {generation.get('config')}")
    if model_path is None or not model_path.is_file():
        raise FileNotFoundError(f"SANA-WM DiT weights not found: {generation.get('model_path')}")
    if text_encoder_path is None or not (text_encoder_path / "config.json").is_file():
        raise FileNotFoundError(
            "SANA-WM stage-1 text encoder not found: "
            f"{generation.get('text_encoder_path') or DEFAULT_TEXT_ENCODER_PATH}"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    pin_hf_from_pretrained(AutoTokenizer, SANA_STAGE1_TEXT_ENCODER_REPO, text_encoder_path)
    pin_hf_from_pretrained(AutoModelForCausalLM, SANA_STAGE1_TEXT_ENCODER_REPO, text_encoder_path)

    logger = get_root_logger()
    device = torch.device(str(generation.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    config = pyrallis.parse(
        config_class=sana_wm.InferenceConfig,
        config_path=str(config_path),
        args=[],
    )
    release_root = model_path.parent.parent
    if (release_root / "vae").is_dir():
        config.vae.vae_pretrained = str(release_root)
    no_refiner = bool(generation.get("no_refiner", False))
    refiner = None
    if not no_refiner:
        if refiner_root is None or not refiner_root.is_dir():
            raise FileNotFoundError(f"SANA-WM refiner not found: {generation.get('refiner_root')}")
        refiner = sana_wm.RefinerSettings(
            root=str(refiner_root),
            gemma_root=str(refiner_gemma_root or (refiner_root / "text_encoder")),
            sink_size=int(generation.get("sink_size", 1)),
            seed=int(generation.get("refiner_seed", 42)),
        )
    pipeline = sana_wm.SanaWMPipeline(
        config=config,
        model_path=str(model_path),
        device=device,
        refiner=refiner,
        offload_vae=bool(generation.get("offload_vae", False)),
        offload_refiner=bool(generation.get("offload_refiner", False)),
        logger=logger,
    )

    requested_frames = int(generation.get("num_frames", 161))
    chunk_num_frames = int(generation.get("chunk_num_frames", 161))
    fps = int(generation.get("fps", 16))
    failed = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        try:
            annotation_path = Path(str(row["annotation_path"])).expanduser().resolve()
            record_camera = bool(row.get("camera_control_recipe") or row.get("record_camera_receipt"))
            camera_hashes = ({name: file_sha(annotation_path / name) for name in ("poses.npy", "intrinsics.npy")}
                             if record_camera else {})
            sample_seed = int(row.get("seed", int(generation.get("seed", 42)) + row_index))
            c2w_full = np.load(annotation_path / "poses.npy").astype(np.float32)
            num_frames = min(requested_frames, int(c2w_full.shape[0]))
            num_frames = sana_wm._snap_num_frames(num_frames, stride=8, upper_bound=int(c2w_full.shape[0]))
            windows = plan_sana_wm_windows(num_frames, chunk_frames=chunk_num_frames)
            image = Image.open(str(row["conditioning_image"])).convert("RGB")
            condition_image = image.convert("RGB")
            intr_src = sana_wm.load_intrinsics(annotation_path / "intrinsics.npy", num_frames)
            chunk_videos: list[np.ndarray] = []
            chunk_c2w: list[np.ndarray] = []
            for window_index, (start, length) in enumerate(windows):
                c2w = c2w_full[start : start + length]
                intr_window = np.asarray(intr_src[start : start + length], dtype=np.float32)
                cropped, src_size, resized_size, crop_offset = sana_wm.resize_and_center_crop(
                    condition_image
                )
                intrinsics_vec4 = sana_wm.transform_intrinsics_for_crop(
                    intr_window,
                    src_size,
                    resized_size,
                    crop_offset,
                )
                params = sana_wm.GenerationParams(
                    num_frames=length,
                    fps=fps,
                    step=int(generation.get("step", 60)),
                    cfg_scale=float(generation.get("cfg_scale", 5.0)),
                    flow_shift=generation.get("flow_shift"),
                    seed=sample_seed + window_index,
                    negative_prompt=str(generation.get("negative_prompt", "")),
                    sampling_algo=str(generation.get("sampling_algo", "flow_euler_ltx")),
                )
                out = pipeline.generate(cropped, str(row["prompt"]), c2w, intrinsics_vec4, params)
                video_hwc = np.asarray(out["video"])
                chunk_videos.append(video_hwc)
                chunk_c2w.append(np.asarray(out["c2w"]))
                condition_image = Image.fromarray(video_hwc[-1]).convert("RGB")
                logger.info(
                    "SANA-WM window %s/%s sample=%s start=%s length=%s wrote=%s",
                    window_index + 1,
                    len(windows),
                    sample_id,
                    start,
                    length,
                    int(video_hwc.shape[0]),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pose_offset = 1 if refiner is not None else 0
            video_hwc = stitch_windowed_frames(
                chunk_videos,
                windows,
                pose_offset=pose_offset,
                target_frames=num_frames,
            )
            if not bool(generation.get("no_action_overlay", True)):
                video_c2w = stitch_windowed_frames(
                    chunk_c2w,
                    windows,
                    pose_offset=pose_offset,
                    target_frames=int(video_hwc.shape[0]),
                )
                video_hwc = sana_wm.apply_overlay(video_hwc, video_c2w)
            temp_dir = Path(str(row["output_path"])).expanduser().resolve().parent / "_sana_wm_outputs"
            temp_path = sana_wm.write_video(temp_dir, str(row["prediction_stem"]), video_hwc, fps, logger)
            copy_output(temp_path, Path(str(row["output_path"])))
            if record_camera:
                write_camera_receipt(row, generation, model_path=model_path, seed=sample_seed,
                    windows=windows, video_shape=video_hwc.shape, camera_hashes=camera_hashes)
            print_status(
                sample_id,
                "generated",
                output_path=str(row["output_path"]),
                num_frames=int(video_hwc.shape[0]),
                fps=fps,
                windows=len(windows),
            )
        except Exception as exc:
            failed += 1
            message = str(exc).strip() or repr(exc)
            print_status(
                sample_id,
                "failed",
                error=f"{type(exc).__name__}: {message}",
                traceback=traceback.format_exc(),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
