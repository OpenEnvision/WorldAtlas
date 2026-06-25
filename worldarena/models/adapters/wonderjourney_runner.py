from __future__ import annotations

import argparse
import importlib.util
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import yaml

from worldarena.common.checkpoints import apply_checkpoint_env


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldArena WonderJourney subprocess runner.")
    parser.add_argument("--batch_spec", default=None, type=str)
    parser.add_argument("--batch_results", default=None, type=str)
    parser.add_argument("--repo_root", default=None, type=str)
    parser.add_argument("--entrypoint", default="run.py", type=str)
    parser.add_argument("--conditioning_image", default=None, type=str)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--sample_name", default=None, type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--scene_name", default=None, type=str)
    parser.add_argument("--entities", nargs="*", default=None)
    parser.add_argument("--style_prompt", default="photorealistic", type=str)
    parser.add_argument("--background_prompt", default="", type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--midas_checkpoint", default=None, type=str)
    parser.add_argument("--stable_diffusion_checkpoint", default=None, type=str)
    parser.add_argument("--stable_diffusion_revision", default=None, type=str)
    parser.add_argument("--rotation_path", default=None, type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--frames", default=5, type=int)
    parser.add_argument("--num_scenes", default=1, type=int)
    parser.add_argument("--num_keyframes", default=2, type=int)
    parser.add_argument("--save_fps", default=10, type=int)
    parser.add_argument("--run_timeout", default=None, type=float)
    parser.add_argument("--depth_model", default="midas_v3.1", type=str)
    parser.add_argument("--camera_speed", default=0.0005, type=float)
    parser.add_argument("--rotation_range", default=0.0, type=float)
    parser.add_argument("--init_focal_length", default=500.0, type=float)
    parser.add_argument("--inpainting_resolution_gen", default=512, type=int)
    parser.add_argument("--inpainting_resolution_interp", default=512, type=int)
    parser.add_argument("--kf2_upsample_coef", default=4, type=int)
    parser.add_argument("--fg_depth_range", default=0.0015, type=float)
    parser.add_argument("--depth_shift", default=0.0001, type=float)
    parser.add_argument("--regenerate_times", default=3, type=int)
    parser.add_argument("--num_finetune_decoder_steps", default=100, type=int)
    parser.add_argument("--num_finetune_decoder_steps_interp", default=30, type=int)
    parser.add_argument("--camera_speed_multiplier_rotation", default=0.0, type=float)
    parser.add_argument("--use_gpt", default=False, type=_parse_bool)
    parser.add_argument("--debug", default=False, type=_parse_bool)
    parser.add_argument("--skip_interp", default=False, type=_parse_bool)
    parser.add_argument("--skip_gen", default=False, type=_parse_bool)
    parser.add_argument("--enable_regenerate", default=False, type=_parse_bool)
    parser.add_argument("--finetune_decoder_gen", default=True, type=_parse_bool)
    parser.add_argument("--finetune_decoder_interp", default=False, type=_parse_bool)
    parser.add_argument("--finetune_depth_model", default=True, type=_parse_bool)
    parser.add_argument("--keep_work_dir", default=False, type=_parse_bool)
    parser.add_argument("--archive_sidecars", default=True, type=_parse_bool)
    return parser.parse_args(argv)


@contextlib.contextmanager
def _work_dir(keep_work_dir: bool):
    if keep_work_dir:
        path = Path(tempfile.mkdtemp(prefix="worldarena_wonderjourney_"))
        try:
            yield path
        finally:
            pass
        return

    with tempfile.TemporaryDirectory(prefix="worldarena_wonderjourney_") as temp_dir_raw:
        yield Path(temp_dir_raw)


def _slugify(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "sample"


def _normalized_env(repo_root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = apply_checkpoint_env(base_env or os.environ.copy())
    pythonpaths = [str(repo_root), str(repo_root / "midas_module")]
    if env.get("PYTHONPATH"):
        pythonpaths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault(
        "WONDERJOURNEY_SAM_CHECKPOINT",
        "/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt/WonderJourney/sam_vit_h_4b8939.pth",
    )
    return env


def _require_repo_layout(repo_root: Path, entrypoint: str) -> tuple[Path, Path]:
    entrypoint_path = (repo_root / entrypoint).resolve()
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"WonderJourney entrypoint not found: {entrypoint_path}")
    base_config_path = (repo_root / "config" / "base-config.yaml").resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"WonderJourney base config not found: {base_config_path}")
    return entrypoint_path, base_config_path


def _runtime_entrypoint(entrypoint_path: Path, work_dir: Path, revision: str | None) -> Path:
    source = entrypoint_path.read_text(encoding="utf-8")
    patched = source

    marker = 'revision="fp16",'
    replacement = f'revision="{revision}",' if revision else ""
    if marker in patched:
        patched = patched.replace(marker, replacement)

    # The benchmark uses prompt_target directly and disables GPT. The upstream
    # demo imports chatGPT4 unconditionally, which reads OPENAI_API_KEY at
    # import time; provide a tiny non-GPT replacement for this path.
    gpt_import = "from util.chatGPT4 import TextpromptGen"
    prompt_stub = 'class TextpromptGen:\n    def __init__(self, *args, **kwargs):\n        pass\n\n    def generate_prompt(self, *, style, entities, background, scene_name):\n        parts = [style, scene_name, *(entities or []), background]\n        return ", ".join(str(part) for part in parts if part)\n\n    def run_conversation(self, *, scene_name, entities, style, background, control_text=None):\n        return {"scene_name": scene_name, "entities": entities or [], "style": style, "background": background}\n\n    def evaluate_image(self, *args, **kwargs):\n        return False, False\n\n    def write_all_content(self, *args, **kwargs):\n        return None\n'
    if gpt_import in patched:
        patched = patched.replace(gpt_import, prompt_stub)

    # Upstream WonderJourney hard-codes a local SAM checkpoint path from the
    # authors' machine. Use the benchmark workspace checkpoint instead.
    segment_import = "from util.segment_utils import create_mask_generator"
    segment_stub = 'import os as _worldarena_os\nfrom segment_anything import sam_model_registry, SamAutomaticMaskGenerator\n\n\ndef create_mask_generator():\n    sam_checkpoint = _worldarena_os.environ.get("WONDERJOURNEY_SAM_CHECKPOINT", "/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt/WonderJourney/sam_vit_h_4b8939.pth")\n    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)\n    sam.to(device="cuda")\n    return SamAutomaticMaskGenerator(\n        model=sam,\n        points_per_side=32,\n        pred_iou_thresh=0.86,\n        stability_score_thresh=0.92,\n        min_mask_region_area=100,\n    )\n'
    if segment_import in patched:
        patched = patched.replace(segment_import, segment_stub)

    oneformer_repo = os.environ.get(
        "WONDERJOURNEY_ONEFORMER_MODEL_REPO",
        "/mnt/cpfs/yangboxue/visual_generation/juanxi/ckpt/oneformer_coco_swin_large",
    )
    patched = patched.replace(
        '"shi-labs/oneformer_coco_swin_large"',
        json.dumps(oneformer_repo),
    )

    if patched == source:
        return entrypoint_path

    patched_path = work_dir / "worldarena_wonderjourney_entrypoint.py"
    patched_path.write_text(patched, encoding="utf-8")
    return patched_path

def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _parse_rotation_path(value: str | None, *, num_scenes: int, num_keyframes: int) -> list[int]:
    required = max(int(num_scenes), 1) * max(int(num_keyframes), 1)
    if value is None or not str(value).strip():
        return [0] * required
    text = str(value).strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in text.split(",") if item.strip()]
    if not isinstance(parsed, list):
        raise ValueError(f"rotation_path must be a list or comma-separated string: {value!r}")
    rotation_path = [int(item) for item in parsed]
    if len(rotation_path) < required:
        rotation_path.extend([0] * (required - len(rotation_path)))
    return rotation_path


def _content_prompt(scene_name: str, entities: list[str]) -> str:
    values = [scene_name, *entities]
    return ", ".join(value for value in values if value)


def _write_example_yaml(
    path: Path,
    *,
    example_name: str,
    image_filename: str,
    style_prompt: str,
    content_prompt: str,
    negative_prompt: str,
    background_prompt: str,
) -> Path:
    payload = [
        {
            "name": example_name,
            "image_filepath": f"examples/images/{image_filename}",
            "style_prompt": style_prompt,
            "content_prompt": content_prompt,
            "negative_prompt": negative_prompt,
            "background": background_prompt,
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _write_example_config(path: Path, *, args: argparse.Namespace, example_name: str) -> Path:
    payload: dict[str, Any] = {
        "runs_dir": f"output/{example_name}",
        "example_name": example_name,
        "seed": int(args.seed),
        "frames": int(args.frames),
        "num_scenes": int(args.num_scenes),
        "num_keyframes": int(args.num_keyframes),
        "use_gpt": bool(args.use_gpt),
        "debug": bool(args.debug),
        "skip_interp": bool(args.skip_interp),
        "skip_gen": bool(args.skip_gen),
        "enable_regenerate": bool(args.enable_regenerate),
        "finetune_decoder_gen": bool(args.finetune_decoder_gen),
        "finetune_decoder_interp": bool(args.finetune_decoder_interp),
        "finetune_depth_model": bool(args.finetune_depth_model),
        "depth_model": str(args.depth_model),
        "camera_speed": float(args.camera_speed),
        "rotation_range": float(args.rotation_range),
        "init_focal_length": float(args.init_focal_length),
        "inpainting_resolution_gen": int(args.inpainting_resolution_gen),
        "inpainting_resolution_interp": int(args.inpainting_resolution_interp),
        "kf2_upsample_coef": int(args.kf2_upsample_coef),
        "fg_depth_range": float(args.fg_depth_range),
        "depth_shift": float(args.depth_shift),
        "regenerate_times": int(args.regenerate_times),
        "num_finetune_decoder_steps": int(args.num_finetune_decoder_steps),
        "num_finetune_decoder_steps_interp": int(args.num_finetune_decoder_steps_interp),
        "camera_speed_multiplier_rotation": float(args.camera_speed_multiplier_rotation),
        "save_fps": int(args.save_fps),
        "rotation_path": _parse_rotation_path(
            args.rotation_path,
            num_scenes=int(args.num_scenes),
            num_keyframes=int(args.num_keyframes),
        ),
    }
    if args.stable_diffusion_checkpoint:
        payload["stable_diffusion_checkpoint"] = str(args.stable_diffusion_checkpoint)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _latest_merged_dir(runs_root: Path) -> Path | None:
    candidates = [path for path in runs_root.glob("*_merged") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _copy_prediction_outputs(
    *,
    output_path: Path,
    merged_dir: Path,
    archive_sidecars: bool,
) -> dict[str, Path]:
    generated_video = merged_dir / "output.mp4"
    if not generated_video.exists():
        raise FileNotFoundError(f"WonderJourney output video was not written: {generated_video}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    shutil.copy2(generated_video, output_path)
    copied = {"prediction": output_path}

    if not archive_sidecars:
        return copied
    for filename, suffix in (
        ("output_reverse.mp4", ".reverse"),
        ("keyframes.mp4", ".keyframes"),
        ("keyframes_reverse.mp4", ".keyframes_reverse"),
    ):
        source = merged_dir / filename
        if not source.exists():
            continue
        destination = output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
        destination.unlink(missing_ok=True)
        shutil.copy2(source, destination)
        copied[suffix.strip(".")] = destination
    return copied


def _validate_single_args(args: argparse.Namespace) -> None:
    required = (
        "repo_root",
        "conditioning_image",
        "output_path",
        "sample_name",
        "scene_name",
        "midas_checkpoint",
    )
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError("missing required WonderJourney runner arguments: " + ", ".join(missing))


def _prepare_run_workspace(args: argparse.Namespace, work_dir: Path) -> tuple[Path, Path, Path, str]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    conditioning_image = Path(args.conditioning_image).expanduser().resolve()
    midas_checkpoint = Path(args.midas_checkpoint).expanduser().resolve()

    if not conditioning_image.exists():
        raise FileNotFoundError(f"WonderJourney conditioning image not found: {conditioning_image}")
    if not midas_checkpoint.exists():
        raise FileNotFoundError(f"WonderJourney MiDaS checkpoint not found: {midas_checkpoint}")

    entrypoint_path, base_config_path = _require_repo_layout(repo_root, args.entrypoint)
    example_name = _slugify(str(args.sample_name))
    entities = [str(value) for value in (args.entities or []) if str(value).strip()]
    content_prompt = _content_prompt(str(args.scene_name), entities)

    examples_root = work_dir / "examples"
    image_suffix = conditioning_image.suffix or ".png"
    local_conditioning = examples_root / "images" / f"{example_name}{image_suffix}"
    local_conditioning.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(conditioning_image, local_conditioning)
    (work_dir / "output").mkdir(parents=True, exist_ok=True)
    _link_or_copy(midas_checkpoint, work_dir / "dpt_beit_large_512.pt")

    _write_example_yaml(
        examples_root / "examples.yaml",
        example_name=example_name,
        image_filename=local_conditioning.name,
        style_prompt=str(args.style_prompt),
        content_prompt=content_prompt,
        negative_prompt=str(args.negative_prompt),
        background_prompt=str(args.background_prompt),
    )
    example_config_path = _write_example_config(
        work_dir / "config" / "worldarena_example.yaml",
        args=args,
        example_name=example_name,
    )
    return entrypoint_path, base_config_path, example_config_path, example_name


def _cache_key(name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> str:
    payload = {
        "name": name,
        "args": [str(item) for item in args],
        "kwargs": {key: str(value) for key, value in sorted(kwargs.items())},
    }
    return json.dumps(payload, sort_keys=True)


def _load_runtime_module(runtime_entrypoint: Path, repo_root: Path):
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "midas_module"))
    module_name = f"worldarena_wonderjourney_runtime_{abs(hash(str(runtime_entrypoint)))}"
    spec = importlib.util.spec_from_file_location(module_name, runtime_entrypoint)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load WonderJourney entrypoint: {runtime_entrypoint}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_singleload_caches(module) -> None:
    cache: dict[str, object] = {}

    original_processor_from_pretrained = module.OneFormerProcessor.from_pretrained

    @classmethod
    def cached_processor_from_pretrained(cls, *args, **kwargs):
        del cls
        key = _cache_key("oneformer_processor", args, kwargs)
        if key not in cache:
            print(f"[WonderJourney singleload] loading OneFormerProcessor once: {args[0] if args else ''}", flush=True)
            cache[key] = original_processor_from_pretrained(*args, **kwargs)
        return cache[key]

    module.OneFormerProcessor.from_pretrained = cached_processor_from_pretrained

    original_segment_from_pretrained = module.OneFormerForUniversalSegmentation.from_pretrained

    @classmethod
    def cached_segment_from_pretrained(cls, *args, **kwargs):
        del cls
        key = _cache_key("oneformer_model", args, kwargs)
        if key not in cache:
            print(f"[WonderJourney singleload] loading OneFormer model once: {args[0] if args else ''}", flush=True)
            cache[key] = original_segment_from_pretrained(*args, **kwargs)
        return cache[key]

    module.OneFormerForUniversalSegmentation.from_pretrained = cached_segment_from_pretrained

    original_mask_generator = module.create_mask_generator

    def cached_mask_generator(*args, **kwargs):
        key = _cache_key("mask_generator", args, kwargs)
        if key not in cache:
            print("[WonderJourney singleload] loading mask generator once", flush=True)
            cache[key] = original_mask_generator(*args, **kwargs)
        return cache[key]

    module.create_mask_generator = cached_mask_generator

    original_inpaint_from_pretrained = module.StableDiffusionInpaintPipeline.from_pretrained

    @classmethod
    def cached_inpaint_from_pretrained(cls, *args, **kwargs):
        del cls
        key = _cache_key("stable_diffusion_inpaint", args, kwargs)
        if key not in cache:
            print(f"[WonderJourney singleload] loading inpaint pipeline once: {args[0] if args else ''}", flush=True)
            cache[key] = original_inpaint_from_pretrained(*args, **kwargs)
        return cache[key]

    module.StableDiffusionInpaintPipeline.from_pretrained = cached_inpaint_from_pretrained

    original_vae_from_pretrained = module.AutoencoderKL.from_pretrained

    @classmethod
    def cached_vae_from_pretrained(cls, *args, **kwargs):
        del cls
        key = _cache_key("vae", args, kwargs)
        if key not in cache:
            print(f"[WonderJourney singleload] loading VAE once: {args[0] if args else ''}", flush=True)
            cache[key] = original_vae_from_pretrained(*args, **kwargs)
        return cache[key]

    module.AutoencoderKL.from_pretrained = cached_vae_from_pretrained

    original_load_model = module.load_model

    def cached_load_model(*args, **kwargs):
        key = _cache_key("midas", args, kwargs)
        if key not in cache:
            print("[WonderJourney singleload] loading MiDaS once", flush=True)
            cache[key] = original_load_model(*args, **kwargs)
        model, transform, net_w, net_h = cache[key]
        if args:
            model = model.to(args[0])
            cache[key] = (model, transform, net_w, net_h)
        return cache[key]

    module.load_model = cached_load_model


def _copy_outputs_or_raise(work_dir: Path, example_name: str, output_path: Path, archive_sidecars: bool) -> None:
    merged_dir = _latest_merged_dir(work_dir / "output" / example_name)
    if merged_dir is None:
        raise FileNotFoundError(
            f"WonderJourney merged output directory was not written under {work_dir}"
        )
    _copy_prediction_outputs(
        output_path=output_path,
        merged_dir=merged_dir,
        archive_sidecars=archive_sidecars,
    )


def _run_one_subprocess(args: argparse.Namespace) -> dict[str, object]:
    _validate_single_args(args)
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    started_at = time.perf_counter()
    with _work_dir(bool(args.keep_work_dir)) as work_dir:
        entrypoint_path, base_config_path, example_config_path, example_name = _prepare_run_workspace(args, work_dir)
        runtime_entrypoint = _runtime_entrypoint(
            entrypoint_path,
            work_dir,
            str(args.stable_diffusion_revision).strip() or None,
        )
        command = [
            sys.executable,
            str(runtime_entrypoint),
            "--base-config",
            str(base_config_path),
            "--example_config",
            str(example_config_path),
        ]
        subprocess.run(
            command,
            check=True,
            cwd=str(work_dir),
            env=_normalized_env(repo_root),
            timeout=args.run_timeout,
        )
        _copy_outputs_or_raise(work_dir, example_name, output_path, bool(args.archive_sidecars))
    return {
        "status": "generated",
        "prediction_path": str(output_path),
        "generation_wall_time_seconds": round(time.perf_counter() - started_at, 6),
    }


def _run_one_inprocess(args: argparse.Namespace, runtime: dict[str, object]) -> dict[str, object]:
    _validate_single_args(args)
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    started_at = time.perf_counter()
    with _work_dir(bool(args.keep_work_dir)) as work_dir:
        entrypoint_path, base_config_path, example_config_path, example_name = _prepare_run_workspace(args, work_dir)
        module = runtime.get("module")
        if module is None:
            runtime_entrypoint = _runtime_entrypoint(
                entrypoint_path,
                work_dir,
                str(args.stable_diffusion_revision).strip() or None,
            )
            module = _load_runtime_module(runtime_entrypoint, repo_root)
            if not hasattr(module.Image, "ANTIALIAS"):
                module.Image.ANTIALIAS = module.Image.Resampling.LANCZOS
            _install_singleload_caches(module)
            runtime["module"] = module

        old_cwd = Path.cwd()
        old_environ = os.environ.copy()
        runtime_env = _normalized_env(repo_root, old_environ)
        os.environ.clear()
        os.environ.update(runtime_env)
        print(f"[WonderJourney batch] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
        try:
            os.chdir(work_dir)
            config = module.OmegaConf.merge(
                module.OmegaConf.load(str(base_config_path)),
                module.OmegaConf.load(str(example_config_path)),
            )
            module.run(config)
        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_environ)
        _copy_outputs_or_raise(work_dir, example_name, output_path, bool(args.archive_sidecars))
    return {
        "status": "generated",
        "prediction_path": str(output_path),
        "generation_wall_time_seconds": round(time.perf_counter() - started_at, 6),
    }


def _run_batch_spec(args: argparse.Namespace) -> None:
    spec_path = Path(str(args.batch_spec)).expanduser().resolve()
    results_path = (
        Path(str(args.batch_results)).expanduser().resolve()
        if args.batch_results
        else spec_path.with_suffix(".results.json")
    )
    rows = [json.loads(line) for line in spec_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    runtime: dict[str, object] = {}
    results: dict[str, dict[str, object]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        request_args = parse_args(list(row["args"]))
        print(f"[WonderJourney batch] generating {sample_id} -> {request_args.output_path}", flush=True)
        try:
            payload = _run_one_inprocess(request_args, runtime)
            payload.setdefault("prompt", row.get("prompt", ""))
        except Exception as exc:
            payload = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "prediction_path": row.get("prediction_path"),
                "prompt": row.get("prompt", ""),
            }
        results[sample_id] = payload
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_spec:
        _run_batch_spec(args)
        return
    _run_one_subprocess(args)


if __name__ == "__main__":
    main()
