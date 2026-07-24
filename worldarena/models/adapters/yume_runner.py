"""Subprocess runner for Yume single-sample inference."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any

from worldarena.common.checkpoints import resolve_checkpoint_path


_YUME_MODEL_DIRS = (
    "Yume-5B-720P",
    "stdstu123--Yume-5B-720P",
)
_CAPTION_MODEL_DIRS = (
    "InternVL3-2B-Instruct",
    "InternVL-2B-Instruct",
)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldAtlas Arena YUME subprocess runner.")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--entrypoint", default="webapp_single_gpu.py", type=str)
    parser.add_argument("--conditioning_image", default=None, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--caption_model_dir", default=None, type=str)
    parser.add_argument("--mode", default="I2V", type=str)
    parser.add_argument("--fps", default=16, type=int)
    parser.add_argument("--sample_steps", default=4, type=int)
    parser.add_argument("--sample_num", default=1, type=int)
    parser.add_argument("--frame_zero", default=32, type=int)
    parser.add_argument("--shift", default=5.0, type=float)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--resolution", default="704x1280", type=str)
    parser.add_argument("--gpu_index", default=0, type=int)
    parser.add_argument("--refine_from_image", default=False, type=_parse_bool)
    parser.add_argument("--load_caption_model", default=False, type=_parse_bool)
    parser.add_argument("--memory_optimization", default=False, type=_parse_bool)
    parser.add_argument("--vae_memory_optimization", default=False, type=_parse_bool)
    parser.add_argument("--camera_movement1", default="None", type=str)
    parser.add_argument("--camera_movement2", default="·", type=str)
    return parser.parse_args()


def _resolve_repo_path(repo_root: Path, value: str | os.PathLike[str] | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    marker_dirs = {"transformer", "vae", "tokenizer", "text_encoder"}
    marker_files = {
        "config.json",
        "generation_config.json",
        "model_index.json",
        "tokenizer_config.json",
    }
    for entry in path.iterdir():
        if entry.is_dir() and entry.name in marker_dirs:
            return True
        if entry.is_file() and (
            entry.name in marker_files
            or entry.suffix in {".bin", ".json", ".pt", ".pth", ".safetensors"}
        ):
            return True
    return False


def _resolve_candidate_dir(path: Path, *, direct_names: tuple[str, ...]) -> Path | None:
    if not path.exists() or not path.is_dir():
        return None
    if path.name in direct_names or _looks_like_model_dir(path):
        return path.resolve()
    return None


def _resolve_model_dir(base_dir: Path | None, *, direct_names: tuple[str, ...]) -> Path | None:
    if base_dir is None:
        return None
    candidates = [base_dir]
    for name in direct_names:
        candidates.append(base_dir / name)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved = _resolve_candidate_dir(candidate, direct_names=direct_names)
        if resolved is not None:
            return resolved
    return None


def _resolve_caption_model_dir(explicit_dir: Path | None) -> Path | None:
    return _resolve_model_dir(explicit_dir, direct_names=_CAPTION_MODEL_DIRS)


def _ensure_flask_importable() -> None:
    """Ensure flask importable."""
    if importlib.util.find_spec("flask") is not None:
        return

    class _Flask:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def route(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        get = route
        post = route

        def run(self, *args, **kwargs) -> None:
            return None

    class _Request:
        args: dict[str, Any] = {}

        @staticmethod
        def get_json(*args, **kwargs) -> dict[str, Any]:
            return {}

    class _Response:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    def _jsonify(*args, **kwargs):
        if args and not kwargs:
            return args[0]
        return kwargs

    def _send_from_directory(*args, **kwargs):
        return None

    flask_stub = ModuleType("flask")
    flask_stub.Flask = _Flask
    flask_stub.jsonify = _jsonify
    flask_stub.request = _Request()
    flask_stub.send_from_directory = _send_from_directory
    flask_stub.Response = _Response
    sys.modules["flask"] = flask_stub


def _load_module(script_path: Path) -> ModuleType:
    _ensure_flask_importable()
    spec = importlib.util.spec_from_file_location("worldarena_yume_entrypoint", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load YUME entrypoint module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_callable(module: ModuleType, name: str) -> object:
    value = getattr(module, name, None)
    if not callable(value):
        raise AttributeError(f"YUME entrypoint is missing callable {name!r}")
    return value


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _prepend_sys_path(path: Path):
    token = str(path)
    sys.path.insert(0, token)
    try:
        yield
    finally:
        try:
            sys.path.remove(token)
        except ValueError:
            pass


def _resolve_generated_path(repo_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _copy_sidecar(generated_path: Path, output_path: Path) -> None:
    for candidate in (
        Path(str(generated_path) + ".meta.json"),
        generated_path.with_suffix(".meta.json"),
    ):
        if candidate.exists():
            shutil.copy2(candidate, Path(str(output_path) + ".meta.json"))
            return


@dataclass(slots=True)
class PersistentYumeRuntime:
    repo_root: Path
    generated_root: Path
    module: ModuleType
    long_gen_args: Any
    long_generate: Any


def _normalize_mode(mode: str | None) -> str:
    return str(mode or "I2V").strip().upper()


def _reset_generation_state(module: ModuleType) -> None:
    state = getattr(module, "LAST", None)
    if not isinstance(state, dict):
        return
    state["last_model_input_latent"] = None
    state["last_model_input_de"] = None
    state["frame_total"] = 0
    state["last_video_path"] = None
    state["last_prompt"] = ""


def _load_persistent_runtime(
    *,
    repo_root: Path,
    checkpoint_dir: str | os.PathLike[str] | None,
    entrypoint: str | os.PathLike[str],
    generated_root: Path,
    gpu_index: int,
    caption_model_dir: str | os.PathLike[str] | None = None,
    load_caption_model: bool = False,
    refine_from_image: bool = False,
) -> PersistentYumeRuntime:
    repo_root = repo_root.expanduser().resolve()
    generated_root = generated_root.expanduser().resolve()
    generated_root.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint_dir = resolve_checkpoint_path(checkpoint_dir, kind="dir", required=True)
    if resolved_checkpoint_dir is None:
        raise ValueError("YUME runner requires checkpoint_dir")
    entrypoint_path = _resolve_repo_path(repo_root, entrypoint)
    if entrypoint_path is None or not entrypoint_path.exists():
        raise FileNotFoundError(f"YUME entrypoint not found: {entrypoint_path}")

    yume_model_dir = _resolve_model_dir(resolved_checkpoint_dir, direct_names=_YUME_MODEL_DIRS)
    if yume_model_dir is None:
        raise FileNotFoundError(
            "YUME model directory not found. Expected a direct model folder or cache entry such as "
            f"{', '.join(_YUME_MODEL_DIRS)} under {resolved_checkpoint_dir}."
        )

    caption_model_path = None
    if load_caption_model or refine_from_image:
        explicit_caption_dir = resolve_checkpoint_path(caption_model_dir, kind="dir", required=True)
        caption_model_path = _resolve_caption_model_dir(explicit_caption_dir)
        if caption_model_path is None:
            raise FileNotFoundError(
                "YUME caption model directory not found. Set generation.caption_model_dir to a "
                f"checkpoint under ckpt/, such as one of {', '.join(_CAPTION_MODEL_DIRS)}."
            )

    with _pushd(repo_root):
        with _prepend_sys_path(repo_root):
            module = _load_module(entrypoint_path)
            setattr(module, "CKPT_DIR", str(yume_model_dir))
            setattr(module, "OUTPUT_DIR", str(generated_root))
            setattr(module, "DEVICE_ID", int(gpu_index))
            if caption_model_path is not None:
                setattr(module, "INTERNVL_PATH", str(caption_model_path))

            load_wan = _require_callable(module, "load_wan")
            long_generate = _require_callable(module, "long_generate")
            long_gen_args = getattr(module, "LongGenArgs", None)
            if long_gen_args is None:
                raise AttributeError("YUME entrypoint is missing LongGenArgs")

            load_wan()
            if load_caption_model or refine_from_image:
                load_caption = _require_callable(module, "load_caption_model")
                load_caption()

    return PersistentYumeRuntime(
        repo_root=repo_root,
        generated_root=generated_root,
        module=module,
        long_gen_args=long_gen_args,
        long_generate=long_generate,
    )


def _run_persistent_generation(
    runtime: PersistentYumeRuntime,
    *,
    conditioning_image: Path | None,
    output_path: Path,
    prompt: str,
    mode: str = "I2V",
    fps: int = 16,
    sample_steps: int = 4,
    sample_num: int = 1,
    frame_zero: int = 32,
    shift: float = 5.0,
    seed: int = -1,
    resolution: str = "704x1280",
    refine_from_image: bool = False,
    memory_optimization: bool = False,
    vae_memory_optimization: bool = False,
    camera_movement1: str = "None",
    camera_movement2: str = "·",
) -> dict[str, Any]:
    mode = _normalize_mode(mode)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode != "T2V":
        if conditioning_image is None:
            raise FileNotFoundError("YUME conditioning image is required for non-T2V generation")
        conditioning_image = conditioning_image.expanduser().resolve()
        if not conditioning_image.exists():
            raise FileNotFoundError(f"YUME conditioning image not found: {conditioning_image}")

    final_prompt = prompt
    try:
        with _pushd(runtime.repo_root):
            with _prepend_sys_path(runtime.repo_root):
                _reset_generation_state(runtime.module)
                payload = runtime.long_gen_args(
                    prompt=prompt,
                    jpg_path=str(conditioning_image) if conditioning_image is not None else None,
                    output_dir=str(runtime.generated_root),
                    fps=int(fps),
                    sample_steps=int(sample_steps),
                    sample_num=int(sample_num),
                    frame_zero=int(frame_zero),
                    shift=float(shift),
                    seed=int(seed),
                    continue_from_last=False,
                    refine_from_image=bool(refine_from_image),
                    caption_path=None,
                    mode=mode,
                    resolution=str(resolution),
                    memory_optimization=bool(memory_optimization),
                    vae_memory_optimization=bool(vae_memory_optimization),
                    camera_movement1=str(camera_movement1),
                    camera_movement2=str(camera_movement2),
                )
                generated_video, final_prompt = runtime.long_generate(payload)

        generated_path = _resolve_generated_path(runtime.repo_root, generated_video)
        if not generated_path.exists():
            raise FileNotFoundError(f"YUME output video was not written: {generated_path}")
        shutil.copy2(generated_path, output_path)
        _copy_sidecar(generated_path, output_path)
        return {
            "prediction_path": str(output_path),
            "prompt": prompt,
            "final_prompt": final_prompt,
        }
    finally:
        _reset_generation_state(runtime.module)
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conditioning_image = _resolve_repo_path(repo_root, args.conditioning_image)

    mode = _normalize_mode(args.mode)
    if mode != "T2V":
        if conditioning_image is None or not conditioning_image.exists():
            raise FileNotFoundError(f"YUME conditioning image not found: {conditioning_image}")

    with tempfile.TemporaryDirectory(prefix="worldarena_yume_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        generated_root = (temp_dir / "outputs").resolve()
        runtime = _load_persistent_runtime(
            repo_root=repo_root,
            checkpoint_dir=args.checkpoint_dir,
            entrypoint=args.entrypoint,
            generated_root=generated_root,
            gpu_index=int(args.gpu_index),
            caption_model_dir=args.caption_model_dir,
            load_caption_model=bool(args.load_caption_model),
            refine_from_image=bool(args.refine_from_image),
        )
        _run_persistent_generation(
            runtime,
            conditioning_image=conditioning_image,
            output_path=output_path,
            prompt=args.prompt,
            mode=mode,
            fps=int(args.fps),
            sample_steps=int(args.sample_steps),
            sample_num=int(args.sample_num),
            frame_zero=int(args.frame_zero),
            shift=float(args.shift),
            seed=int(args.seed),
            resolution=str(args.resolution),
            refine_from_image=bool(args.refine_from_image),
            memory_optimization=bool(args.memory_optimization),
            vae_memory_optimization=bool(args.vae_memory_optimization),
            camera_movement1=str(args.camera_movement1),
            camera_movement2=str(args.camera_movement2),
        )


if __name__ == "__main__":
    main()
