"""Subprocess runtime for MeT3R reconstruction-consistency metrics."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image

from worldarena.common.checkpoints import (
    apply_checkpoint_env,
    checkpoint_path,
    filter_checkpoint_env_overrides,
    hf_local_dir,
)

MET3R_RUNTIME_WORKER_ENV = "WORLDARENA_MET3R_RUNTIME_WORKER"
MET3R_DISTANCE_DIRECTIONS = {
    "cosine": "lower_is_better",
    "lpips": "lower_is_better",
    "rmse": "lower_is_better",
    "mse": "lower_is_better",
    "psnr": "higher_is_better",
    "ssim": "higher_is_better",
}
MET3R_BACKBONE_REPOS = {
    "mast3r": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    "dust3r": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
}


def _project_root() -> Path:
    """Project root -> Path."""
    return Path(__file__).resolve().parents[2]


def _runtime_cache_dir(name: str) -> Path:
    path = _project_root() / ".cache" / "benchmark" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_met3r_root() -> Path:
    """Resolve met3r root -> Path."""
    return (_project_root() / "thirdparty" / "met3r").resolve()


def _resolve_runtime_python_bin(runtime: dict[str, Any] | None) -> str | None:
    payload = dict(runtime or {})
    raw = str(payload.get("python_bin") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    if raw.startswith(".") or "/" in raw:
        return str((_project_root() / path).absolute())
    return raw


def resolve_met3r_runtime_python(runtime: dict[str, Any] | None = None) -> str | None:
    return _resolve_runtime_python_bin(runtime)


def inspect_met3r_layout() -> dict[str, Any]:
    """Inspect met3r layout -> dict[str, Any]."""
    root = resolve_met3r_root()
    package_dir = root / "met3r"
    init_path = package_dir / "__init__.py"
    metric_path = package_dir / "met3r.py"
    mast3r_root = root / "mast3r"
    mast3r_package = mast3r_root / "mast3r"
    dust3r_package = mast3r_root / "dust3r"
    source_ready = init_path.exists() and metric_path.exists()
    submodule_ready = mast3r_package.is_dir() and dust3r_package.is_dir()
    ready = source_ready and submodule_ready
    error = None
    if not source_ready:
        error = f"MEt3R checkout is missing package files under {package_dir}"
    elif not submodule_ready:
        error = (
            "MEt3R checkout is missing initialized MASt3R/DUSt3R submodules; "
            "run `git submodule update --init --recursive` inside the met3r repo"
        )
    return {
        "root": str(root),
        "package_dir": str(package_dir),
        "metric_path": str(metric_path),
        "mast3r_root": str(mast3r_root),
        "mast3r_package": str(mast3r_package),
        "dust3r_package": str(dust3r_package),
        "source_ready": source_ready,
        "submodule_ready": submodule_ready,
        "ready": ready,
        "error": error,
    }


def _coerce_runtime_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _default_device() -> str:
    """Default device -> str."""
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_img_size(value: Any, default: int | None = 256) -> int | None:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return int(value)


def _metric_options(runtime: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(runtime or {})
    return {
        "device": str(payload.get("device") or _default_device()),
        "batch_size": max(1, int(payload.get("batch_size", 1))),
        "img_size": _resolve_img_size(payload.get("img_size"), default=256),
        "use_norm": _coerce_runtime_bool(payload.get("use_norm"), default=True),
        "backbone": str(payload.get("backbone") or "mast3r"),
        "feature_backbone": str(payload.get("feature_backbone") or "dino16"),
        "feature_backbone_weights": str(
            payload.get("feature_backbone_weights") or hf_local_dir("mhamilton723/FeatUp")
        ),
        "upsampler": str(payload.get("upsampler") or "featup"),
        "distance": str(payload.get("distance") or "cosine"),
        "freeze": _coerce_runtime_bool(payload.get("freeze"), default=True),
    }


def _snapshot_dir(cache_root: Path) -> Path | None:
    refs_main = cache_root / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = cache_root / "snapshots" / revision
        if snapshot.is_dir():
            return snapshot
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.is_dir():
        return None
    snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
    return snapshots[0] if len(snapshots) == 1 else None


def _ensure_backbone_weights_available(backbone: str) -> None:
    repo_id = MET3R_BACKBONE_REPOS.get(backbone)
    if repo_id is None:
        return
    cache_root = checkpoint_path(
        "huggingface",
        "hub",
        f"models--{repo_id.replace('/', '--')}",
        kind="dir",
        required=False,
    )
    snapshot = _snapshot_dir(cache_root)
    if snapshot is None:
        raise FileNotFoundError(f"MEt3R {backbone} weights are not cached under {cache_root}")
    if not any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(
            f"MEt3R {backbone} weights are incomplete under {snapshot}; "
            "expected model.safetensors or pytorch_model.bin"
        )


def _distance_direction(distance: str) -> str:
    return MET3R_DISTANCE_DIRECTIONS.get(distance.lower(), "lower_is_better")


def _subprocess_runtime_env(runtime: dict[str, Any] | None) -> dict[str, str]:
    env = apply_checkpoint_env()
    env[MET3R_RUNTIME_WORKER_ENV] = "1"
    pythonpath_entries = [str(_project_root())]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    for key, value in filter_checkpoint_env_overrides(dict((runtime or {}).get("env", {}))).items():
        env[str(key)] = str(value)
    return env


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype == np.uint8:
        contiguous = np.ascontiguousarray(array)
        return contiguous if contiguous.flags.writeable else contiguous.copy()
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _frame_pairs(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    pair_count = min(len(reference_frames), len(prediction_frames))
    if pair_count <= 0:
        raise ValueError("MEt3R requires at least one reference/prediction frame pair")
    return list(zip(reference_frames[:pair_count], prediction_frames[:pair_count]))


def _serialize_frame_pairs(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    pair_dir: Path,
) -> list[dict[str, str]]:
    pair_dir.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, str]] = []
    for index, (reference_frame, prediction_frame) in enumerate(
        _frame_pairs(reference_frames, prediction_frames)
    ):
        reference_path = pair_dir / f"{index:04d}.reference.png"
        prediction_path = pair_dir / f"{index:04d}.prediction.png"
        Image.fromarray(_to_uint8(reference_frame)).save(reference_path)
        Image.fromarray(_to_uint8(prediction_frame)).save(prediction_path)
        payload.append(
            {
                "reference_path": str(reference_path),
                "prediction_path": str(prediction_path),
            }
        )
    return payload


def _cleanup_ipc_paths(
    request_path: Path,
    response_path: Path,
    pair_dir: Path,
    *,
    keep_ipc: bool,
) -> None:
    if keep_ipc:
        return
    request_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    shutil.rmtree(pair_dir, ignore_errors=True)


def _maybe_run_in_subprocess(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    *,
    runtime: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if os.getenv(MET3R_RUNTIME_WORKER_ENV) == "1":
        return None
    python_bin = _resolve_runtime_python_bin(runtime)
    if python_bin is None:
        return None

    ipc_dir = _runtime_cache_dir("met3r_runtime_ipc")
    token = uuid.uuid4().hex
    pair_dir = ipc_dir / f"pairs.{token}"
    request_path = ipc_dir / f"met3r.{token}.request.json"
    response_path = ipc_dir / f"met3r.{token}.response.json"
    pairs = _serialize_frame_pairs(reference_frames, prediction_frames, pair_dir)
    request_path.write_text(
        json.dumps({"pairs": pairs, "runtime": dict(runtime or {})}, ensure_ascii=False),
        encoding="utf-8",
    )
    command = [
        python_bin,
        "-m",
        "worldarena.benchmark.met3r_runtime_worker",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    keep_ipc = _coerce_runtime_bool((runtime or {}).get("keep_ipc"), default=False)
    try:
        completed = subprocess.run(
            command,
            cwd=str(_project_root()),
            env=_subprocess_runtime_env(runtime),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = " | ".join(part for part in (stderr, stdout) if part)
        raise RuntimeError(f"MEt3R runtime subprocess failed: {details}") from exc

    if not response_path.exists():
        logs = " | ".join(
            part for part in ((completed.stderr or "").strip(), (completed.stdout or "").strip()) if part
        )
        raise RuntimeError(f"MEt3R runtime subprocess did not write a response: {logs}")

    response = json.loads(response_path.read_text(encoding="utf-8"))
    try:
        if not bool(response.get("ok")):
            error = str(response.get("error") or "MEt3R runtime subprocess failed")
            traceback_text = str(response.get("traceback") or "").strip()
            stdout = str(response.get("stdout") or "").strip()
            stderr = str(response.get("stderr") or "").strip()
            raise RuntimeError(
                " | ".join(part for part in (error, traceback_text, stderr, stdout) if part)
            )
        return dict(response["result"])
    finally:
        _cleanup_ipc_paths(request_path, response_path, pair_dir, keep_ipc=keep_ipc)


@contextmanager
def _met3r_import_context(root: Path) -> Iterator[None]:
    root_str = str(root)
    sys.path.insert(0, root_str)
    try:
        yield
    finally:
        while root_str in sys.path:
            sys.path.remove(root_str)


def _clear_met3r_modules() -> None:
    """Clear met3r modules."""
    for name in list(sys.modules):
        if name == "met3r" or name.startswith("met3r."):
            sys.modules.pop(name, None)


@contextmanager
def _met3r_torch_hub_context() -> Iterator[None]:
    """Met3r torch hub context -> Iterator[None]."""
    import torch

    original_load = torch.hub.load

    def patched_load(repo_or_dir: Any, *args: Any, **kwargs: Any) -> Any:
        raw_repo = str(repo_or_dir)
        repo_path = Path(raw_repo).expanduser()
        if not repo_path.is_absolute() and (raw_repo.startswith(".") or "/" in raw_repo):
            repo_path = _project_root() / repo_path
        if repo_path.exists():
            kwargs.setdefault("source", "local")
            return original_load(str(repo_path), *args, **kwargs)
        if "mhamilton723--FeatUp" in raw_repo:
            return original_load("mhamilton723/FeatUp", *args, **kwargs)
        return original_load(repo_or_dir, *args, **kwargs)

    torch.hub.load = patched_load
    try:
        yield
    finally:
        torch.hub.load = original_load


def _adapt_dino_spatial_features(metric: Any, upsampler: str) -> None:
    """Expose DINO patch tokens as a spatial map for non-FeatUp interpolation."""
    if upsampler == "featup":
        return
    feature_model = getattr(metric, "feature_model", None)
    if feature_model is None or not hasattr(feature_model, "get_intermediate_layers"):
        return

    import torch

    class DinoSpatialFeatureAdapter(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, images: Any) -> Any:
            tokens = self.model.get_intermediate_layers(images, n=1)[0]
            patch_size = self.model.patch_embed.patch_size
            if isinstance(patch_size, Sequence):
                patch_height, patch_width = (int(patch_size[0]), int(patch_size[1]))
            else:
                patch_height = patch_width = int(patch_size)
            grid_height = int(images.shape[-2]) // patch_height
            grid_width = int(images.shape[-1]) // patch_width
            patch_count = grid_height * grid_width
            if int(tokens.shape[1]) == patch_count + 1:
                tokens = tokens[:, 1:]
            elif int(tokens.shape[1]) != patch_count:
                raise ValueError(
                    "DINO token count does not match the input patch grid: "
                    f"tokens={int(tokens.shape[1])}, grid={grid_height}x{grid_width}"
                )
            return tokens.transpose(1, 2).reshape(
                int(tokens.shape[0]),
                int(tokens.shape[2]),
                grid_height,
                grid_width,
            )

    metric.feature_model = DinoSpatialFeatureAdapter(feature_model)


@lru_cache(maxsize=4)
def _load_met3r_model(
    root: str,
    device: str,
    img_size: int | None,
    use_norm: bool,
    backbone: str,
    feature_backbone: str,
    feature_backbone_weights: str,
    upsampler: str,
    distance: str,
    freeze: bool,
):
    os.environ.update(apply_checkpoint_env())
    layout = inspect_met3r_layout()
    if not layout["ready"]:
        raise RuntimeError(str(layout["error"]))
    _ensure_backbone_weights_available(backbone)

    _clear_met3r_modules()
    try:
        with _met3r_import_context(Path(root)):
            module = importlib.import_module("met3r")
            metric_class = getattr(module, "MEt3R")
            with _met3r_torch_hub_context():
                metric = metric_class(
                    img_size=img_size,
                    use_norm=use_norm,
                    backbone=backbone,
                    feature_backbone=feature_backbone,
                    feature_backbone_weights=feature_backbone_weights,
                    upsampler=upsampler,
                    distance=distance,
                    freeze=freeze,
                )
                _adapt_dino_spatial_features(metric, upsampler)
    except Exception:
        _clear_met3r_modules()
        raise
    if hasattr(metric, "to"):
        metric = metric.to(device)
    if hasattr(metric, "eval"):
        metric.eval()
    return metric


def _pair_batch_to_tensor(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    device: str,
    img_size: int | None,
):
    import torch
    from torch.nn import functional as F

    def resize_image(image: Any, target_size: tuple[int, int]) -> Any:
        if tuple(image.shape[-2:]) == target_size:
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    batch: list[Any] = []
    for reference_frame, prediction_frame in pairs:
        reference_tensor = torch.from_numpy(_to_uint8(reference_frame)).permute(2, 0, 1).float()
        prediction_tensor = torch.from_numpy(_to_uint8(prediction_frame)).permute(2, 0, 1).float()
        target_size = (
            (int(img_size), int(img_size))
            if img_size is not None
            else tuple(int(value) for value in prediction_tensor.shape[-2:])
        )
        reference_tensor = resize_image(reference_tensor, target_size)
        prediction_tensor = resize_image(prediction_tensor, target_size)
        batch.append(torch.stack([reference_tensor, prediction_tensor], dim=0))
    tensor = torch.stack(batch, dim=0) / 127.5 - 1.0
    return tensor.to(device)


def _tensor_to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(item) for item in array.tolist()]


def _empty_overlap_penalty(distance: str) -> float:
    """Return the worst raw score when MEt3R has no comparable pixels.

    MEt3R's weighted reducer divides by ``mask.sum() + eps``.  An empty mask
    therefore produces a raw zero, which is an artificial perfect score for
    lower-is-better distances.  Treat no overlap as the configured distance's
    worst endpoint instead, while retaining an explicit zero overlap ratio in
    the details payload.
    """
    distance_name = distance.lower()
    if distance_name in {"psnr", "ssim"}:
        return 0.0
    if distance_name == "mse":
        return 4.0
    return 2.0


def compute_met3r_consistency(
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = dict(runtime or {})
    delegated = _maybe_run_in_subprocess(
        reference_frames,
        prediction_frames,
        runtime=runtime,
    )
    if delegated is not None:
        return delegated

    layout = inspect_met3r_layout()
    if not layout["ready"]:
        raise RuntimeError(str(layout["error"]))

    options = _metric_options(runtime)
    pairs = _frame_pairs(reference_frames, prediction_frames)
    metric = _load_met3r_model(
        str(resolve_met3r_root()),
        str(options["device"]),
        options["img_size"],
        bool(options["use_norm"]),
        str(options["backbone"]),
        str(options["feature_backbone"]),
        str(options["feature_backbone_weights"]),
        str(options["upsampler"]),
        str(options["distance"]),
        bool(options["freeze"]),
    )

    import torch

    frame_scores: list[float] = []
    overlap_ratios: list[float] = []
    empty_overlap_pair_indices: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), int(options["batch_size"])):
            batch_pairs = pairs[start : start + int(options["batch_size"])]
            batch = _pair_batch_to_tensor(
                batch_pairs,
                str(options["device"]),
                options["img_size"],
            )
            outputs = metric(
                images=batch,
                return_overlap_mask=True,
                return_score_map=False,
                return_projections=False,
            )
            batch_scores = outputs[0] if isinstance(outputs, tuple) else outputs
            if not isinstance(outputs, tuple) or len(outputs) < 2:
                raise RuntimeError("MEt3R did not return the requested overlap mask")
            overlap_mask = outputs[1]
            batch_score_values = _tensor_to_float_list(batch_scores)
            batch_overlap_ratios = _tensor_to_float_list(
                overlap_mask.float().flatten(start_dim=1).mean(dim=1)
            )
            if len(batch_score_values) != len(batch_overlap_ratios):
                raise RuntimeError(
                    "MEt3R returned mismatched score and overlap-mask batch sizes"
                )
            for offset, (score, overlap_ratio) in enumerate(
                zip(batch_score_values, batch_overlap_ratios)
            ):
                if not np.isfinite(overlap_ratio) or overlap_ratio <= 0.0:
                    empty_overlap_pair_indices.append(start + offset)
                    overlap_ratio = 0.0
                    score = _empty_overlap_penalty(str(options["distance"]))
                elif not np.isfinite(score):
                    raise ValueError(
                        "MEt3R produced a non-finite score for frame pair "
                        f"{start + offset}"
                    )
                frame_scores.append(float(score))
                overlap_ratios.append(float(overlap_ratio))

    raw = float(np.mean(frame_scores))
    return {
        "raw": raw,
        "backend": "met3r",
        "details": {
            "frame_scores": [round(value, 4) for value in frame_scores],
            "overlap_ratios": [round(value, 6) for value in overlap_ratios],
            "mean_overlap_ratio": float(np.mean(overlap_ratios)),
            "pair_count": len(frame_scores),
            "valid_pair_count": len(frame_scores) - len(empty_overlap_pair_indices),
            "empty_overlap_pair_count": len(empty_overlap_pair_indices),
            "empty_overlap_pair_indices": empty_overlap_pair_indices,
            "empty_overlap_policy": "worst_raw_score",
            "source_root": str(resolve_met3r_root()),
            "device": str(options["device"]),
            "batch_size": int(options["batch_size"]),
            "img_size": options["img_size"],
            "use_norm": bool(options["use_norm"]),
            "backbone": str(options["backbone"]),
            "feature_backbone": str(options["feature_backbone"]),
            "feature_backbone_weights": str(options["feature_backbone_weights"]),
            "upsampler": str(options["upsampler"]),
            "distance": str(options["distance"]),
            "freeze": bool(options["freeze"]),
            "native_direction": _distance_direction(str(options["distance"])),
        },
    }


def compute_met3r_consistency_from_paths(
    pairs: Sequence[dict[str, str]],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_frames: list[np.ndarray] = []
    prediction_frames: list[np.ndarray] = []
    for pair in pairs:
        reference_frames.append(
            np.asarray(Image.open(pair["reference_path"]).convert("RGB"))
        )
        prediction_frames.append(
            np.asarray(Image.open(pair["prediction_path"]).convert("RGB"))
        )
    return compute_met3r_consistency(
        reference_frames,
        prediction_frames,
        runtime=runtime,
    )


__all__ = [
    "MET3R_RUNTIME_WORKER_ENV",
    "compute_met3r_consistency",
    "compute_met3r_consistency_from_paths",
    "inspect_met3r_layout",
    "resolve_met3r_root",
    "resolve_met3r_runtime_python",
]
