from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_RUNTIME_ENV = "WORLDARENA_METRIC_SOURCE_ROOT"


@dataclass(frozen=True, slots=True)
class MetricSourceLayout:
    source_root: Path
    package_name: str
    overlay_root: Path
    droid_root: Path | None
    lietorch_root: Path | None


OFFICIAL_IMPORT_TARGETS = {
    "camera_error": (
        "benchmark.metrics.third_party.camera_error_metrics",
        "CameraErrorMetric",
    ),
    "reprojection_error": (
        "benchmark.metrics.third_party.reprojection_error_metrics",
        "ReprojectionErrorMetric",
    ),
    "optical_flow_aepe": (
        "benchmark.metrics.third_party.flow_aepe_metrics",
        "OpticalFlowAverageEndPointErrorMetric",
    ),
    "motion_magnitude": (
        "benchmark.metrics.third_party.flow_metrics",
        "OpticalFlowMetric",
    ),
    "motion_smoothness": (
        "benchmark.metrics.third_party.motion_smoothness_metrics",
        "MotionSmoothnessMetric",
    ),
    "object_detection": (
        "benchmark.metrics.third_party.object_detection_metrics",
        "ObjectDetectionMetric",
    ),
    "motion_accuracy": (
        "benchmark.metrics.third_party.motion_accuracy_metrics",
        "MotionAccuracyMetric",
    ),
}
OFFICIAL_METRIC_NAMES = tuple(OFFICIAL_IMPORT_TARGETS)


def _runtime_cache_dir(name: str) -> Path:
    path = PROJECT_ROOT / ".cache" / "benchmark" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
        return
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def _discover_metric_source_root_from(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    for candidate in [directory, *sorted(path for path in directory.iterdir() if path.is_dir())]:
        metric_file = next(
            candidate.rglob("benchmark/metrics/third_party/camera_error_metrics.py"),
            None,
        )
        if metric_file is not None:
            return candidate.resolve()
    return None


def _select_candidate_path(candidates: list[Path], *, package_prefix: str | None = None) -> Path | None:
    if not candidates:
        return None
    if package_prefix is not None:
        for candidate in candidates:
            if package_prefix in str(candidate):
                return candidate
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def _prepare_overlay_root(
    source_root: Path,
    package_dir: Path,
    grounding_package_dir: Path | None,
    sam2_package_dir: Path | None,
    sea_raft_root: Path | None,
    vfi_mamba_root: Path | None,
) -> Path:
    key = hashlib.sha1(str(source_root).encode("utf-8")).hexdigest()[:12]
    overlay_root = _runtime_cache_dir("metric_source_overlay") / key
    overlay_root.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(overlay_root / package_dir.name, package_dir)
    if grounding_package_dir is not None:
        _ensure_symlink(overlay_root / "groundingdino", grounding_package_dir)
    if sam2_package_dir is not None:
        _ensure_symlink(overlay_root / "sam2", sam2_package_dir)
    if sea_raft_root is not None and (sea_raft_root / "core").exists():
        _ensure_symlink(overlay_root / "core", sea_raft_root / "core")
    if vfi_mamba_root is not None and (vfi_mamba_root / "model").exists():
        _ensure_symlink(overlay_root / "model", vfi_mamba_root / "model")
    return overlay_root


@lru_cache(maxsize=1)
def resolve_metric_source_layout() -> MetricSourceLayout:
    configured_root = os.getenv(OFFICIAL_RUNTIME_ENV)
    candidate_roots: list[Path] = []
    if configured_root:
        candidate_roots.append(Path(configured_root).expanduser().resolve())
    third_party_root = PROJECT_ROOT / "third_party"
    if third_party_root.exists():
        candidate_roots.append(third_party_root)

    source_root: Path | None = None
    for root in candidate_roots:
        discovered = _discover_metric_source_root_from(root)
        if discovered is not None:
            source_root = discovered
            break
    if source_root is None:
        raise FileNotFoundError(
            "official metric source tree not found; set "
            f"{OFFICIAL_RUNTIME_ENV} or symlink a compatible checkout under third_party/"
        )

    metric_file = next(source_root.rglob("benchmark/metrics/third_party/camera_error_metrics.py"))
    package_name = metric_file.relative_to(source_root).parts[0]
    package_dir = source_root / package_name

    droid_root = _select_candidate_path(
        [path.parent for path in source_root.rglob("droid.py") if path.parent.name == "droid_slam"],
        package_prefix=package_name,
    )
    lietorch_root = _select_candidate_path(
        [path.parent.parent for path in source_root.rglob("lietorch/__init__.py")],
        package_prefix="DROID-SLAM",
    )
    sea_raft_root = _select_candidate_path(
        [path.parents[2] for path in source_root.rglob("config/eval/spring-M.json")],
        package_prefix="SEA-RAFT",
    )
    grounding_package_dir = _select_candidate_path(
        [path.parent for path in source_root.rglob("groundingdino/__init__.py")],
        package_prefix="GroundingDINO",
    )
    sam2_package_dir = _select_candidate_path(
        [path.parent for path in source_root.rglob("sam2/__init__.py")],
        package_prefix=package_name,
    )
    vfi_mamba_root = _select_candidate_path(
        [
            path.parent
            for path in source_root.rglob("VFIMamba/Trainer_finetune.py")
            if path.parent.name == "VFIMamba"
        ],
        package_prefix="VFIMamba",
    )
    overlay_root = _prepare_overlay_root(
        source_root=source_root,
        package_dir=package_dir,
        grounding_package_dir=grounding_package_dir,
        sam2_package_dir=sam2_package_dir,
        sea_raft_root=sea_raft_root,
        vfi_mamba_root=vfi_mamba_root,
    )
    return MetricSourceLayout(
        source_root=source_root,
        package_name=package_name,
        overlay_root=overlay_root,
        droid_root=droid_root,
        lietorch_root=lietorch_root,
    )


@contextmanager
def metric_runtime_context() -> Iterator[MetricSourceLayout]:
    layout = resolve_metric_source_layout()
    previous_cwd = Path.cwd()
    inserted_paths = [
        path
        for path in (
            layout.overlay_root,
            layout.droid_root,
            layout.lietorch_root,
        )
        if path is not None
    ]
    try:
        for path in reversed(inserted_paths):
            sys.path.insert(0, str(path))
        os.chdir(layout.overlay_root)
        yield layout
    finally:
        os.chdir(previous_cwd)
        for path in inserted_paths:
            path_str = str(path)
            while path_str in sys.path:
                sys.path.remove(path_str)


@lru_cache(maxsize=None)
def _load_metric_class(metric_name: str) -> type[Any]:
    with metric_runtime_context() as layout:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            pass
        module_suffix, class_name = OFFICIAL_IMPORT_TARGETS[metric_name]
        package_root = layout.overlay_root / layout.package_name
        benchmark_root = package_root / "benchmark"
        metrics_root = benchmark_root / "metrics"
        third_party_root = metrics_root / "third_party"
        _ensure_namespace_package(layout.package_name, package_root)
        _ensure_namespace_package(f"{layout.package_name}.benchmark", benchmark_root)
        _ensure_namespace_package(f"{layout.package_name}.benchmark.metrics", metrics_root)
        _ensure_namespace_package(
            f"{layout.package_name}.benchmark.metrics.third_party",
            third_party_root,
        )
        _install_base_metric_shim(layout.package_name)
        module_name = f"{layout.package_name}.{module_suffix}"
        module_path = package_root / Path(module_suffix.replace(".", "/")).with_suffix(".py")
        module = _load_module_from_path(module_name, module_path)
        return getattr(module, class_name)


def _ensure_namespace_package(module_name: str, path: Path) -> None:
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = module_name
    sys.modules[module_name] = module


def _install_base_metric_shim(package_name: str) -> None:
    module_name = f"{package_name}.benchmark.metrics.base_metrics"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
        torch = None  # type: ignore[assignment]

    class BaseMetric:
        def __init__(self) -> None:
            self._metric = None
            self._device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

        def _process_image(self, rendered_images: list[str | Any]) -> Any:
            from PIL import Image
            from torchvision import transforms

            preprocessing = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                ]
            )
            rendered_images_: list[Any] = []
            for image in rendered_images:
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                rendered_images_.append(preprocessing(image))
            if torch is None:
                return rendered_images_
            return torch.stack(rendered_images_).to(self._device)

        def _process_images(self, rendered_images: list[str | Any], reference_image: str | Any) -> Any:
            from PIL import Image
            from torchvision import transforms

            preprocessing = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                ]
            )
            if isinstance(reference_image, str):
                reference_image = Image.open(reference_image).convert("RGB")
            reference_image = preprocessing(reference_image)

            rendered_images_: list[Any] = []
            reference_images_: list[Any] = []
            for image in rendered_images:
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                rendered_images_.append(preprocessing(image))
                reference_images_.append(reference_image)

            if torch is None:
                return rendered_images_, reference_images_
            return (
                torch.stack(rendered_images_).to(self._device),
                torch.stack(reference_images_).to(self._device),
            )

        def _process_np_to_tensor(self, rendered_image: np.ndarray, reference_image: np.ndarray) -> Any:
            from torchvision.transforms import ToTensor

            img1 = ToTensor()(rendered_image).unsqueeze(0)
            img2 = ToTensor()(reference_image).unsqueeze(0)
            if torch is None:
                return img1, img2
            return img1.to(self._device), img2.to(self._device)

        def _compute_scores(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - interface shim
            raise NotImplementedError

    module.BaseMetric = BaseMetric  # type: ignore[attr-defined]
    sys.modules[module_name] = module


def _load_module_from_path(module_name: str, module_path: Path) -> types.ModuleType:
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        if isinstance(existing, types.ModuleType):
            return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module spec for {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _video_frame_cache_key(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def materialize_frame_paths(media_path: str) -> list[str]:
    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"media path does not exist: {path}")
    if path.is_dir():
        image_paths = [
            image_path
            for image_path in sorted(path.iterdir())
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        return [str(image_path) for image_path in image_paths]
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return [str(path)]

    frame_dir = _runtime_cache_dir("metric_frames") / _video_frame_cache_key(path)
    index_path = frame_dir / "index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        frame_paths = [str(frame_dir / name) for name in payload.get("frames", [])]
        if frame_paths and all(Path(frame_path).exists() for frame_path in frame_paths):
            return frame_paths

    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    frame_names: list[str] = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_name = f"{frame_index:06d}.png"
        frame_path = frame_dir / frame_name
        cv2.imwrite(str(frame_path), frame)
        frame_names.append(frame_name)
        frame_index += 1
    capture.release()
    if not frame_names:
        raise RuntimeError(f"video contains no decodable frames: {path}")
    index_path.write_text(json.dumps({"frames": frame_names}), encoding="utf-8")
    return [str(frame_dir / frame_name) for frame_name in frame_names]


def _normalize_empirical(score: float, spec: dict[str, Any]) -> float:
    empirical_max = float(spec["empirical_max"])
    empirical_min = float(spec.get("empirical_min", 0.0))
    normalized = (max(empirical_min, min(empirical_max, score)) - empirical_min) / (
        empirical_max - empirical_min
    )
    if not bool(spec.get("higher_is_better", True)):
        normalized = 1.0 - normalized
    return max(0.0, min(1.0, float(normalized)))


def _normalize_empirical_components(score: list[float], spec: dict[str, Any]) -> float:
    normalized_components: list[float] = []
    for score_i, empirical_max, empirical_min, higher_is_better in zip(
        score,
        spec["empirical_max"],
        spec.get("empirical_min", [0.0] * len(score)),
        spec["higher_is_better"],
    ):
        normalized_component = _normalize_empirical(
            float(score_i),
            {
                "empirical_max": float(empirical_max),
                "empirical_min": float(empirical_min),
                "higher_is_better": bool(higher_is_better),
            },
        )
        normalized_components.append(normalized_component)
    reducer = str(spec.get("reducer", "geometric_mean"))
    if reducer == "arithmetic_mean":
        return float(sum(normalized_components) / len(normalized_components))
    product = 1.0
    for value in normalized_components:
        product *= value
    return float(product ** (1.0 / len(normalized_components)))


def _normalize_zscore(score: float, spec: dict[str, Any]) -> float:
    avg = float(spec["avg"])
    std = float(spec["std"])
    z_max = float(spec["z_max"])
    z_min = float(spec["z_min"])
    score_range = spec.get("range", [0.25, 0.75])
    x_prime = ((float(score) - avg) / std - z_min) / (z_max - z_min)
    if not bool(spec.get("higher_is_better", True)):
        x_prime = 1.0 - x_prime
    normalized = float(score_range[0]) + (float(score_range[1]) - float(score_range[0])) * x_prime
    return max(0.0, min(1.0, normalized))


def normalize_official_score(score: Any, spec: dict[str, Any]) -> float:
    if "avg" in spec:
        return _normalize_zscore(float(score), spec)
    if isinstance(score, (list, tuple)):
        return _normalize_empirical_components([float(value) for value in score], spec)
    return _normalize_empirical(float(score), spec)


def _native_score_details(score: Any) -> dict[str, Any]:
    if isinstance(score, (list, tuple)):
        return {"native_components": [round(float(value), 4) for value in score]}
    return {"native_score": round(float(score), 4)}


def _run_metric(metric_name: str, *args: Any, **kwargs: Any) -> Any:
    metric_class = _load_metric_class(metric_name)
    try:
        return metric_class(*args, **kwargs)
    except Exception:
        _empty_cuda_cache()
        raise


def inspect_metric_source_layout() -> dict[str, Any]:
    try:
        layout = resolve_metric_source_layout()
    except Exception as exc:
        return {
            "ready": False,
            "error": str(exc),
            "configured_env": os.getenv(OFFICIAL_RUNTIME_ENV),
        }

    return {
        "ready": True,
        "configured_env": os.getenv(OFFICIAL_RUNTIME_ENV),
        "source_root": str(layout.source_root),
        "package_name": layout.package_name,
        "overlay_root": str(layout.overlay_root),
        "droid_root": str(layout.droid_root) if layout.droid_root is not None else None,
        "lietorch_root": str(layout.lietorch_root) if layout.lietorch_root is not None else None,
    }


def inspect_official_metric_import(metric_name: str) -> dict[str, Any]:
    if metric_name not in OFFICIAL_IMPORT_TARGETS:
        raise KeyError(f"unsupported official metric: {metric_name}")

    try:
        layout = resolve_metric_source_layout()
    except Exception as exc:
        return {
            "ready": False,
            "error": str(exc),
        }

    module_suffix, class_name = OFFICIAL_IMPORT_TARGETS[metric_name]
    module_name = f"{layout.package_name}.{module_suffix}"
    module_path = (
        layout.source_root
        / layout.package_name
        / Path(module_suffix.replace(".", "/")).with_suffix(".py")
    )
    if not module_path.exists():
        return {
            "ready": False,
            "module": module_name,
            "class_name": class_name,
            "source_file": str(module_path),
            "error": f"metric source file not found: {module_path}",
        }

    try:
        module_ast = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    except (OSError, SyntaxError) as exc:
        return {
            "ready": False,
            "module": module_name,
            "class_name": class_name,
            "source_file": str(module_path),
            "error": str(exc),
        }

    if not any(
        isinstance(node, ast.ClassDef) and node.name == class_name for node in module_ast.body
    ):
        return {
            "ready": False,
            "module": module_name,
            "class_name": class_name,
            "source_file": str(module_path),
            "error": f"class {class_name} not declared in {module_path}",
        }

    return {
        "ready": True,
        "class_name": class_name,
        "module": module_name,
        "source_file": str(module_path),
    }


def compute_camera_error(
    prediction_path: str,
    cameras_gt: np.ndarray,
    scale: float,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    frame_paths = materialize_frame_paths(prediction_path)
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
        raise ModuleNotFoundError("torch is required for camera_error") from exc

    with metric_runtime_context():
        metric = _run_metric("camera_error")
        try:
            score = metric._compute_scores(frame_paths, torch.as_tensor(cameras_gt), scale=scale)
        finally:
            del metric
            _empty_cuda_cache()
    raw_components = [float(score[0]), float(score[1])]
    return {
        "raw": {
            "rotation_error": round(raw_components[0], 4),
            "translation_error": round(raw_components[1], 4),
        },
        "normalized": normalize_official_score(raw_components, normalization),
        "backend": "droid_slam",
        "details": {
            "frame_count": len(frame_paths),
            "camera_scale": float(scale),
            **_native_score_details(raw_components),
        },
    }


def compute_reprojection_error(
    prediction_path: str,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    frame_paths = materialize_frame_paths(prediction_path)
    with metric_runtime_context():
        metric = _run_metric("reprojection_error")
        try:
            score = float(metric._compute_scores(frame_paths))
        finally:
            del metric
            _empty_cuda_cache()
    return {
        "raw": score,
        "normalized": normalize_official_score(score, normalization),
        "backend": "droid_slam",
        "details": {"frame_count": len(frame_paths), **_native_score_details(score)},
    }


def compute_optical_flow_aepe(
    prediction_path: str,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    frame_paths = materialize_frame_paths(prediction_path)
    with metric_runtime_context():
        metric = _run_metric("optical_flow_aepe")
        try:
            score = float(metric._compute_scores(frame_paths))
        finally:
            del metric
            _empty_cuda_cache()
    return {
        "raw": score,
        "normalized": normalize_official_score(score, normalization),
        "backend": "sea_raft",
        "details": {"frame_count": len(frame_paths), **_native_score_details(score)},
    }


def compute_motion_magnitude(
    prediction_path: str,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    frame_paths = materialize_frame_paths(prediction_path)
    with metric_runtime_context():
        metric = _run_metric("motion_magnitude")
        try:
            score = float(metric._compute_scores(frame_paths))
        finally:
            del metric
            _empty_cuda_cache()
    return {
        "raw": score,
        "normalized": normalize_official_score(score, normalization),
        "backend": "sea_raft",
        "details": {"frame_count": len(frame_paths), **_native_score_details(score)},
    }


def compute_motion_smoothness(
    prediction_path: str,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    frame_paths = materialize_frame_paths(prediction_path)
    with metric_runtime_context():
        metric = _run_metric("motion_smoothness")
        try:
            score = metric._compute_scores(frame_paths)
        finally:
            del metric
            _empty_cuda_cache()
    components = [float(score[0]), float(score[1]), float(score[2])]
    return {
        "raw": {
            "mse": round(components[0], 4),
            "ssim": round(components[1], 4),
            "lpips": round(components[2], 4),
        },
        "normalized": normalize_official_score(components, normalization),
        "backend": "vfi_mamba",
        "details": {
            "frame_count": len(frame_paths),
            "native_components": {
                "mse": round(components[0], 4),
                "ssim": round(components[1], 4),
                "lpips": round(components[2], 4),
            },
        },
    }


def compute_object_detection(
    prediction_path: str,
    prompt_string: str,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    frame_paths = materialize_frame_paths(prediction_path)
    with metric_runtime_context():
        metric = _run_metric("object_detection")
        try:
            score = float(metric._compute_scores(frame_paths, prompt_string))
        finally:
            del metric
            _empty_cuda_cache()
    return {
        "raw": score,
        "normalized": normalize_official_score(score, normalization),
        "backend": "grounding_dino",
        "details": {
            "frame_count": len(frame_paths),
            "prompt_string": prompt_string,
            **_native_score_details(score),
        },
    }


def compute_motion_accuracy(
    prediction_path: str,
    mask_dir: str,
    normalization: dict[str, Any],
    *,
    generate_type: str = "i2v",
) -> dict[str, Any]:
    frame_paths = materialize_frame_paths(prediction_path)
    mask_paths = [
        str(path)
        for path in sorted(Path(mask_dir).iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not mask_paths:
        raise FileNotFoundError(f"no masks found in {mask_dir}")
    with metric_runtime_context():
        metric = _run_metric("motion_accuracy", generate_type)
        try:
            score = float(metric._compute_scores(frame_paths, mask_paths, []))
        finally:
            del metric
            _empty_cuda_cache()
    return {
        "raw": score,
        "normalized": normalize_official_score(score, normalization),
        "backend": "sam2_sea_raft",
        "details": {
            "frame_count": len(frame_paths),
            "mask_count": len(mask_paths),
            "mask_dir": mask_dir,
            "generate_type": generate_type,
            **_native_score_details(score),
        },
    }


__all__ = [
    "OFFICIAL_RUNTIME_ENV",
    "OFFICIAL_METRIC_NAMES",
    "compute_camera_error",
    "compute_motion_accuracy",
    "compute_motion_magnitude",
    "compute_motion_smoothness",
    "compute_object_detection",
    "compute_optical_flow_aepe",
    "compute_reprojection_error",
    "inspect_metric_source_layout",
    "inspect_official_metric_import",
    "materialize_frame_paths",
    "metric_runtime_context",
    "normalize_official_score",
    "resolve_metric_source_layout",
]
