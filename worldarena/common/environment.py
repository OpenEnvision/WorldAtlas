from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.official_runtime import (
    inspect_metric_source_layout,
    inspect_official_metric_import,
)
from worldarena.common.runtime_layout import inspect_runtime_layout


DEPENDENCY_PROFILES = {
    "core": {
        "numpy": "numpy",
        "opencv": "cv2",
        "pandas": "pandas",
        "yaml": "yaml",
    },
    "reference_metrics": {
        "torch": "torch",
        "torchmetrics": "torchmetrics",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "pyiqa": "pyiqa",
    },
    "official_video_metrics": {
        "hydra": "hydra",
        "omegaconf": "omegaconf",
        "cvxpy": "cvxpy",
        "iopath": "iopath",
        "pycocotools": "pycocotools",
        "spacy": "spacy",
        "scipy": "scipy",
        "timm": "timm",
        "einops": "einops",
        "causal_conv1d": "causal_conv1d",
        "mamba_ssm": "mamba_ssm",
        "tqdm": "tqdm",
    },
    "model_adapters": {
        "diffusers": "diffusers",
        "decord": "decord",
        "dashscope": "dashscope",
        "easydict": "easydict",
        "ftfy": "ftfy",
        "imageio": "imageio",
        "modelscope": "modelscope",
        "protobuf": "google.protobuf",
        "pynvml": "pynvml",
        "safetensors": "safetensors",
        "sentencepiece": "sentencepiece",
    },
}
BUILTIN_METRIC_REQUIREMENTS = {
    "prompt_alignment": ("torch", "torchmetrics", "transformers"),
    "style_consistency": ("torch", "torchvision"),
    "perceptual_quality": ("torch", "pyiqa"),
    "reference_appearance_alignment": ("torch", "transformers"),
    "anchor_consistency": ("torch", "transformers"),
}
OFFICIAL_METRIC_REQUIREMENTS = {
    "camera_error": ("torch", "hydra", "omegaconf", "scipy", "cvxpy"),
    "object_detection": ("torch", "spacy", "spacy_model", "pycocotools"),
    "reprojection_error": ("torch", "hydra", "omegaconf", "scipy"),
    "optical_flow_aepe": ("torch", "hydra", "omegaconf", "scipy"),
    "motion_accuracy": ("torch", "spacy", "spacy_model", "pycocotools", "iopath"),
    "motion_magnitude": ("torch", "hydra", "omegaconf", "scipy"),
    "motion_smoothness": ("torch", "timm", "einops", "causal_conv1d", "mamba_ssm", "tqdm"),
}
OFFICIAL_METRIC_ORDER = (
    "camera_error",
    "object_detection",
    "reprojection_error",
    "optical_flow_aepe",
    "motion_accuracy",
    "motion_magnitude",
    "motion_smoothness",
)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _dependency_profile_status() -> dict[str, dict[str, bool]]:
    return {
        profile_name: {
            dependency_name: _module_available(module_name)
            for dependency_name, module_name in dependencies.items()
        }
        for profile_name, dependencies in DEPENDENCY_PROFILES.items()
    }


def _spacy_model_available() -> bool:
    if not _module_available("spacy"):
        return False
    try:
        import spacy
    except ModuleNotFoundError:
        return False
    return bool(spacy.util.is_package("en_core_web_sm"))


def _gpu_status() -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_used_mb": parts[2],
                "memory_total_mb": parts[3],
                "utilization_gpu_pct": parts[4],
            }
        )
    return rows


def _flatten_dependency_status(
    dependency_profiles: dict[str, dict[str, bool]],
    *,
    spacy_model_available: bool,
) -> dict[str, bool]:
    flattened: dict[str, bool] = {}
    for dependencies in dependency_profiles.values():
        flattened.update(dependencies)
    flattened["spacy_model"] = spacy_model_available
    return flattened


def _builtin_metric_status(package_status: dict[str, bool]) -> dict[str, Any]:
    metrics_by_dimension: dict[str, dict[str, Any]] = {}
    for metric_name, requirements in BUILTIN_METRIC_REQUIREMENTS.items():
        missing = [dependency for dependency in requirements if not package_status.get(dependency, False)]
        dimension = METRIC_DIMENSIONS.get(metric_name, "uncategorized")
        metrics_by_dimension.setdefault(dimension, {})[metric_name] = {
            "backend_type": "builtin",
            "ready": not missing,
            "requirements": list(requirements),
            "missing": missing,
        }
    return metrics_by_dimension


def _official_metric_status(
    package_status: dict[str, bool],
    metric_source: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metrics_by_dimension: dict[str, dict[str, Any]] = {}
    for metric_name in OFFICIAL_METRIC_ORDER:
        requirements = OFFICIAL_METRIC_REQUIREMENTS.get(metric_name, ())
        missing = [dependency for dependency in requirements if not package_status.get(dependency, False)]
        source_entry = inspect_official_metric_import(metric_name)
        ready = bool(metric_source.get("ready")) and bool(source_entry.get("ready")) and not missing
        error_parts: list[str] = []
        if not metric_source.get("ready"):
            error_parts.append(str(metric_source.get("error", "metric source not ready")))
        if not source_entry.get("ready"):
            error_parts.append(str(source_entry.get("error", "metric entry not ready")))
        if missing:
            error_parts.append(f"missing dependencies: {', '.join(missing)}")
        dimension = METRIC_DIMENSIONS.get(metric_name, "uncategorized")
        metrics_by_dimension.setdefault(dimension, {})[metric_name] = {
            "backend_type": "official_source",
            "ready": ready,
            "requirements": list(requirements),
            "missing": missing,
            "metric_source_ready": bool(metric_source.get("ready")),
            "source_entry_ready": bool(source_entry.get("ready")),
            "module": source_entry.get("module"),
            "class_name": source_entry.get("class_name"),
            "source_file": source_entry.get("source_file"),
            "error": "; ".join(error_parts) if error_parts else None,
        }
    return metrics_by_dimension


def inspect_environment(project_root: Path) -> dict[str, Any]:
    layout = inspect_runtime_layout(project_root)
    dependency_profiles = _dependency_profile_status()
    spacy_model_available = _spacy_model_available()
    package_status = _flatten_dependency_status(
        dependency_profiles,
        spacy_model_available=spacy_model_available,
    )
    metric_source = inspect_metric_source_layout()

    metric_backends = _builtin_metric_status(package_status)
    for dimension, metric_map in _official_metric_status(package_status, metric_source).items():
        metric_backends.setdefault(dimension, {}).update(metric_map)

    return {
        "project_root": str(project_root.resolve()),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "layout": layout,
        "dependency_profiles": {
            **dependency_profiles,
            "spacy_model": {"en_core_web_sm": spacy_model_available},
        },
        "metric_source": metric_source,
        "metrics_by_dimension": metric_backends,
        "gpus": _gpu_status(),
    }


__all__ = ["inspect_environment"]
