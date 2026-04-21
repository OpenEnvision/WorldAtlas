from __future__ import annotations

import json
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from worldarena.benchmark.metrics.catalog import METRIC_DIMENSIONS
from worldarena.benchmark.official_runtime import (
    inspect_metric_source_layout,
    inspect_official_metric_import,
    resolve_official_runtime_python,
)
from worldarena.benchmark.pulse_runtime import inspect_pulse_of_motion_layout
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
        "omegaconf": "omegaconf",
        "protobuf": "google.protobuf",
        "pynvml": "pynvml",
        "safetensors": "safetensors",
        "sentencepiece": "sentencepiece",
    },
    "temporal_calibration": {
        "torch": "torch",
        "huggingface_hub": "huggingface_hub",
        "omegaconf": "omegaconf",
        "pytorch_lightning": "pytorch_lightning",
        "einops": "einops",
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
TEMPORAL_METRIC_REQUIREMENTS = {
    "physical_fps_error": ("torch", "omegaconf", "huggingface_hub", "pytorch_lightning", "einops"),
    "physical_fps_pct_error": ("torch", "omegaconf", "huggingface_hub", "pytorch_lightning", "einops"),
    "physical_fps_intra_video_cv": (
        "torch",
        "omegaconf",
        "huggingface_hub",
        "pytorch_lightning",
        "einops",
    ),
}


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


def _load_official_runtime_config(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "config" / "benchmark.yaml"
    if not config_path.exists():
        return {"python_bin": "auto"}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {"python_bin": "auto"}
    benchmark_payload = payload.get("benchmark", {})
    metric_runtime = benchmark_payload.get("metric_runtime", {})
    official_runtime = metric_runtime.get("official_source", {})
    if not isinstance(official_runtime, dict):
        return {"python_bin": "auto"}
    return {"python_bin": "auto", **official_runtime}


def _probe_dependency_status_in_runtime(
    python_bin: str,
    modules: dict[str, str],
    *,
    env_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    script = """
import importlib.util
import json
import platform
import sys

modules = json.loads(sys.argv[1])
status = {name: bool(importlib.util.find_spec(module)) for name, module in modules.items()}
spacy_model = False
if status.get("spacy"):
    try:
        import spacy
        spacy_model = bool(spacy.util.is_package("en_core_web_sm"))
    except Exception:
        spacy_model = False
print(json.dumps({
    "python": {
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    },
    "modules": status,
    "spacy_model": spacy_model,
}))
""".strip()
    env = os.environ.copy()
    for key, value in dict(env_overrides or {}).items():
        env[str(key)] = str(value)
    try:
        result = subprocess.run(
            [python_bin, "-c", script, json.dumps(modules, ensure_ascii=False)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stdout = getattr(exc, "stdout", "") or ""
        stderr = getattr(exc, "stderr", "") or ""
        error = " | ".join(part for part in (str(exc), stderr.strip(), stdout.strip()) if part)
        return {
            "ready": False,
            "python_bin": python_bin,
            "error": error,
            "modules": {},
            "spacy_model": False,
        }
    payload = json.loads(result.stdout)
    return {
        "ready": True,
        "python_bin": python_bin,
        "python": payload["python"],
        "modules": dict(payload["modules"]),
        "spacy_model": bool(payload["spacy_model"]),
        "error": None,
    }


def _official_runtime_status(project_root: Path) -> dict[str, Any]:
    runtime_config = _load_official_runtime_config(project_root)
    python_bin = resolve_official_runtime_python(runtime_config)
    if python_bin is None:
        return {
            "python_bin": None,
            "config": runtime_config,
            "ready": False,
            "uses_side_runtime": False,
            "modules": {},
            "spacy_model": False,
            "error": None,
        }
    modules = {
        "torch": "torch",
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
    }
    probe = _probe_dependency_status_in_runtime(
        python_bin,
        modules,
        env_overrides=runtime_config.get("env"),
    )
    probe["config"] = runtime_config
    probe["uses_side_runtime"] = True
    return probe


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


def _temporal_metric_status(
    package_status: dict[str, bool],
    pulse_layout: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metrics_by_dimension: dict[str, dict[str, Any]] = {}
    for metric_name, requirements in TEMPORAL_METRIC_REQUIREMENTS.items():
        missing = [dependency for dependency in requirements if not package_status.get(dependency, False)]
        dimension = METRIC_DIMENSIONS.get(metric_name, "uncategorized")
        metrics_by_dimension.setdefault(dimension, {})[metric_name] = {
            "backend_type": "external_source",
            "ready": bool(pulse_layout.get("ready")) and not missing,
            "requirements": list(requirements),
            "missing": missing,
            "source_root": pulse_layout.get("root"),
            "error": None
            if bool(pulse_layout.get("ready")) and not missing
            else "; ".join(
                part
                for part in [
                    None if pulse_layout.get("ready") else str(pulse_layout.get("error")),
                    None if not missing else f"missing dependencies: {', '.join(missing)}",
                ]
                if part
            ),
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
    official_runtime = _official_runtime_status(project_root)
    official_package_status = dict(package_status)
    if official_runtime.get("ready"):
        official_package_status.update(dict(official_runtime.get("modules", {})))
        official_package_status["spacy_model"] = bool(official_runtime.get("spacy_model"))
    metric_source = inspect_metric_source_layout()
    pulse_layout = inspect_pulse_of_motion_layout()

    metric_backends = _builtin_metric_status(package_status)
    for dimension, metric_map in _official_metric_status(official_package_status, metric_source).items():
        metric_backends.setdefault(dimension, {}).update(metric_map)
    for dimension, metric_map in _temporal_metric_status(package_status, pulse_layout).items():
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
        "official_runtime": official_runtime,
        "pulse_of_motion": pulse_layout,
        "metrics_by_dimension": metric_backends,
        "gpus": _gpu_status(),
    }


__all__ = ["inspect_environment"]
