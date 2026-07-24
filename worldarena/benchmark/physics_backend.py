"""Physics simulation backend for WorldArena physics benchmark metrics."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from worldarena.benchmark.physics_manifest import REQUIRED_CORE_METRICS
from worldarena.benchmark.physics_simulator_derived import (
    append_derived_simulator_metrics,
    ensure_pred_worldlines,
)
from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.benchmark.taxonomy import (
    artifact_type_for_modality,
    benchmark_split,
    control_signals_for_sample,
    task_family_for_suite,
)


WORLD_GT_REQUIRED_FILENAMES: tuple[str, ...] = ("rgb.mp4",)
WORLD_GT_METRIC_FILENAMES: tuple[str, ...] = (
    "rgb.mp4",
    "depth.npz",
    "flow.npz",
    "mask.npz",
    "occlusion.npz",
    "cameras.npz",
    "points4d.npz",
    "worldlines.npz",
)
WORLD_GT_OPTIONAL_FILENAMES: tuple[str, ...] = (
    "novel_time_depth.npz",
    "novel_time_rgb.npz",
)
WORLD_GT_DIRNAMES: tuple[str, ...] = ("rgb_frames",)
PRED_WORLD_REQUIRED_FILENAMES: tuple[str, ...] = (
    "depth.npz",
    "mask.npz",
    "flow.npz",
    "cameras.npz",
)
PRED_WORLD_OPTIONAL_FILENAMES: tuple[str, ...] = (
    "occlusion.npz",
    "points4d.npz",
    "worldlines.npz",
    "novel_time_depth.npz",
    "novel_time_rgb.npz",
)
SYNC_FILE_NAMES: tuple[str, ...] = (
    WORLD_GT_METRIC_FILENAMES + WORLD_GT_OPTIONAL_FILENAMES + ("export_info.json",)
)
SYNC_DIR_NAMES: tuple[str, ...] = WORLD_GT_DIRNAMES
AUTO_PRED_WORLD_ENV = "WORLDARENA_PHYSICS_DISABLE_AUTO_PRED_WORLD"
PHYSICS_PYTHON_BIN_ENV = "WORLDARENA_PHYSICS_PYTHON_BIN"
ONLINE_PHYSICS_INTERNAL_SAMPLE_ID = "0000"


@dataclass(frozen=True)
class PhysicsCaseSpec:
    physics_pipeline_root: Path
    physics_data_root: Path
    source_sample_id: str
    benchmark_sample_id: str
    case_root: Path
    simulator_root: Path
    submitted_video_path: Path
    metadata_template_path: Path
    world_gt_root: Path
    export_benchmark_name: str
    render_split: str


def _utc_timestamp_slug() -> str:
    """Utc timestamp slug -> str."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sample_payload(sample: BenchmarkSample | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(sample, BenchmarkSample):
        return sample.to_dict()
    return sample


def _present_metadata(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _clean_metadata_overrides(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        str(key): value
        for key, value in dict(payload).items()
        if str(key)
        not in {
            "conditioning_tracks",
            "video_prompt",
            "fg_prompt",
            "primary",
            "secondary",
            "submitted_video_path",
        }
        and _present_metadata(value)
    }


def _coalesce_metadata_str(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _conditioning_tracks_from_metadata(payload: Mapping[str, Any]) -> list[str]:
    value = payload.get("conditioning_tracks")
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else ["i2v"]
    if isinstance(value, (list, tuple)):
        tracks = [str(item).strip() for item in value if str(item).strip()]
        return tracks or ["i2v"]
    return ["i2v"]


def parse_case_spec(sample: BenchmarkSample | Mapping[str, Any]) -> PhysicsCaseSpec:
    payload = dict(_sample_payload(sample).get("physics_case") or {})
    if not payload:
        raise ValueError("sample metadata is missing physics_case")
    return PhysicsCaseSpec(
        physics_pipeline_root=Path(payload["physics_pipeline_root"]).expanduser().resolve(),
        physics_data_root=Path(payload["physics_data_root"]).expanduser().resolve(),
        source_sample_id=str(payload["source_sample_id"]),
        benchmark_sample_id=str(payload.get("benchmark_sample_id") or payload["source_sample_id"]),
        case_root=Path(payload["case_root"]).expanduser().resolve(),
        simulator_root=Path(payload["simulator_root"]).expanduser().resolve(),
        submitted_video_path=Path(payload["submitted_video_path"]).expanduser().resolve(),
        metadata_template_path=Path(payload["metadata_template_path"]).expanduser().resolve(),
        world_gt_root=Path(payload["world_gt_root"]).expanduser().resolve(),
        export_benchmark_name=str(payload.get("physics_export_benchmark_name", "worldarena_physics")),
        render_split=str(payload.get("physics_render_split", "original_length")),
    )


def _physics_data_root(case: PhysicsCaseSpec) -> Path:
    return case.physics_data_root.resolve()


def _exported_world_gt_root(case: PhysicsCaseSpec) -> Path:
    return (
        _physics_data_root(case)
        / "OUT_Benchmark"
        / case.export_benchmark_name
        / "cases"
        / case.source_sample_id
        / "world_gt"
    )


def world_gt_video_ready(world_gt_root: Path) -> bool:
    return all((world_gt_root / filename).exists() for filename in WORLD_GT_REQUIRED_FILENAMES)


def world_gt_metrics_ready(world_gt_root: Path) -> bool:
    return all((world_gt_root / filename).exists() for filename in WORLD_GT_METRIC_FILENAMES)


def pred_world_ready(pred_world_root: Path) -> bool:
    return all((pred_world_root / filename).exists() for filename in PRED_WORLD_REQUIRED_FILENAMES)


def _inspect_named_paths(paths: Mapping[str, Path]) -> dict[str, Any]:
    items = {
        name: {
            "path": str(path.resolve()),
            "exists": path.exists(),
        }
        for name, path in paths.items()
    }
    missing = [name for name, payload in items.items() if not payload["exists"]]
    return {
        "ready": not missing,
        "missing": missing,
        "items": items,
    }


def inspect_world_gt_root(world_gt_root: Path) -> dict[str, Any]:
    metric_artifacts = _inspect_named_paths(
        {name: world_gt_root / name for name in WORLD_GT_METRIC_FILENAMES}
    )
    extra_artifacts = _inspect_named_paths(
        {name: world_gt_root / name for name in WORLD_GT_OPTIONAL_FILENAMES}
    )
    support_artifacts = _inspect_named_paths(
        {name: world_gt_root / name for name in WORLD_GT_DIRNAMES + ("export_info.json",)}
    )
    return {
        "root": str(world_gt_root.resolve()),
        "exists": world_gt_root.exists(),
        "video_ready": world_gt_video_ready(world_gt_root),
        "full_metrics_ready": metric_artifacts["ready"],
        "metric_artifacts": metric_artifacts,
        "extra_artifacts": extra_artifacts,
        "support_artifacts": support_artifacts,
    }


def inspect_pred_world_root(pred_world_root: Path) -> dict[str, Any]:
    required_artifacts = _inspect_named_paths(
        {name: pred_world_root / name for name in PRED_WORLD_REQUIRED_FILENAMES}
    )
    optional_artifacts = _inspect_named_paths(
        {name: pred_world_root / name for name in PRED_WORLD_OPTIONAL_FILENAMES}
    )
    return {
        "root": str(pred_world_root.resolve()),
        "exists": pred_world_root.exists(),
        "ready": required_artifacts["ready"],
        "required_artifacts": required_artifacts,
        "optional_artifacts": optional_artifacts,
    }


def _physics_subprocess_env(case: PhysicsCaseSpec, *, data_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["WORLDARENA_PHYSICS_PIPELINE_ROOT"] = str(case.physics_pipeline_root)
    env["WORLDARENA_PHYSICS_DATA_ROOT"] = str((data_root or _physics_data_root(case)).resolve())
    return env


def _resolve_physics_python_bin(case: PhysicsCaseSpec) -> str:
    override = os.environ.get(PHYSICS_PYTHON_BIN_ENV, "").strip()
    if override:
        return str(Path(override).expanduser())

    case_payload = None
    try:
        case_payload = json.loads(case.metadata_template_path.read_text(encoding="utf-8"))
    except Exception:
        case_payload = None
    if isinstance(case_payload, Mapping):
        python_bin = str(case_payload.get("python_bin", "")).strip()
        if python_bin:
            return str(Path(python_bin).expanduser())
    return sys.executable


def _benchmark_export_command(case: PhysicsCaseSpec) -> str:
    python_bin = _resolve_physics_python_bin(case)
    return " ".join(
        [
            shlex_quote(python_bin),
            shlex_quote(str(case.physics_pipeline_root / "benchmark_export.py")),
            "--benchmark-name",
            shlex_quote(case.export_benchmark_name),
            "--sample-ids",
            shlex_quote(case.source_sample_id),
            "--render-split",
            shlex_quote(case.render_split),
            "--overwrite",
        ]
    )


def _benchmark_export_invocation(case: PhysicsCaseSpec) -> tuple[list[str], Path]:
    python_bin = _resolve_physics_python_bin(case)
    return (
        [
            python_bin,
            str(case.physics_pipeline_root / "benchmark_export.py"),
            "--benchmark-name",
            case.export_benchmark_name,
            "--sample-ids",
            case.source_sample_id,
            "--render-split",
            case.render_split,
            "--overwrite",
        ],
        case.physics_pipeline_root,
    )


def _benchmark_eval_invocation(
    case: PhysicsCaseSpec,
    *,
    manifest_path: Path,
    pred_dir: Path,
    output_path: Path,
    pred_world_root: Path | None = None,
) -> tuple[list[str], Path]:
    python_bin = _resolve_physics_python_bin(case)
    command = [
        python_bin,
        str(case.physics_pipeline_root / "benchmark_eval.py"),
        "--manifest",
        str(manifest_path),
        "--pred-dir",
        str(pred_dir),
        "--output",
        str(output_path),
        "--skip-rgb-warp-lpips",
    ]
    if pred_world_root is not None:
        command.extend(["--pred-world-root", str(pred_world_root)])
    return command, case.physics_pipeline_root


def _main_part2_invocation(case: PhysicsCaseSpec) -> tuple[list[str], Path]:
    python_bin = _resolve_physics_python_bin(case)
    return (
        [
            python_bin,
            str(case.physics_pipeline_root / "main_part2.py"),
            "--video",
            case.source_sample_id,
        ],
        case.physics_pipeline_root,
    )


def _run_logged_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    step_name: str,
) -> tuple[bool, str | None]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "step": step_name,
                "command": command,
                "cwd": str(cwd),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode == 0:
        return True, None
    return (
        False,
        (
            f"{step_name} failed with code {completed.returncode}. "
            f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        ),
    )


def ensure_submission_staged(
    sample: BenchmarkSample | Mapping[str, Any],
    generated_video_path: str | Path,
) -> PhysicsCaseSpec:
    payload = _sample_payload(sample)
    case = parse_case_spec(sample)
    case.simulator_root.mkdir(parents=True, exist_ok=True)
    case.submitted_video_path.parent.mkdir(parents=True, exist_ok=True)
    generated_video_path = Path(generated_video_path).expanduser().resolve()
    if generated_video_path != case.submitted_video_path:
        shutil.copy2(generated_video_path, case.submitted_video_path)

    metadata = {}
    if case.metadata_template_path.exists():
        metadata = json.loads(case.metadata_template_path.read_text(encoding="utf-8"))
    if not metadata:
        physics_case = dict(payload.get("physics_case") or {})
        inline_template = physics_case.get("metadata_template")
        if isinstance(inline_template, Mapping):
            metadata = dict(inline_template)
    if not metadata:
        physics_spec = dict(payload.get("physics_spec") or {})
        inline_template = physics_spec.get("metadata_template")
        if isinstance(inline_template, Mapping):
            metadata = dict(inline_template)
    if not metadata:
        physics_case = dict(payload.get("physics_case") or {})
        physics_spec = dict(payload.get("physics_spec") or {})
        case_taxonomy = dict(physics_spec.get("case_taxonomy") or physics_case.get("case_taxonomy") or {})
        conditioning_tracks = ["i2v"] if str(payload.get("track") or "").strip() == "physics_i2v" else ["t2v"]
        metadata = {
            "conditioning_tracks": conditioning_tracks,
            "video_prompt": str(
                payload.get("prompt_current") or payload.get("prompt_target") or ""
            ).strip(),
            "fg_prompt": str(physics_spec.get("fg_prompt") or "").strip(),
            "primary": str(physics_spec.get("primary_object") or "").strip(),
            "secondary": str(physics_spec.get("secondary_object") or "").strip(),
        }
        metadata.update(case_taxonomy)
    metadata["submitted_video_path"] = str(case.submitted_video_path)
    case.metadata_template_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return case


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _sync_world_gt(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for name in SYNC_FILE_NAMES:
        _remove_path(target_root / name)
    for name in SYNC_DIR_NAMES:
        _remove_path(target_root / name)
    for name in SYNC_FILE_NAMES:
        source = source_root / name
        if source.exists():
            (target_root / name).symlink_to(source.resolve())
    for name in SYNC_DIR_NAMES:
        source = source_root / name
        if source.exists():
            (target_root / name).symlink_to(source.resolve(), target_is_directory=True)


def try_export_world_gt(case: PhysicsCaseSpec) -> tuple[bool, str | None]:
    if world_gt_metrics_ready(case.world_gt_root):
        return True, None

    exported_root = _exported_world_gt_root(case)
    if world_gt_metrics_ready(exported_root) or world_gt_video_ready(exported_root):
        _sync_world_gt(exported_root, case.world_gt_root)
        if world_gt_metrics_ready(case.world_gt_root):
            return True, None
        return False, "synced partial world_gt but metric artifacts are still missing"

    export_script = case.physics_pipeline_root / "benchmark_export.py"
    if not export_script.exists():
        return False, f"missing physics export script: {export_script}"

    command, cwd = _benchmark_export_invocation(case)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=_physics_subprocess_env(case),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return (
            False,
            (
                "physics export failed. "
                f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
            ),
        )

    if world_gt_metrics_ready(exported_root) or world_gt_video_ready(exported_root):
        _sync_world_gt(exported_root, case.world_gt_root)
    if world_gt_metrics_ready(case.world_gt_root):
        return True, None
    return False, f"physics export completed but world_gt is still incomplete: {exported_root}"


def inspect_physics_case(sample: BenchmarkSample | Mapping[str, Any]) -> dict[str, Any]:
    case = parse_case_spec(sample)
    scripts = _inspect_named_paths(
        {
            "benchmark_export.py": case.physics_pipeline_root / "benchmark_export.py",
            "benchmark_eval.py": case.physics_pipeline_root / "benchmark_eval.py",
            "main_part2.py": case.physics_pipeline_root / "main_part2.py",
        }
    )
    local_world_gt = inspect_world_gt_root(case.world_gt_root)
    exported_world_gt = inspect_world_gt_root(_exported_world_gt_root(case))

    if local_world_gt["full_metrics_ready"]:
        status = "ready"
    elif exported_world_gt["full_metrics_ready"]:
        status = "needs_sync"
    elif not scripts["items"]["benchmark_export.py"]["exists"] or not scripts["items"]["benchmark_eval.py"]["exists"]:
        status = "missing_backend"
    elif local_world_gt["video_ready"]:
        status = "partial_world_gt"
    else:
        status = "needs_export"

    remediation: list[dict[str, Any]] = []
    if status == "needs_export":
        remediation.append(
            {
                "action": "run_benchmark_export",
                "message": "export the simulator-backed world_gt before trusting physics metrics",
                "command": _benchmark_export_command(case),
            }
        )
    elif status == "needs_sync":
        remediation.append(
            {
                "action": "sync_world_gt",
                "message": "a complete exported world_gt exists and can be synced locally",
            }
        )

    return {
        "sample_id": case.benchmark_sample_id,
        "source_sample_id": case.source_sample_id,
        "track": _sample_payload(sample).get("track"),
        "status": status,
        "physics_pipeline_root": str(case.physics_pipeline_root),
        "physics_data_root": str(_physics_data_root(case)),
        "render_split": case.render_split,
        "scripts": scripts,
        "local_world_gt": local_world_gt,
        "exported_world_gt": exported_world_gt,
        "remediation": remediation,
    }


def preflight_physics_benchmark(samples: list[BenchmarkSample | Mapping[str, Any]]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample in samples:
        case = parse_case_spec(sample)
        export_message = None
        if not world_gt_metrics_ready(case.world_gt_root):
            _, export_message = try_export_world_gt(case)
        diagnostics = inspect_physics_case(sample)
        record = {
            "sample_id": case.benchmark_sample_id,
            "status": diagnostics["status"],
            "world_gt_root": diagnostics["local_world_gt"]["root"],
            "missing_metric_artifacts": diagnostics["local_world_gt"]["metric_artifacts"]["missing"],
            "export_message": export_message,
            "remediation": diagnostics["remediation"],
        }
        checked.append(record)
        if diagnostics["status"] != "ready":
            failures.append(record)
    return {
        "ready": not failures,
        "num_samples": len(checked),
        "samples": checked,
        "failures": failures,
    }


def _build_eval_manifest_item(case: PhysicsCaseSpec, stem: str) -> dict[str, Any]:
    world_gt = {"root": str(case.world_gt_root.resolve())}
    key_map = {
        "rgb.mp4": "rgb_video_path",
        "depth.npz": "depth_path",
        "flow.npz": "flow_path",
        "mask.npz": "mask_path",
        "cameras.npz": "camera_path",
        "points4d.npz": "points4d_path",
        "worldlines.npz": "worldlines_path",
        "occlusion.npz": "occlusion_path",
    }
    for filename, key in key_map.items():
        path = case.world_gt_root / filename
        if path.exists():
            world_gt[key] = str(path.resolve())
    for name in ("novel_time_depth", "novel_time_rgb"):
        path = case.world_gt_root / f"{name}.npz"
        if path.exists():
            world_gt[f"{name}_path"] = str(path.resolve())
    rgb_frames_dir = case.world_gt_root / "rgb_frames"
    if rgb_frames_dir.exists():
        world_gt["rgb_frames_dir"] = str(rgb_frames_dir.resolve())
    return {
        "sample_id": stem,
        "stem": stem,
        "world_gt": world_gt,
    }


def _run_eval_script(
    *,
    case: PhysicsCaseSpec,
    pred_video_path: Path,
    output_path: Path,
    pred_world_root: Path | None = None,
    sample_stem: str | None = None,
) -> dict[str, Any]:
    eval_script = case.physics_pipeline_root / "benchmark_eval.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"missing physics eval script: {eval_script}")

    with tempfile.TemporaryDirectory(prefix="worldarena_physics_eval_") as temp_dir:
        temp_root = Path(temp_dir)
        pred_dir = temp_root / "pred"
        pred_dir.mkdir(parents=True, exist_ok=True)
        stem = sample_stem or pred_video_path.stem
        staged_pred = pred_dir / f"{stem}{pred_video_path.suffix}"
        if staged_pred.resolve() != pred_video_path.resolve():
            shutil.copy2(pred_video_path, staged_pred)
        manifest_path = temp_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "benchmark_name": "worldarena_physics_eval",
                    "items": [_build_eval_manifest_item(case, stem)],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        command, cwd = _benchmark_eval_invocation(
            case,
            manifest_path=manifest_path,
            pred_dir=pred_dir,
            output_path=output_path,
            pred_world_root=pred_world_root,
        )
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=_physics_subprocess_env(case),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "physics evaluation failed. "
                f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
            )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _required_core_metric_names(sample: BenchmarkSample | Mapping[str, Any]) -> tuple[str, ...]:
    payload = _sample_payload(sample)
    physics_spec = dict(payload.get("physics_spec") or {})
    metric_contract = dict(physics_spec.get("metric_contract") or {})
    required = metric_contract.get("required_core_metrics")
    if required:
        return tuple(str(metric_name) for metric_name in required)
    return REQUIRED_CORE_METRICS


def _pred_world_artifacts_required(sample: BenchmarkSample | Mapping[str, Any]) -> bool:
    payload = _sample_payload(sample)
    physics_spec = dict(payload.get("physics_spec") or {})
    metric_contract = dict(physics_spec.get("metric_contract") or {})
    return bool(metric_contract.get("require_pred_world_artifacts"))


def _metric_contract_violations(
    sample: BenchmarkSample | Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    missing_metrics = [
        metric_name
        for metric_name in _required_core_metric_names(sample)
        if not isinstance(result.get(metric_name), (int, float)) or isinstance(result.get(metric_name), bool)
    ]
    messages: list[str] = []
    if not bool(result.get("frame_count_match", True)):
        messages.append("prediction frame count does not match the reference rollout")
    frame_coverage = result.get("frame_coverage")
    if isinstance(frame_coverage, (int, float)) and float(frame_coverage) < 1.0 - 1e-6:
        messages.append(f"prediction covers only {float(frame_coverage):.3f} of the reference rollout")
    if missing_metrics:
        messages.append("missing required core physics metrics: " + ", ".join(missing_metrics))
    return {
        "required_core_metrics": list(_required_core_metric_names(sample)),
        "missing_required_metrics": missing_metrics,
        "messages": messages,
    }


def _auto_pred_world_enabled() -> bool:
    """Auto pred world enabled -> bool."""
    return os.environ.get(AUTO_PRED_WORLD_ENV, "").strip().lower() not in {"1", "true", "yes"}


def _stage_runtime_inputs(
    case: PhysicsCaseSpec,
    *,
    generated_video_path: Path,
    data_root: Path,
) -> dict[str, str]:
    input_data_dir = data_root / "INPUT_DATA"
    videos_dir = input_data_dir / "Videos"
    meta_dir = input_data_dir / "Metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    staged_video_path = videos_dir / f"{case.source_sample_id}.mp4"
    shutil.copy2(generated_video_path, staged_video_path)

    metadata: dict[str, Any] = {}
    if case.metadata_template_path.exists():
        metadata = json.loads(case.metadata_template_path.read_text(encoding="utf-8"))
    metadata["submitted_video_path"] = str(generated_video_path)
    meta_path = meta_dir / f"{case.source_sample_id}.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "staged_video_path": str(staged_video_path),
        "metadata_path": str(meta_path),
    }


def _load_indexed_npz_array(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if "data" in payload:
        data = np.asarray(payload["data"])
    elif len(payload.files) == 1:
        data = np.asarray(payload[payload.files[0]])
    else:
        raise KeyError(f"could not infer data array from {path}; found keys {payload.files}")
    inds = np.asarray(payload["inds"]) if "inds" in payload else np.arange(len(data))
    order = np.argsort(inds)
    return np.asarray(data[order]), np.asarray(inds[order])


def _decode_archived_array(raw: bytes, *, suffix: str) -> np.ndarray:
    if suffix == ".npy":
        return np.asarray(np.load(io.BytesIO(raw), allow_pickle=False))
    if suffix == ".npz":
        payload = np.load(io.BytesIO(raw), allow_pickle=False)
        if "data" in payload:
            return np.asarray(payload["data"])
        if len(payload.files) == 1:
            return np.asarray(payload[payload.files[0]])
        raise KeyError(f"could not infer npz payload; found keys {payload.files}")
    if suffix in {".png", ".jpg", ".jpeg"}:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"failed to decode archived image with suffix {suffix}")
        if image.ndim == 3:
            image = image[..., 0]
        return np.asarray(image)
    if suffix == ".exr":
        try:
            import Imath
            import OpenEXR

            with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as temp_file:
                temp_file.write(raw)
                temp_path = Path(temp_file.name)
            try:
                exr_file = OpenEXR.InputFile(str(temp_path))
                header = exr_file.header()
                data_window = header["dataWindow"]
                width = data_window.max.x - data_window.min.x + 1
                height = data_window.max.y - data_window.min.y + 1
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                channel_names = tuple(header.get("channels", {}).keys())
                for channel_name in ("Z", "Y", "R"):
                    if channel_name in channel_names:
                        return np.frombuffer(
                            exr_file.channel(channel_name, pixel_type),
                            dtype=np.float32,
                        ).reshape(height, width)
                if channel_names:
                    return np.frombuffer(
                        exr_file.channel(channel_names[0], pixel_type),
                        dtype=np.float32,
                    ).reshape(height, width)
                raise ValueError("failed to locate any EXR channels")
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            image = cv2.imdecode(
                np.frombuffer(raw, dtype=np.uint8),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
            if image is None:
                raise ValueError("failed to decode archived exr payload")
            if image.ndim == 3:
                image = image[..., 0]
            return np.asarray(image)
    raise ValueError(f"unsupported archived array suffix: {suffix}")


def _read_zip_arrays(zip_path: Path, *, suffixes: tuple[str, ...]) -> list[np.ndarray]:
    with zipfile.ZipFile(zip_path) as archive:
        member_names = sorted(
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in suffixes
        )
        if not member_names:
            raise FileNotFoundError(
                f"no members with suffixes {suffixes} found in archive {zip_path}"
            )
        return [
            _decode_archived_array(
                archive.read(name),
                suffix=Path(name).suffix.lower(),
            )
            for name in member_names
        ]


def _load_mask_map(txt_path: Path) -> dict[int, str]:
    mask_map: dict[int, str] = {}
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        mask_id, mask_name = line.split(": ", 1)
        mask_map[int(mask_id)] = mask_name.strip()
    return mask_map


def _resize_array_hw(
    array: np.ndarray,
    *,
    target_hw: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    target_h, target_w = target_hw
    if array.shape[:2] == (target_h, target_w):
        return array
    return cv2.resize(array, (target_w, target_h), interpolation=interpolation)


def _mask_ids_for_objects(
    mask_map: Mapping[int, str],
    *,
    primary_object: str | None,
    secondary_object: str | None,
) -> tuple[list[int], list[int]]:
    normalized = {mask_id: name.strip().lower() for mask_id, name in mask_map.items()}
    primary = (primary_object or "").strip().lower()
    secondary = (secondary_object or "").strip().lower()
    object_ids = [
        mask_id
        for mask_id, mask_name in normalized.items()
        if mask_name in {primary, secondary} and mask_name
    ]
    background_ids = [
        mask_id
        for mask_id, mask_name in normalized.items()
        if mask_name in {"background", "bg", "sky"}
    ]
    if not object_ids:
        object_ids = [
            mask_id
            for mask_id, mask_name in normalized.items()
            if mask_name not in {"background", "bg", "sky"}
        ]
    return object_ids, background_ids


def _build_instance_mask_sequence(
    mask_frames: list[np.ndarray],
    *,
    mask_map: Mapping[int, str],
    primary_object: str | None,
    secondary_object: str | None,
    target_hw: tuple[int, int],
) -> np.ndarray:
    object_ids, background_ids = _mask_ids_for_objects(
        mask_map,
        primary_object=primary_object,
        secondary_object=secondary_object,
    )
    masks = []
    for frame in mask_frames:
        frame_ids = np.asarray(frame)
        if frame_ids.ndim == 3:
            frame_ids = frame_ids[..., 0]
        frame_ids = frame_ids.astype(np.int32)
        if object_ids:
            mask = np.isin(frame_ids, object_ids)
        elif background_ids:
            mask = ~np.isin(frame_ids, background_ids)
        else:
            mask = frame_ids > 0
        mask = _resize_array_hw(
            mask.astype(np.uint8),
            target_hw=target_hw,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        masks.append(mask)
    return np.stack(masks, axis=0).astype(bool)


def _load_video_gray_frames(
    video_path: Path,
    *,
    target_hw: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video for flow export: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if target_hw is not None and gray.shape[:2] != target_hw:
                gray = _resize_array_hw(
                    gray,
                    target_hw=target_hw,
                    interpolation=cv2.INTER_LINEAR,
                )
            frames.append(gray.astype(np.uint8))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def _compute_dense_flow_sequence(gray_frames: list[np.ndarray]) -> np.ndarray:
    if len(gray_frames) <= 1:
        if not gray_frames:
            return np.zeros((0, 0, 0, 2), dtype=np.float32)
        height, width = gray_frames[0].shape[:2]
        return np.zeros((0, height, width, 2), dtype=np.float32)
    flows: list[np.ndarray] = []
    for idx in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[idx],
            gray_frames[idx + 1],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        flows.append(np.asarray(flow, dtype=np.float32))
    return np.stack(flows, axis=0).astype(np.float32)


def _normalize_pose_sequence(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 2:
        poses = poses[None]
    if poses.ndim != 3:
        raise ValueError(f"expected pose sequence with 3 dims, got {poses.shape}")
    if poses.shape[-2:] == (4, 4):
        return poses.astype(np.float32)
    if poses.shape[-2:] == (3, 4):
        bottom = np.repeat(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)[None, None, :],
            poses.shape[0],
            axis=0,
        )
        return np.concatenate([poses, bottom], axis=1).astype(np.float32)
    raise ValueError(f"unsupported pose layout: {poses.shape}")


def _intrinsics_matrix_to_vec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected 3x3 intrinsics matrix, got {matrix.shape}")
    return np.asarray([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]], dtype=np.float32)


def _normalize_intrinsics_sequence(intrinsics: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.ndim == 1:
        if intrinsics.shape[0] == 4:
            return intrinsics[None].astype(np.float32)
        raise ValueError(f"unsupported intrinsics vector shape: {intrinsics.shape}")
    if intrinsics.ndim == 2:
        if intrinsics.shape[-1] == 4:
            return intrinsics.astype(np.float32)
        if intrinsics.shape == (3, 3):
            return _intrinsics_matrix_to_vec(intrinsics)[None]
        raise ValueError(f"unsupported intrinsics layout: {intrinsics.shape}")
    if intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3):
        return np.stack([_intrinsics_matrix_to_vec(matrix) for matrix in intrinsics], axis=0)
    raise ValueError(f"unsupported intrinsics layout: {intrinsics.shape}")


def export_pred_world_from_runtime(
    *,
    sample_id: str,
    runtime_data_root: Path,
    generated_video_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root = runtime_data_root / "OUT_ViPE_Raw"
    metadata_path = runtime_data_root / "INPUT_DATA" / "Metadata" / f"{sample_id}.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        primary_object = str(metadata.get("primary", "")).strip() or None
        secondary_object = str(metadata.get("secondary", "")).strip() or None
        depth_frames = _read_zip_arrays(
            raw_root / "depth" / f"{sample_id}.zip",
            suffixes=(".exr", ".npy", ".npz"),
        )
        mask_frames = _read_zip_arrays(
            raw_root / "mask" / f"{sample_id}.zip",
            suffixes=(".png", ".npy", ".npz"),
        )
        poses, _ = _load_indexed_npz_array(raw_root / "pose" / f"{sample_id}.npz")
        intrinsics, _ = _load_indexed_npz_array(raw_root / "intrinsics" / f"{sample_id}.npz")
        mask_map = (
            _load_mask_map(raw_root / "mask" / f"{sample_id}.txt")
            if (raw_root / "mask" / f"{sample_id}.txt").exists()
            else {}
        )

        depth_seq = []
        for frame in depth_frames:
            depth = np.asarray(frame, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth_seq.append(depth)
        if not depth_seq:
            raise RuntimeError("no depth frames available for predicted-world export")

        target_hw = depth_seq[0].shape[:2]
        depth_seq = [
            _resize_array_hw(depth, target_hw=target_hw, interpolation=cv2.INTER_NEAREST).astype(np.float32)
            for depth in depth_seq
        ]
        mask_seq = _build_instance_mask_sequence(
            mask_frames,
            mask_map=mask_map,
            primary_object=primary_object,
            secondary_object=secondary_object,
            target_hw=target_hw,
        )
        gray_frames = _load_video_gray_frames(generated_video_path, target_hw=target_hw)
        poses = _normalize_pose_sequence(poses)
        intrinsics = _normalize_intrinsics_sequence(intrinsics)

        num_frames = min(len(depth_seq), len(mask_seq), len(poses), len(intrinsics), len(gray_frames))
        if num_frames <= 0:
            raise RuntimeError("could not align any frames for predicted-world export")

        depth_video = np.stack(depth_seq[:num_frames], axis=0).astype(np.float32)
        mask_video = np.asarray(mask_seq[:num_frames], dtype=bool)
        camera_c2w = np.asarray(poses[:num_frames], dtype=np.float32)
        camera_K = np.asarray(intrinsics[:num_frames], dtype=np.float32)
        gray_frames = gray_frames[:num_frames]
        flow_seq = _compute_dense_flow_sequence(gray_frames)
        occlusion_seq = (
            ~(mask_video[:-1] & mask_video[1:])
            if num_frames > 1
            else np.zeros((0, *target_hw), dtype=bool)
        )

        np.savez_compressed(output_root / "depth.npz", depth=depth_video)
        np.savez_compressed(output_root / "mask.npz", mask=mask_video)
        np.savez_compressed(output_root / "flow.npz", flow=flow_seq.astype(np.float32))
        np.savez_compressed(output_root / "cameras.npz", c2w=camera_c2w, K=camera_K)
        np.savez_compressed(output_root / "occlusion.npz", occlusion=occlusion_seq.astype(bool))
    except Exception as exc:
        return {
            "success": False,
            "pred_world_root": str(output_root),
            "error": f"{type(exc).__name__}: {exc}",
        }

    inspection = inspect_pred_world_root(output_root)
    payload = {
        "success": inspection["ready"],
        "pred_world_root": str(output_root),
        "primary_object": primary_object,
        "secondary_object": secondary_object,
        "num_frames": int(num_frames),
        "required_artifacts": list(PRED_WORLD_REQUIRED_FILENAMES),
        "missing_required_artifacts": inspection["required_artifacts"]["missing"],
    }
    if not inspection["ready"]:
        payload["error"] = (
            "predicted-world export completed, but required artifacts are missing: "
            + ", ".join(inspection["required_artifacts"]["missing"])
        )
    (output_root / "pred_world_info.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _generate_pred_world_artifacts(
    *,
    case: PhysicsCaseSpec,
    generated_video_path: Path,
    reports_root: Path,
) -> dict[str, Any]:
    pred_world_root = reports_root / "pred_world" / case.benchmark_sample_id
    if pred_world_ready(pred_world_root):
        return {
            "enabled": True,
            "success": True,
            "cache_hit": True,
            "pred_world_root": str(pred_world_root),
        }
    if not _auto_pred_world_enabled():
        return {
            "enabled": False,
            "success": False,
            "cache_hit": False,
            "pred_world_root": str(pred_world_root),
            "error": f"automatic predicted-world export is disabled by {AUTO_PRED_WORLD_ENV}",
        }
    main_part2_script = case.physics_pipeline_root / "main_part2.py"
    if not main_part2_script.exists():
        return {
            "enabled": True,
            "success": False,
            "cache_hit": False,
            "pred_world_root": str(pred_world_root),
            "error": f"missing physics reconstruction script: {main_part2_script}",
        }

    runtime_data_root = reports_root / "_pred_world_runtime" / case.benchmark_sample_id / "data_root"
    runtime_data_root.mkdir(parents=True, exist_ok=True)
    stage_info = _stage_runtime_inputs(
        case,
        generated_video_path=generated_video_path,
        data_root=runtime_data_root,
    )
    env = _physics_subprocess_env(case, data_root=runtime_data_root)
    command, cwd = _main_part2_invocation(case)
    ok, error = _run_logged_subprocess(
        command,
        cwd=cwd,
        env=env,
        log_path=reports_root / "logs" / "main_part2.json",
        step_name="main_part2",
    )
    if not ok:
        return {
            "enabled": True,
            "success": False,
            "cache_hit": False,
            "pred_world_root": str(pred_world_root),
            "runtime_data_root": str(runtime_data_root),
            "stage_info": stage_info,
            "error": error,
        }
    export_info = export_pred_world_from_runtime(
        sample_id=case.source_sample_id,
        runtime_data_root=runtime_data_root,
        generated_video_path=generated_video_path,
        output_root=pred_world_root,
    )
    return {
        "enabled": True,
        "success": bool(export_info.get("success")),
        "cache_hit": False,
        "pred_world_root": str(pred_world_root),
        "runtime_data_root": str(runtime_data_root),
        "stage_info": stage_info,
        "error": export_info.get("error"),
    }


def _run_online_physics_pipeline(
    *,
    case: PhysicsCaseSpec,
    generated_video_path: Path,
    reports_root: Path,
) -> dict[str, Any]:
    runtime_data_root = _physics_data_root(case)
    runtime_data_root.mkdir(parents=True, exist_ok=True)
    stage_info = _stage_runtime_inputs(
        case,
        generated_video_path=generated_video_path,
        data_root=runtime_data_root,
    )
    env = _physics_subprocess_env(case, data_root=runtime_data_root)
    main_command, main_cwd = _main_part2_invocation(case)
    main_ok, main_error = _run_logged_subprocess(
        main_command,
        cwd=main_cwd,
        env=env,
        log_path=reports_root / "logs" / "main_part2.json",
        step_name="main_part2",
    )
    if not main_ok:
        return {
            "success": False,
            "runtime_data_root": str(runtime_data_root),
            "stage_info": stage_info,
            "error": main_error,
        }
    export_command, export_cwd = _benchmark_export_invocation(case)
    export_ok, export_error = _run_logged_subprocess(
        export_command,
        cwd=export_cwd,
        env=env,
        log_path=reports_root / "logs" / "benchmark_export.json",
        step_name="benchmark_export",
    )
    if not export_ok:
        return {
            "success": False,
            "runtime_data_root": str(runtime_data_root),
            "stage_info": stage_info,
            "error": export_error,
        }
    exported_world_gt_root = _exported_world_gt_root(case)
    if not world_gt_metrics_ready(exported_world_gt_root):
        return {
            "success": False,
            "runtime_data_root": str(runtime_data_root),
            "exported_world_gt_root": str(exported_world_gt_root),
            "stage_info": stage_info,
            "error": "simulator-backed world_gt export completed, but the artifact set is incomplete",
        }
    _sync_world_gt(exported_world_gt_root, case.world_gt_root)
    return {
        "success": True,
        "runtime_data_root": str(runtime_data_root),
        "exported_world_gt_root": str(exported_world_gt_root),
        "stage_info": stage_info,
    }


def evaluate_generated_video(
    *,
    sample: BenchmarkSample | Mapping[str, Any],
    generated_video_path: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    payload = _sample_payload(sample)
    case = ensure_submission_staged(sample, generated_video_path)
    generated_video_path = Path(generated_video_path).expanduser().resolve()
    reports_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else case.case_root / "evaluation"
    )
    reports_root.mkdir(parents=True, exist_ok=True)

    export_error = None
    pipeline_run: dict[str, Any] | None = None
    diagnostics = inspect_physics_case(sample)
    if not diagnostics["local_world_gt"]["full_metrics_ready"]:
        _, export_error = try_export_world_gt(case)
        diagnostics = inspect_physics_case(sample)
    if not diagnostics["local_world_gt"]["full_metrics_ready"]:
        pipeline_run = _run_online_physics_pipeline(
            case=case,
            generated_video_path=generated_video_path,
            reports_root=reports_root,
        )
        if not pipeline_run.get("success"):
            export_error = str(pipeline_run.get("error") or export_error or "")
        diagnostics = inspect_physics_case(sample)

    if not diagnostics["local_world_gt"]["video_ready"]:
        return {
            "sample_id": str(payload.get("sample_id", generated_video_path.stem)),
            "track": payload.get("track"),
            "generated_video_path": str(generated_video_path),
            "world_gt_root": str(case.world_gt_root),
            "error": export_error or "physics reference world_gt is unavailable for this case",
            "missing_world_artifacts": diagnostics["local_world_gt"]["metric_artifacts"]["missing"],
            "physics_status": diagnostics["status"],
            "physics_diagnostics": diagnostics,
            "physics_remediation": diagnostics["remediation"],
            "pipeline_run": pipeline_run,
            "success": False,
        }

    if pipeline_run and pipeline_run.get("success"):
        pred_world_root = reports_root / "pred_world" / case.benchmark_sample_id
        if pred_world_ready(pred_world_root):
            pred_world_info = {
                "enabled": True,
                "success": True,
                "cache_hit": True,
                "pred_world_root": str(pred_world_root),
                "runtime_data_root": pipeline_run.get("runtime_data_root"),
            }
        else:
            export_info = export_pred_world_from_runtime(
                sample_id=case.source_sample_id,
                runtime_data_root=Path(str(pipeline_run["runtime_data_root"])).expanduser().resolve(),
                generated_video_path=generated_video_path,
                output_root=pred_world_root,
            )
            pred_world_info = {
                "enabled": True,
                "success": bool(export_info.get("success")),
                "cache_hit": False,
                "pred_world_root": str(pred_world_root),
                "runtime_data_root": pipeline_run.get("runtime_data_root"),
                "error": export_info.get("error"),
            }
    else:
        pred_world_info = _generate_pred_world_artifacts(
            case=case,
            generated_video_path=generated_video_path,
            reports_root=reports_root,
        )
    pred_world_root = (
        Path(pred_world_info["pred_world_root"]).expanduser().resolve()
        if pred_world_info.get("success")
        else None
    )
    if pred_world_root is not None:
        try:
            pred_worldline_info = ensure_pred_worldlines(
                world_gt_root=case.world_gt_root,
                pred_world_root=pred_world_root,
            )
        except Exception as exc:
            pred_worldline_info = {
                "created": False,
                "path": str(pred_world_root / "worldlines.npz"),
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        pred_worldline_info = None
    pred_world_eval_root = pred_world_root.parent if pred_world_root is not None else None
    report_path = reports_root / f"{generated_video_path.stem}_physics_report.json"
    payload_out = _run_eval_script(
        case=case,
        pred_video_path=generated_video_path,
        output_path=report_path,
        pred_world_root=pred_world_eval_root,
        sample_stem=str(payload.get("sample_id", generated_video_path.stem)),
    )

    per_video = list(payload_out.get("per_video") or [])
    result = dict(per_video[0]) if per_video else {}
    result["sample_id"] = str(payload.get("sample_id", generated_video_path.stem))
    result["track"] = payload.get("track")
    result["generated_video_path"] = str(generated_video_path)
    result["world_gt_root"] = str(case.world_gt_root)
    result["report_path"] = str(report_path.resolve())
    result["physics_pipeline_root"] = str(case.physics_pipeline_root)
    result["physics_status"] = diagnostics["status"]
    if pipeline_run is not None:
        result["pipeline_run"] = pipeline_run
    result["pred_world_status"] = {
        "enabled": bool(pred_world_info.get("enabled")),
        "success": bool(pred_world_info.get("success")),
        "cache_hit": bool(pred_world_info.get("cache_hit")),
        "pred_world_root": pred_world_info.get("pred_world_root"),
        "runtime_data_root": pred_world_info.get("runtime_data_root"),
        "worldlines": pred_worldline_info,
    }
    result["world_gt_status"] = {
        "video_ready": diagnostics["local_world_gt"]["video_ready"],
        "full_metrics_ready": diagnostics["local_world_gt"]["full_metrics_ready"],
        "missing_metric_artifacts": diagnostics["local_world_gt"]["metric_artifacts"]["missing"],
    }
    result.setdefault("missing_world_artifacts", [])
    result["missing_world_artifacts"] = sorted(
        dict.fromkeys(
            list(result["missing_world_artifacts"])
            + diagnostics["local_world_gt"]["metric_artifacts"]["missing"]
        )
    )
    if export_error is not None:
        result["world_gt_sync_message"] = export_error
    if pred_world_info.get("error"):
        result["pred_world_error"] = str(pred_world_info["error"])
    append_derived_simulator_metrics(
        result,
        sample=sample,
        world_gt_root=case.world_gt_root,
        pred_world_root=pred_world_root,
    )

    contract_violations = _metric_contract_violations(sample, result)
    result["metric_contract"] = contract_violations
    if contract_violations["missing_required_metrics"]:
        result["missing_required_metrics"] = contract_violations["missing_required_metrics"]
    contract_messages = list(contract_violations["messages"])
    if _pred_world_artifacts_required(sample) and not bool(pred_world_info.get("success")):
        message = "predicted world artifacts are required for this physics evaluation"
        if pred_world_info.get("error"):
            message += ". " + str(pred_world_info["error"])
        contract_messages.append(message)
    if contract_messages:
        result["error"] = " ".join(contract_messages)
    if diagnostics["status"] != "ready" or pred_world_info.get("error") or "error" in result:
        result["physics_diagnostics"] = diagnostics
        result["physics_remediation"] = diagnostics["remediation"]

    result["success"] = (
        diagnostics["status"] == "ready"
        and "error" not in result
        and (
            not _pred_world_artifacts_required(sample)
            or bool(pred_world_info.get("success"))
        )
    )
    return result


def build_online_physics_sample(
    *,
    prompt: str,
    image_path: str | Path,
    output_root: str | Path,
    physics_pipeline_root: str | Path,
    benchmark_sample_id: str | None = None,
    fg_prompt: str | None = None,
    primary_object: str | None = None,
    secondary_object: str | None = None,
    primary_obj_rot_axis: Any = None,
    metadata_overrides: Mapping[str, Any] | None = None,
) -> BenchmarkSample:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"reference image not found: {image_path}")

    benchmark_sample_id = benchmark_sample_id or f"online_physics__{_utc_timestamp_slug()}"
    output_root = Path(output_root).expanduser().resolve()
    case_root = output_root / "cases" / benchmark_sample_id
    simulator_root = case_root / "simulator"
    world_gt_root = case_root / "world_gt"
    simulator_root.mkdir(parents=True, exist_ok=True)
    world_gt_root.mkdir(parents=True, exist_ok=True)
    submitted_video_path = simulator_root / "submitted_video.mp4"
    metadata_template_path = simulator_root / "metadata_template.json"
    raw_metadata_overrides = dict(metadata_overrides or {})
    cleaned_overrides = _clean_metadata_overrides(metadata_overrides)
    case_taxonomy = cleaned_overrides.copy()
    resolved_fg_prompt = fg_prompt or prompt.split(",", 1)[0].strip()
    resolved_primary_object = primary_object or "object"
    resolved_secondary_object = secondary_object or ""
    metadata_template = {
        "conditioning_tracks": _conditioning_tracks_from_metadata(raw_metadata_overrides),
        "video_prompt": prompt,
        "fg_prompt": resolved_fg_prompt,
        "primary": resolved_primary_object,
        "secondary": resolved_secondary_object,
        "submitted_video_path": str(submitted_video_path.resolve()),
    }
    if primary_obj_rot_axis is not None:
        metadata_template["primary_obj_rot_axis"] = primary_obj_rot_axis
    metadata_template.update(cleaned_overrides)
    metadata_template_path.write_text(
        json.dumps(metadata_template, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return BenchmarkSample(
        sample_id=benchmark_sample_id,
        suite="experimental",
        split=benchmark_split(),
        asset_level="video_nonformal",
        modality="video",
        is_formal=False,
        eligible_for_official=False,
        category_path="physics/online/i2v",
        path=str(image_path),
        relative_path=image_path.name,
        reference_path=str(image_path),
        prediction_stem=benchmark_sample_id,
        conditioning_strategy="external_image",
        prompt_current=prompt,
        prompt_target=prompt,
        style=None,
        environment=_coalesce_metadata_str(cleaned_overrides, "environment", "environment_family", "surface_family"),
        scene=_coalesce_metadata_str(cleaned_overrides, "scene", "scene_family", "camera_family"),
        motion_category=_coalesce_metadata_str(cleaned_overrides, "motion_family", "interaction_family")
        or primary_object
        or "physics",
        source_name="worldarena_online_physics",
        source_type="generated_case",
        source_group_id=benchmark_sample_id,
        license_bucket="internal",
        benchmark_family="worldarena_physics",
        task_family=task_family_for_suite("experimental", track="physics_i2v"),
        artifact_type=artifact_type_for_modality("video"),
        control_signals=control_signals_for_sample(
            modality="video",
            conditioning_strategy="external_image",
            has_pose=False,
        ),
        has_annotation=True,
        has_instruction=True,
        has_pose=False,
        has_mask=False,
        mask_count=0,
        width=None,
        height=None,
        fps=8.0,
        duration_seconds=None,
        duration_bucket=None,
        conditioning_path=str(image_path),
        track="physics_i2v",
        physics_spec={
            "benchmark_family": "worldarena_physics",
            "backend": "simulator_reference",
            "source_sample_id": ONLINE_PHYSICS_INTERNAL_SAMPLE_ID,
            "fg_prompt": resolved_fg_prompt,
            "primary_object": resolved_primary_object,
            "secondary_object": resolved_secondary_object,
            "metadata_template": metadata_template,
            "metric_contract": {
                "required_core_metrics": list(REQUIRED_CORE_METRICS),
                "require_full_frame_coverage": True,
                "require_pred_world_artifacts": True,
            },
            "case_taxonomy": case_taxonomy,
        },
        physics_case={
            "physics_pipeline_root": str(Path(physics_pipeline_root).expanduser().resolve()),
            "physics_data_root": str((output_root / "runtime" / benchmark_sample_id / "data_root").resolve()),
            "source_sample_id": ONLINE_PHYSICS_INTERNAL_SAMPLE_ID,
            "benchmark_sample_id": benchmark_sample_id,
            "case_root": str(case_root.resolve()),
            "simulator_root": str(simulator_root.resolve()),
            "submitted_video_path": str(submitted_video_path.resolve()),
            "metadata_template_path": str(metadata_template_path.resolve()),
            "metadata_template": metadata_template,
            "world_gt_root": str(world_gt_root.resolve()),
            "physics_export_benchmark_name": f"worldarena_online_physics_{benchmark_sample_id}",
            "physics_render_split": "original_length",
            "case_taxonomy": case_taxonomy,
        },
    )


def evaluate_online_physics_i2v(
    *,
    prompt: str,
    image_path: str | Path,
    generated_video_path: str | Path,
    output_root: str | Path,
    physics_pipeline_root: str | Path,
    fg_prompt: str | None = None,
    primary_object: str | None = None,
    secondary_object: str | None = None,
    primary_obj_rot_axis: Any = None,
    metadata_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample = build_online_physics_sample(
        prompt=prompt,
        image_path=image_path,
        output_root=output_root,
        physics_pipeline_root=physics_pipeline_root,
        fg_prompt=fg_prompt,
        primary_object=primary_object,
        secondary_object=secondary_object,
        primary_obj_rot_axis=primary_obj_rot_axis,
        metadata_overrides=metadata_overrides,
    )
    reports_root = Path(output_root).expanduser().resolve() / "reports" / sample.sample_id
    reports_root.mkdir(parents=True, exist_ok=True)

    generated_video_path = Path(generated_video_path).expanduser().resolve()
    staged_case = ensure_submission_staged(sample, generated_video_path)
    pipeline_run = _run_online_physics_pipeline(
        case=staged_case,
        generated_video_path=staged_case.submitted_video_path.resolve(),
        reports_root=reports_root,
    )
    diagnostics = inspect_physics_case(sample)
    if not pipeline_run.get("success"):
        return {
            "success": False,
            "sample_id": sample.sample_id,
            "track": sample.track,
            "generated_video_path": str(staged_case.submitted_video_path.resolve()),
            "physics_status": diagnostics["status"],
            "physics_diagnostics": diagnostics,
            "physics_remediation": diagnostics["remediation"],
            "pipeline_run": pipeline_run,
            "error": str(pipeline_run.get("error") or "online physics pipeline failed"),
        }

    pred_world_root = reports_root / "pred_world" / staged_case.benchmark_sample_id
    pred_world_info = export_pred_world_from_runtime(
        sample_id=staged_case.source_sample_id,
        runtime_data_root=Path(pipeline_run["runtime_data_root"]).expanduser().resolve(),
        generated_video_path=staged_case.submitted_video_path.resolve(),
        output_root=pred_world_root,
    )
    if pred_world_info.get("success"):
        try:
            pred_worldline_info = ensure_pred_worldlines(
                world_gt_root=staged_case.world_gt_root,
                pred_world_root=pred_world_root,
            )
        except Exception as exc:
            pred_worldline_info = {
                "created": False,
                "path": str(pred_world_root / "worldlines.npz"),
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        pred_worldline_info = None
    pred_world_eval_root = pred_world_root.parent if pred_world_info.get("success") else None
    report_path = reports_root / f"{staged_case.benchmark_sample_id}_physics_report.json"
    payload = _run_eval_script(
        case=staged_case,
        pred_video_path=staged_case.submitted_video_path.resolve(),
        output_path=report_path,
        pred_world_root=pred_world_eval_root,
        sample_stem=staged_case.benchmark_sample_id,
    )

    per_video = list(payload.get("per_video") or [])
    result = dict(per_video[0]) if per_video else {}
    result["sample_id"] = sample.sample_id
    result["track"] = sample.track
    result["generated_video_path"] = str(staged_case.submitted_video_path.resolve())
    result["report_path"] = str(report_path.resolve())
    result["world_gt_root"] = str(staged_case.world_gt_root.resolve())
    result["pipeline_run"] = pipeline_run
    result["pred_world_status"] = inspect_pred_world_root(pred_world_root)
    result["pred_world_status"]["worldlines"] = pred_worldline_info
    result["world_gt_status"] = inspect_world_gt_root(staged_case.world_gt_root)
    append_derived_simulator_metrics(
        result,
        sample=sample,
        world_gt_root=staged_case.world_gt_root,
        pred_world_root=pred_world_root if pred_world_info.get("success") else None,
    )
    contract_violations = _metric_contract_violations(sample, result)
    result["metric_contract"] = contract_violations
    if contract_violations["messages"]:
        result["error"] = " ".join(contract_violations["messages"])
    result["success"] = "error" not in result and bool(pred_world_info.get("success"))
    return result


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)
