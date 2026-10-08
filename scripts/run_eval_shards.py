from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Iterable, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sharded WorldAtlas evaluation across GPUs.")
    parser.add_argument("--config", default="config/benchmark.yaml")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--manifest", default="artifacts/benchmark/manifest/worldatlas_manifest.jsonl")
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suites", default="image_dynamic")
    parser.add_argument(
        "--metrics",
        default="",
        help="Comma-separated metric override applied to each selected suite.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-launch sample limit.")
    parser.add_argument(
        "--skip-existing-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip shard directories that already have per_sample_scores.jsonl and run_manifest.json.",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Seconds between launch heartbeat messages while shards are running.",
    )
    parser.add_argument(
        "--max-concurrent-shards",
        type=int,
        default=0,
        help="Maximum shard processes to run at once. 0 means launch all shards.",
    )
    parser.add_argument(
        "--max-concurrent-per-gpu",
        type=int,
        default=0,
        help="Maximum simultaneous shard processes assigned to one GPU. 0 means unlimited.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=None,
        help=(
            "Replica shard processes to place on each GPU. Total shards = "
            "len(--gpus) * this value. Default 1, or WORLDATLAS_EVAL_WORKERS_PER_GPU."
        ),
    )
    parser.add_argument(
        "--motion-frame-max-side",
        type=int,
        default=None,
        help="Optional max image side for motion metric frame caches.",
    )
    parser.add_argument(
        "--motion-smoothness-frame-max-side",
        type=int,
        default=None,
        help="Optional max image side for motion_smoothness frame caches.",
    )
    # internal per-shard mode
    parser.add_argument("--internal-shard", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--cuda-retries",
        type=int,
        default=2,
        help=(
            "Restart a failed CUDA worker in a fresh process, resuming healthy "
            "samples from checkpoints. Reprojection floors this at 16."
        ),
    )
    return parser.parse_args()


DEFAULT_CUDA_RETRIES = 2
REPROJECTION_MIN_CUDA_RETRIES = 16


def sample_checkpoint_path(out_dir: Path, fingerprint: str) -> Path:
    """Always persist healthy sample scores so CUDA process restarts can resume.

    ``--no-skip-existing-shards`` only refuses to skip a *completed* shard
    jsonl. Intra-shard checkpoints are required after CUDA_RESTART_EXIT_CODE
    even when that flag is set.
    """
    return Path(out_dir) / "sample_checkpoints" / fingerprint


def resolve_cuda_retries(metrics: Iterable[str], requested: int | None) -> int:
    retries = DEFAULT_CUDA_RETRIES if requested is None else max(int(requested), 0)
    if "reprojection_error" in {str(name) for name in metrics}:
        return max(retries, REPROJECTION_MIN_CUDA_RETRIES)
    return retries


def should_retry_cuda_worker(exit_code: int, attempts: int, cuda_retries: int) -> bool:
    from worldatlas.benchmark.runtime_health import CUDA_RESTART_EXIT_CODE

    return int(exit_code) == CUDA_RESTART_EXIT_CODE and int(attempts) <= max(int(cuda_retries), 0)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def expand_shard_assignments(gpu_ids: list[str], workers_per_gpu: int) -> list[tuple[int, str]]:
    """Map (shard_index, gpu_id) so each GPU gets ``workers_per_gpu`` replica processes."""
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")
    workers = max(int(workers_per_gpu), 1)
    assignments: list[tuple[int, str]] = []
    shard_index = 0
    for _ in range(workers):
        for gpu_id in gpu_ids:
            assignments.append((shard_index, gpu_id))
            shard_index += 1
    return assignments


def resolve_workers_per_gpu(explicit: int | None) -> int:
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    return _positive_int_env("WORLDATLAS_EVAL_WORKERS_PER_GPU", 1)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_project_path(path: str | None, project_root: Path) -> Path | None:
    from worldatlas.models.config import resolve_model_config_path

    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    # Use the same config file for scoring and for report/resume fingerprints,
    # even when the command still names the old flat model config path.
    return resolve_model_config_path(candidate)


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_source_hashes(project_root: Path) -> dict[str, str]:
    from worldatlas.benchmark.integrity import evaluation_source_hashes
    return evaluation_source_hashes(project_root)


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import cv2

        versions["opencv"] = str(cv2.__version__)
    except Exception:
        pass
    try:
        import numpy

        versions["numpy"] = str(numpy.__version__)
    except Exception:
        pass
    return versions


def _git_head(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _suites_set(raw: str) -> set[str] | None:
    items = {item.strip() for item in raw.split(",") if item.strip()}
    return items or None


def _metric_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


EVAL_CUDA_CACHE_ROOT = Path("/tmp/worldatlas_cuda_cache")
MET3R_WARMUP_SECONDS = 900
MET3R_WARMUP_SOFT_SECONDS = 30
MET3R_WARMUP_POLL_SECONDS = 2
FEATUP_NATIVE_DIR_ENV = "WORLDATLAS_FEATUP_NATIVE_DIR"
FEATUP_FAULT_LOG_ENV = "WORLDATLAS_EVAL_FAULT_LOG"


def includes_met3r(metrics: Iterable[str] | None) -> bool:
    return any("met3r" in str(name).lower() for name in (metrics or []))


def met3r_warmup_ready_text(text: str) -> bool:
    """True only after FeatUp finished loading or a sample actually scored ok."""
    if "met3r_model_ready" in text:
        return True
    for line in text.splitlines():
        if "[progress]" not in line or "sample_start" in line:
            continue
        # A failed first sample is not ready: FeatUp never loaded.
        if " sample " in line and "status=ok" in line:
            return True
    return False


def met3r_warmup_load_failed_text(text: str) -> bool:
    """True when FeatUp/MeT3R load failed; remaining shards must not launch."""
    for line in text.splitlines():
        if "met3r_model_load" in line and "status=fail" in line:
            return True
    return False


def take_met3r_warmup_assignment(pending: list[tuple[int, str]]) -> tuple[int, str] | None:
    if not pending:
        return None
    for index, (shard_index, _) in enumerate(pending):
        if shard_index == 0:
            return pending.pop(index)
    return pending.pop(0)


def met3r_warmup_blocks_remaining(status: str) -> bool:
    """Remaining shards launch only after FeatUp finished loading."""
    return status != "ready"


def met3r_warmup_skips_merge(status: str) -> bool:
    """Do not wrap a native warmup death as a merge coverage ValueError."""
    return status in {"exited", "timeout", "load_failed"}


def _met3r_warmup_seconds() -> int:
    raw = os.environ.get("WORLDATLAS_MET3R_WARMUP_SECONDS", str(MET3R_WARMUP_SECONDS))
    try:
        return max(15, int(raw))
    except ValueError:
        return MET3R_WARMUP_SECONDS


def torch_runtime_library_dirs(python_executable: str | None = None) -> list[str]:
    """torch/lib (libc10) plus the env lib (libstdc++ GLIBCXX_3.4.29)."""
    dirs: list[str] = []
    exe = Path(python_executable or sys.executable).resolve()
    env_root = exe.parent.parent
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    conda = os.environ.get("CONDA_PREFIX") or os.environ.get("WORLDATLAS_EVAL_ENV")
    candidates = [
        env_root / "lib" / version / "site-packages" / "torch" / "lib",
        env_root / "lib",
    ]
    if conda:
        conda_root = Path(conda)
        candidates.extend(
            [
                conda_root / "lib" / version / "site-packages" / "torch" / "lib",
                conda_root / "lib",
            ]
        )
    for candidate in candidates:
        resolved = str(candidate)
        if candidate.is_dir() and resolved not in dirs:
            dirs.append(resolved)
    return dirs


def _prepend_env_path(env: dict[str, str], key: str, *parts: str) -> None:
    existing = [item for item in env.get(key, "").split(os.pathsep) if item]
    unique_prefix: list[str] = []
    for part in parts:
        if part and part not in unique_prefix:
            unique_prefix.append(part)
    rest = [item for item in existing if item not in unique_prefix]
    env[key] = os.pathsep.join(unique_prefix + rest)


def apply_eval_harden_env(
    env: dict[str, str] | None = None,
    *,
    shard_index: int | None = None,
    python_executable: str | None = None,
) -> dict[str, str]:
    """Point CUDA/MPL/font caches at /tmp and put torch libc10 on the linker path.

    Safe for Trajan and other metrics. Required so FeatUp's NFS-built
    ``adaptive_conv_cuda_impl*.so`` can resolve ``libc10.so`` / ``libstdc++``.
    """
    env = dict(os.environ if env is None else env)
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    env.setdefault("MPLBACKEND", "Agg")
    cache_root = EVAL_CUDA_CACHE_ROOT
    extensions = cache_root / "torch_extensions"
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        extensions.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    env.setdefault("CUDA_CACHE_PATH", str(cache_root))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(extensions))
    token = f"{os.getpid()}_{shard_index if shard_index is not None else 'parent'}"
    mpl_dir = Path(f"/tmp/worldatlas_mpl_{token}")
    xdg_dir = Path(f"/tmp/worldatlas_xdg_{token}")
    font_dir = Path(f"/tmp/worldatlas_fontconfig_{token}")
    for path in (mpl_dir, xdg_dir, font_dir):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["XDG_CACHE_HOME"] = str(xdg_dir)
    env.setdefault("XDG_CONFIG_HOME", str(xdg_dir))
    env["FONTCONFIG_CACHE"] = str(font_dir)
    _prepend_env_path(env, "LD_LIBRARY_PATH", *torch_runtime_library_dirs(python_executable))
    featup_dir = env.get(FEATUP_NATIVE_DIR_ENV, "").strip()
    if featup_dir:
        _prepend_env_path(env, "PYTHONPATH", featup_dir)
    return env


def _launch_metric_names(args: argparse.Namespace, project_root: Path) -> list[str]:
    explicit = _metric_list(args.metrics)
    if explicit:
        return explicit

    from worldatlas.benchmark import load_config
    from worldatlas.benchmark.config import apply_model_metric_overrides
    from worldatlas.models import load_model_config

    config_path = _resolve_project_path(args.config, project_root)
    if config_path is None:
        return []
    model_config = load_model_config(Path(args.model_config)) if args.model_config else None
    config = apply_model_metric_overrides(load_config(config_path), model_config)
    suites = _suites_set(args.suites)
    names: list[str] = []
    seen: set[str] = set()
    for suite_name, metric_names in config.metrics_by_suite.items():
        if suites is not None and suite_name not in suites:
            continue
        for metric_name in metric_names:
            if metric_name not in seen:
                seen.add(metric_name)
                names.append(metric_name)
    return names


def _apply_metric_override(config, suites: set[str] | None, metrics: Iterable[str]) -> None:
    metric_names = list(metrics)
    if not metric_names:
        return
    config.metrics_by_suite = dict(config.metrics_by_suite)
    target_suites = suites or set(config.metrics_by_suite)
    for suite in target_suites:
        config.metrics_by_suite[suite] = list(metric_names)


def _bind_runtime_to_visible_gpu(config, metrics: Iterable[str], args: argparse.Namespace) -> None:
    metric_names = set(metrics)
    cacheable_metrics = {
        "motion_accuracy",
        "motion_magnitude",
        "motion_smoothness",
        "optical_flow_aepe",
    }
    selected_cacheable_metrics = metric_names & cacheable_metrics
    if not selected_cacheable_metrics:
        return
    config.metric_runtime = dict(config.metric_runtime)
    for metric_name in selected_cacheable_metrics:
        runtime = dict(config.metric_runtime.get(metric_name, {}))
        runtime.setdefault("cache_instance", True)
        if args.motion_frame_max_side:
            runtime["frame_max_side"] = args.motion_frame_max_side
        elif metric_name == "optical_flow_aepe":
            runtime.setdefault(
                "frame_max_side",
                int(os.environ.get("WORLDATLAS_AEPE_FRAME_MAX_SIDE", "1920") or 1920),
            )
        elif metric_name == "motion_smoothness":
            runtime.setdefault(
                "frame_max_side",
                int(os.environ.get("WORLDATLAS_MOTION_SM_FRAME_MAX_SIDE", "1920") or 1920),
            )
        if metric_name == "optical_flow_aepe":
            raw_batch = os.environ.get("WORLDATLAS_AEPE_BATCH_SIZE", "").strip()
            if raw_batch:
                runtime["batch_size"] = max(1, int(raw_batch))
            else:
                runtime.setdefault("batch_size", 2)
        if metric_name == "motion_smoothness" and args.motion_smoothness_frame_max_side:
            runtime["frame_max_side"] = args.motion_smoothness_frame_max_side
        if metric_name == "motion_accuracy":
            runtime.setdefault("sam2_device", "cuda:0")
            runtime.setdefault("flow_device", "cuda:0")
            runtime.setdefault("grounding_dino_device", "cuda:0")
        config.metric_runtime[metric_name] = runtime


def _shard_resume_signature(
    args: argparse.Namespace,
    *,
    shard_index: int,
    num_shards: int,
    predictions_root: Path,
) -> dict:
    project_root = _project_root()
    manifest_path = _resolve_project_path(args.manifest, project_root)
    signature = {
        "integrity_version": "content_bound_v2",
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "suites": args.suites,
        "metrics": _metric_list(args.metrics),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "predictions_root": str(predictions_root.resolve()),
        "limit": args.limit,
        "motion_frame_max_side": getattr(args, "motion_frame_max_side", None),
        "motion_smoothness_frame_max_side": getattr(args, "motion_smoothness_frame_max_side", None),
        "manifest_sha256": _sha256_file(manifest_path),
        "config_sha256": _sha256_file(_resolve_project_path(getattr(args, "config", None), project_root)),
        "model_config_sha256": _sha256_file(_resolve_project_path(getattr(args, "model_config", None), project_root)),
        "evaluation_source_sha256": _evaluation_source_hashes(project_root),
        "runtime_env": {key: os.environ.get(key) for key in (
            "TORCH_CUDNN_V8_API_DISABLED", "WORLDATLAS_RAFT_CUDNN_ENABLED",
        )},
    }
    from worldatlas.benchmark.integrity import sample_input_content, digest_json, scoring_environment, package_versions
    signature["scoring_environment"] = scoring_environment()
    signature['package_versions'] = package_versions()
    if manifest_path is not None and manifest_path.is_file():
        from worldatlas.benchmark.manifest import load_manifest
        from worldatlas.benchmark.runner import _resolve_prediction_path
        selected_suites = _suites_set(args.suites)
        selected = [s for s in load_manifest(manifest_path) if not selected_suites or s.suite in selected_suites]
        selected = selected[shard_index::num_shards]
        if args.limit is not None:
            selected = selected[:args.limit]
        signature["input_content_sha256"] = digest_json([
            sample_input_content(s, _resolve_prediction_path(predictions_root, s)) for s in selected])
    requested = _metric_list(args.metrics)
    from worldatlas.benchmark.integrity import checkpoint_content_for_metrics
    signature['motion_checkpoint_content'] = checkpoint_content_for_metrics(
        requested or {'motion_accuracy', 'motion_magnitude', 'optical_flow_aepe'}, project_root)
    if not requested or any(name.startswith(("memory_", "keystroke_", "rollout_")) for name in requested):
        signature["memory_metric_source_sha256"] = {
            relative: _sha256_file(project_root / relative)
            for relative in (
                "worldatlas/benchmark/memory_pairing.py",
                "worldatlas/benchmark/sidecars.py",
                "worldatlas/benchmark/metrics/memory_loop.py",
                "worldatlas/benchmark/metrics/long_horizon_diagnostics.py",
            )
        }
        signature["memory_metric_config_sha256"] = {
            key: _sha256_file(_resolve_project_path(getattr(args, key, None), project_root))
            for key in ("config", "model_config")
        }
    return signature


def _write_shard_resume_signature(
    shard_dir: Path,
    signature: dict,
) -> None:
    manifest_path = shard_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    payload["shard_resume_signature"] = signature
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _metric_summary_has_score(entry: object) -> bool:
    """True when a summary entry is a usable numeric score, not a failed/empty slot."""
    if isinstance(entry, bool) or entry is None:
        return False
    if isinstance(entry, (int, float)):
        return entry == entry
    if not isinstance(entry, dict):
        return False
    mean = entry.get("mean", entry.get("normalized_mean"))
    if isinstance(mean, bool) or not isinstance(mean, (int, float)) or mean != mean:
        return False
    count = entry.get("count")
    if count is None:
        return True
    try:
        return int(count) > 0
    except (TypeError, ValueError):
        return False


def _summary_covers_requested_metrics(summary: object, requested: list[str]) -> tuple[bool, str]:
    suites = summary if isinstance(summary, dict) else {}
    missing: list[str] = []
    for name in requested:
        found = any(
            isinstance(metrics, dict) and _metric_summary_has_score(metrics.get(name))
            for metrics in suites.values()
        )
        if not found:
            missing.append(name)
    if missing:
        return False, "existing shard summary missing requested metrics: " + ",".join(missing)
    return True, "compatible"


def _can_skip_existing_shard(
    shard_dir: Path,
    signature: dict,
) -> tuple[bool, str]:
    scores_path = shard_dir / "per_sample_scores.jsonl"
    manifest_path = shard_dir / "run_manifest.json"
    if not scores_path.exists() or not manifest_path.exists():
        return False, "missing outputs"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid manifest: {exc}"
    if payload.get("source_changes_during_run"):
        return False, "source changed during previous run"
    existing = payload.get("shard_resume_signature")
    if existing != signature:
        return False, "resume signature mismatch"
    requested = [str(name) for name in (signature.get("metrics") or []) if str(name).strip()]
    from worldatlas.benchmark.runtime_health import sample_score_failed

    configured = payload.get("requested_metrics") or {}
    if not requested and not configured:
        return False, "existing full-suite report lacks requested metric coverage"
    try:
        rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid sample scores: {exc}"
    if len(rows) != payload.get("items", len(rows)) or (not rows and payload.get("items") != 0):
        return False, "incomplete sample scores"
    if len({row.get("sample_id") for row in rows}) != len(rows):
        return False, "duplicate sample scores"
    for row in rows:
        required = requested or configured.get(row.get("suite"))
        if required is None or sample_score_failed(row, required):
            return False, "failed or missing requested metrics: " + ",".join(required or [])
    return True, "compatible"


def run_shard(args: argparse.Namespace) -> int:
    os.environ.update(apply_eval_harden_env(os.environ, shard_index=args.shard_index))
    try:
        import faulthandler

        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from worldatlas.benchmark import load_config, load_manifest, run_benchmark
    from worldatlas.benchmark.config import apply_model_metric_overrides
    from worldatlas.common.progress import log_exception, log_progress
    from worldatlas.models import load_model_config

    config = load_config(Path(args.config))
    model_config = load_model_config(Path(args.model_config)) if args.model_config else None
    config = apply_model_metric_overrides(config, model_config)
    manifest = load_manifest(Path(args.manifest).resolve())
    suites = _suites_set(args.suites)
    metrics = _metric_list(args.metrics)
    _apply_metric_override(config, suites, metrics)
    runtime_metrics = metrics or [name for suite, names in config.metrics_by_suite.items()
                                  if not suites or suite in suites for name in names]
    _bind_runtime_to_visible_gpu(config, runtime_metrics, args)

    filtered = [s for s in manifest if not suites or s.suite in suites]
    shard = filtered[args.shard_index :: args.num_shards]
    if args.limit is not None:
        shard = shard[: args.limit]

    predictions_root = Path(args.predictions_root).resolve()
    out_dir = Path(args.output_dir).resolve() / f"shard{args.shard_index}"
    model_name = args.model_name or (model_config.name if model_config else predictions_root.name)
    os.environ.setdefault("WORLDATLAS_EVAL_SHARD_INDEX", str(args.shard_index))
    os.environ.setdefault("WORLDATLAS_EVAL_NUM_SHARDS", str(args.num_shards))
    log_progress(
        "shard_start",
        job_id=os.environ.get("WORLDATLAS_JOB_ID"),
        model=model_name,
        suite=args.suites,
        metrics=",".join(metrics) if metrics else "config",
        samples=len(shard),
        shard=f"{args.shard_index}/{args.num_shards}",
        gpu=os.environ.get("WORLDATLAS_EVAL_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES"),
        python=sys.executable,
        output_dir=str(out_dir),
    )

    from worldatlas.benchmark.runtime_health import CUDA_RESTART_EXIT_CODE, CudaContextError, sample_score_failed
    signature = _shard_resume_signature(
        args, shard_index=args.shard_index, num_shards=args.num_shards, predictions_root=predictions_root,
    )
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    try:
        result = run_benchmark(
            config=config, manifest=shard, predictions_root=predictions_root,
            output_dir=out_dir, model_name=model_name, suites=suites, limit=None,
            run_suite_level_metrics=False,
            checkpoint_path=sample_checkpoint_path(out_dir, fingerprint),
        )
    except CudaContextError as exc:
        log_exception("cuda_worker_restart_required", exc, shard=args.shard_index,
                      exit_code=CUDA_RESTART_EXIT_CODE)
        return CUDA_RESTART_EXIT_CODE
    _write_shard_resume_signature(out_dir, signature)
    required = result["run_manifest"]["requested_metrics"]
    return int(any(sample_score_failed(score.to_dict(), required.get(score.suite, []))
                   for score in result["sample_scores"]))


def _load_scores(path: Path):
    from worldatlas.benchmark.schemas import MetricOutput, SampleScore

    scores = []
    if not path.exists():
        return scores
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        metrics = {name: MetricOutput(**mo) for name, mo in d.get("metrics", {}).items()}
        scores.append(
            SampleScore(
                sample_id=d["sample_id"],
                suite=d["suite"],
                metrics=metrics,
                path=d.get("path", ""),
                prediction_path=d.get("prediction_path"),
                style=d.get("style"),
                environment=d.get("environment"),
                scene=d.get("scene"),
                motion_category=d.get("motion_category"),
                error=d.get("error"),
                input_fingerprint=d.get("input_fingerprint"),
            )
        )
    return scores


def merge(args: argparse.Namespace, num_shards: int) -> dict:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from datetime import datetime, timezone

    from worldatlas.benchmark import load_config, load_manifest
    from worldatlas.benchmark.aggregates import aggregate_results
    from worldatlas.benchmark.config import apply_model_metric_overrides
    from worldatlas.benchmark.metrics.registry import split_suite_level_metrics
    from worldatlas.benchmark.reference_corpus import PredictionEntry, resolve_reference_index
    from worldatlas.benchmark.report import write_run_outputs
    from worldatlas.benchmark.runner import apply_suite_level_metrics
    from worldatlas.benchmark.taxonomy import prediction_modality_for_task
    from worldatlas.models import load_model_config

    base = Path(args.output_dir).resolve()
    all_scores = []
    shard_source_changes = {}
    for i in range(num_shards):
        all_scores.extend(_load_scores(base / f"shard{i}" / "per_sample_scores.jsonl"))
        shard_manifest = base / f"shard{i}" / "run_manifest.json"
        if shard_manifest.is_file():
            changes = json.loads(shard_manifest.read_text()).get("source_changes_during_run", [])
            if changes:
                shard_source_changes[str(i)] = changes

    model_config = load_model_config(Path(args.model_config)) if args.model_config else None
    model_name = args.model_name or (model_config.name if model_config else Path(args.predictions_root).name)
    config_path = _resolve_project_path(args.config, project_root)
    model_config_path = _resolve_project_path(args.model_config, project_root)
    manifest_path = _resolve_project_path(args.manifest, project_root)
    predictions_root = Path(args.predictions_root).resolve()
    if config_path is None or manifest_path is None:
        raise ValueError("config and manifest paths are required for shard merge")

    config = apply_model_metric_overrides(load_config(config_path), model_config)
    selected_suites = _suites_set(args.suites)
    _apply_metric_override(config, selected_suites, _metric_list(args.metrics))
    manifest_samples = load_manifest(manifest_path)
    manifest_by_id = {
        sample.sample_id: sample
        for sample in manifest_samples
    }
    if len(manifest_by_id) != len(manifest_samples):
        raise ValueError('Duplicate sample IDs in merge manifest')
    filtered = [sample for sample in manifest_samples if not selected_suites or sample.suite in selected_suites]
    expected_ids = {sample.sample_id for i in range(num_shards)
                    for sample in (filtered[i::num_shards][:args.limit] if args.limit is not None else filtered[i::num_shards])}
    actual_ids = [score.sample_id for score in all_scores]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != expected_ids:
        missing = sorted(expected_ids - set(actual_ids))
        extra = sorted(set(actual_ids) - expected_ids)
        raise ValueError(
            "Shard merge has duplicate, missing or unexpected sample coverage: "
            f"expected={len(expected_ids)} actual={len(actual_ids)} unique={len(set(actual_ids))} "
            f"missing={len(missing)} extra={len(extra)}"
        )
    if any(score.suite != manifest_by_id[score.sample_id].suite for score in all_scores):
        raise ValueError('Shard score suite differs from the selected manifest')
    suite_level_metric_names: dict[str, list[str]] = {}
    for suite_name, metric_names in config.metrics_by_suite.items():
        if selected_suites is not None and suite_name not in selected_suites:
            continue
        _, suite_metrics = split_suite_level_metrics(metric_names)
        suite_level_metric_names[suite_name] = suite_metrics

    suite_predictions: dict[str, list[PredictionEntry]] = defaultdict(list)
    suite_sample_ids: dict[str, list[str]] = defaultdict(list)
    for score in all_scores:
        if not suite_level_metric_names.get(score.suite):
            continue
        suite_sample_ids[score.suite].append(score.sample_id)
        sample = manifest_by_id.get(score.sample_id)
        if sample is None or score.prediction_path is None:
            continue
        prediction_modality = prediction_modality_for_task(
            suite=sample.suite,
            source_modality=sample.modality,
            task_family=sample.task_family,
            artifact_type=sample.artifact_type,
        )
        if prediction_modality != "video":
            continue
        suite_predictions[score.suite].append(
            PredictionEntry(
                path=Path(score.prediction_path),
                sample_id=score.sample_id,
                style=sample.style,
                environment=sample.environment,
            )
        )

    aggregate_payload = aggregate_results(all_scores)
    reference_index = None
    if any(suite_level_metric_names.values()):
        reference_index = resolve_reference_index(config)
        apply_suite_level_metrics(
            aggregate_payload=aggregate_payload,
            suite_level_metric_names=suite_level_metric_names,
            suite_predictions=suite_predictions,
            suite_sample_ids=suite_sample_ids,
            config=config,
            output_dir=base,
            reference_index=reference_index,
        )
    suite_level_failure_summary = {
        suite: {
            metric_name: details
            for metric_name, details in aggregate_payload.get("failure_summary", {})
            .get(suite, {})
            .items()
            if metric_name in suite_level_metric_names.get(suite, [])
        }
        for suite in suite_level_metric_names
    }
    suite_level_failure_summary = {
        suite: metrics
        for suite, metrics in suite_level_failure_summary.items()
        if metrics
    }
    suites = sorted({s.suite for s in all_scores})

    run_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_changes_by_shard": shard_source_changes,
        "model_name": model_name,
        "predictions_root": str(predictions_root),
        "output_dir": str(base),
        "items": len(all_scores),
        "missing_predictions": sum(1 for s in all_scores if s.error == "missing prediction"),
        "suites": suites,
        "reported_metrics": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload["per_metric_summary"].items()
        },
        "reported_dimensions": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload.get("per_dimension_summary", {}).items()
            if m
        },
        "reported_diagnostics": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload.get("diagnostic_metric_summary", {}).items()
            if m
        },
        "reported_failures": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload.get("failure_summary", {}).items()
            if m
        },
        "reported_not_applicable": {
            suite: sorted(m.keys())
            for suite, m in aggregate_payload.get("not_applicable_summary", {}).items()
            if m
        },
        "num_shards": num_shards,
        "metric_implementation_revisions": aggregate_payload.get("metric_implementation_revisions", {}),
        "metric_override": _metric_list(args.metrics),
        "limit": args.limit,
        "reference_corpus": (None if reference_index is None else reference_index.describe()),
        "reproducibility": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "git_head": _git_head(project_root),
            "python": sys.executable,
            "python_version": sys.version,
            "runtime_versions": _runtime_versions(),
            "evaluation_source_sha256": _evaluation_source_hashes(project_root),
            "cwd": str(project_root),
            "inputs": {
                "config": str(config_path) if config_path is not None else None,
                "config_sha256": _sha256_file(config_path),
                "model_config": str(model_config_path) if model_config_path is not None else None,
                "model_config_sha256": _sha256_file(model_config_path),
                "manifest": str(manifest_path) if manifest_path is not None else None,
                "manifest_sha256": _sha256_file(manifest_path),
                "predictions_root": str(predictions_root),
            },
            "launch": {
                "gpus": args.gpus,
                "workers_per_gpu": resolve_workers_per_gpu(args.workers_per_gpu),
                "max_concurrent_shards": args.max_concurrent_shards,
                "max_concurrent_per_gpu": args.max_concurrent_per_gpu,
                "skip_existing_shards": args.skip_existing_shards,
                "suites": args.suites,
                "metrics": args.metrics,
                "motion_frame_max_side": args.motion_frame_max_side,
                "motion_smoothness_frame_max_side": args.motion_smoothness_frame_max_side,
            },
        },
    }
    write_run_outputs(
        output_dir=base,
        run_manifest=run_manifest,
        sample_scores=all_scores,
        aggregate_payload=aggregate_payload,
        model_name=model_name,
    )
    return {
        "run_manifest": run_manifest,
        "per_metric_summary": aggregate_payload["per_metric_summary"],
        "diagnostic_metric_summary": aggregate_payload.get("diagnostic_metric_summary", {}),
        "failure_summary": aggregate_payload.get("failure_summary", {}),
        "suite_level_failure_summary": suite_level_failure_summary,
    }


def _last_progress_snippet(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 8192))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line.strip() for line in text.splitlines() if "[progress]" in line]
    if not lines:
        return None
    return lines[-1]


def _tee_shard_stdout(
    proc: subprocess.Popen[str],
    log_file: TextIO,
    shard_index: int,
    gpu_id: str,
) -> None:
    prefix = f"[shard{shard_index}:gpu{gpu_id}] "
    stdout = proc.stdout
    if stdout is None:
        return
    try:
        for line in stdout:
            log_file.write(line)
            log_file.flush()
            display = line if line.endswith("\n") else f"{line}\n"
            print(f"{prefix}{display}", end="", flush=True)
    finally:
        try:
            stdout.close()
        except OSError:
            pass


def launch(args: argparse.Namespace) -> int:
    os.environ.update(apply_eval_harden_env(os.environ, shard_index=None))
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    workers_per_gpu = resolve_workers_per_gpu(args.workers_per_gpu)
    assignments = expand_shard_assignments(gpus, workers_per_gpu)
    num_shards = len(assignments)
    base = Path(args.output_dir).resolve()
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_log = base / "run_eval_shards.parent.log"
    parent_log.write_text(
        f"launch_start python={sys.executable} output_dir={base}\n",
        encoding="utf-8",
    )

    from worldatlas.benchmark.metric_errors import MetricLoadError
    from worldatlas.common.progress import log_progress

    metrics = _launch_metric_names(args, project_root)
    log_progress(
        "job_start",
        job_id=os.environ.get("WORLDATLAS_JOB_ID"),
        model=args.model_name,
        suite=args.suites,
        metrics=",".join(metrics) if metrics else "config",
        shards=num_shards,
        gpus=",".join(gpus),
        workers_per_gpu=workers_per_gpu,
        python=sys.executable,
        output_dir=str(base),
        predictions_root=str(Path(args.predictions_root).resolve()),
    )
    from worldatlas.benchmark.local_checkpoints import prepare_metric_checkpoints

    try:
        env_updates = prepare_metric_checkpoints(metrics)
    except MetricLoadError as exc:
        message = f"ERROR: {exc}"
        print(message, flush=True)
        parent_log.write_text(
            parent_log.read_text(encoding="utf-8") + message + "\n",
            encoding="utf-8",
        )
        log_progress("metric_load", status="fail", error=str(exc))
        return 2
    os.environ.update(env_updates)
    for key, value in env_updates.items():
        log_progress("metric_load", status="local_checkpoint", key=key, path=value)

    pending = []
    attempts: dict[int, int] = defaultdict(int)
    skipped = 0
    for shard_index, gpu_id in assignments:
        shard_dir = base / f"shard{shard_index}"
        skip_existing = False
        skip_reason = "disabled"
        if args.skip_existing_shards:
            skip_existing, skip_reason = _can_skip_existing_shard(
                shard_dir,
                _shard_resume_signature(
                    args,
                    shard_index=shard_index,
                    num_shards=num_shards,
                    predictions_root=Path(args.predictions_root).resolve(),
                ),
            )
        if skip_existing:
            log_progress(
                "shard_skip",
                shard=f"{shard_index}/{num_shards}",
                gpu=gpu_id,
                status="skip",
                reason=skip_reason,
                output_dir=str(shard_dir),
            )
            print(f"[run_eval_shards] shard={shard_index} gpu={gpu_id} skip existing {shard_dir}", flush=True)
            skipped += 1
            continue
        if args.skip_existing_shards and (shard_dir / "per_sample_scores.jsonl").exists():
            log_progress(
                "shard_rerun",
                shard=f"{shard_index}/{num_shards}",
                gpu=gpu_id,
                status="rerun",
                reason=skip_reason,
                output_dir=str(shard_dir),
            )
        pending.append((shard_index, gpu_id))

    max_concurrent = args.max_concurrent_shards if args.max_concurrent_shards and args.max_concurrent_shards > 0 else len(pending)
    max_concurrent = max(max_concurrent, 1)
    max_concurrent_per_gpu = max(args.max_concurrent_per_gpu, 0)
    requested_retries = getattr(args, "cuda_retries", DEFAULT_CUDA_RETRIES)
    args.cuda_retries = resolve_cuda_retries(metrics, requested_retries)
    if "reprojection_error" in metrics:
        # DROID's dense frontend can consume most of an 80 GiB GPU. Keep
        # the shard layout (and checkpoints), but run each GPU's shards serially.
        max_concurrent_per_gpu = 1
        args.max_concurrent_per_gpu = 1
        log_progress("gpu_concurrency", metrics="reprojection_error",
                     max_concurrent_per_gpu=1, reason="DROID memory budget")
        log_progress(
            "cuda_retries",
            metrics="reprojection_error",
            cuda_retries=args.cuda_retries,
            reason="DROID leak requires process recycle; resume healthy samples",
        )

    def pop_launchable():
        if not pending:
            return None
        if max_concurrent_per_gpu == 0:
            return pending.pop(0)
        running_per_gpu: dict[str, int] = {}
        for _, gpu_id, _, _, _, _ in remaining:
            running_per_gpu[gpu_id] = running_per_gpu.get(gpu_id, 0) + 1
        for index, (_, gpu_id) in enumerate(pending):
            if running_per_gpu.get(gpu_id, 0) < max_concurrent_per_gpu:
                return pending.pop(index)
        return None

    def launch_one(shard_index: int, gpu_id: str):
        attempts[shard_index] += 1
        env = apply_eval_harden_env(
            os.environ.copy(),
            shard_index=shard_index,
            python_executable=sys.executable,
        )
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["WORLDATLAS_EVAL_GPU"] = gpu_id
        env["WORLDATLAS_EVAL_SHARD_INDEX"] = str(shard_index)
        env["WORLDATLAS_EVAL_NUM_SHARDS"] = str(num_shards)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("OPENCV_FFMPEG_THREADS", "1")
        if "segment_continuity" in metrics:
            env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
            env.setdefault("TF_NUM_INTEROP_THREADS", "1")
            env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env["WORLDATLAS_CHECKPOINT_ROOT"] = str(project_root / "ckpt")
        pythonpath = [str(project_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        featup_dir = env.get(FEATUP_NATIVE_DIR_ENV, "").strip()
        if featup_dir:
            _prepend_env_path(env, "PYTHONPATH", featup_dir)
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--internal-shard",
            "--config", args.config,
            "--manifest", args.manifest,
            "--predictions-root", args.predictions_root,
            "--output-dir", str(base),
            "--suites", args.suites,
            "--num-shards", str(num_shards),
            "--shard-index", str(shard_index),
        ]
        if args.model_config:
            cmd.extend(["--model-config", args.model_config])
        if not args.skip_existing_shards:
            cmd.append("--no-skip-existing-shards")
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.metrics:
            cmd.extend(["--metrics", args.metrics])
        if args.motion_frame_max_side:
            cmd.extend(["--motion-frame-max-side", str(args.motion_frame_max_side)])
        if args.motion_smoothness_frame_max_side:
            cmd.extend(["--motion-smoothness-frame-max-side", str(args.motion_smoothness_frame_max_side)])
        if args.model_name:
            cmd.extend(["--model-name", args.model_name])
        attempt_suffix = "" if attempts[shard_index] == 1 else f".attempt{attempts[shard_index]}"
        log_path = log_dir / f"eval_shard{shard_index}_gpu{gpu_id}{attempt_suffix}.log"
        env[FEATUP_FAULT_LOG_ENV] = str(log_path.with_name(log_path.name + ".fault"))
        log_file = log_path.open("a", encoding="utf-8")
        log_progress(
            "shard_launch",
            shard=f"{shard_index}/{num_shards}",
            gpu=gpu_id,
            log=str(log_path),
            python=sys.executable,
            metrics=",".join(metrics) if metrics else "config",
        )
        print(f"[run_eval_shards] shard={shard_index} gpu={gpu_id} log={log_path}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        tee_thread = threading.Thread(
            target=_tee_shard_stdout,
            args=(proc, log_file, shard_index, gpu_id),
            name=f"eval-shard-tee-{shard_index}",
            daemon=True,
        )
        tee_thread.start()
        return (shard_index, gpu_id, proc, log_file, log_path, tee_thread)

    def wait_met3r_warmup(handle) -> str:
        _shard_index, _gpu_id, proc, _log_file, log_path, _tee_thread = handle
        started = time.time()
        deadline = started + _met3r_warmup_seconds()
        next_note = started + MET3R_WARMUP_SOFT_SECONDS
        while True:
            if proc.poll() is not None:
                return "exited"
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if met3r_warmup_ready_text(text):
                return "ready"
            if met3r_warmup_load_failed_text(text):
                if proc.poll() is None:
                    proc.terminate()
                return "load_failed"
            now = time.time()
            if now >= deadline:
                return "timeout"
            if now >= next_note:
                log_progress(
                    "met3r_warmup",
                    status="still_waiting",
                    shard=f"{_shard_index}/{num_shards}",
                    gpu=_gpu_id,
                    elapsed_s=int(now - started),
                    last_progress=_last_progress_snippet(log_path),
                    remaining=len(pending),
                )
                print(
                    f"[run_eval_shards] MeT3R warmup still waiting shard={_shard_index} "
                    f"elapsed_s={int(now - started)} last_progress={_last_progress_snippet(log_path)}",
                    flush=True,
                )
                next_note = now + MET3R_WARMUP_SOFT_SECONDS
            time.sleep(MET3R_WARMUP_POLL_SECONDS)

    failed = False
    warmup_abort = False
    remaining = []
    last_heartbeat = 0.0
    finished = skipped
    if includes_met3r(metrics) and pending:
        warmup = take_met3r_warmup_assignment(pending)
        assert warmup is not None
        warmup_timeout = _met3r_warmup_seconds()
        log_progress(
            "met3r_warmup",
            status="start",
            shard=f"{warmup[0]}/{num_shards}",
            gpu=warmup[1],
            timeout_s=warmup_timeout,
            remaining=len(pending),
        )
        print(
            f"[run_eval_shards] MeT3R warmup shard={warmup[0]} gpu={warmup[1]} "
            f"timeout_s={warmup_timeout} remaining={len(pending)}",
            flush=True,
        )
        handle = launch_one(*warmup)
        remaining.append(handle)
        status = wait_met3r_warmup(handle)
        log_progress(
            "met3r_warmup",
            status=status,
            shard=f"{warmup[0]}/{num_shards}",
            gpu=warmup[1],
            remaining=len(pending),
        )
        print(
            f"[run_eval_shards] MeT3R warmup {status} shard={warmup[0]} gpu={warmup[1]} "
            f"remaining={len(pending)}",
            flush=True,
        )
        if met3r_warmup_blocks_remaining(status):
            pending.clear()
            warmup_abort = met3r_warmup_skips_merge(status)
            exit_code = handle[2].returncode
            last_progress = _last_progress_snippet(handle[4])
            log_progress(
                "met3r_warmup",
                status="fail_fast",
                shard=f"{warmup[0]}/{num_shards}",
                gpu=warmup[1],
                reason=(
                    "warmup shard exited before model ready; not launching remaining"
                    if status == "exited"
                    else "warmup FeatUp/MeT3R load failed; not launching remaining"
                    if status == "load_failed"
                    else "warmup timed out before model ready; not launching remaining"
                ),
                exit_code=exit_code,
                last_progress=last_progress,
            )
            print(
                f"[run_eval_shards] MeT3R warmup {status}; not launching remaining shards "
                f"exit_code={exit_code} last_progress={last_progress}",
                flush=True,
            )
    while pending or remaining:
        while pending and len(remaining) < max_concurrent:
            launchable = pop_launchable()
            if launchable is None:
                break
            remaining.append(launch_one(*launchable))
        still = []
        for shard_index, gpu_id, proc, log_file, log_path, tee_thread in remaining:
            code = proc.poll()
            if code is None:
                still.append((shard_index, gpu_id, proc, log_file, log_path, tee_thread))
                continue
            tee_thread.join(timeout=5.0)
            log_file.close()
            log_progress(
                "shard_exit",
                shard=f"{shard_index}/{num_shards}",
                gpu=gpu_id,
                status="ok" if code == 0 else "fail",
                exit_code=code,
                log=str(log_path),
            )
            print(f"[run_eval_shards] shard={shard_index} gpu={gpu_id} exited code={code}", flush=True)
            if should_retry_cuda_worker(code, attempts[shard_index], getattr(args, "cuda_retries", DEFAULT_CUDA_RETRIES)):
                pending.append((shard_index, gpu_id))
                log_progress("shard_retry", shard=shard_index, gpu=gpu_id,
                             next_attempt=attempts[shard_index] + 1, reason="fresh CUDA process; resume healthy samples")
                continue
            failed = failed or code != 0
            finished += 1
        remaining = still
        if pending or remaining:
            now = time.time()
            if now - last_heartbeat >= max(args.heartbeat_interval, 1):
                active = [
                    f"{shard_index}:gpu{gpu_id}:pid{proc.pid}"
                    for shard_index, gpu_id, proc, _, _, _ in remaining[:8]
                ]
                suffix = " ..." if len(remaining) > 8 else ""
                latest = []
                for shard_index, gpu_id, _, _, log_path, _ in remaining[:8]:
                    snippet = _last_progress_snippet(log_path)
                    if snippet:
                        latest.append(f"shard{shard_index}:{snippet}")
                log_progress(
                    "heartbeat",
                    finished_or_skipped=f"{finished}/{num_shards}",
                    running=len(remaining),
                    pending=len(pending),
                    active=",".join(active) + suffix,
                    latest="; ".join(latest) if latest else None,
                )
                print(
                    "[run_eval_shards] HEARTBEAT "
                    f"finished_or_skipped={finished}/{num_shards} running={len(remaining)} pending={len(pending)} "
                    f"active={','.join(active)}{suffix}",
                    flush=True,
                )
                last_heartbeat = now
            time.sleep(5)

    if warmup_abort:
        log_progress(
            "job_done",
            job_id=os.environ.get("WORLDATLAS_JOB_ID"),
            shards=num_shards,
            status="fail",
            reason="met3r_warmup_did_not_become_ready",
            output_dir=str(base),
        )
        print(
            "[run_eval_shards] MeT3R warmup did not become ready; skipping merge "
            "(native FeatUp load failed or timed out)",
            flush=True,
        )
        return 1
    if failed:
        log_progress(
            "job_done",
            job_id=os.environ.get("WORLDATLAS_JOB_ID"),
            shards=num_shards,
            status="fail",
            reason="shard_failed_skip_merge",
            output_dir=str(base),
        )
        print(
            "[run_eval_shards] one or more shards failed; skipping merge "
            "(identity reject, OOM, or worker crash is not a coverage merge)",
            flush=True,
        )
        return 1
    result = merge(args, num_shards)
    merge_failed = bool(result.get("failure_summary") or result.get("suite_level_failure_summary")
                        or result.get("run_manifest", {}).get("missing_predictions"))
    log_progress(
        "job_done",
        job_id=os.environ.get("WORLDATLAS_JOB_ID"),
        shards=num_shards,
        status="fail" if failed or merge_failed else "ok",
        output_dir=str(base),
    )
    print("[run_eval_shards] MERGED per_metric_summary:")
    print(json.dumps(result["per_metric_summary"], indent=2, ensure_ascii=False))
    return 1 if failed or merge_failed else 0


def main() -> int:
    args = parse_args()
    if args.internal_shard:
        return run_shard(args)
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
