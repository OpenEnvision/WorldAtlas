#!/usr/bin/env python3
"""Validate a model config for checkpoint-resident batch production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worldarena.models.config import load_model_config  # noqa: E402
from worldarena.models.registry import build_model_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a WorldArena model config before multi-GPU batch production. "
            "The strict mode accepts only adapters verified to load checkpoints once per worker."
        )
    )
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--total-gpus", default=8, type=int)
    parser.add_argument("--require-load-once", action="store_true")
    parser.add_argument(
        "--validate-paths", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    return parser.parse_args()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return normalized.lower() or "model"


def _gpus_per_worker(generation: dict[str, Any]) -> int:
    for key in ("gpus_per_worker", "nproc_per_node", "num_gpus"):
        value = generation.get(key)
        if value is not None:
            count = int(value)
            if count <= 0:
                raise ValueError(f"generation.{key} must be positive, got {count}")
            return count
    return 1


def _validate_runtime_paths(config) -> None:
    missing: list[str] = []
    if config.repo_root is not None and not config.repo_root.is_dir():
        missing.append(f"repo_root={config.repo_root}")
    if config.checkpoint_dir is not None and not config.checkpoint_dir.exists():
        missing.append(f"checkpoint_dir={config.checkpoint_dir}")

    python_bin = str(config.python_bin)
    if "/" in python_bin:
        candidate = Path(python_bin).expanduser()
        if not candidate.is_file():
            missing.append(f"python_bin={candidate}")
    elif shutil.which(python_bin) is None:
        missing.append(f"python_bin={python_bin}")

    if missing:
        raise FileNotFoundError("missing model runtime path(s): " + ", ".join(missing))


def inspect_model(
    model_config: Path,
    *,
    total_gpus: int,
    require_load_once: bool,
    validate_paths: bool,
) -> dict[str, Any]:
    if total_gpus <= 0:
        raise ValueError(f"total_gpus must be positive, got {total_gpus}")
    config = load_model_config(model_config.expanduser().resolve())
    adapter = build_model_adapter(config)
    supports_batch = adapter.supports_batch_generation()
    checkpoint_policy = adapter.batch_checkpoint_load_policy()
    if require_load_once and (not supports_batch or checkpoint_policy != "load_once"):
        raise RuntimeError(
            f"{config.name} is not approved for checkpoint-resident production: "
            f"supports_batch={supports_batch}, checkpoint_policy={checkpoint_policy!r}"
        )
    if validate_paths:
        _validate_runtime_paths(config)

    gpus_per_worker = _gpus_per_worker(config.generation)
    if gpus_per_worker > total_gpus:
        raise ValueError(
            f"{config.name} needs {gpus_per_worker} GPUs per worker, "
            f"but only {total_gpus} were requested"
        )
    if total_gpus % gpus_per_worker:
        raise ValueError(
            f"total_gpus={total_gpus} must be divisible by "
            f"gpus_per_worker={gpus_per_worker}"
        )

    return {
        "model_name": config.name,
        "model_slug": _slug(config.name),
        "family": config.family,
        "supported_suites": list(config.supported_suites),
        "supports_batch_generation": supports_batch,
        "checkpoint_load_policy": checkpoint_policy,
        "gpus_per_worker": gpus_per_worker,
        "worker_count": total_gpus // gpus_per_worker,
        "total_gpus": total_gpus,
        "python_bin": config.python_bin,
        "repo_root": str(config.repo_root) if config.repo_root is not None else None,
        "checkpoint_dir": (
            str(config.checkpoint_dir) if config.checkpoint_dir is not None else None
        ),
    }


def _safe_line(value: object) -> str:
    text = str(value)
    if "\n" in text or "\r" in text:
        raise ValueError(f"preflight line value contains a newline: {text!r}")
    return text


def main() -> int:
    args = parse_args()
    payload = inspect_model(
        args.model_config,
        total_gpus=args.total_gpus,
        require_load_once=args.require_load_once,
        validate_paths=args.validate_paths,
    )
    if args.format == "lines":
        values = (
            payload["model_name"],
            payload["model_slug"],
            payload["family"],
            payload["checkpoint_load_policy"],
            payload["gpus_per_worker"],
            payload["worker_count"],
            ",".join(payload["supported_suites"]),
        )
        print("\n".join(_safe_line(value) for value in values))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
