#!/usr/bin/env python3
"""Check the installed CLI and public configs without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from worldatlas.benchmark.config import load_config
from worldatlas.cli import build_parser
from worldatlas.models.config import load_model_config
from worldatlas.models.registry import build_model_adapter


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, help="Also check this model's interpreter, source and weight directories")
    args = parser.parse_args()
    build_parser()
    load_config(ROOT / "config/benchmark.yaml")
    failures = []
    paths = [p for p in sorted((ROOT / "config/models").rglob("*.yaml"))
             if not any(part == "backup" for part in p.parts)
             and not any(token in p.stem for token in ("_local", "_debug", "_juanxi"))]
    for path in paths:
        try:
            build_model_adapter(load_model_config(path))
        except Exception as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    if args.model_config:
        config = load_model_config(args.model_config)
        if shutil.which(config.python_bin) is None:
            failures.append(f"Interpreter is not executable: {config.python_bin}")
        for label, path in [("source", config.repo_root), ("checkpoint", config.checkpoint_dir)]:
            if path is not None and not path.is_dir():
                failures.append(f"Missing {label} directory: {path}")
        if shutil.which("ffmpeg") is None:
            failures.append("ffmpeg is missing from PATH")
    print(json.dumps({"python": sys.version.split()[0], "model_configs": len(paths),
                      "failures": failures, "gpu_inference_checked": False}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
