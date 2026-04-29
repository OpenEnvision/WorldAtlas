from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.flow_sources import inspect_flow_source_layout


LONG_SEQUENCE_MODULE_ROOT = Path(__file__).resolve().parent
LONG_SEQUENCE_PARAMS_PATH = LONG_SEQUENCE_MODULE_ROOT / "long_sequence_params.py"


def inspect_long_sequence_layout() -> dict[str, Any]:
    module_root = LONG_SEQUENCE_MODULE_ROOT.resolve()
    params_path = LONG_SEQUENCE_PARAMS_PATH.resolve()
    flow_layout = inspect_flow_source_layout()
    params_ready = params_path.exists()
    ready = params_ready and bool(flow_layout["ready"])
    error = None
    if not params_ready:
        error = (
            "WorldArena long-sequence native modules are incomplete; restore "
            "worldarena/benchmark/long_sequence_params.py"
        )
    elif not flow_layout["ready"]:
        error = str(flow_layout["error"])
    return {
        "module_root": str(module_root),
        "params_path": str(params_path),
        "config_mode": "builtin",
        "amt_root": str(flow_layout["amt_root"]),
        "raft_root": str(flow_layout["raft_root"]),
        "amt_ready": bool(flow_layout["amt_ready"]),
        "raft_ready": bool(flow_layout["raft_ready"]),
        "ready": ready,
        "error": error,
    }


__all__ = [
    "LONG_SEQUENCE_MODULE_ROOT",
    "LONG_SEQUENCE_PARAMS_PATH",
    "inspect_long_sequence_layout",
]
