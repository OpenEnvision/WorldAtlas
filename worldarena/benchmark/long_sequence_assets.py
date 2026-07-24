"""Asset preparation helpers for long-horizon sequence metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.benchmark.flow_sources import inspect_flow_source_layout


LONG_SEQUENCE_MODULE_ROOT = Path(__file__).resolve().parent


def inspect_long_sequence_layout() -> dict[str, Any]:
    """Inspect long sequence layout -> dict[str, Any]."""
    module_root = LONG_SEQUENCE_MODULE_ROOT.resolve()
    flow_layout = inspect_flow_source_layout()
    ready = bool(flow_layout["ready"])
    error = None
    if not flow_layout["ready"]:
        error = str(flow_layout["error"])
    return {
        "module_root": str(module_root),
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
    "inspect_long_sequence_layout",
]
