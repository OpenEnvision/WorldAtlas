from __future__ import annotations

import worldarena.benchmark.long_sequence_runtime as long_sequence_runtime
from worldarena.benchmark.long_sequence_runtime import (
    inspect_long_sequence_layout,
    resolve_long_sequence_root,
)


def test_resolve_long_sequence_root_defaults_to_native_module_root() -> None:
    resolved = resolve_long_sequence_root()
    expected = long_sequence_runtime.PROJECT_ROOT / "worldarena" / "benchmark"

    assert resolved == expected.resolve()
    assert "long_sequence_backend" not in str(resolved)
    assert "long_sequence_vendor" not in str(resolved)


def test_inspect_long_sequence_layout_reports_native_assets_ready() -> None:
    layout = inspect_long_sequence_layout()

    assert layout["config_mode"] == "builtin"
    assert "thirdparty/AMT" in layout["amt_root"]
    assert "thirdparty/RAFT" in layout["raft_root"]
    assert layout["ready"] is (layout["amt_ready"] and layout["raft_ready"])
    if layout["ready"]:
        assert layout["error"] is None
    else:
        assert isinstance(layout["error"], str)
