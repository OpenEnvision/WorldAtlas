from __future__ import annotations

import sys
from pathlib import Path

from worldarena.benchmark.flow_sources import isolated_upstream_import_context


def _write_utils_tree(root: Path, value: str) -> None:
    package_dir = root / "utils"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "utils.py").write_text(
        f"VALUE = {value!r}\n",
        encoding="utf-8",
    )


def test_isolated_upstream_import_context_cleans_conflicting_modules(tmp_path: Path) -> None:
    root_a = tmp_path / "repo_a"
    root_b = tmp_path / "repo_b"
    _write_utils_tree(root_a, "a")
    _write_utils_tree(root_b, "b")

    with isolated_upstream_import_context(root_a):
        from utils.utils import VALUE as value_a

    assert value_a == "a"
    assert "utils" not in sys.modules
    assert "utils.utils" not in sys.modules

    with isolated_upstream_import_context(root_b):
        from utils.utils import VALUE as value_b

    assert value_b == "b"
    assert "utils" not in sys.modules
    assert "utils.utils" not in sys.modules
