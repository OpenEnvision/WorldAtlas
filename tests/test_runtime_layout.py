from __future__ import annotations

from pathlib import Path

import pytest

from worldarena.common.runtime_layout import inspect_runtime_layout, prepare_runtime_layout


def test_prepare_runtime_layout_creates_local_cache_and_env_paths(tmp_path: Path) -> None:
    result = prepare_runtime_layout(tmp_path)

    cache_root = tmp_path / "cache" / "hfd"
    third_party_root = tmp_path / "third_party"

    assert cache_root.is_dir()
    assert third_party_root.is_dir()
    assert (cache_root / "huggingface" / "hub").is_dir()
    assert (cache_root / "transformers").is_dir()
    assert result["cache_mode"] == "directory"
    assert result["recommended_env"] == {
        "HF_HOME": str(cache_root),
        "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
    }
    assert result["runtime_pythons"]["managed"]["path"].endswith(".venv/bin/python")


def test_prepare_runtime_layout_links_shared_cache_and_repo_roots(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    shared_cache = tmp_path / "shared-cache"
    lingbot_source = tmp_path / "lingbot-source"
    lingbot_source.mkdir(parents=True, exist_ok=True)

    result = prepare_runtime_layout(
        project_root,
        shared_hf_cache=str(shared_cache),
        repo_links={"lingbot-world": str(lingbot_source)},
    )

    cache_link = project_root / "cache" / "hfd"
    repo_link = project_root / "third_party" / "lingbot-world"

    assert cache_link.is_symlink()
    assert cache_link.resolve() == shared_cache.resolve()
    assert repo_link.is_symlink()
    assert repo_link.resolve() == lingbot_source.resolve()
    assert result["linked_repositories"]["lingbot-world"] == {
        "link_path": str(repo_link),
        "source_path": str(lingbot_source.resolve()),
    }


def test_prepare_runtime_layout_refuses_conflicting_cache_without_force(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    existing_cache = project_root / "cache" / "hfd"
    existing_cache.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="already exists"):
        prepare_runtime_layout(project_root, shared_hf_cache=str(tmp_path / "shared-cache"))


def test_inspect_runtime_layout_reports_existing_layout_without_mutation(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    cache_root = project_root / "cache" / "hfd"
    metric_source = project_root / "third_party" / "metric-source"
    runtime_python = project_root / ".runtime312" / "bin" / "python"

    cache_root.mkdir(parents=True, exist_ok=True)
    metric_source.mkdir(parents=True, exist_ok=True)
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    result = inspect_runtime_layout(project_root)

    assert result["cache_mode"] == "directory"
    assert result["third_party_exists"] is True
    assert result["linked_repositories"]["metric-source"]["is_symlink"] is False
    assert result["runtime_pythons"]["runtime312"]["exists"] is True
