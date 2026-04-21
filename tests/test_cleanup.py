from pathlib import Path

from worldarena.common.cleanup import collect_cleanup_targets


def test_collect_cleanup_targets_skips_missing_entries(tmp_path: Path) -> None:
    (tmp_path / ".cache").mkdir()
    (tmp_path / "worldarena" / "__pycache__").mkdir(parents=True)
    (tmp_path / "tests" / "__pycache__").mkdir(parents=True)
    (tmp_path / "worldarena.egg-info").mkdir()
    (tmp_path / "docs" / "generated").mkdir(parents=True)

    targets = collect_cleanup_targets(tmp_path, include_generated=False)

    assert tmp_path / ".cache" in targets
    assert tmp_path / "worldarena" / "__pycache__" in targets
    assert tmp_path / "tests" / "__pycache__" in targets
    assert tmp_path / "worldarena.egg-info" in targets
    assert tmp_path / "docs" / "generated" not in targets


def test_collect_cleanup_targets_can_include_generated_outputs(tmp_path: Path) -> None:
    (tmp_path / "docs" / "generated").mkdir(parents=True)
    (tmp_path / "artifacts" / "dataset_stats").mkdir(parents=True)

    targets = collect_cleanup_targets(tmp_path, include_generated=True)

    assert tmp_path / "docs" / "generated" in targets
    assert tmp_path / "artifacts" / "dataset_stats" in targets
