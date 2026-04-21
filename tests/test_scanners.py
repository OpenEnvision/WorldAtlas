from pathlib import Path

from worldarena.datasets.scanners import bucketize_duration, should_skip_path


def test_bucketize_duration() -> None:
    assert bucketize_duration(None) is None
    assert bucketize_duration(2.5) == "0-5s"
    assert bucketize_duration(5.0) == "5-10s"
    assert bucketize_duration(12.0) == "10-20s"
    assert bucketize_duration(21.0) == "20s+"


def test_should_skip_path() -> None:
    assert should_skip_path(Path("dynamic/.cache/file.jpg"), [".", "_backup"]) is True
    assert should_skip_path(Path("dynamic/_backup_before_v1/file.jpg"), [".", "_backup"]) is True
    assert should_skip_path(Path("dynamic/images/articulated/000.jpg"), [".", "_backup"]) is False
