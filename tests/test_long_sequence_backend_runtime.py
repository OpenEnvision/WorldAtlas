from __future__ import annotations

from pathlib import Path

import pytest

from worldarena.benchmark import long_sequence_dispatch as runtime


def test_resolve_long_sequence_evaluator_rejects_unknown_dimension() -> None:
    with pytest.raises(KeyError):
        runtime.resolve_long_sequence_evaluator("unknown_dimension")


def test_evaluate_long_sequence_dimensions_uses_worldarena_dispatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    split_root = tmp_path / "input" / "split_clip"
    output_root = tmp_path / "output"
    manifest_path = output_root / "demo_full_info.json"
    saved: dict[str, object] = {}
    calls: list[tuple[str, str, object, bool]] = []

    monkeypatch.setattr(
        runtime,
        "prepare_long_video_clips",
        lambda videos_path, **kwargs: split_root,
    )
    monkeypatch.setattr(
        runtime,
        "build_long_video_manifest",
        lambda split_clip_root, *, output_path, name, dimension_list: manifest_path,
    )
    monkeypatch.setattr(
        runtime,
        "_save_json",
        lambda data, path: saved.update({"data": data, "path": path}),
    )

    def fake_resolve(dimension: str):
        def _evaluate(json_dir: str, device: object, submodules: object, **kwargs: object):
            calls.append((dimension, json_dir, submodules, bool(kwargs.get("dev_flag"))))
            return (0.5, [], [{"video_path": f"{dimension}.mp4", "video_results": 0.5}])

        return _evaluate

    monkeypatch.setattr(runtime, "resolve_long_sequence_evaluator", fake_resolve)

    result = runtime.evaluate_long_sequence_dimensions(
        videos_path=tmp_path / "input",
        output_path=output_root,
        name="demo",
        dimensions=["subject_consistency", "motion_smoothness", "dynamic_degree"],
        device="cpu",
        dev_flag=True,
    )

    assert set(result) == {"subject_consistency", "motion_smoothness", "dynamic_degree"}
    assert calls == [
        (
            "subject_consistency",
            str(manifest_path),
            None,
            True,
        ),
        (
            "motion_smoothness",
            str(manifest_path),
            None,
            True,
        ),
        (
            "dynamic_degree",
            str(manifest_path),
            None,
            True,
        ),
    ]
    assert saved["data"] == result
    assert saved["path"] == output_root / "demo_eval_results.json"
