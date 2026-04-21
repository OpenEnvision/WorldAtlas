from __future__ import annotations

from pathlib import Path

from worldarena.common.environment import inspect_environment


def test_inspect_environment_groups_metrics_by_dimension(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "worldarena.common.environment.inspect_runtime_layout",
        lambda project_root: {
            "project_root": str(project_root),
            "cache_mode": "directory",
            "runtime_pythons": {},
        },
    )
    monkeypatch.setattr(
        "worldarena.common.environment.inspect_metric_source_layout",
        lambda: {
            "ready": True,
            "source_root": "/tmp/source",
            "package_name": "worldscore",
        },
    )
    monkeypatch.setattr(
        "worldarena.common.environment.inspect_official_metric_import",
        lambda metric_name: {"ready": metric_name != "object_detection", "module": f"fake.{metric_name}", "error": None if metric_name != "object_detection" else "missing spaCy"},
    )
    monkeypatch.setattr(
        "worldarena.common.environment._dependency_profile_status",
        lambda: {
            "core": {"numpy": True},
            "reference_metrics": {
                "torch": True,
                "torchmetrics": True,
                "torchvision": True,
                "transformers": True,
                "pyiqa": True,
            },
            "official_video_metrics": {
                "hydra": True,
                "omegaconf": True,
                "cvxpy": True,
                "spacy": False,
                "scipy": True,
                "timm": True,
                "einops": True,
                "tqdm": True,
            },
            "model_adapters": {"diffusers": False},
        },
    )
    monkeypatch.setattr("worldarena.common.environment._spacy_model_available", lambda: False)
    monkeypatch.setattr("worldarena.common.environment._gpu_status", lambda: [{"index": "0"}])

    result = inspect_environment(tmp_path)

    assert result["metric_source"]["ready"] is True
    assert result["dependency_profiles"]["spacy_model"]["en_core_web_sm"] is False
    assert result["metrics_by_dimension"]["camera_control"]["camera_error"]["ready"] is True
    assert result["metrics_by_dimension"]["object_control"]["object_detection"]["ready"] is False
    assert result["metrics_by_dimension"]["content_alignment"]["prompt_alignment"]["ready"] is True
    assert result["gpus"] == [{"index": "0"}]
