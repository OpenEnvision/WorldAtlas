from __future__ import annotations

import numpy as np

from worldarena.benchmark import reference_runtime


def _frame(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_compute_reference_perceptual_quality_uses_shared_component_backend(
    monkeypatch,
) -> None:
    component_results = {
        "clip_iqa": {
            "backend": "clipiqa_plus",
            "raw": 0.3,
            "frame_scores": [0.2, 0.4],
            "native_frame_scores": [0.2, 0.4],
            "native_score_range": [0.0, 1.0],
        },
        "aesthetic": {
            "backend": "laion_aes",
            "raw": 0.7,
            "frame_scores": [0.6, 0.8],
            "native_frame_scores": [6.0, 8.0],
            "native_score_range": [0.0, 10.0],
        },
    }

    monkeypatch.setattr(
        reference_runtime,
        "compute_quality_component_scores",
        lambda frames, *, component_name: component_results[component_name],
    )

    result = reference_runtime.compute_reference_perceptual_quality([_frame(1), _frame(2)])

    assert result["raw"] == 0.5
    assert result["backend"] == "pyiqa_stack"
    assert result["details"]["frame_scores"] == [0.4, 0.6]
    assert result["details"]["required_components"] == ["clip_iqa", "aesthetic"]
    assert result["details"]["components"]["clip_iqa"]["backend"] == "clipiqa_plus"
    assert result["details"]["components"]["aesthetic"]["native_score_range"] == [0.0, 10.0]
