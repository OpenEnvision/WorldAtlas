from __future__ import annotations

import numpy as np
import pytest

import worldarena.benchmark_core.reference_runtime as reference_runtime


class _DummyReferenceClipScore:
    def __call__(self, rendered_image, text_prompt) -> float:
        assert rendered_image is not None
        assert text_prompt == "prompt"
        return 80.0

    def reset(self) -> None:
        return None


class _DummyReferenceGramMatrix:
    pass


class _DummyScalarMetric:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, rendered_images) -> float:
        assert rendered_images is not None
        return self.value


def _frame(value: int = 0) -> np.ndarray:
    frame = np.full((8, 8, 3), value, dtype=np.uint8)
    return frame


def test_reference_prompt_alignment_scales_native_scores(monkeypatch) -> None:
    monkeypatch.setattr(
        reference_runtime,
        "_load_reference_clipscore_backend",
        lambda: ("clipscore_torchmetrics", _DummyReferenceClipScore(), "cpu"),
    )
    monkeypatch.setattr(
        reference_runtime,
        "_frame_to_uint8_tensor",
        lambda frame, device: frame,
    )

    result = reference_runtime.compute_reference_prompt_alignment([_frame()], "prompt")

    assert result["backend"] == "clipscore_torchmetrics"
    assert result["raw"] == 0.8
    assert result["details"]["frame_scores"] == [0.8]
    assert result["details"]["native_frame_scores"] == [80.0]
    assert result["details"]["native_score_range"] == [0.0, 100.0]


def test_reference_style_consistency_maps_distance_to_similarity(monkeypatch) -> None:
    monkeypatch.setattr(
        reference_runtime,
        "_style_distance",
        lambda reference_frame, prediction_frame: ("vgg19_gram", 0.25, [0.2, 0.3]),
    )

    result = reference_runtime.compute_reference_style_consistency([_frame()], [_frame(12)])

    assert result["backend"] == "vgg19_gram"
    assert result["raw"] == 0.8
    assert result["details"]["frame_scores"] == [0.8]
    assert result["details"]["native_frame_distances"] == [0.25]
    assert result["details"]["native_layer_distances"] == [[0.2, 0.3]]


def test_reference_perceptual_quality_aggregates_components(monkeypatch) -> None:
    components = {
        "clip_iqa": ("clipiqa_plus", _DummyScalarMetric(0.6), "cpu", 1.0),
        "aesthetic": ("laion_aes", _DummyScalarMetric(7.0), "cpu", 10.0),
    }
    monkeypatch.setattr(
        reference_runtime,
        "_load_reference_quality_component",
        lambda component_name: components.get(component_name),
    )
    monkeypatch.setattr(
        reference_runtime,
        "_frame_to_float_tensor",
        lambda frame, device: frame,
    )

    result = reference_runtime.compute_reference_perceptual_quality([_frame(), _frame(32)])

    assert result["backend"] == "pyiqa_stack"
    assert result["raw"] == pytest.approx(0.65)
    assert result["details"]["frame_scores"] == [0.65, 0.65]
    assert result["details"]["required_components"] == ["clip_iqa", "aesthetic"]
    assert result["details"]["components"]["clip_iqa"]["backend"] == "clipiqa_plus"
    assert result["details"]["components"]["aesthetic"]["backend"] == "laion_aes"


def test_reference_perceptual_quality_requires_all_components(monkeypatch) -> None:
    components = {
        "clip_iqa": ("clipiqa_plus", _DummyScalarMetric(0.6), "cpu", 1.0),
        "aesthetic": None,
    }
    monkeypatch.setattr(
        reference_runtime,
        "_load_reference_quality_component",
        lambda component_name: components.get(component_name),
    )

    assert not reference_runtime.reference_perceptual_quality_available()
    with pytest.raises(ModuleNotFoundError, match="aesthetic"):
        reference_runtime.compute_reference_perceptual_quality([_frame()])
