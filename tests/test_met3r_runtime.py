from __future__ import annotations

import numpy as np
import pytest

from worldarena.benchmark import met3r_runtime


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        ("cosine", 2.0),
        ("lpips", 2.0),
        ("rmse", 2.0),
        ("mse", 4.0),
        ("psnr", 0.0),
        ("ssim", 0.0),
    ],
)
def test_empty_overlap_penalty_uses_worst_raw_endpoint(
    distance: str,
    expected: float,
) -> None:
    assert met3r_runtime._empty_overlap_penalty(distance) == expected


def test_empty_overlap_is_penalized_instead_of_reported_as_perfect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class FakeMetric:
        def __call__(self, **_: object):
            scores = torch.tensor([0.0, 0.4], dtype=torch.float32)
            masks = torch.stack(
                [
                    torch.zeros((2, 2), dtype=torch.float32),
                    torch.ones((2, 2), dtype=torch.float32),
                ]
            )
            return scores, masks

    options = {
        "device": "cpu",
        "img_size": 2,
        "use_norm": True,
        "backbone": "mast3r",
        "feature_backbone": "dino_vits16",
        "feature_backbone_weights": "unused",
        "upsampler": "bilinear",
        "distance": "cosine",
        "freeze": True,
        "batch_size": 2,
    }
    monkeypatch.setattr(met3r_runtime, "_maybe_run_in_subprocess", lambda *args, **kwargs: None)
    monkeypatch.setattr(met3r_runtime, "inspect_met3r_layout", lambda: {"ready": True})
    monkeypatch.setattr(met3r_runtime, "_metric_options", lambda runtime: options)
    monkeypatch.setattr(met3r_runtime, "_load_met3r_model", lambda *args, **kwargs: FakeMetric())
    monkeypatch.setattr(
        met3r_runtime,
        "_pair_batch_to_tensor",
        lambda pairs, device, img_size: torch.zeros((len(pairs), 2, 3, 2, 2)),
    )

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    result = met3r_runtime.compute_met3r_consistency(
        [frame, frame],
        [frame, frame],
        runtime={},
    )

    assert result["raw"] == pytest.approx(1.2)
    assert result["details"]["overlap_ratios"] == [0.0, 1.0]
    assert result["details"]["valid_pair_count"] == 1
    assert result["details"]["empty_overlap_pair_count"] == 1
    assert result["details"]["empty_overlap_pair_indices"] == [0]
    assert result["details"]["empty_overlap_policy"] == "worst_raw_score"
