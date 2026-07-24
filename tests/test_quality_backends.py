from __future__ import annotations

import numpy as np

from worldarena.benchmark import quality_backends


def test_ensure_numpy_compat_for_imgaug_restores_sctypes(monkeypatch) -> None:
    monkeypatch.delattr(np, "sctypes", raising=False)

    quality_backends._ensure_numpy_compat_for_imgaug()

    assert hasattr(np, "sctypes")
    assert np.float32 in np.sctypes["float"]
    assert np.complex64 in np.sctypes["complex"]


def test_hpsv3_scores_frames_in_batches_and_selects_reward_mean(monkeypatch) -> None:
    class FakeInferencer:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.next_reward = 1.0

        def reward(self, prompts, image_paths):
            assert len(prompts) == len(image_paths)
            self.calls.append(len(image_paths))
            rewards = []
            for _ in image_paths:
                rewards.append([self.next_reward, 99.0])
                self.next_reward += 1.0
            return np.asarray(rewards, dtype=np.float32)

    inferencer = FakeInferencer()
    monkeypatch.setattr(
        quality_backends,
        "load_hpsv3_backend",
        lambda **kwargs: (
            "hpsv3",
            inferencer,
            "cpu",
            "/tmp/HPSv3.safetensors",
            "/tmp/HPSv3.yaml",
        ),
    )

    frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(5)]
    result = quality_backends.compute_hpsv3_norm_scores(
        frames,
        p1=0.0,
        p99=10.0,
        batch_size=2,
    )

    assert inferencer.calls == [2, 2, 1]
    assert result["raw_rewards"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["raw_reward_mean"] == 3.0
    assert result["raw"] == 0.3
