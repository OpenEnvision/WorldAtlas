from __future__ import annotations

import numpy as np

from worldarena.benchmark import quality_backends


def test_ensure_numpy_compat_for_imgaug_restores_sctypes(monkeypatch) -> None:
    monkeypatch.delattr(np, "sctypes", raising=False)

    quality_backends._ensure_numpy_compat_for_imgaug()

    assert hasattr(np, "sctypes")
    assert np.float32 in np.sctypes["float"]
    assert np.complex64 in np.sctypes["complex"]
