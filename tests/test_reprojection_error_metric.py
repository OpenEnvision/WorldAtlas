from __future__ import annotations

import sys
from types import SimpleNamespace
from types import ModuleType

import torch


if "droid" not in sys.modules:
    droid_stub = ModuleType("droid")
    droid_stub.Droid = object
    sys.modules["droid"] = droid_stub

from worldarena.benchmark.official_backends.reprojection_error import (
    _bounded_neighborhood_edges,
    _valid_reprojection_errors,
)


def test_valid_reprojection_errors_excludes_invisible_zero_confidence_and_nonfinite_points() -> None:
    coords = torch.tensor(
        [[[[[3.0, 4.0]], [[2.0, 0.0]], [[float("nan"), 1.0]]]]]
    )
    target = torch.zeros_like(coords)
    visible = torch.tensor([[[[[1.0]], [[0.0]], [[1.0]]]]])
    weight = torch.tensor(
        [[[[[1.0, 1.0]], [[1.0, 1.0]], [[0.0, 0.0]]]]]
    )
    graph = SimpleNamespace(
        ii=torch.tensor([0]),
        jj=torch.tensor([1]),
        target=target,
        weight=weight,
        video=SimpleNamespace(reproject=lambda _ii, _jj: (coords, visible)),
    )

    errors = _valid_reprojection_errors(graph)

    assert errors.tolist() == [5.0]


def test_reprojection_metric_uses_bounded_dense_ba_factor_budget() -> None:
    from worldarena.benchmark.official_backends.reprojection_error import ReprojectionErrorMetric

    metric = ReprojectionErrorMetric()

    assert metric._args.stride == 4
    assert metric._args.max_factors == 512


def test_bounded_neighborhood_edges_are_bidirectional_and_respect_budget() -> None:
    ii, jj = _bounded_neighborhood_edges(10, radius=2, budget=12, device="cpu")

    pairs = set(zip(ii.tolist(), jj.tolist()))
    assert len(pairs) == 12
    assert all(abs(source - target) <= 2 for source, target in pairs)
    assert all((target, source) in pairs for source, target in pairs)
