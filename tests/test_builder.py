from pathlib import Path

import plotly.graph_objects as go

from worldarena.docs.builder import ChartWriter, _artifact_path


def test_chart_writer_uses_configured_src_prefix(tmp_path: Path) -> None:
    writer = ChartWriter(tmp_path, "../generated/charts")

    snippet = writer.render(
        "captions",
        "coverage",
        "Caption Coverage",
        go.Figure(),
    )

    assert 'src="../generated/charts/captions-coverage.html"' in snippet
    assert (tmp_path / "captions-coverage.html").exists()


def test_artifact_path_builds_relative_links() -> None:
    assert _artifact_path("../generated/data", "summary.json") == "../generated/data/summary.json"
