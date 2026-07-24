from __future__ import annotations

from worldarena.docs.site_export import _extract_summary, _strip_metric_examples_block


def test_strip_metric_examples_block_removes_injected_gallery() -> None:
    content = "\n".join(
        [
            "# Metric",
            "Intro paragraph.",
            "<!-- metric-examples:start -->",
            "## Examples",
            "<div>gallery</div>",
            "<!-- metric-examples:end -->",
            "## References",
            "More text.",
        ]
    )

    stripped = _strip_metric_examples_block(content)

    assert "metric-examples:start" not in stripped
    assert "metric-examples:end" not in stripped
    assert "Intro paragraph." in stripped
    assert "## References" in stripped


def test_extract_summary_ignores_metric_examples_section() -> None:
    content = "\n".join(
        [
            "# Metric",
            "Intro paragraph.",
            "<!-- metric-examples:start -->",
            "## Representative Case Comparisons",
            "<div>gallery</div>",
            "<!-- metric-examples:end -->",
            "## References",
            "More text.",
        ]
    )

    summary = _extract_summary(content)

    assert "Intro paragraph." in summary
    assert "Representative Case Comparisons" not in summary
    assert "metric-examples:end" not in summary


def test_extract_summary_excludes_references_section() -> None:
    content = "\n".join(
        [
            "# Rollout Drift",
            "Intro paragraph.",
            "<!-- metric-examples:start -->",
            "## Representative Case Comparisons",
            "<div>gallery</div>",
            "<!-- metric-examples:end -->",
            "## References {#reference}",
            "Citation text.",
        ]
    )

    summary = _extract_summary(content)

    assert "Intro paragraph." in summary
    assert "References" not in summary
    assert "Citation text." not in summary
