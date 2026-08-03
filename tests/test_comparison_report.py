"""Tests for comparison/report.py: rendering a table to Markdown/HTML/CSV/plot."""

from __future__ import annotations

import pandas as pd
import pytest

from comparison import report as report_module
from comparison.report import Reporter

requires_matplotlib = pytest.mark.skipif(
    report_module.plt is None, reason="matplotlib is not installed"
)


def _table():
    return pd.DataFrame({"model_name": ["svm", "logreg"], "f1": [0.7, 0.9]})


class TestToMarkdown:
    """Markdown rendering, no external dependency required."""

    def test_includes_title_and_header_row(self):
        text = Reporter(_table(), title="My Report").to_markdown()

        assert "# My Report" in text
        assert "| model_name | f1 |" in text
        assert "| svm | 0.7000 |" in text


class TestToHtml:
    """Standalone HTML rendering."""

    def test_includes_title_and_table(self):
        text = Reporter(_table(), title="My Report").to_html()

        assert "<title>My Report</title>" in text
        assert "<table" in text


class TestToCsv:
    """CSV export, creating parent directories as needed."""

    def test_writes_csv_creating_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "leaderboard.csv"
        result = Reporter(_table()).to_csv(path)

        assert result == path
        assert "svm,0.7" in path.read_text(encoding="utf-8")


def _grid():
    frame = pd.DataFrame(
        {"fast": [0.7, 0.6], "svm": [0.8, float("nan")]},
        index=["svm", "rf"],
    )
    frame.index.name = "model"
    frame.columns.name = "pipeline"
    return frame


class TestPlotBar:
    """Optional matplotlib bar chart, skipped silently when unavailable."""

    def test_saves_png_when_matplotlib_present(self, tmp_path):
        path = tmp_path / "plot.png"
        result = Reporter(_table()).plot_bar(path)

        if report_module.plt is None:
            assert result is None
        else:
            assert result == path
            assert path.is_file()

    def test_returns_none_when_matplotlib_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(report_module, "plt", None)
        result = Reporter(_table()).plot_bar(tmp_path / "plot.png")

        assert result is None
        assert not (tmp_path / "plot.png").exists()

    @requires_matplotlib
    def test_labels_bars_with_model_names_not_positions(self, monkeypatch, tmp_path):
        """The identity column names the bars; a RangeIndex would give 0..N-1."""
        ax = _plot_and_capture_axes(monkeypatch, tmp_path, Reporter(_table()).plot_bar)

        assert [t.get_text() for t in ax.get_xticklabels()] == ["svm", "logreg"]
        assert ax.get_xlabel() == "model_name"
        assert ax.get_ylabel() == "f1"

    @requires_matplotlib
    def test_labels_bars_from_the_index_when_it_carries_the_names(self, monkeypatch, tmp_path):
        table = _table().set_index("model_name")
        ax = _plot_and_capture_axes(monkeypatch, tmp_path, Reporter(table).plot_bar)

        assert [t.get_text() for t in ax.get_xticklabels()] == ["svm", "logreg"]


class TestPlotMatrix:
    """The grid shape renders as a labelled heatmap, not one arbitrary column."""

    def test_returns_none_when_matplotlib_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(report_module, "plt", None)
        result = Reporter(_grid()).plot_matrix(tmp_path / "grid.png")

        assert result is None
        assert not (tmp_path / "grid.png").exists()

    def test_returns_none_for_an_empty_table(self, tmp_path):
        assert Reporter(pd.DataFrame()).plot_matrix(tmp_path / "grid.png") is None

    @requires_matplotlib
    def test_labels_both_axes_with_model_and_pipeline_names(self, monkeypatch, tmp_path):
        ax = _plot_and_capture_axes(monkeypatch, tmp_path, Reporter(_grid()).plot_matrix)

        assert [t.get_text() for t in ax.get_xticklabels()] == ["fast", "svm"]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["svm", "rf"]
        assert ax.get_xlabel() == "pipeline"
        assert ax.get_ylabel() == "model"

    @requires_matplotlib
    def test_annotates_populated_cells_and_skips_excluded_pairings(self, monkeypatch, tmp_path):
        ax = _plot_and_capture_axes(monkeypatch, tmp_path, Reporter(_grid()).plot_matrix)

        assert {t.get_text() for t in ax.texts} == {"0.700", "0.600", "0.800"}


def _plot_and_capture_axes(monkeypatch, tmp_path, plot_method):
    """Run a plot method and return the Axes it drew.

    plt.close() unregisters the figure but keeps the object, so intercepting the
    close call is enough to inspect the artists the method actually drew.
    """
    captured = []
    real_close = report_module.plt.close

    def capture(figure):
        captured.append(figure)
        real_close(figure)

    monkeypatch.setattr(report_module.plt, "close", capture)
    plot_method(tmp_path / "plot.png")
    return captured[0].axes[0]
