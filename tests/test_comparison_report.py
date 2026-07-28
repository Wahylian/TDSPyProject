"""Tests for comparison/report.py: rendering a table to Markdown/HTML/CSV/plot."""

from __future__ import annotations

import pandas as pd

from comparison import report as report_module
from comparison.report import Reporter


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
