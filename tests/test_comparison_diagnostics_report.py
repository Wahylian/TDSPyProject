"""Tests for comparison/diagnostics_report.py: per-run diagnostics rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from comparison import diagnostics_report as diag_module
from comparison.diagnostics_report import diagnostics_markdown, plot_confusion_matrix
from comparison.records import RunRecord


def _record(diagnostics):
    return RunRecord(
        model_name="rf", run_id="r1", timestamp="2026-01-01T00:00:00",
        pipeline_used="svm", pipeline_spec=None, pipeline_steps=(),
        scoring="f1", sample_sizes={}, hyperparameters={}, best_val_score=0.7,
        evaluation_metrics={"f1": 0.7}, baseline_metrics={"f1": 0.5},
        diagnostics=diagnostics, source_path=Path("x"),
    )


class TestDiagnosticsMarkdown:
    """Rendering the available diagnostics fields as Markdown."""

    def test_renders_confusion_matrix_and_report(self):
        record = _record({
            "confusion_matrix": [[2, 1], [0, 3]],
            "classification_report": "precision recall f1-score",
        })
        text = diagnostics_markdown(record)

        assert "Confusion matrix" in text
        assert "[[2, 1], [0, 3]]" in text
        assert "precision recall f1-score" in text

    def test_renders_feature_importances_and_oob(self):
        record = _record({"feature_importances": [0.1, 0.9, 0.3], "oob_score": 0.82})
        text = diagnostics_markdown(record)

        assert "feature[1]: 0.9000" in text
        assert "OOB score:** 0.8200" in text

    def test_renders_hyperparameter_scores_and_learning_curve(self):
        record = _record({
            "hyperparameter_scores": [
                {"params": {"clf__C": 1.0}, "mean_val_score": 0.8, "std_val_score": 0.01},
            ],
            "learning_curve": {"train_sizes": [10, 20], "val_scores_mean": [0.6, 0.7]},
        })
        text = diagnostics_markdown(record)

        assert "clf__C" in text
        assert "- 10: 0.6000" in text

    def test_empty_diagnostics_says_so(self):
        text = diagnostics_markdown(_record({}))
        assert "No diagnostics recorded" in text


class TestPlotConfusionMatrix:
    """Optional confusion-matrix heatmap, skipped silently when unavailable."""

    def test_returns_none_without_confusion_matrix(self, tmp_path):
        result = plot_confusion_matrix(_record({}), tmp_path / "cm.png")
        assert result is None

    def test_saves_png_when_matplotlib_present(self, tmp_path):
        record = _record({"confusion_matrix": [[2, 1], [0, 3]]})
        path = tmp_path / "cm.png"
        result = plot_confusion_matrix(record, path)

        if diag_module.plt is None:
            assert result is None
        else:
            assert result == path
            assert path.is_file()

    def test_returns_none_when_matplotlib_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(diag_module, "plt", None)
        record = _record({"confusion_matrix": [[2, 1], [0, 3]]})
        result = plot_confusion_matrix(record, tmp_path / "cm.png")

        assert result is None

    @pytest.mark.skipif(diag_module.plt is None, reason="matplotlib is not installed")
    def test_ticks_are_class_names_not_continuous_positions(self, tmp_path, monkeypatch):
        record = _record({"confusion_matrix": [[2, 1], [0, 3]]})
        captured = []
        real_close = diag_module.plt.close
        monkeypatch.setattr(
            diag_module.plt, "close",
            lambda figure: (captured.append(figure), real_close(figure)),
        )
        plot_confusion_matrix(record, tmp_path / "cm.png")
        ax = captured[0].axes[0]

        assert [t.get_text() for t in ax.get_xticklabels()] == ["real", "fake"]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["real", "fake"]
