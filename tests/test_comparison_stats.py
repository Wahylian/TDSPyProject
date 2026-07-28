"""Tests for comparison/stats.py: ranking, paired tests, and lift-delta fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from comparison.records import RunRecord
from comparison.stats import StatisticalComparison


def _record(model, run_id, f1, baseline_f1=0.5, per_sample_scores=None):
    """A minimal RunRecord for stats logic, independent of any real training run."""
    diagnostics = {}
    if per_sample_scores is not None:
        diagnostics["per_sample_scores"] = per_sample_scores
    return RunRecord(
        model_name=model, run_id=run_id, timestamp="2026-01-01T00:00:00",
        pipeline_used="svm", pipeline_spec=None, pipeline_steps=(),
        scoring="f1", sample_sizes={"train": 1, "val": 1, "test": 1},
        hyperparameters={}, best_val_score=f1,
        evaluation_metrics={"f1": f1, "precision": f1, "recall": f1},
        baseline_metrics={"f1": baseline_f1, "precision": baseline_f1, "recall": baseline_f1},
        diagnostics=diagnostics, source_path=Path("x"),
    )


class TestRank:
    """Sorting records by a metric."""

    def test_sorts_descending_and_drops_missing_metric(self):
        records = [
            _record("svm", "a", 0.7),
            _record("logreg", "b", 0.9),
            RunRecord(
                model_name="rf", run_id="c", timestamp="t", pipeline_used="svm",
                pipeline_spec=None, pipeline_steps=(), scoring="f1",
                sample_sizes={}, hyperparameters={}, best_val_score=0.5,
                evaluation_metrics={}, baseline_metrics={}, diagnostics={},
                source_path=Path("x"),
            ),
        ]
        ranked = StatisticalComparison(records).rank(metric="f1")

        assert [r.run_id for r in ranked] == ["b", "a"]


class TestCompare:
    """Pairwise comparison between two runs by run_id."""

    def test_falls_back_to_lift_delta_without_per_sample_scores(self):
        records = [_record("svm", "a", 0.7, baseline_f1=0.5), _record("logreg", "b", 0.9, baseline_f1=0.5)]
        result = StatisticalComparison(records).compare("a", "b", metric="f1")

        assert result["method"] == "lift_delta"
        assert result["lift_delta"] == pytest.approx((0.7 - 0.5) - (0.9 - 0.5))

    def test_runs_paired_test_when_per_sample_scores_present(self):
        records = [
            _record("svm", "a", 0.7, per_sample_scores=[1, 0, 1, 1, 0, 1, 1, 0]),
            _record("logreg", "b", 0.9, per_sample_scores=[1, 1, 1, 0, 0, 1, 0, 1]),
        ]
        result = StatisticalComparison(records).compare("a", "b", metric="f1")

        assert result["method"] == "wilcoxon_paired"
        assert "p_value" in result

    def test_unknown_run_id_raises(self):
        records = [_record("svm", "a", 0.7)]
        with pytest.raises(KeyError):
            StatisticalComparison(records).compare("a", "missing")
