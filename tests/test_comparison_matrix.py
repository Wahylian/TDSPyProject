"""Tests for comparison/matrix.py: leaderboard, resilience, and grid tables."""

from __future__ import annotations

from pathlib import Path

import pytest

from comparison.matrix import ComparisonMatrix
from comparison.records import RunRecord


def _record(model, pipeline, f1, baseline_f1=0.5, run_id="r1"):
    """A minimal RunRecord for matrix logic, independent of any real training run."""
    return RunRecord(
        model_name=model, run_id=run_id, timestamp="2026-01-01T00:00:00",
        pipeline_used=pipeline, pipeline_spec=None, pipeline_steps=(),
        scoring="f1", sample_sizes={"train": 1, "val": 1, "test": 1},
        hyperparameters={}, best_val_score=f1,
        evaluation_metrics={"f1": f1, "precision": f1, "recall": f1},
        baseline_metrics={"f1": baseline_f1, "precision": baseline_f1, "recall": baseline_f1},
        diagnostics={}, source_path=Path("x"),
    )


class TestLeaderboard:
    """N×1: ranking models by a metric."""

    def test_ranks_by_metric_descending(self):
        records = [_record("svm", "svm", 0.7), _record("logreg", "svm", 0.9)]
        board = ComparisonMatrix(records).leaderboard(metric="f1")

        assert list(board["model_name"]) == ["logreg", "svm"]
        assert board.loc[0, "lift_f1"] == pytest.approx(0.9 - 0.5)

    def test_pipeline_filter_narrows_records(self):
        records = [_record("svm", "svm", 0.7), _record("svm", "fast", 0.9)]
        board = ComparisonMatrix(records).leaderboard(metric="f1", pipeline="fast")

        assert list(board["f1"]) == [0.9]

    def test_best_run_kept_when_model_has_multiple_runs(self):
        records = [_record("svm", "svm", 0.6, run_id="a"), _record("svm", "svm", 0.8, run_id="b")]
        board = ComparisonMatrix(records).leaderboard(metric="f1")

        assert board.loc[0, "run_id"] == "b"

    def test_empty_records_yields_empty_frame(self):
        board = ComparisonMatrix([]).leaderboard(metric="f1")
        assert board.empty


class TestResilience:
    """1×N: one model's score across the pipelines it was run on."""

    def test_sweeps_pipelines_for_one_model(self):
        records = [
            _record("svm", "svm", 0.7),
            _record("svm", "fast", 0.6),
            _record("logreg", "svm", 0.95),  # different model, excluded
        ]
        board = ComparisonMatrix(records).resilience("svm", metric="f1")

        assert set(board["pipeline_used"]) == {"svm", "fast"}
        assert list(board["f1"]) == [0.7, 0.6]


class TestGrid:
    """N×M: model x pipeline pivot, classical/torch blocks kept separate."""

    def test_pivots_classical_models_and_pipelines(self):
        records = [
            _record("svm", "svm", 0.7),
            _record("svm", "fast", 0.6),
            _record("logreg", "svm", 0.9),
        ]
        grid = ComparisonMatrix(records).grid(metric="f1")

        assert grid.loc["svm", "svm"] == 0.7
        assert grid.loc["svm", "fast"] == 0.6
        assert grid.loc["logreg", "svm"] == 0.9

    def test_custom_pipeline_runs_excluded_from_grid(self):
        records = [_record("svm", "custom", 0.99)]
        grid = ComparisonMatrix(records).grid(metric="f1")
        assert grid.empty

    def test_empty_records_yields_empty_frame(self):
        grid = ComparisonMatrix([]).grid(metric="f1")
        assert grid.empty

    def test_pretrained_torch_model_paired_with_its_pixel_pipeline(self):
        """cnn_pretrained/vit_pretrained land in the torch block alongside
        pixels_pretrained, not dropped as a classical/torch family mismatch."""
        from trainbase import MODEL_REGISTRY
        if "cnn_pretrained" not in MODEL_REGISTRY:
            pytest.skip("torchvision not installed")

        records = [_record("cnn_pretrained", "pixels_pretrained", 0.8)]
        grid = ComparisonMatrix(records).grid(metric="f1")

        assert grid.loc["cnn_pretrained", "pixels_pretrained"] == 0.8
