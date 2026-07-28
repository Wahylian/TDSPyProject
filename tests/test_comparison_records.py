"""Tests for comparison/records.py: parsing metadata.json into RunRecord."""

from __future__ import annotations

import json

import pytest

from comparison.records import RunRecord


def _write_metadata(path, **overrides):
    """Write a minimal, schema-shaped metadata.json, overridable per test."""
    metadata = {
        "model_name": "svm",
        "run_id": "20260722_115739",
        "timestamp": "2026-07-22T11:57:39",
        "pipeline_used": "svm",
        "pipeline_spec": None,
        "pipeline_steps": [
            ["grayscale", {}],
            ["reduce", {"method": "vec-pca", "n_components": 150}],
            ["scale", {}],
        ],
        "scoring": "f1",
        "sample_sizes": {"train": 100, "val": 50, "test": 50},
        "hyperparameters": {"clf__C": 1.0},
        "best_val_score": 0.75,
        "evaluation_metrics": {
            "accuracy": 0.75, "precision": 0.73, "recall": 0.72,
            "pr_auc": 0.80, "roc_auc": 0.83,
        },
        "baseline_metrics": {
            "accuracy": 0.54, "precision": 0.0, "recall": 0.0,
            "pr_auc": 0.46, "roc_auc": 0.5,
        },
        "diagnostics": {"confusion_matrix": [[2, 1], [1, 2]]},
    }
    metadata.update(overrides)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


class TestFromMetadata:
    """Parsing a metadata.json file into a RunRecord."""

    def test_parses_all_fields(self, tmp_path):
        """Every top-level field round-trips into the record."""
        path = _write_metadata(tmp_path / "metadata.json")
        record = RunRecord.from_metadata(path)

        assert record.model_name == "svm"
        assert record.run_id == "20260722_115739"
        assert record.pipeline_used == "svm"
        assert record.sample_sizes == {"train": 100, "val": 50, "test": 50}
        assert record.hyperparameters == {"clf__C": 1.0}
        assert record.best_val_score == 0.75
        assert record.source_path == path

    def test_reconstructs_f1_from_precision_recall(self, tmp_path):
        """f1 is absent from the raw JSON but present on the record, harmonic mean."""
        path = _write_metadata(tmp_path / "metadata.json")
        record = RunRecord.from_metadata(path)

        assert "f1" not in json.loads(path.read_text())["evaluation_metrics"]
        expected = 2 * 0.73 * 0.72 / (0.73 + 0.72)
        assert record.metric("f1") == pytest.approx(expected)

    def test_baseline_f1_is_zero_when_precision_and_recall_are_zero(self, tmp_path):
        """A baseline that never predicts positive gets f1=0, not a division error."""
        path = _write_metadata(tmp_path / "metadata.json")
        record = RunRecord.from_metadata(path)

        assert record.baseline("f1") == 0.0

    def test_has_reduce_step_true_for_classical_pipeline(self, tmp_path):
        """A pipeline with a 'reduce' op is detected as such."""
        path = _write_metadata(tmp_path / "metadata.json")
        record = RunRecord.from_metadata(path)

        assert record.has_reduce_step() is True

    def test_has_reduce_step_false_for_raw_pixel_pipeline(self, tmp_path):
        """A pipeline with no 'reduce' op (raw pixels) is detected as such."""
        path = _write_metadata(
            tmp_path / "metadata.json",
            pipeline_used="pixels",
            pipeline_steps=[["grayscale", {}], ["vectorize", {}]],
        )
        record = RunRecord.from_metadata(path)

        assert record.has_reduce_step() is False

    def test_lift_is_metric_minus_baseline(self, tmp_path):
        """lift() subtracts the baseline value from the test-set metric."""
        path = _write_metadata(tmp_path / "metadata.json")
        record = RunRecord.from_metadata(path)

        assert record.lift("accuracy") == pytest.approx(0.75 - 0.54)

    def test_lift_is_none_when_metric_missing(self, tmp_path):
        """lift() returns None rather than raising when a metric key is absent."""
        path = _write_metadata(
            tmp_path / "metadata.json",
            evaluation_metrics={"accuracy": 0.75},
            baseline_metrics={"accuracy": 0.54},
        )
        record = RunRecord.from_metadata(path)

        assert record.lift("roc_auc") is None
