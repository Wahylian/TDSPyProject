"""Tests for comparison/loader.py: discovering runs under an artifacts root."""

from __future__ import annotations

import json

from comparison.loader import RunLoader


def _write_metadata(run_dir, model_name, pipeline_used, **overrides):
    """Write a minimal, schema-shaped metadata.json into <run_dir>/metadata.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "run_id": run_dir.name,
        "timestamp": "2026-07-22T11:57:39",
        "pipeline_used": pipeline_used,
        "pipeline_spec": None,
        "pipeline_steps": [["grayscale", {}]],
        "scoring": "f1",
        "sample_sizes": {"train": 10, "val": 5, "test": 5},
        "hyperparameters": {},
        "best_val_score": 0.7,
        "evaluation_metrics": {"accuracy": 0.7, "precision": 0.6, "recall": 0.6},
        "baseline_metrics": {"accuracy": 0.5, "precision": 0.0, "recall": 0.0},
        "diagnostics": {},
    }
    metadata.update(overrides)
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


class TestLoad:
    """Globbing and parsing every metadata.json under the artifacts root."""

    def test_loads_valid_runs(self, tmp_path):
        """Real registry names (svm model, svm pipeline) round-trip into records."""
        _write_metadata(tmp_path / "svm" / "run_a", "svm", "svm")
        _write_metadata(tmp_path / "logreg" / "run_b", "logreg", "svm")

        records = RunLoader(tmp_path).load()

        assert {r.model_name for r in records} == {"svm", "logreg"}

    def test_skips_unknown_model(self, tmp_path, caplog):
        """A model no longer in MODEL_REGISTRY is skipped with a warning, not raised."""
        _write_metadata(tmp_path / "xgboost" / "run_a", "xgboost", "svm")

        with caplog.at_level("WARNING"):
            records = RunLoader(tmp_path).load()

        assert records == []
        assert "xgboost" in caplog.text

    def test_skips_unknown_pipeline(self, tmp_path, caplog):
        """A pipeline no longer in PIPELINE_REGISTRY is skipped with a warning."""
        _write_metadata(tmp_path / "svm" / "run_a", "svm", "nonexistent_pipeline")

        with caplog.at_level("WARNING"):
            records = RunLoader(tmp_path).load()

        assert records == []

    def test_custom_pipeline_used_is_not_skipped(self, tmp_path):
        """pipeline_used='custom' (from --pipeline-spec) bypasses PIPELINE_REGISTRY lookup."""
        _write_metadata(tmp_path / "svm" / "run_a", "svm", "custom")

        records = RunLoader(tmp_path).load()

        assert len(records) == 1

    def test_skips_malformed_json(self, tmp_path, caplog):
        """Invalid JSON is skipped with a warning rather than raising."""
        run_dir = tmp_path / "svm" / "run_bad"
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text("{not valid json", encoding="utf-8")

        with caplog.at_level("WARNING"):
            records = RunLoader(tmp_path).load()

        assert records == []

    def test_skips_missing_required_key(self, tmp_path, caplog):
        """A metadata.json missing a required key is skipped with a warning."""
        run_dir = tmp_path / "svm" / "run_bad"
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(json.dumps({"model_name": "svm"}), encoding="utf-8")

        with caplog.at_level("WARNING"):
            records = RunLoader(tmp_path).load()

        assert records == []

    def test_filters_by_model(self, tmp_path):
        """The model filter narrows the loaded set."""
        _write_metadata(tmp_path / "svm" / "run_a", "svm", "svm")
        _write_metadata(tmp_path / "logreg" / "run_b", "logreg", "svm")

        records = RunLoader(tmp_path).load(model="logreg")

        assert [r.model_name for r in records] == ["logreg"]

    def test_filters_by_pipeline(self, tmp_path):
        """The pipeline filter narrows the loaded set."""
        _write_metadata(tmp_path / "svm" / "run_a", "svm", "svm")
        _write_metadata(tmp_path / "svm" / "run_b", "svm", "fast")

        records = RunLoader(tmp_path).load(pipeline="fast")

        assert [r.pipeline_used for r in records] == ["fast"]

    def test_filters_by_required_metric(self, tmp_path):
        """require_metric drops runs where that metric is missing or None."""
        _write_metadata(tmp_path / "svm" / "run_a", "svm", "svm", evaluation_metrics={"accuracy": 0.7})
        _write_metadata(
            tmp_path / "svm" / "run_b", "svm", "svm",
            evaluation_metrics={"accuracy": 0.7, "roc_auc": 0.8},
        )

        records = RunLoader(tmp_path).load(require_metric="roc_auc")

        assert [r.run_id for r in records] == ["run_b"]

    def test_empty_root_returns_empty_list(self, tmp_path):
        """A root with no metadata.json anywhere yields an empty list, no error."""
        records = RunLoader(tmp_path / "does_not_exist").load()

        assert records == []
