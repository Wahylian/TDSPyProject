"""Tests for trainbase/artifacts.py, run persistence and metadata assembly.

save_artifacts writes a run bundle (model.joblib, feature_pipeline.joblib,
metadata.json) into <base>/<model>/<run_id>/; these pin the layout, dir creation,
rerun isolation, JSON round-trip, and joblib reload. build_metadata's schema is
pinned separately. A DummyClassifier and a trivial ImagePipeline stand in so
pickling is real (mocks don't pickle).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from preprocessing import ImagePipeline
from trainbase.artifacts import HEADLINE_METRICS, build_metadata, save_artifacts


@pytest.fixture
def fitted_artifacts():
    """A fitted dummy model, a real pipeline, and a metadata dict to persist."""
    model = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((4, 2)), np.array([0, 1, 0, 0])
    )
    pipeline = ImagePipeline([("grayscale", {}), ("vectorize", {})])
    metadata = {
        "model_name": "demo",
        "run_id": "20260720_120000",
        "evaluation_metrics": {"accuracy": 0.75},
        "diagnostics": {"confusion_matrix": [[2, 0], [1, 1]]},
    }
    return model, pipeline, metadata


class TestSaveArtifacts:
    """Serializing the model, feature pipeline, and metadata into a run dir."""

    def test_writes_run_bundle(self, tmp_path, fitted_artifacts):
        """The three artifacts land in <base>/<model>/<run_id>/, which is returned."""
        model, pipeline, metadata = fitted_artifacts
        run_dir = save_artifacts(
            model, pipeline, metadata, tmp_path, model_name="svm", run_id="20260720_120000"
        )

        assert run_dir == tmp_path / "svm" / "20260720_120000"
        assert (run_dir / "model.joblib").is_file()
        assert (run_dir / "feature_pipeline.joblib").is_file()
        assert (run_dir / "metadata.json").is_file()

    def test_creates_missing_parent_dirs(self, tmp_path, fitted_artifacts):
        """A non-existent base directory and the model/run subdirs are created."""
        model, pipeline, metadata = fitted_artifacts
        base = tmp_path / "artifacts"
        assert not base.exists()

        run_dir = save_artifacts(
            model, pipeline, metadata, base, model_name="rf", run_id="20260720_130000"
        )

        assert (run_dir / "metadata.json").is_file()

    def test_reruns_are_isolated(self, tmp_path, fitted_artifacts):
        """Two runs of the same model write to distinct run dirs, no overwrite."""
        model, pipeline, metadata = fitted_artifacts
        first = save_artifacts(model, pipeline, metadata, tmp_path, "svm", "20260720_120000")
        second = save_artifacts(model, pipeline, metadata, tmp_path, "svm", "20260720_120001")

        assert first != second
        assert (first / "metadata.json").is_file()
        assert (second / "metadata.json").is_file()

    def test_metadata_json_round_trips_to_equal_dict(self, tmp_path, fitted_artifacts):
        """The metadata JSON reloads to a dict equal to the one passed in."""
        model, pipeline, metadata = fitted_artifacts
        run_dir = save_artifacts(model, pipeline, metadata, tmp_path, "lr", "20260720_120000")

        with (run_dir / "metadata.json").open(encoding="utf-8") as f:
            reloaded = json.load(f)
        assert reloaded == metadata

    def test_joblib_artifacts_reload_into_equivalent_objects(self, tmp_path, fitted_artifacts):
        """The model and pipeline reload into working, equivalent objects."""
        model, pipeline, metadata = fitted_artifacts
        run_dir = save_artifacts(model, pipeline, metadata, tmp_path, "svm", "20260720_120000")

        reloaded_pipeline = joblib.load(run_dir / "feature_pipeline.joblib")
        reloaded_model = joblib.load(run_dir / "model.joblib")

        assert reloaded_pipeline.operations == pipeline.operations
        # Majority class of the fit labels was 0.
        assert list(reloaded_model.predict(np.zeros((3, 2)))) == [0, 0, 0]


class TestBuildMetadata:
    """Assembling the uniform metadata record."""

    def _call(self):
        test_metrics = {
            "accuracy": 0.9, "precision": 0.88, "recall": 0.91, "f1": 0.89,
            "pr_auc": 0.93, "roc_auc": 0.95,
            "confusion_matrix": [[45, 5], [4, 46]],
            "classification_report": "report-text",
        }
        baseline = {"accuracy": 0.54, "precision": 0.0, "recall": 0.0, "pr_auc": 0.46, "roc_auc": 0.5}
        diagnostics = {
            "feature_importances": None, "oob_score": None,
            "hyperparameter_scores": [{"params": {"clf__C": 1.0}, "mean_val_score": 0.8, "std_val_score": 0.0}],
            "learning_curve": None,
        }
        return build_metadata(
            model_name="svm", run_id="20260720_120000", timestamp="2026-07-20T12:00:00",
            pipeline_used="svm", pipeline_spec=None, pipeline_steps=[["grayscale", {}]],
            scoring="f1", sample_sizes={"train": 100, "val": 50, "test": 50},
            hyperparameters={"clf__C": 1.0}, best_val_score=0.8,
            test_metrics=test_metrics, baseline_metrics=baseline, diagnostics=diagnostics,
        )

    def test_headline_metrics_extracted(self):
        """evaluation_metrics holds exactly the five standardized headline keys."""
        meta = self._call()
        assert set(meta["evaluation_metrics"]) == set(HEADLINE_METRICS)
        assert set(meta["baseline_metrics"]) == set(HEADLINE_METRICS)
        assert "f1" not in meta["evaluation_metrics"]
        assert meta["evaluation_metrics"]["pr_auc"] == 0.93

    def test_confusion_matrix_folded_into_diagnostics(self):
        """The confusion matrix and report move under diagnostics."""
        meta = self._call()
        assert meta["diagnostics"]["confusion_matrix"] == [[45, 5], [4, 46]]
        assert meta["diagnostics"]["classification_report"] == "report-text"
        assert "hyperparameter_scores" in meta["diagnostics"]

    def test_top_level_schema_is_uniform(self):
        """Every expected top-level key is present."""
        meta = self._call()
        expected = {
            "model_name", "run_id", "timestamp", "pipeline_used", "pipeline_spec",
            "pipeline_steps", "scoring", "sample_sizes", "hyperparameters",
            "best_val_score", "evaluation_metrics", "baseline_metrics", "diagnostics",
        }
        assert set(meta) == expected

    def test_metadata_is_json_serializable(self):
        """The assembled record serializes cleanly."""
        json.dumps(self._call())
