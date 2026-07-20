"""
Tests for ``trainbase/artifacts.py`` — run persistence.

:func:`save_artifacts` writes the three files that make a run reproducible: the
fitted classifier (joblib), the fitted feature pipeline (joblib), and the metrics (JSON). 
These tests pin the filenames, that a missing (nested) output directory
is created, that the metrics round-trip byte-for-value through JSON, and that the
two joblib artifacts reload into equivalent objects.

A tiny fitted ``DummyClassifier`` and a trivial real ``ImagePipeline`` stand in
for the heavy real artifacts so the round-trip stays fast and pickling is real
(no mocks, which don't pickle).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from preprocessing import ImagePipeline
from trainbase.artifacts import save_artifacts


@pytest.fixture
def fitted_artifacts():
    """A fitted dummy model, a real pipeline, and a metrics dict to persist."""
    model = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((4, 2)), np.array([0, 1, 0, 0])
    )
    pipeline = ImagePipeline([("grayscale", {}), ("vectorize", {})])
    metrics = {"model": "demo", "accuracy": 0.75, "confusion_matrix": [[2, 0], [1, 1]]}
    return model, pipeline, metrics


class TestSaveArtifacts:
    """Serializing the model, feature pipeline, and metrics."""

    def test_writes_three_named_artifacts(self, tmp_path, fitted_artifacts):
        """All three artifacts are written with the ``{model_name}_*`` names.

        Args:
            tmp_path: pytest temp output dir (fixture).
            fitted_artifacts: (model, pipeline, metrics) to persist (fixture).
        """
        model, pipeline, metrics = fitted_artifacts
        save_artifacts(model, pipeline, metrics, tmp_path, model_name="svm")

        assert (tmp_path / "svm_model.joblib").is_file()
        assert (tmp_path / "svm_feature_pipeline.joblib").is_file()
        assert (tmp_path / "svm_metrics.json").is_file()

    def test_creates_missing_nested_output_dir(self, tmp_path, fitted_artifacts):
        """A non-existent (nested) ``output_dir`` is created before writing.

        Args:
            tmp_path: pytest temp dir root (fixture).
            fitted_artifacts: (model, pipeline, metrics) to persist (fixture).
        """
        model, pipeline, metrics = fitted_artifacts
        nested = tmp_path / "runs" / "exp1"
        assert not nested.exists()

        save_artifacts(model, pipeline, metrics, nested, model_name="rf")

        assert (nested / "rf_metrics.json").is_file()

    def test_metrics_json_round_trips_to_equal_dict(self, tmp_path, fitted_artifacts):
        """The metrics JSON reloads to a dict equal to the one passed in.

        Args:
            tmp_path: pytest temp output dir (fixture).
            fitted_artifacts: (model, pipeline, metrics) to persist (fixture).
        """
        model, pipeline, metrics = fitted_artifacts
        save_artifacts(model, pipeline, metrics, tmp_path, model_name="lr")

        with (tmp_path / "lr_metrics.json").open(encoding="utf-8") as f:
            reloaded = json.load(f)
        assert reloaded == metrics

    def test_joblib_artifacts_reload_into_equivalent_objects(self, tmp_path, fitted_artifacts):
        """The model and pipeline reload into working, equivalent objects.

        The reloaded pipeline keeps its operation list, and the reloaded model
        still predicts the most-frequent class (0) it was fit on.

        Args:
            tmp_path: pytest temp output dir (fixture).
            fitted_artifacts: (model, pipeline, metrics) to persist (fixture).
        """
        model, pipeline, metrics = fitted_artifacts
        save_artifacts(model, pipeline, metrics, tmp_path, model_name="svm")

        reloaded_pipeline = joblib.load(tmp_path / "svm_feature_pipeline.joblib")
        reloaded_model = joblib.load(tmp_path / "svm_model.joblib")

        assert reloaded_pipeline.operations == pipeline.operations
        # Majority class of the fit labels was 0; the reloaded model still says 0.
        assert list(reloaded_model.predict(np.zeros((3, 2)))) == [0, 0, 0]
