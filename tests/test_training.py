"""Tests for trainbase/training.py, model assembly and validation-driven tuning.

Covers build_estimator (the one-step 'clf' Pipeline) and tune_hyperparameters
(grid search over an explicit validation holdout via PredefinedSplit, then refit).
Runs on the tiny separable feature_split with capped grids, so a real but instant
GridSearchCV exercises the actual selection logic.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from trainbase.training import build_estimator, n_jobs_for, tune_hyperparameters


class TestBuildEstimator:
    """Assembling the single-step classifier pipeline."""

    @pytest.mark.parametrize("model_name", ["svm", "rf", "logreg"])
    def test_returns_single_step_clf_pipeline(self, model_name):
        """Each registered model becomes a one-step Pipeline named 'clf'."""
        estimator = build_estimator(model_name)
        assert isinstance(estimator, Pipeline)
        assert [name for name, _ in estimator.steps] == ["clf"]

    def test_clf_step_is_the_registered_estimator_type(self):
        """build_estimator('logreg') wraps a LogisticRegression."""
        estimator = build_estimator("logreg")
        assert isinstance(estimator.named_steps["clf"], LogisticRegression)

    def test_unknown_model_raises_key_error(self):
        """An unregistered model name surfaces as KeyError."""
        with pytest.raises(KeyError):
            build_estimator("nope")


class TestTuneHyperparameters:
    """Validation-holdout grid search and refit."""

    def test_returns_fitted_search_whose_best_estimator_predicts(self, feature_split):
        """A small grid yields a fitted GridSearchCV whose winner predicts."""
        s = feature_split
        estimator = build_estimator("logreg")
        grid = {"clf__C": [0.1, 10.0]}

        search = tune_hyperparameters(
            estimator, grid, s.X_train, s.y_train, s.X_val, s.y_val
        )

        assert isinstance(search, GridSearchCV)
        assert search.best_params_["clf__C"] in (0.1, 10.0)
        preds = search.best_estimator_.predict(s.X_test)
        assert preds.shape == (len(s.y_test),)

    def test_selection_uses_only_the_validation_fold(self, feature_split):
        """Candidates are scored on val alone, so exactly one CV split is searched."""
        s = feature_split
        search = tune_hyperparameters(
            build_estimator("logreg"),
            {"clf__C": [1.0]},
            s.X_train, s.y_train, s.X_val, s.y_val,
        )
        results = search.cv_results_
        assert "split0_test_score" in results
        assert "split1_test_score" not in results

    def test_empty_grid_fits_single_default_candidate(self, feature_split):
        """An empty grid means no tuning: one default candidate is still fit."""
        s = feature_split
        search = tune_hyperparameters(
            build_estimator("logreg"), {}, s.X_train, s.y_train, s.X_val, s.y_val
        )
        assert search.best_params_ == {}
        preds = search.best_estimator_.predict(s.X_test)
        assert len(preds) == len(s.y_test)

    def test_separable_data_tunes_to_perfect_validation_score(self, feature_split):
        """On cleanly separable data the best validation F1 reaches 1.0."""
        s = feature_split
        search = tune_hyperparameters(
            build_estimator("logreg"),
            {"clf__C": [0.1, 1.0, 10.0]},
            s.X_train, s.y_train, s.X_val, s.y_val,
        )
        assert search.best_score_ == pytest.approx(1.0)


class TestNJobsFor:
    """Sequential n_jobs for GPU-bound torch models; parallel for classical ones.

    A shared GPU means parallel candidates would contend for one device instead
    of speeding anything up, so torch image models get n_jobs=1.
    """

    def test_classical_model_gets_parallel_n_jobs(self):
        assert n_jobs_for(build_estimator("logreg")) == -1

    def test_torch_model_gets_sequential_n_jobs(self):
        pytest.importorskip("torch")
        assert n_jobs_for(build_estimator("cnn")) == 1

    def test_tune_hyperparameters_passes_sequential_n_jobs_for_torch(self, pixel_split):
        pytest.importorskip("torch")
        s = pixel_split
        estimator = build_estimator("cnn")
        estimator.set_params(clf__epochs=1)
        search = tune_hyperparameters(
            estimator, {"clf__lr": [1e-3]}, s.X_train, s.y_train, s.X_val, s.y_val
        )
        assert search.n_jobs == 1
