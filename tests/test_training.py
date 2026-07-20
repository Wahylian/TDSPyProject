"""
Tests for ``trainbase/training.py`` — model assembly and validation-driven tuning.

Two units:

* :func:`build_estimator` — wraps the chosen registry classifier in a one-step
  ``Pipeline`` under the name ``"clf"`` (so the ``clf__`` grids line up).
* :func:`tune_hyperparameters` — grid-searches that estimator selecting on an
  explicit *validation* holdout (via ``PredefinedSplit``), then refits the
  winner on train+val.

Everything runs on the tiny separable ``feature_split`` fixture with grids capped
to 1-2 candidates, so a real (but instant) ``GridSearchCV`` exercises the actual
selection logic rather than a mock of it.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from trainbase.training import build_estimator, tune_hyperparameters


class TestBuildEstimator:
    """Assembling the single-step classifier pipeline."""

    @pytest.mark.parametrize("model_name", ["svm", "rf", "logreg"])
    def test_returns_single_step_clf_pipeline(self, model_name):
        """Each registered model becomes a one-step ``Pipeline`` named ``"clf"``.

        The lone step must be named ``"clf"`` (so ``clf__`` grids resolve) and
        wrap a fresh, unfitted estimator.

        Args:
            model_name: A registered model key from the sweep.
        """
        estimator = build_estimator(model_name)
        assert isinstance(estimator, Pipeline)
        assert [name for name, _ in estimator.steps] == ["clf"]

    def test_clf_step_is_the_registered_estimator_type(self):
        """``build_estimator("logreg")`` wraps a ``LogisticRegression``."""
        estimator = build_estimator("logreg")
        assert isinstance(estimator.named_steps["clf"], LogisticRegression)

    def test_unknown_model_raises_key_error(self):
        """An unregistered model name surfaces as ``KeyError``."""
        with pytest.raises(KeyError):
            build_estimator("nope")


class TestTuneHyperparameters:
    """Validation-holdout grid search and refit."""

    def test_returns_fitted_search_whose_best_estimator_predicts(self, feature_split):
        """A small grid yields a fitted ``GridSearchCV`` with a usable winner.

        On the separable fixture the search must complete, expose
        ``best_params_`` drawn from the grid, and its refit ``best_estimator_``
        must predict one label per test row.

        Args:
            feature_split: small separable train/val/test split (fixture).
        """
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
        """Candidates are scored on val alone, so exactly one CV split is searched.

        ``PredefinedSplit`` marks train rows ``-1`` (never validated) and val
        rows ``0`` (the single fold). The per-candidate CV results must therefore
        record exactly one split (``split0_test_score`` present, no
        ``split1_test_score``), proving a true holdout rather than k-fold over a
        mixed pool.

        Args:
            feature_split: small separable train/val/test split (fixture).
        """
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
        """An empty grid means "no tuning": one default candidate is still fit.

        The function must not error on an empty grid — it should run the single
        default configuration and return a refit ``best_estimator_``.

        Args:
            feature_split: small separable train/val/test split (fixture).
        """
        s = feature_split
        search = tune_hyperparameters(
            build_estimator("logreg"), {}, s.X_train, s.y_train, s.X_val, s.y_val
        )
        assert search.best_params_ == {}
        preds = search.best_estimator_.predict(s.X_test)
        assert len(preds) == len(s.y_test)

    def test_separable_data_tunes_to_perfect_validation_score(self, feature_split):
        """On cleanly separable data the best validation F1 reaches 1.0.

        A sanity check that the val fold is genuinely driving selection: the
        winning config separates the held-out blobs perfectly.

        Args:
            feature_split: small separable train/val/test split (fixture).
        """
        s = feature_split
        search = tune_hyperparameters(
            build_estimator("logreg"),
            {"clf__C": [0.1, 1.0, 10.0]},
            s.X_train, s.y_train, s.X_val, s.y_val,
        )
        assert search.best_score_ == pytest.approx(1.0)
