"""
Tests for ``trainbase/diagnostics.py`` — algorithm-appropriate run diagnostics.

:func:`collect_diagnostics` gathers cheap, model-specific diagnostics from a
fitted ``GridSearchCV`` and its best estimator: tree-based feature importances
and the OOB score (present only for tree models), the hyperparameter-grid
validation scores (always), and an opt-in learning curve. These tests fit real
(tiny) searches on the separable fixture and pin which keys are populated for a
tree vs. a linear model, and that every result is JSON-serializable.
"""

from __future__ import annotations

import json

from trainbase.diagnostics import collect_diagnostics
from trainbase.model_registry import MODEL_REGISTRY
from trainbase.training import build_estimator, tune_hyperparameters


def _fit_search(model_name, s):
    """Fit the registry model's GridSearchCV on the tiny feature split."""
    estimator = build_estimator(model_name)
    grid = MODEL_REGISTRY[model_name].param_grid
    return tune_hyperparameters(
        estimator, grid, s.X_train, s.y_train, s.X_val, s.y_val, scoring="f1"
    )


class TestCollectDiagnostics:
    """Which diagnostics get populated, per model family."""

    def test_random_forest_exposes_importances_and_oob(self, feature_split):
        """A tree model reports feature importances (one per feature) and an OOB score."""
        s = feature_split
        search = _fit_search("rf", s)
        diag = collect_diagnostics(search, search.best_estimator_, s.X_train, s.y_train)

        assert diag["feature_importances"] is not None
        assert len(diag["feature_importances"]) == s.X_train.shape[1]
        assert diag["oob_score"] is not None
        assert diag["learning_curve"] is None  # not requested
        assert diag["hyperparameter_scores"]   # always populated, non-empty
        json.dumps(diag)

    def test_linear_model_has_null_tree_diagnostics(self, feature_split):
        """A non-tree model leaves the tree-only diagnostics as ``None`` (uniform schema)."""
        s = feature_split
        search = _fit_search("logreg", s)
        diag = collect_diagnostics(search, search.best_estimator_, s.X_train, s.y_train)

        assert diag["feature_importances"] is None
        assert diag["oob_score"] is None
        assert set(diag) == {
            "feature_importances", "oob_score", "hyperparameter_scores", "learning_curve",
        }
        json.dumps(diag)

    def test_hyperparameter_scores_shape(self, feature_split):
        """Each grid entry carries its params and mean/std validation score."""
        s = feature_split
        search = _fit_search("logreg", s)
        diag = collect_diagnostics(search, search.best_estimator_, s.X_train, s.y_train)

        entry = diag["hyperparameter_scores"][0]
        assert set(entry) == {"params", "mean_val_score", "std_val_score"}
        assert isinstance(entry["mean_val_score"], float)

    def test_include_curves_adds_learning_curve(self, feature_split):
        """With ``include_curves`` the learning curve is a well-formed dict (or None, best-effort)."""
        s = feature_split
        search = _fit_search("logreg", s)
        diag = collect_diagnostics(
            search, search.best_estimator_, s.X_train, s.y_train, include_curves=True
        )

        lc = diag["learning_curve"]
        # Best-effort: on this deliberately tiny split it may be None; if present
        # it must be well-formed and aligned.
        if lc is not None:
            assert set(lc) == {"train_sizes", "train_scores_mean", "val_scores_mean"}
            assert len(lc["train_sizes"]) == len(lc["train_scores_mean"]) == len(lc["val_scores_mean"])
        json.dumps(diag)
