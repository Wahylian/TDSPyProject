"""Tests for trainbase/evaluation.py, the metrics suite and naive baseline.

Covers _positive_scores (predict_proba > decision_function > None), evaluate
(the JSON-serializable metrics dict), and baseline_metrics (the most-frequent
floor). Stub models keep the score-source paths deterministic; evaluate is also
checked end-to-end against a real LogisticRegression on the separable fixture.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from trainbase.evaluation import (
    CLASS_NAMES,
    _positive_scores,
    baseline_metrics,
    evaluate,
)

# The headline scalar metrics every evaluation dict must report.
HEADLINE_KEYS = {"accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"}


class _ProbaModel:
    """Stub exposing predict and predict_proba (the preferred score source)."""

    def __init__(self, n: int):
        self._n = n

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        # Column 1 is the positive-class probability.
        p1 = np.linspace(0.1, 0.9, len(X))
        return np.column_stack([1.0 - p1, p1])


class _DecisionModel:
    """Stub exposing predict and decision_function only (the fallback source)."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def decision_function(self, X):
        return np.linspace(-2.0, 2.0, len(X))


class _BareModel:
    """Stub exposing predict only, so ROC-AUC must be skipped."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestPositiveScores:
    """Selecting the per-sample positive-class score for ROC-AUC."""

    def test_prefers_predict_proba_column_one(self):
        """When available, the positive-class column of predict_proba is used."""
        X = np.zeros((5, 2))
        scores = _positive_scores(_ProbaModel(5), X)
        np.testing.assert_allclose(scores, np.linspace(0.1, 0.9, 5))

    def test_falls_back_to_decision_function(self):
        """Without predict_proba the raw decision_function is returned."""
        X = np.zeros((5, 2))
        scores = _positive_scores(_DecisionModel(), X)
        np.testing.assert_allclose(scores, np.linspace(-2.0, 2.0, 5))

    def test_returns_none_when_no_score_source(self):
        """A model with neither method yields None."""
        assert _positive_scores(_BareModel(), np.zeros((3, 2))) is None


class TestEvaluate:
    """The full metrics dict for a fitted classifier."""

    def test_class_names_are_real_then_fake(self):
        """CLASS_NAMES matches the manifest's 0=real, 1=fake convention."""
        assert CLASS_NAMES == ["real", "fake"]

    def test_metrics_dict_is_complete_typed_and_json_serializable(self, feature_split):
        """A fitted model yields a complete, typed, JSON-safe metrics dict."""
        s = feature_split
        model = LogisticRegression(max_iter=1000).fit(s.X_train, s.y_train)

        metrics = evaluate(model, s.X_test, s.y_test, model_label="lr")

        assert HEADLINE_KEYS <= set(metrics)
        for key in HEADLINE_KEYS:
            assert isinstance(metrics[key], float)
        assert metrics["model"] == "lr"
        assert metrics["n_test"] == len(s.y_test)
        assert isinstance(metrics["confusion_matrix"], list)
        assert isinstance(metrics["classification_report"], str)
        # save_artifacts writes this, so it must serialize.
        json.dumps(metrics)

    def test_separable_data_scores_perfectly(self, feature_split):
        """On the separable fixture a linear model reports accuracy/F1/AUC of 1.0."""
        s = feature_split
        model = LogisticRegression(max_iter=1000).fit(s.X_train, s.y_train)
        metrics = evaluate(model, s.X_test, s.y_test)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["pr_auc"] == pytest.approx(1.0)
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_auc_metrics_are_none_without_score_source(self):
        """A predict-only model gives pr_auc/roc_auc of None, not an error."""
        y_test = np.array([0, 1, 0, 1])
        metrics = evaluate(_BareModel(), np.zeros((4, 2)), y_test)
        assert metrics["pr_auc"] is None
        assert metrics["roc_auc"] is None
        assert metrics["n_test"] == 4


class TestBaselineMetrics:
    """The naive most-frequent baseline."""

    def test_majority_baseline_accuracy_equals_majority_fraction(self):
        """The dummy predicts the train-majority class; accuracy = its test fraction."""
        X_train = np.zeros((5, 3), dtype=np.float32)
        y_train = np.array([0, 0, 0, 0, 1])           # majority class 0
        X_test = np.zeros((5, 3), dtype=np.float32)
        y_test = np.array([0, 0, 0, 1, 1])            # 3/5 are class 0

        metrics = baseline_metrics(X_train, y_train, X_test, y_test)

        assert metrics["accuracy"] == pytest.approx(0.6)
        assert metrics["model"] == "baseline (most_frequent)"
