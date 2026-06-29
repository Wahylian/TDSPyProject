"""
Tests for ``trainbase/evaluation.py`` — the metrics suite and naive baseline.

Three units:

* ``_positive_scores`` — picks the positive-class score source for ROC-AUC
  (``predict_proba`` > ``decision_function`` > ``None``).
* :func:`evaluate` — assembles the JSON-serializable metrics dict for a fitted
  model on the test split.
* :func:`baseline_metrics` — fits/evaluates a most-frequent ``DummyClassifier``
  as the floor any real model must beat.

Lightweight stub models (predict + optional score method) keep ``_positive_scores``
and the "no scores -> roc_auc=None" path deterministic without fitting anything;
``evaluate`` itself is also checked end-to-end against a real fitted
``LogisticRegression`` on the separable fixture.
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
HEADLINE_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


# --- Stub models for the score-source / no-score paths ----------------------
class _ProbaModel:
    """Stub exposing ``predict`` and ``predict_proba`` (the preferred source)."""

    def __init__(self, n: int):
        self._n = n

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        # Two columns; column 1 is the positive-class probability.
        p1 = np.linspace(0.1, 0.9, len(X))
        return np.column_stack([1.0 - p1, p1])


class _DecisionModel:
    """Stub exposing ``predict`` and ``decision_function`` only (the fallback)."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def decision_function(self, X):
        return np.linspace(-2.0, 2.0, len(X))


class _BareModel:
    """Stub exposing ``predict`` only — no score source (ROC-AUC must be skipped)."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestPositiveScores:
    """Selecting the per-sample positive-class score for ROC-AUC."""

    def test_prefers_predict_proba_column_one(self):
        """When available, the positive-class column of ``predict_proba`` is used."""
        X = np.zeros((5, 2))
        scores = _positive_scores(_ProbaModel(5), X)
        np.testing.assert_allclose(scores, np.linspace(0.1, 0.9, 5))

    def test_falls_back_to_decision_function(self):
        """Without ``predict_proba`` the raw ``decision_function`` is returned."""
        X = np.zeros((5, 2))
        scores = _positive_scores(_DecisionModel(), X)
        np.testing.assert_allclose(scores, np.linspace(-2.0, 2.0, 5))

    def test_returns_none_when_no_score_source(self):
        """A model with neither method yields ``None`` (ROC-AUC skipped gracefully)."""
        assert _positive_scores(_BareModel(), np.zeros((3, 2))) is None


class TestEvaluate:
    """The full metrics dict for a fitted classifier."""

    def test_class_names_are_real_then_fake(self):
        """``CLASS_NAMES`` matches the manifest's 0=real, 1=fake convention."""
        assert CLASS_NAMES == ["real", "fake"]

    def test_metrics_dict_is_complete_typed_and_json_serializable(self, feature_split):
        """A fitted model yields a complete, correctly-typed, JSON-safe metrics dict.

        All headline metrics are present as floats, ``n_test`` matches the test
        size, the confusion matrix is a nested list, the report is a string, and
        the whole dict survives ``json.dumps`` (it is saved as an artifact).

        Args:
            feature_split: small separable train/val/test split (fixture).
        """
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
        # Must be serializable — this is what save_artifacts writes.
        json.dumps(metrics)

    def test_separable_data_scores_perfectly(self, feature_split):
        """On the separable fixture a linear model reports accuracy/F1 of 1.0.

        Confirms ``evaluate`` wires predictions to metrics correctly (not just
        that keys exist).

        Args:
            feature_split: small separable train/val/test split (fixture).
        """
        s = feature_split
        model = LogisticRegression(max_iter=1000).fit(s.X_train, s.y_train)
        metrics = evaluate(model, s.X_test, s.y_test)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_roc_auc_is_none_without_score_source(self):
        """A predict-only model produces ``roc_auc=None`` rather than erroring.

        ROC-AUC needs continuous scores; when the model exposes none, the field
        is ``None`` and the rest of the suite still computes.
        """
        y_test = np.array([0, 1, 0, 1])
        metrics = evaluate(_BareModel(), np.zeros((4, 2)), y_test)
        assert metrics["roc_auc"] is None
        assert metrics["n_test"] == 4


class TestBaselineMetrics:
    """The naive most-frequent baseline."""

    def test_majority_baseline_accuracy_equals_majority_fraction(self):
        """The dummy predicts the train-majority class; accuracy = its test fraction.

        Train is majority class 0 (4 vs 1), so the baseline always predicts 0;
        on a test split that is 3/5 class 0, accuracy must be exactly 0.6, and
        the result is labelled as the most-frequent baseline.
        """
        X_train = np.zeros((5, 3), dtype=np.float32)
        y_train = np.array([0, 0, 0, 0, 1])           # majority class 0
        X_test = np.zeros((5, 3), dtype=np.float32)
        y_test = np.array([0, 0, 0, 1, 1])            # 3/5 are class 0

        metrics = baseline_metrics(X_train, y_train, X_test, y_test)

        assert metrics["accuracy"] == pytest.approx(0.6)
        assert metrics["model"] == "baseline (most_frequent)"
