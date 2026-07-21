"""Tests for trainbase.linear_models.ThresholdedLinearRegression.

The wrapper turns plain least-squares LinearRegression into a binary classifier
(regress 0/1 targets, threshold at 0.5), exposing the sklearn classifier surface
the registry contract and evaluation suite expect.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from trainbase.linear_models import ThresholdedLinearRegression
from trainbase.model_registry import RANDOM_STATE


def test_unfitted_raises_not_fitted():
    with pytest.raises(NotFittedError):
        check_is_fitted(ThresholdedLinearRegression())


def test_exposes_random_state_default_seed():
    assert ThresholdedLinearRegression().get_params()["random_state"] == RANDOM_STATE


def test_fit_predict_separates_classes(feature_split):
    clf = ThresholdedLinearRegression().fit(feature_split.X_train, feature_split.y_train)
    acc = (clf.predict(feature_split.X_test) == feature_split.y_test).mean()
    assert acc > 0.8


def test_decision_function_shape_and_ordering(feature_split):
    clf = ThresholdedLinearRegression().fit(feature_split.X_train, feature_split.y_train)
    scores = clf.decision_function(feature_split.X_test)
    assert scores.shape == (len(feature_split.y_test),)
    # class-1 test points should score higher on average than class-0 ones.
    y = feature_split.y_test
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_predict_labels_are_from_classes(feature_split):
    clf = ThresholdedLinearRegression().fit(feature_split.X_train, feature_split.y_train)
    assert set(np.unique(clf.predict(feature_split.X_test))).issubset({0, 1})


def test_clone_and_get_params_roundtrip():
    clf = ThresholdedLinearRegression(fit_intercept=False, threshold=0.3)
    twin = clone(clf)
    assert twin.get_params() == clf.get_params()


def test_rejects_non_binary():
    X = np.random.default_rng(0).random((9, 3))
    y = np.array([0, 1, 2] * 3)
    with pytest.raises(ValueError):
        ThresholdedLinearRegression().fit(X, y)
