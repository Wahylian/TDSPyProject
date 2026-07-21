"""Linear-regression-as-classifier estimator for the model registry.

``ThresholdedLinearRegression`` wraps plain ordinary-least-squares
:class:`~sklearn.linear_model.LinearRegression` into a binary classifier: it
regresses 0/1 class targets and thresholds the continuous output at 0.5. This
gives the project a literal "linear regression" classifier alongside the
least-squares :class:`~sklearn.linear_model.RidgeClassifier`, while exposing the
sklearn classifier surface (``predict`` / ``decision_function`` / ``classes_``)
that the registry contract and the evaluation suite rely on.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from .model_registry import RANDOM_STATE


class ThresholdedLinearRegression(BaseEstimator, ClassifierMixin):
    """Binary classifier: least-squares regression thresholded at ``threshold``.

    Args:
        fit_intercept: Passed through to the internal ``LinearRegression``.
        threshold: Decision boundary on the regression output; predictions are
            the positive class where ``raw >= threshold``.
        random_state: Held only to satisfy the registry contract (every entry
            exposes the shared seed). OLS is deterministic, so it is unused.
    """

    def __init__(self, fit_intercept: bool = True, threshold: float = 0.5,
                 random_state: int = RANDOM_STATE):
        self.fit_intercept = fit_intercept
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        """Fit OLS on 0/1-encoded targets; store ``classes_`` and ``reg_``."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                "ThresholdedLinearRegression supports binary classification "
                f"only; got {len(self.classes_)} classes."
            )
        # Regress the indicator of the positive (second) class.
        y01 = (y == self.classes_[1]).astype(float)
        self.reg_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.reg_.fit(X, y01)
        return self

    def decision_function(self, X):
        """Continuous score, centered on the threshold (for PR-AUC / ROC-AUC)."""
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        return self.reg_.predict(X) - self.threshold

    def predict(self, X):
        """Predict class labels from ``classes_`` by thresholding the score."""
        idx = (self.decision_function(X) >= 0).astype(int)
        return self.classes_[idx]
