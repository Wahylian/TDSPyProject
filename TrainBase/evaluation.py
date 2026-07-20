"""Evaluation suite for a fitted classifier.

Computes the headline classification metrics (accuracy, precision, recall, F1,
ROC-AUC) plus a confusion matrix and per-class report, and provides a naive
majority-class baseline so a trained model's numbers have a floor to beat. 
Every metric is returned as a JSON-serializable dict suitable for saving as an
artifact, and is also logged in a human-readable block.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model_registry import RANDOM_STATE

logger = logging.getLogger(__name__)

# Human-readable class names, indexed by integer label (0 = real, 1 = fake),
# matching the manifest produced by create_split.py.
CLASS_NAMES = ["real", "fake"]


def _positive_scores(model: BaseEstimator, X: np.ndarray) -> Optional[np.ndarray]:
    """Return per-sample scores for the positive class (label 1 = fake).

    Used for ROC-AUC. Prefers calibrated probabilities when available, otherwise
    falls back to the raw decision function. 
    Returns ``None`` if the model exposes neither (so ROC-AUC can be skipped gracefully).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_label: str = "model",
) -> Dict[str, object]:
    """Compute the full evaluation suite for a fitted model on the test split.

    Reports accuracy, precision, recall, F1 and ROC-AUC, plus a confusion matrix
    and a per-class classification report (logged for human inspection).

    Args:
        model: A fitted estimator with ``predict`` (and ideally
            ``predict_proba``/``decision_function`` for ROC-AUC).
        X_test, y_test: Held-out test features/labels.
        model_label: Name used in log lines and the returned dict.

    Returns:
        A JSON-serializable dict of metrics (confusion matrix and report
        included), suitable for saving as an artifact.
    """
    y_pred = model.predict(X_test)
    scores = _positive_scores(model, X_test)

    metrics: Dict[str, object] = {
        "model": model_label,
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        # ROC-AUC needs continuous scores; None if the model can't provide them.
        "roc_auc": float(roc_auc_score(y_test, scores)) if scores is not None else None,
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
    )

    # --- Human-readable log block ------------------------------------------
    logger.info("=" * 60)
    logger.info("Evaluation: %s", model_label)
    logger.info("  Accuracy : %.4f", metrics["accuracy"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall   : %.4f", metrics["recall"])
    logger.info("  F1-score : %.4f", metrics["f1"])
    if metrics["roc_auc"] is not None:
        logger.info("  ROC-AUC  : %.4f", metrics["roc_auc"])
    # Confusion matrix laid out with labelled rows (true) and columns (pred).
    logger.info("  Confusion matrix (rows=true, cols=pred) [%s]:", ", ".join(CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, cm):
        logger.info("    %-5s %s", name, row.tolist())
    logger.info("  Classification report:\n%s", metrics["classification_report"])
    logger.info("=" * 60)

    return metrics


def baseline_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    """Fit and evaluate a naive majority-class baseline for context.

    A ``DummyClassifier(strategy="most_frequent")`` always predicts the majority
    training class. Any real model must clear this bar to be worth anything; on
    this ~54/46 split that bar is ~0.54 accuracy. (Switch the strategy to
    ``"constant"`` with ``constant=1`` to instead model an "always predict fake"
    baseline.)
    """
    logger.info("Fitting naive majority-class baseline (DummyClassifier)...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    return evaluate(dummy, X_test, y_test, model_label="baseline (most_frequent)")
