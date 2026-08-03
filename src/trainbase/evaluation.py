"""Evaluation suite for a fitted classifier.

Computes accuracy, precision, recall, F1, PR-AUC, ROC-AUC plus a confusion
matrix and per-class report, and a naive majority-class baseline as a floor.
Metrics are returned as a JSON-serializable dict and also logged.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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
    """Return positive-class scores (label 1 = fake) for PR-AUC/ROC-AUC.

    Prefers predict_proba, falls back to decision_function, and returns None if
    the model exposes neither so the AUC metrics can be skipped.
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
    """Run the full metric suite for a fitted model on the test split.

    Returns a JSON-serializable dict (metrics, confusion matrix, per-class
    report). AUC metrics need a probability or decision score; see _positive_scores.
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
        # AUC metrics need continuous scores; None when the model provides none.
        "pr_auc": float(average_precision_score(y_test, scores)) if scores is not None else None,
        "roc_auc": float(roc_auc_score(y_test, scores)) if scores is not None else None,
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
    )

    # Human-readable log block.
    logger.info("=" * 60)
    logger.info("Evaluation: %s", model_label)
    logger.info("  Accuracy : %.4f", metrics["accuracy"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall   : %.4f", metrics["recall"])
    logger.info("  F1-score : %.4f", metrics["f1"])
    if metrics["pr_auc"] is not None:
        logger.info("  PR-AUC   : %.4f", metrics["pr_auc"])
    if metrics["roc_auc"] is not None:
        logger.info("  ROC-AUC  : %.4f", metrics["roc_auc"])
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
    """Fit and evaluate a majority-class DummyClassifier as a floor to beat.

    On this ~54/46 split that bar is ~0.54 accuracy.
    """
    logger.info("Fitting naive majority-class baseline (DummyClassifier)...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    return evaluate(dummy, X_test, y_test, model_label="baseline (most_frequent)")
