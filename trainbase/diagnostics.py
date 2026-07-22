"""Model-specific diagnostics for a fitted training run.

Beyond the headline metrics, captures cheap extras for graphing and comparison:
tree feature importances, the Random Forest OOB score, per-configuration
validation scores from the hyperparameter search, and an opt-in learning curve.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline

from .model_registry import RANDOM_STATE

logger = logging.getLogger(__name__)

# Fixed grid keeping the opt-in learning curve affordable even for a kernel SVM.
_LEARNING_CURVE_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]
_LEARNING_CURVE_CV = 3


def _jsonable(value: object) -> object:
    """Coerce a hyperparameter value to something ``json.dump`` can handle."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _hyperparameter_scores(search: GridSearchCV) -> List[Dict[str, object]]:
    """Per-configuration mean/std validation scores from GridSearchCV.cv_results_."""
    results = search.cv_results_
    scores: List[Dict[str, object]] = []
    for params, mean, std in zip(
        results["params"], results["mean_test_score"], results["std_test_score"]
    ):
        scores.append(
            {
                "params": {k: _jsonable(v) for k, v in params.items()},
                "mean_val_score": float(mean),
                "std_val_score": float(std),
            }
        )
    return scores


def _learning_curve(
    best_model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scoring: str,
) -> Optional[Dict[str, object]]:
    """Train/val scores over growing training-set sizes, best-effort.

    Refits a clone of the tuned model on growing subsets; any failure is logged
    and returns None rather than crashing the run.
    """
    try:
        sizes, train_scores, val_scores = learning_curve(
            clone(best_model),
            X_train,
            y_train,
            train_sizes=_LEARNING_CURVE_SIZES,
            cv=_LEARNING_CURVE_CV,
            scoring=scoring,
            shuffle=True,               # representative subsets, not the sorted prefix
            random_state=RANDOM_STATE,  # made deterministic by the seed
            n_jobs=-1,
        )
    except Exception as exc:  # best-effort diagnostic; never fail the run
        logger.warning("Learning curve skipped (%s: %s)", type(exc).__name__, exc)
        return None
    return {
        "train_sizes": [int(n) for n in sizes],
        "train_scores_mean": [float(s) for s in train_scores.mean(axis=1)],
        "val_scores_mean": [float(s) for s in val_scores.mean(axis=1)],
    }


def collect_diagnostics(
    search: GridSearchCV,
    best_model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    include_curves: bool = False,
    scoring: str = "f1",
) -> Dict[str, object]:
    """Gather JSON-serializable diagnostics for the tuned model.

    Always includes the hyperparameter-grid scores, plus feature importances and
    OOB score when the estimator exposes them. include_curves adds the learning
    curve (the only diagnostic costing extra fits). Absent diagnostics are None.
    """
    clf = best_model.named_steps["clf"]

    importances = getattr(clf, "feature_importances_", None)
    oob_score = getattr(clf, "oob_score_", None)

    return {
        "feature_importances": (
            [float(v) for v in importances] if importances is not None else None
        ),
        "oob_score": float(oob_score) if oob_score is not None else None,
        "hyperparameter_scores": _hyperparameter_scores(search),
        "learning_curve": (
            _learning_curve(best_model, X_train, y_train, scoring)
            if include_curves
            else None
        ),
    }
