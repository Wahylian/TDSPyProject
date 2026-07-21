"""Algorithm-appropriate diagnostics for a fitted training run.

Beyond the headline metrics in :mod:`trainbase.evaluation`, a run can capture
cheap, model-specific diagnostics that make performance graphing and model
comparison possible: tree-based feature importances, the Random Forest
out-of-bag score, the per-configuration validation scores from the
hyperparameter search, and (opt-in) a learning curve over sample sizes.

The current model registry holds only classical sklearn estimators, so there is
no per-epoch loss/metric history to record. An iterative model (e.g. a neural
network) would add that history here as a separate branch.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Bounded, fixed grid so the (opt-in) learning curve stays affordable even for
# an expensive estimator like a kernel SVM.
_LEARNING_CURVE_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]
_LEARNING_CURVE_CV = 3


def _jsonable(value: object) -> object:
    """Coerce a hyperparameter value to something ``json.dump`` can handle."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _hyperparameter_scores(search: GridSearchCV) -> List[Dict[str, object]]:
    """Per-configuration validation scores from the fitted GridSearchCV.

    Serves as the "validation curve across the hyperparameter grid": each grid
    point with its mean/std validation score. Free — GridSearchCV already
    computed it in ``cv_results_``.
    """
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
    """Train/val scores over increasing training-set sizes (opt-in, best-effort).

    Refits a clone of the tuned model on growing subsets. Wrapped so any failure
    (too few samples per class, an estimator that rejects a subset, etc.) is
    logged and yields ``None`` rather than crashing the run.
    """
    try:
        sizes, train_scores, val_scores = learning_curve(
            clone(best_model),
            X_train,
            y_train,
            train_sizes=_LEARNING_CURVE_SIZES,
            cv=_LEARNING_CURVE_CV,
            scoring=scoring,
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

    Always includes the hyperparameter-grid validation scores, plus feature
    importances and the OOB score when the estimator exposes them (tree-based
    models). When ``include_curves`` is set, also computes a bounded learning
    curve over sample sizes (the one diagnostic that costs extra model fits).

    Args:
        search: The fitted GridSearchCV from tuning.
        best_model: Its ``best_estimator_`` (a one-step ``clf`` Pipeline).
        X_train, y_train: Training features/labels (for the learning curve).
        include_curves: Compute the sample-size learning curve if True.
        scoring: Metric name used for the learning curve.

    Returns:
        A dict with uniform keys (values ``None`` when a diagnostic does not
        apply to this estimator or was not requested).
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
