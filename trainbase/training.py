"""Model assembly and hyperparameter tuning.

The feature front-end already delivers reduced, standardized vectors, so the
estimator here is just the classifier. build_estimator wraps it in a one-step
sklearn Pipeline; tune_hyperparameters grid-searches on the validation split
(not k-fold CV), then refits the winner on train+val.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline

from .model_registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)


def build_estimator(model_name: str) -> Pipeline:
    """Wrap the registry classifier in a one-step 'clf' Pipeline.

    Reduction and scaling happen upstream, so this is just the classifier. The
    one-step Pipeline keeps the GridSearchCV grids 'clf__'-prefixed.
    """
    spec = MODEL_REGISTRY[model_name]
    return Pipeline(steps=[("clf", spec.factory())])


def tune_hyperparameters(
    estimator: Pipeline,
    param_grid: Dict[str, list],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scoring: str = "f1",
) -> GridSearchCV:
    """Grid-search hyperparameters, scoring on the validation split.

    Uses PredefinedSplit over concatenated train+val (train marked -1, val fold
    0), so every candidate is fit on train and scored on val — a true holdout,
    not k-fold CV. refit=True then refits the winner on train+val. best_estimator_
    is that refit model.
    """
    # Concatenate the two splits; the fold array keeps their roles distinct.
    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    test_fold = np.concatenate(
        [np.full(len(X_train), -1, dtype=int), np.zeros(len(X_val), dtype=int)]
    )
    predefined = PredefinedSplit(test_fold)

    n_candidates = (
        max(1, int(np.prod([len(v) for v in param_grid.values()]))) if param_grid else 1
    )
    logger.info(
        "Tuning %d candidate configuration(s) on the val split (scoring='%s')...",
        n_candidates,
        scoring,
    )

    search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=predefined,
        refit=True,       # refit the winner on train+val before test
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)

    logger.info("Best validation %s: %.4f", scoring, search.best_score_)
    logger.info("Best params: %s", search.best_params_ or "(defaults; empty grid)")
    return search
