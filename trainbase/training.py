"""Model assembly and hyperparameter tuning.

The feature front-end (:mod:`trainbase.features`) already delivers reduced,
standardized vectors, so the estimator here is *just the classifier*. Two
functions cover the model side of a run:

  * :func:`build_estimator` wraps the chosen registry classifier in a one-step
    sklearn :class:`~sklearn.pipeline.Pipeline` under the name ``"clf"``.
  * :func:`tune_hyperparameters` grid-searches that estimator, selecting on the
    *validation* split (not k-fold CV), then refits the winner on train+val.
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
    """Assemble the sklearn ``Pipeline`` — just the classifier.

    PCA and per-feature standardization are done upstream by the feature
    pipeline, so the features reaching this estimator are already reduced and
    scaled. The estimator is therefore a single-step ``Pipeline`` wrapping the
    classifier under the name ``"clf"``.

    The one-step ``Pipeline`` is kept (rather than a bare estimator) so the
    ``GridSearchCV`` grids stay ``clf__``-prefixed and the classifier-swap
    mechanism in :data:`MODEL_REGISTRY` is unchanged.

    Args:
        model_name: Key into :data:`MODEL_REGISTRY`.

    Returns:
        An unfitted sklearn :class:`~sklearn.pipeline.Pipeline` (``clf`` only).

    Raises:
        KeyError: If ``model_name`` is not registered (caught by the caller).
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
    """Grid-search hyperparameters, selecting on the *validation* split.

    The ``val`` split explicitly drives tuning via
    :class:`~sklearn.model_selection.PredefinedSplit`: train and val are
    concatenated, and the fold definition marks train rows as "never validate"
    (``-1``) and val rows as the single validation fold (``0``). So every
    candidate is *fit on train* and *scored on val* — a true holdout, not k-fold
    CV on a mixed pool.

    With ``refit=True`` (default), ``GridSearchCV`` then refits the best
    configuration on train+val combined, the standard "use all the data you
    tuned with" step before test evaluation.

    Args:
        estimator: The unfitted pipeline from :func:`build_estimator`.
        param_grid: Hyperparameter grid (``clf__`` prefixed keys).
        X_train, y_train: Training features/labels (used to fit candidates).
        X_val, y_val: Validation features/labels (used to score candidates).
        scoring: Metric to optimize. F1 balances precision/recall on this
            roughly-balanced task.

    Returns:
        The fitted :class:`GridSearchCV`, whose ``best_estimator_`` is the model
        refit on train+val.
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
        refit=True,       # refit the winner on train+val before we touch test
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)

    logger.info("Best validation %s: %.4f", scoring, search.best_score_)
    logger.info("Best params: %s", search.best_params_ or "(defaults; empty grid)")
    return search
