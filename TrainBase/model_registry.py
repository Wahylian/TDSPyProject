"""
Helper For 'train_model.py' 

Contains the Registry for ML models that the project can train on.

How to add a new classifier to the model Registry (Example): 
    To add a Gradiant Boosting model:
    1. Import it at the top of the file 
    2. Add an entry of the following structure to the MODEL_REGISTRY:
        "gb" :  ModelSpec(
               factory=lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
               param_grid={"clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [100, 200]},
            )
    3. Run with ``--model gb``.

    Grid keys are prefixed with ``clf__`` because the estimator is the ``"clf"`` step of the sklearn
    ``Pipeline`` (see ``build_estimator``).
"""



from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# -- sk-learn ----------------------------------------------------------------
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# -- Constants ---------------------------------------------------------------

# A single seed threaded through every random operation (subsampling, PCA, the
# estimators) so the whole run is reproducible.
RANDOM_STATE = 42

# =============================================================================
# Model registry — the single place to add or swap classifiers.
# =============================================================================
@dataclass
class ModelSpec:
    """One entry in :data:`MODEL_REGISTRY`.

    Attributes:
        factory: A zero-argument callable returning a *fresh*, unfitted
            estimator. It is a factory (not a pre-built instance) so each run
            gets its own clean estimator and we never accidentally reuse fitted
            state across runs.
        param_grid: The hyperparameter grid handed to ``GridSearchCV``. Keys are
            prefixed with ``clf__`` because the estimator is the ``"clf"`` step
            of the sklearn ``Pipeline`` built in :func:`build_estimator`. An
            empty grid means "no tuning" (the model is fit with its defaults).
    """

    factory: Callable[[], BaseEstimator]
    param_grid: Dict[str, list] = field(default_factory=dict)


# To add a new classifier: import it above, then add one entry here. Nothing
# else in the script needs to change — selection is purely by the ``--model``
# flag, and the PCA->scale feature front-end / tuning / evaluation are shared.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # Soft-margin kernel SVM — the focus of this script.
    #   * C       : soft-margin strength (low C = wider margin, more tolerant).
    #   * kernel  : 'rbf' (non-linear) vs 'linear' (gamma is ignored for linear,
    #               but leaving it in the grid is harmless).
    #   * gamma   : RBF kernel width; 'scale' and 'auto' are data-derived defaults.
    # probability=False keeps fitting fast; we use decision_function for ROC-AUC.
    "svm": ModelSpec(
        factory=lambda: SVC(probability=False, random_state=RANDOM_STATE),
        param_grid={
            "clf__C": [0.1, 1.0, 10.0],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
        },
    ),
    # Random Forest — a strong, scale-insensitive baseline. Included to
    # demonstrate how trivially the classifier swaps out (`--model rf`).
    "rf": ModelSpec(
        factory=lambda: RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        param_grid={
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 20],
        },
    ),
    # Plain logistic regression — a fast linear reference point.
    "logreg": ModelSpec(
        factory=lambda: LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        param_grid={"clf__C": [0.1, 1.0, 10.0]},
    ),
}