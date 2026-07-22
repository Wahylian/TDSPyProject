"""Registry of trainable classifiers for train_model.py.

To add a model: import the estimator, add a ModelSpec entry keyed by a CLI name,
and run with --model <name>. Grid keys are prefixed 'clf__' because the estimator
is the "clf" step of the sklearn Pipeline (see build_estimator). Deep models
(cnn/vit) register only when PyTorch is installed; pair them with a raw-pixel
pipeline, e.g. --model cnn --pipeline pixels.
"""



from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

# One seed threaded through every random operation for reproducibility.
RANDOM_STATE = 42


@dataclass
class ModelSpec:
    """One MODEL_REGISTRY entry: an estimator factory plus its tuning grid.

    factory returns a fresh unfitted estimator each call, so no fitted state
    leaks across runs. param_grid feeds GridSearchCV with 'clf__'-prefixed keys;
    an empty grid means no tuning.
    """

    factory: Callable[[], BaseEstimator]
    param_grid: Dict[str, list] = field(default_factory=dict)


from .linear_models import ThresholdedLinearRegression

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # Soft-margin kernel SVM, the focus of this script. probability=False keeps
    # fitting fast; ROC-AUC uses decision_function instead.
    "svm": ModelSpec(
        factory=lambda: SVC(probability=False, random_state=RANDOM_STATE),
        param_grid={
            "clf__C": [0.1, 1.0, 10.0],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
        },
    ),
    # Scale-insensitive bagging baseline. oob_score exposes clf.oob_score_ for diagnostics.
    "rf": ModelSpec(
        factory=lambda: RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, oob_score=True
        ),
        param_grid={
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 20],
        },
    ),
    # Fast linear reference point.
    "logreg": ModelSpec(
        factory=lambda: LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        param_grid={"clf__C": [0.1, 1.0, 10.0]},
    ),
    # Hard-margin SVM: a huge C removes the slack. LinearSVC is the fast linear form.
    "hard_svm": ModelSpec(
        factory=lambda: LinearSVC(C=1e6, max_iter=10_000, random_state=RANDOM_STATE),
        param_grid={"clf__C": [1e4, 1e6]},
    ),
    # Same hard margin via the kernel SVC with a linear kernel, for a like-for-like compare.
    "hard_svm_kernel": ModelSpec(
        factory=lambda: SVC(kernel="linear", C=1e6, max_iter=10_000, random_state=RANDOM_STATE),
        param_grid={"clf__C": [1e4, 1e6]},
    ),
    # Least-squares classifier: regresses the class targets and thresholds.
    "ridge": ModelSpec(
        factory=lambda: RidgeClassifier(random_state=RANDOM_STATE),
        param_grid={"clf__alpha": [0.1, 1.0, 10.0]},
    ),
    # The most literal linear-regression baseline: regress 0/1 targets, threshold at 0.5.
    "linreg": ModelSpec(
        factory=lambda: ThresholdedLinearRegression(random_state=RANDOM_STATE),
        param_grid={"clf__fit_intercept": [True, False]},
    ),
    # Histogram gradient boosting: fast, scale-insensitive non-linear baseline on the PCA features.
    "hgb": ModelSpec(
        factory=lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid={
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_iter": [100, 200],
        },
    ),
    # Iterative MLP exposing loss_curve_ for the per-epoch training-history diagnostic.
    # max_iter is raised so small splits converge without a ConvergenceWarning.
    "mlp": ModelSpec(
        factory=lambda: MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
        param_grid={
            "clf__alpha": [1e-4, 1e-3],
            "clf__hidden_layer_sizes": [(100,), (64, 32)],
        },
    ),
}


# Deep models register only when torch is importable, so the project runs
# without the optional dependency; --model cnn/vit then appear automatically.
try:
    from .torch_models import build_torch_registry

    MODEL_REGISTRY.update(build_torch_registry())
except ImportError:  # pragma: no cover - exercised only when torch is absent
    pass