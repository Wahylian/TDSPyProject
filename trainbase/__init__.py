"""Registry-driven training backend behind train_model.py.

Supplies the reusable pieces a run needs — a catalogue of trainable models and
a catalogue of preprocessing pipelines — while keeping the training script thin.

Selection is by name through two registries rather than hardcoded branches, so
a new model or pipeline is added as one registry entry, not a new code path. A
single shared seed (RANDOM_STATE) keeps runs deterministic, and each ModelSpec
co-locates an estimator factory with its hyperparameter grid.

Extend by registering a ModelSpec in MODEL_REGISTRY or a pipeline factory in
PIPELINE_REGISTRY; the feature, tuning, and evaluation machinery pick it up.
"""

from .model_registry import (
    RANDOM_STATE,
    ModelSpec,
    MODEL_REGISTRY,
)
from .pipeline_registry import PIPELINE_REGISTRY
from .features import (
    build_feature_pipeline,
    load_images,
    fit_features,
    transform_features,
)
from .training import (
    build_estimator,
    tune_hyperparameters,
)
from .evaluation import (
    CLASS_NAMES,
    evaluate,
    baseline_metrics,
)
from .diagnostics import collect_diagnostics
from .artifacts import build_metadata, save_artifacts

__all__ = [
    # registries & their entry type
    "RANDOM_STATE",
    "ModelSpec",
    "MODEL_REGISTRY",
    "PIPELINE_REGISTRY",
    # feature front-end
    "build_feature_pipeline",
    "load_images",
    "fit_features",
    "transform_features",
    # model assembly & tuning
    "build_estimator",
    "tune_hyperparameters",
    # evaluation
    "CLASS_NAMES",
    "evaluate",
    "baseline_metrics",
    # diagnostics
    "collect_diagnostics",
    # persistence
    "build_metadata",
    "save_artifacts",
]
