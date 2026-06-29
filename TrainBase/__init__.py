"""trainbase — a modular, registry-driven backend for model training.

trainbase is the helper package behind ``train_model.py``.
It supplies the reusable building blocks a training run needs — the catalogue of trainable
models and the catalogue of image-to-vector preprocessing pipelines — while
keeping the training script itself thin and model-agnostic.

Design philosophy
-----------------
The package is built around *registries* rather than hardcoded branches.
A training run never asks "is this an SVM? a random forest?";
It asks the registry for whatever entry a short string name maps to and works with the
result through a common interface.
This yields a few deliberate properties:

* **Extensible by design.** 
    New models and pipelines are added as data — 
    one new entry in a registry — not by editing the control flow of the training code.
    The set of available models and pipelines is therefore dynamic:
    it is whatever the registries currently contain, and it grows simply by registering more entries.
* **Selection by name.** 
    Every model and pipeline is keyed by a CLI-friendly string, 
    so the choice of estimator or preprocessing front-end is a single flag rather than a code path.
* **Reproducibility.** 
    A single shared random seed (:data:`RANDOM_STATE`) is threaded through every randomised operation, so runs are deterministic.
* **Separation of concerns.**
    Each model entry (:class:`ModelSpec`) bundles a *factory* for a fresh, unfitted estimator together with its hyperparameter
    grid, keeping per-run state clean and tuning configuration co-located with the estimator it belongs to.

Extending the package
----------------------
To add a model, register a new :class:`ModelSpec` in :data:`MODEL_REGISTRY`;
To add a preprocessing front-end, register a new pipeline factory in :data:`PIPELINE_REGISTRY`.
No other part of the training code needs to change —
the shared feature front-end, tuning, and evaluation machinery pick the new entry up automatically.
See the module docstrings of ``model_registry`` and ``pipeline_registry`` for worked examples.
"""

# -- Model side: the estimator catalogue and its entry type. -----------------
from .model_registry import (
    RANDOM_STATE,
    ModelSpec,
    MODEL_REGISTRY,
)

# -- Pipeline side: the preprocessing front-end catalogue. -------------------
from .pipeline_registry import PIPELINE_REGISTRY

# -- Feature front-end: build the pipeline, extract & cache features. --------
from .features import (
    build_feature_pipeline,
    load_images,
    fit_features,
    transform_features,
)

# -- Model assembly & validation-driven hyperparameter tuning. ---------------
from .training import (
    build_estimator,
    tune_hyperparameters,
)

# -- Evaluation suite & naive baseline. --------------------------------------
from .evaluation import (
    CLASS_NAMES,
    evaluate,
    baseline_metrics,
)

# -- Run persistence (fitted pipeline + classifier + metrics). ---------------
from .artifacts import save_artifacts

# The explicit public API of the package. Anything not listed here is an
# implementation detail and may change without notice.
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
    # persistence
    "save_artifacts",
]
