"""
train_svm.py — Train a Soft-Margin SVM to classify images as real vs. fake.
================================================================================

This script ties the project's existing building blocks together into a single,
reproducible training pipeline:

    extract_features.get_feature_stream  ->  streams (image, label) per split
    prebuilt_pipelines.PrebuiltPipelines ->  turns each image into a flat vector
    preprocessing.ImagePipeline          ->  PCA reduction ('reduce') + per-feature
                                             standardization ('scale'), both fit on
                                             train and reused on val/test
    scikit-learn (this file)             ->  tunes & evaluates the classifier

It trains a *Soft-SVM* — an SVM with a soft margin, i.e. the regularization
parameter ``C`` that trades off margin width against misclassification. Both
``SVC`` (kernel SVM) and ``LinearSVC`` are soft-margin SVMs; we use ``SVC`` so
that the kernel and ``gamma`` can be tuned as the task requires.

--------------------------------------------------------------------------------
Why subsample + PCA instead of training on everything?
--------------------------------------------------------------------------------
The train split has ~78k images and the SVM pipeline emits 16,384 features each.
A dense 78k x 16,384 float64 matrix is ~10 GB, and a *kernel* SVM costs roughly
O(n^2)-O(n^3) in the number of samples — a grid search over that would run for
days. So by default we:

  1. take a bounded random subsample of the (already shuffled) train stream,
  2. compress the 16,384 raw features with PCA and standardize the result —
     both performed by the project's ``ImagePipeline`` (``vec-pca`` reduce step
     then a ``scale`` step), fit on the training features and reused unchanged on
     val/test so all splits share one feature space, and
  3. grid-search a kernel ``SVC`` over {C, kernel, gamma}.

Every cap is a CLI flag (``--max-train-samples``, ``--pca-components``, ...), so
scaling up is a one-line change once you have the compute/RAM for it.

NOTE (effectiveness lever): a single 5k-sample SVM leaves most of the data
unused. A natural extension is *bagging* — train several SVMs, each on a
distinct ~5k chunk of the full 78k, and average their decisions (e.g.
``sklearn.ensemble.BaggingClassifier`` wrapping the SVM, or a manual ensemble
over chunks). That uses far more of the dataset while keeping each base
estimator cheap to fit. It is intentionally left out here to keep this script
focused, but the modular structure below makes it straightforward to add.

--------------------------------------------------------------------------------
How to run
--------------------------------------------------------------------------------
    # Defaults: SVM, 5k train / 2k val / 5k test subsample, PCA->150 dims.
    python train_svm.py

    # Use more data and more PCA components (slower, usually better):
    python train_svm.py --max-train-samples 10000 --pca-components 300

    # Evaluate on the FULL test split (0 means "use all rows"):
    python train_svm.py --max-test-samples 0

    # Swap the classifier (see MODEL_REGISTRY below) — no other change needed:
    python train_svm.py --model rf

    # Pick a different feature pipeline (see PIPELINE_REGISTRY below):
    python train_svm.py --pipeline fast

Artifacts (the fitted classifier, the fitted feature pipeline — which holds the
PCA basis *and* the scaling statistics — and metrics) are written to ``outputs/``
by default (override with ``--output-dir``).

--------------------------------------------------------------------------------
How to adapt this for another algorithm
--------------------------------------------------------------------------------
The classifier is decoupled from everything else via ``MODEL_REGISTRY`` (see its
definition for the exact recipe). To add, say, a gradient-boosting model:

    1. Import it at the top of this file.
    2. Add one entry to MODEL_REGISTRY:
           "gb": ModelSpec(
               factory=lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
               param_grid={"clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [100, 200]},
           )
    3. Run with ``--model gb``.

The feature front-end (PCA *and* standardization are done upstream by the
feature pipeline, not in the estimator), the train/val tuning, the evaluation
suite, and the artifact saving all stay exactly the same. Grid keys are prefixed
with ``clf__`` because the estimator is the ``"clf"`` step of the sklearn
``Pipeline`` (see ``build_estimator``).

Requirements:
    pip install numpy opencv-python scikit-learn joblib
    (plus this project's ``preprocessing`` package and its dependencies)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# --- Project building blocks (imported, not reimplemented) --------------------
from extract_features import get_feature_stream
from prebuilt_pipelines import PrebuiltPipelines
from preprocessing import ImagePipeline

# --- scikit-learn ------------------------------------------------------------
import joblib
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# A single seed threaded through every random operation (subsampling, PCA, the
# estimators) so the whole run is reproducible.
RANDOM_STATE = 42

# Human-readable class names, indexed by integer label (0 = real, 1 = fake),
# matching the manifest produced by create_split.py.
CLASS_NAMES = ["real", "fake"]

logger = logging.getLogger("train_svm")


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


# =============================================================================
# Feature pipeline registry — the per-image image -> vector transforms.
# =============================================================================
# Maps a CLI name to a PrebuiltPipelines factory. These are the per-image *base*
# pipelines (image -> flat vector). build_feature_pipeline() appends the project's
# batch-level 'vec-pca' reduce step to the chosen base, so a single ImagePipeline
# does extraction *and* PCA reduction end to end. To add one, point at any
# per-image PrebuiltPipelines factory.
PIPELINE_REGISTRY: Dict[str, Callable[[], ImagePipeline]] = {
    "svm": PrebuiltPipelines.svm_pipeline,      # 128x128 grayscale -> 16,384
    "fast": PrebuiltPipelines.fast_pipeline,    # 64x64 grayscale  ->  4,096
    "hq": PrebuiltPipelines.hq_pipeline,        # 224x224 grayscale -> 50,176
    "no_denoise": PrebuiltPipelines.no_denoise_pipeline,
}


# =============================================================================
# Data loading & feature extraction
# =============================================================================
def load_images(split: str, max_samples: int = 0) -> Tuple[List[np.ndarray], np.ndarray]:
    """Stream one split's raw images + labels into memory (subsampled).

    Images come from :func:`extract_features.get_feature_stream`, which yields
    ``(image, label)`` pairs in a seeded, shuffled order — so taking the first
    ``max_samples`` items is an unbiased random subsample.

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.
        max_samples: Cap on the number of images to read; ``0`` means "all".

    Returns:
        ``(images, y)`` — a list of decoded BGR image arrays and an int label
        array aligned with it.

    Raises:
        FileNotFoundError: If the manifest CSV is missing (from the stream).
        RuntimeError: If the split yields no usable images.
    """
    images: List[np.ndarray] = []
    labels: List[int] = []
    for image, label in get_feature_stream(split, random_seed=RANDOM_STATE):
        images.append(image)
        labels.append(label)
        if len(images) % 1000 == 0:
            logger.info("  ... %d images loaded", len(images))
        if max_samples and len(images) >= max_samples:
            break

    if not images:
        raise RuntimeError(f"No usable images found for split '{split}'.")
    return images, np.asarray(labels, dtype=int)


def fit_features(
    pipeline: ImagePipeline,
    max_samples: int,
    cache_dir: Optional[Path],
    cache_prefix: str,
) -> Tuple[ImagePipeline, np.ndarray, np.ndarray]:
    """Fit the feature pipeline on TRAIN and return ``(pipeline, X, y)``.

    Runs ``pipeline.fit_transform`` on the training images: the per-image
    transforms produce flat vectors, the ``vec-pca`` reduce learns its PCA basis
    on those vectors, and the trailing ``scale`` learns the per-feature mean/std
    on the PCA output — all stored on the pipeline for later reuse.

    Caches the fitted pipeline *and* the reduced train features together, so a
    rerun reloads both without re-decoding images (and the reloaded pipeline can
    still transform val/test). The cache is keyed by the feature-defining params
    (pipeline + PCA dims + sample cap), so changing any of them recomputes.

    Returns:
        ``(fitted_pipeline, X_train, y_train)``.
    """
    feat_path, pipe_path = _train_cache_paths(cache_dir, cache_prefix, max_samples)
    if feat_path and feat_path.is_file() and pipe_path.is_file():
        logger.info("Loading cached fitted pipeline + train features from %s", feat_path)
        data = np.load(feat_path)
        return joblib.load(pipe_path), data["X"], data["y"]

    images, y = load_images("train", max_samples)
    logger.info("Fitting feature pipeline (incl. PCA) on %d train images...", len(images))
    X = pipeline.fit_transform(images)
    logger.info("  -> train features %s", X.shape)

    if feat_path:
        np.savez_compressed(feat_path, X=X, y=y)
        joblib.dump(pipeline, pipe_path)
        logger.info("Cached train features + fitted pipeline to %s", cache_dir)
    return pipeline, X, y


def transform_features(
    split: str,
    pipeline: ImagePipeline,
    max_samples: int,
    cache_dir: Optional[Path],
    cache_prefix: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a held-out split with the already-fitted pipeline.

    Runs ``pipeline.transform`` so val/test images are projected with the *same*
    PCA basis learned on train. Reduced features are cached per split.

    Returns:
        ``(X, y)`` for the requested split.
    """
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_prefix}_{split}_n{max_samples}.npz"
        if cache_path.is_file():
            logger.info("Loading cached features for '%s' from %s", split, cache_path)
            data = np.load(cache_path)
            return data["X"], data["y"]

    images, y = load_images(split, max_samples)
    logger.info("Transforming %d '%s' images with the train PCA basis...", len(images), split)
    X = pipeline.transform(images)
    logger.info("  -> %s features %s", split, X.shape)

    if cache_path is not None:
        np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def _train_cache_paths(
    cache_dir: Optional[Path], cache_prefix: str, max_samples: int
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return the (features, fitted-pipeline) cache paths for the train split."""
    if cache_dir is None:
        return None, None
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cache_prefix}_train_n{max_samples}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}_pipeline.joblib"


# =============================================================================
# Model construction
# =============================================================================
def build_feature_pipeline(name: str, n_components: int) -> ImagePipeline:
    """Build the single feature pipeline: per-image transforms + PCA + scale.

    Takes the chosen per-image *base* pipeline from :data:`PIPELINE_REGISTRY`
    (image -> flat vector) and appends the project's own batch-level ``vec-pca``
    reduce step followed by a ``scale`` (per-feature standardization) step. One
    :class:`ImagePipeline` then does everything end to end:

      * ``pipeline.fit_transform(train_images)`` runs the per-image transforms,
        learns the PCA basis on the resulting vectors, then learns the per-feature
        mean/std on the PCA output — all stored on the pipeline.
      * ``pipeline.transform(val/test images)`` reuses that *same* basis and the
        *same* scaling statistics, so every split lands in one consistent feature
        space (no leakage), and
      * ``pipeline.process(one_image)`` applies the whole chain to a single image
        for inference.

    Both the PCA and the standardization are done by the project's ImagePipeline
    — scikit-learn's ``PCA`` and ``StandardScaler`` are not used. The estimator
    (see :func:`build_estimator`) is therefore just the classifier.

    Args:
        name: Key into :data:`PIPELINE_REGISTRY` (the per-image base pipeline).
        n_components: Target PCA width. The ``vec-pca`` step clamps it to
            ``min(n_samples, n_features)`` at fit time.

    Returns:
        An unfitted :class:`ImagePipeline` (base transforms + ``vec-pca`` +
        ``scale``).
    """
    pipeline = PIPELINE_REGISTRY[name]()
    pipeline.add_operation(
        "reduce",
        {"method": "vec-pca", "n_components": n_components, "random_state": RANDOM_STATE},
    )
    # Standardize the PCA features per column (zero mean, unit variance), fit on
    # train and reused on val/test — the role scikit-learn's StandardScaler used
    # to play, now folded into the one feature pipeline so the estimator is just
    # the classifier.
    pipeline.add_operation("scale", {})
    return pipeline


def build_estimator(model_name: str) -> Pipeline:
    """Assemble the sklearn ``Pipeline`` — now just the classifier.

    Both PCA *and* per-feature standardization are done upstream by the project's
    :class:`ImagePipeline` (see :func:`build_feature_pipeline`), so the features
    reaching this estimator are already reduced and scaled. The estimator is
    therefore a single-step ``Pipeline`` wrapping the classifier under the name
    ``"clf"``.

    The one-step ``Pipeline`` is kept (rather than returning a bare estimator) so
    the ``GridSearchCV`` param grids stay ``clf__``-prefixed and the
    classifier-swap mechanism in :data:`MODEL_REGISTRY` is unchanged.

    Args:
        model_name: Key into :data:`MODEL_REGISTRY`.

    Returns:
        An unfitted sklearn :class:`~sklearn.pipeline.Pipeline` (``clf`` only).

    Raises:
        KeyError: If ``model_name`` is not registered (caught in ``main``).
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

    The task requires that the ``val`` split explicitly drive tuning. We achieve
    that with :class:`~sklearn.model_selection.PredefinedSplit`: train and val
    are concatenated, and the fold definition marks train rows as "never
    validate" (``-1``) and val rows as the single validation fold (``0``). So
    every candidate is *fit on train* and *scored on val* — a true holdout, not
    k-fold CV on a mixed pool.

    With ``refit=True`` (the default), ``GridSearchCV`` then refits the best
    configuration on train+val combined, the standard "use all the data you
    tuned with" final step before test evaluation.

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

    n_candidates = max(1, int(np.prod([len(v) for v in param_grid.values()]))) if param_grid else 1
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


# =============================================================================
# Evaluation
# =============================================================================
def _positive_scores(model: BaseEstimator, X: np.ndarray) -> Optional[np.ndarray]:
    """Return per-sample scores for the positive class (label 1 = fake).

    Used for ROC-AUC. Prefers calibrated probabilities when available, otherwise
    falls back to the raw SVM decision function. Returns ``None`` if the model
    exposes neither (so ROC-AUC can be skipped gracefully).
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


# =============================================================================
# Artifact saving
# =============================================================================
def save_artifacts(
    model: BaseEstimator,
    feature_pipeline: ImagePipeline,
    metrics: Dict[str, object],
    output_dir: Path,
    model_name: str,
) -> None:
    """Serialize the fitted feature pipeline, the model, and the metrics JSON.

    Two joblib artifacts together capture the full inference path from a raw
    image: the fitted ``feature_pipeline`` (the project ImagePipeline that does
    the per-image transforms *and* holds the trained PCA basis *and* the scaling
    statistics) turns an image into a reduced, standardized vector via
    ``feature_pipeline.process(image)``, and the fitted sklearn ``model`` (the
    classifier) scores it. End to end:
    ``model.predict([feature_pipeline.process(image)])``.

    Args:
        model: The fitted best estimator (a one-step ``Pipeline``: classifier).
        feature_pipeline: The fitted :class:`ImagePipeline` (transforms + PCA +
            scale).
        metrics: The evaluation dict from :func:`evaluate` (plus baseline).
        output_dir: Directory to write into (created if missing).
        model_name: Used in the artifact filenames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}_model.joblib"
    pipeline_path = output_dir / f"{model_name}_feature_pipeline.joblib"
    metrics_path = output_dir / f"{model_name}_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(feature_pipeline, pipeline_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved model     -> %s", model_path)
    logger.info("Saved pipeline  -> %s", pipeline_path)
    logger.info("Saved metrics   -> %s", metrics_path)


# =============================================================================
# Orchestration
# =============================================================================
def main(args: argparse.Namespace) -> None:
    """Run the end-to-end pipeline: load -> tune -> evaluate -> save.

    Each step is verified before the next proceeds: feature extraction must
    yield samples, tuning must select a model, and evaluation must clear the
    naive baseline (logged for comparison).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Validate selections up front so we fail fast with a clear message. ---
    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown --model '{args.model}'. Choices: {sorted(MODEL_REGISTRY)}"
        )
    if args.pipeline not in PIPELINE_REGISTRY:
        raise SystemExit(
            f"Unknown --pipeline '{args.pipeline}'. Choices: {sorted(PIPELINE_REGISTRY)}"
        )

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    # Cache key for the feature space: pipeline + PCA dims (+ '_std' marking the
    # features are now standardized in-pipeline). Changing any recomputes;
    # per-split sample caps are folded into each file name. The '_std' suffix
    # also invalidates any pre-standardization cache left from older runs.
    cache_prefix = f"{args.pipeline}_pca{args.pca_components}_std"

    # One feature pipeline does everything: per-image transforms + vec-pca reduce.
    feature_pipeline = build_feature_pipeline(args.pipeline, args.pca_components)
    logger.info("Feature pipeline '%s': %r", args.pipeline, feature_pipeline)

    # --- 1. Build features. The pipeline is FIT on train (learning the PCA
    #        basis) and REUSED (transform) on val/test, so all splits share one
    #        consistent feature space — no leakage, no second pipeline.
    try:
        feature_pipeline, X_train, y_train = fit_features(
            feature_pipeline, args.max_train_samples, cache_dir, cache_prefix
        )
        X_val, y_val = transform_features(
            "val", feature_pipeline, args.max_val_samples, cache_dir, cache_prefix
        )
        X_test, y_test = transform_features(
            "test", feature_pipeline, args.max_test_samples, cache_dir, cache_prefix
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Could not load data: {exc}")

    # --- 2. Tune hyperparameters on the validation split. --------------------
    estimator = build_estimator(args.model)
    search = tune_hyperparameters(
        estimator,
        MODEL_REGISTRY[args.model].param_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        scoring=args.scoring,
    )
    best_model = search.best_estimator_

    # --- 3. Evaluate: naive baseline first, then the tuned model. ------------
    baseline = baseline_metrics(X_train, y_train, X_test, y_test)
    model_metrics = evaluate(best_model, X_test, y_test, model_label=args.model)

    # Quick sanity check the model actually beats "always predict majority".
    if model_metrics["accuracy"] <= baseline["accuracy"]:
        logger.warning(
            "Model accuracy (%.4f) does NOT beat the naive baseline (%.4f).",
            model_metrics["accuracy"],
            baseline["accuracy"],
        )

    # --- 4. Save artifacts (fitted feature pipeline + model + metrics). ------
    report = {
        "model_name": args.model,
        "pipeline": args.pipeline,
        "pca_components": args.pca_components,
        "best_params": search.best_params_,
        "best_val_score": float(search.best_score_),
        "test_metrics": model_metrics,
        "baseline_metrics": baseline,
    }
    save_artifacts(best_model, feature_pipeline, report, output_dir, args.model)
    logger.info("Done.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse the command-line interface.

    All sample caps accept ``0`` to mean "use the entire split".
    """
    parser = argparse.ArgumentParser(
        description="Train a Soft-SVM (or other registered model) to classify "
        "images as real vs. fake.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="svm",
        help=f"Classifier to train. Choices: {sorted(MODEL_REGISTRY)}.",
    )
    parser.add_argument(
        "--pipeline",
        default="svm",
        help=f"Feature pipeline. Choices: {sorted(PIPELINE_REGISTRY)}.",
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=5000,
        help="Cap on training images (0 = all). Kept small so the kernel SVM "
        "is tractable; raise it if you have the compute.",
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=2000,
        help="Cap on validation images used for tuning (0 = all).",
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=5000,
        help="Cap on test images for evaluation (0 = all).",
    )
    parser.add_argument(
        "--pca-components", type=int, default=150,
        help="PCA target dimensionality (project ImagePipeline 'vec-pca' reduce, "
        "fit on train and reused on val/test).",
    )
    parser.add_argument(
        "--scoring", default="f1",
        help="Metric GridSearchCV optimizes on the validation split.",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directory for the saved model and metrics.",
    )
    parser.add_argument(
        "--cache-dir", default="outputs/feature_cache",
        help="Directory for the feature cache (set empty '' to disable).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
