"""
Feature standardization for the preprocessing pipeline.

This module provides the optional batch-level ``'scale'`` step. It standardizes
each feature *column* across the training samples — subtracting the per-feature
mean and dividing by the per-feature standard deviation — then reuses those same
statistics on held-out data.

Where it sits in the chain
--------------------------
``'scale'`` is a *batch-level* op like ``'reduce'`` (it must learn statistics
across multiple samples), and it normally runs *after* ``'reduce'`` so it
standardizes the final reduced feature vectors. It is the in-pipeline equivalent
of scikit-learn's ``StandardScaler`` placed right before a scale-sensitive
classifier (e.g. an SVM): conditioning per-feature magnitudes so no single
component dominates the kernel or decision boundary.

This is distinct from the per-image ``'normalize'`` step (:func:`normalize_image`),
which rescales the pixels *within one image* before vectorization. ``'normalize'``
equalizes images to one another; ``'scale'`` equalizes feature columns to one
another across the dataset. They operate on different axes and are complementary,
not redundant.

Batch-level fittable-op protocol
--------------------------------
Like :func:`reduce_dimensions`, :func:`standardize_features` implements the
small protocol :class:`ImagePipeline` uses for its batch-level steps::

    fit:    standardize_features(X, return_reducer=True)  -> (X_scaled, scaler)
    apply:  standardize_features(X, reducer=scaler)        -> X_scaled

so ``ImagePipeline.fit_transform`` learns the statistics on the training batch
and ``transform`` / ``process`` reuse them. The fitted object is a feature
standardizer; the kwargs are named ``reducer`` / ``return_reducer`` only to match
the shared batch-op protocol (the same names :func:`reduce_dimensions` uses).

Shape conventions
-----------------
This step works on flat feature vectors only:

    single sample   1D (n_features,)        -> returned 1D, sample axis dropped
    batch           2D (n_samples, n_features)

Fitting needs the *batch* form (multiple samples). A single 1D sample is only
valid together with a pre-fit ``reducer`` (the inference-time path). Place
``'scale'`` after ``'vectorize'`` / ``'reduce'`` so the input is already flat.

Pure NumPy — no scikit-learn dependency.
"""

from typing import Optional

import numpy as np


class _FeatureStandardizer:
    """Per-feature zero-mean / unit-variance scaler (pure NumPy).

    Mirrors the ``fit`` / ``transform`` / ``fit_transform`` surface of
    scikit-learn's ``StandardScaler`` closely enough for the pipeline's batch-op
    protocol, but without the dependency. The per-column statistics are learned
    on the fitted batch and reused on any later batch or single sample.
    """

    is_matrix = False

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None   # (n_features,)
        self.scale_: Optional[np.ndarray] = None  # (n_features,)

    def fit(self, X: np.ndarray) -> "_FeatureStandardizer":
        # X: (n_samples, n_features). Stats are computed in float64 for numerical
        # stability, then applied back to float32 features in transform().
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        self.mean_ = (
            X.mean(axis=0) if self.with_mean else np.zeros(n_features, dtype=np.float64)
        )
        if self.with_std:
            scale = X.std(axis=0)
            # Guard zero-variance columns: a 0 std would divide to inf/nan. Set
            # them to 1.0 (as sklearn's StandardScaler does), so a constant
            # feature is merely centered, never blown up.
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(n_features, dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # (n_samples, n_features) -> standardized (n_samples, n_features).
        standardized = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return standardized.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _apply_scaler(features: np.ndarray, scaler, return_reducer: bool):
    """Apply an already-fitted scaler, preserving single-vs-batch shape.

    A single 1D sample (``(n_features,)``) is wrapped to a one-row batch,
    transformed, and unwrapped back to 1D — the inference-time path matching how
    :func:`reduce_dimensions` handles a lone vector.
    """
    was_single = features.ndim == 1
    batch = features[np.newaxis, :] if was_single else features
    scaled = scaler.transform(batch)
    if was_single:
        scaled = scaled[0]
    return (scaled, scaler) if return_reducer else scaled


def standardize_features(
    features: np.ndarray,
    with_mean: bool = True,
    with_std: bool = True,
    reducer: Optional[object] = None,
    return_reducer: bool = False,
):
    """
    Standardize flat feature vectors to zero mean and unit variance per column.

    This is the batch-level ``'scale'`` step of an image preprocessing pipeline,
    applied to the flat feature matrix produced by ``'vectorize'`` / ``'reduce'``.
    It fits one mean and one standard deviation per feature on the training batch
    and reuses them on held-out data, so every split is scaled consistently.

    Args:
        features: The data to standardize. A 2D ``(n_samples, n_features)`` matrix
            to fit on, or a single 1D ``(n_features,)`` vector together with a
            pre-fit ``reducer`` (inference time).
        with_mean: If True (default), center each feature to zero mean.
        with_std: If True (default), scale each feature to unit variance.
            Zero-variance features keep a scale of 1.0 (only centered).
        reducer: A pre-fit scaler (e.g. one returned with ``return_reducer=True``).
            When given, ``with_mean`` / ``with_std`` are ignored and only
            ``reducer.transform(features)`` runs — the path for applying the
            training-time statistics to new data (val/test split, or a single
            vector at inference time). Named ``reducer`` to match the shared
            batch-op protocol used by :func:`reduce_dimensions`.
        return_reducer: If True, return ``(scaled_features, fitted_scaler)`` so
            the scaler can be reused. If False (default), return only the scaled
            features.

    Returns:
        ``scaled_features`` (or the ``(scaled_features, scaler)`` tuple when
        ``return_reducer=True``). A single 1D sample is returned 1D.

    Raises:
        ValueError: If asked to *fit* on a single 1D sample (no ``reducer``), or
            if the input is not 1D/2D.
    """
    # Pre-fit scaler: transform-only, regardless of with_mean/with_std.
    if reducer is not None:
        return _apply_scaler(features, reducer, return_reducer)

    if features.ndim == 1:
        raise ValueError(
            "standardize_features cannot be fit on a single 1D sample "
            "(n_samples=1). To use this step:\n"
            "  • supply a pre-fit `reducer` (recommended at inference time), or\n"
            "  • call batch_process / ImagePipeline.fit_transform on a batch."
        )
    if features.ndim != 2:
        raise ValueError(
            f"standardize_features expects a 2D (n_samples, n_features) matrix, "
            f"got shape {features.shape}. Place 'scale' after 'vectorize'/'reduce'."
        )

    scaler = _FeatureStandardizer(with_mean=with_mean, with_std=with_std).fit(features)
    scaled = scaler.transform(features)
    return (scaled, scaler) if return_reducer else scaled
