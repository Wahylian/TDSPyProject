"""Batch-level 'scale' step: per-column standardization, pure NumPy.

Subtracts the per-feature mean and divides by the per-feature std learned on the
training batch, reusing those statistics on held-out data. Distinct from the
per-image 'normalize' step: 'normalize' equalizes pixels within one image,
'scale' equalizes feature columns across the dataset.

It normally runs after 'reduce' (the scikit-learn StandardScaler role before a
scale-sensitive classifier like an SVM) and follows the same batch-op protocol
as reduce_dimensions: fit with return_reducer=True, apply with reducer=scaler.
Works on flat vectors only; a single 1D sample is valid only with a pre-fit reducer.
"""

from typing import Optional

import numpy as np


class _FeatureStandardizer:
    """Per-feature zero-mean/unit-variance scaler, a dependency-free StandardScaler.

    Learns per-column statistics on the fitted batch and reuses them on later data.
    """

    is_matrix = False

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None   # (n_features,)
        self.scale_: Optional[np.ndarray] = None  # (n_features,)

    def fit(self, X: np.ndarray) -> "_FeatureStandardizer":
        # Compute stats in float64 for stability; transform() casts back to float32.
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        self.mean_ = (
            X.mean(axis=0) if self.with_mean else np.zeros(n_features, dtype=np.float64)
        )
        if self.with_std:
            scale = X.std(axis=0)
            # Zero-variance columns keep scale 1.0 (as sklearn does) so a constant
            # feature is only centered, never divided to inf/nan.
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(n_features, dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return standardized.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _apply_scaler(features: np.ndarray, scaler, return_reducer: bool):
    """Apply a fitted scaler, keeping single-vs-batch shape.

    A 1D sample is wrapped to a one-row batch and unwrapped back, matching how
    reduce_dimensions handles a lone vector.
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
    """Standardize flat feature vectors to per-column zero mean and unit variance.

    Fits one mean/std per feature on a 2D (n_samples, n_features) batch and, with
    return_reducer=True, returns the fitted scaler for reuse. Pass a pre-fit
    reducer to transform new data (val/test or a single 1D vector) with the same
    statistics; with_mean/with_std are then ignored. Fitting requires a batch.
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
