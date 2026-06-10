"""
Dimensionality reduction for vectorized image features.

This module appends an optional dimensionality-reduction step to the end of the
preprocessing pipeline (after grayscaling and vectorization). Reducing the
feature dimension before training a traditional ML model can substantially
speed up training and may improve generalization when the original feature
vector is very high-dimensional (e.g. 150 528 raw RGB pixels).

Three modes are supported, selected via the ``method`` parameter:

- ``None`` (or the string ``'none'``) — **bypass**. The input is returned
  unchanged. This makes it safe to drop the ``'reduce'`` step into any
  existing pipeline as a no-op for backward compatibility.
- ``'pca'`` — Principal Component Analysis (``sklearn.decomposition.PCA``).
  A linear, *data-dependent* projection that maximises preserved variance.
  Requires a batch of samples to fit (``n_samples >= n_components``).
- ``'johnson_lindenstrauss'`` — random projection
  (``sklearn.random_projection.GaussianRandomProjection``). A *data-independent*
  projection whose distortion is bounded by the Johnson–Lindenstrauss lemma;
  it is much cheaper to fit than PCA and works well when the goal is to
  preserve pairwise distances rather than variance.

Why this lives in its own module:
    PCA / random-projection are inherently *batch-level* operations — they
    need a whole feature matrix to fit. Keeping them isolated from the
    per-image transforms in ``transforms.py`` makes the distinction explicit.

Why ``scikit-learn`` is lazy-imported:
    Users who set ``method=None`` (or never use ``'reduce'``) should not pay
    the import cost or be forced to install scikit-learn for the bypass case.

Backward compatibility:
    The new ``'reduce'`` pipeline op is appended at the end of pipelines. With
    ``method=None`` it is a perfect no-op, so every pipeline that existed
    before this change continues to produce bit-identical output.
"""

from typing import Optional, Union

import numpy as np


# Canonical method names. ``None`` is the documented default; the lower-case
# string aliases (``'none'``, ``'pca'``, ``'johnson_lindenstrauss'``) make the
# step easy to configure from YAML/JSON where Python's ``None`` cannot be
# expressed directly.
_VALID_METHODS = (None, 'none', 'pca', 'johnson_lindenstrauss')


def _normalize_method(method: Optional[str]) -> Optional[str]:
    """Map user-facing method values to a canonical internal form."""
    if method is None:
        return None
    if not isinstance(method, str):
        raise TypeError(
            f"reduce_dimensions: method must be None or str, got {type(method).__name__}"
        )
    # Case-insensitive match; also accept hyphenated 'johnson-lindenstrauss'.
    normalized = method.strip().lower().replace('-', '_')
    if normalized == 'none':
        return None
    if normalized not in ('pca', 'johnson_lindenstrauss'):
        raise ValueError(
            f"Unsupported reduction method: {method!r}. "
            f"Choose from: None, 'pca', 'johnson_lindenstrauss'."
        )
    return normalized


def _ensure_2d(features: np.ndarray) -> np.ndarray:
    """
    Promote a 1D feature vector to a (1, n_features) batch.

    PCA / random projection both expect a 2D ``(n_samples, n_features)``
    matrix. ``reduce_dimensions`` is callable from inside ``ImagePipeline``
    where a single 1D vector is the natural input, so we wrap it here.
    """
    if features.ndim == 1:
        return features.reshape(1, -1)
    if features.ndim == 2:
        return features
    raise ValueError(
        f"reduce_dimensions expects 1D or 2D input, got shape {features.shape}"
    )


def reduce_dimensions(
    features: np.ndarray,
    method: Optional[str] = None,
    n_components: Union[int, str] = 128,
    random_state: int = 42,
    reducer: Optional[object] = None,
    return_reducer: bool = False,
):
    """
    Reduce feature dimensionality using PCA, Johnson–Lindenstrauss, or a no-op bypass.

    This is intended to be the final step of an image preprocessing pipeline,
    applied *after* the image has been vectorized. It operates on a feature
    matrix of shape ``(n_samples, n_features)``. When called inside
    ``ImagePipeline.process(image)`` on a single 1D vector, it is treated as a
    pass-through unless an already-fitted ``reducer`` is supplied — see the
    ``Single-sample behaviour`` section below.

    Args:
        features: Feature matrix, shape ``(n_samples, n_features)``, or a single
            1D vector of shape ``(n_features,)``.
        method: One of ``None``, ``'pca'``, ``'johnson_lindenstrauss'``.
            ``None`` (the default) bypasses reduction and returns the input
            unchanged. String aliases ``'none'`` and ``'johnson-lindenstrauss'``
            are also accepted, case-insensitively.
        n_components: Target dimensionality. For PCA, must satisfy
            ``n_components <= min(n_samples, n_features)``. For JL, the special
            value ``'auto'`` lets sklearn pick a dimension that satisfies the
            Johnson–Lindenstrauss bound for the given sample count.
            Default: 128.
        random_state: Seed forwarded to PCA/JL so projections are reproducible.
            Default: 42.
        reducer: Optional pre-fit reducer (e.g. one returned from a previous
            call with ``return_reducer=True``). When supplied, this function
            skips fitting and only applies ``reducer.transform(features)``.
            Use this to apply the *same* projection that was fit on the
            training set to the test set, which is required to keep features
            comparable across splits.
        return_reducer: If True, returns ``(reduced_features, fitted_reducer)``
            so the reducer can be reused for transforming additional data
            (e.g. the test split). If False (default), returns just the
            reduced features.

    Returns:
        ``reduced_features``, or ``(reduced_features, reducer)`` if
        ``return_reducer=True``. When ``method`` is ``None``, ``reducer`` is
        ``None`` in the tuple variant. Single 1D inputs are returned as 1D;
        2D inputs are returned as 2D.

    Raises:
        ValueError: If ``method`` is not one of the supported values, or if
            the input shape is not 1D/2D, or if PCA is requested on a single
            sample without a pre-fit reducer.
        ImportError: If method requires scikit-learn and it is not installed.

    Single-sample behaviour:
        Fitting PCA/JL on a single 1D vector is mathematically meaningless
        (no covariance / no pairwise distances). Therefore, when this function
        is reached inside ``ImagePipeline.process(single_image)``:
          * ``method=None`` returns the vector unchanged (true no-op).
          * ``method='pca'`` / ``'johnson_lindenstrauss'`` without a pre-fit
            ``reducer`` raises ``ValueError`` with a helpful message pointing
            the caller at ``batch_process`` (which applies the reduction at
            the batch level).
          * If a pre-fit ``reducer`` is supplied, the single vector is
            projected with it — this is the supported path for applying a
            training-time PCA/JL to one new image at inference time.

    Examples:
        # 1. Bypass (None) — identity passthrough, no sklearn required.
        >>> X = np.random.rand(100, 50_000).astype(np.float32)
        >>> reduce_dimensions(X, method=None) is X  # same array, no copy
        True

        # 2. PCA on a batch.
        >>> X_train = np.random.rand(200, 4096).astype(np.float32)
        >>> X_train_reduced, reducer = reduce_dimensions(
        ...     X_train, method='pca', n_components=64, return_reducer=True
        ... )
        >>> X_train_reduced.shape
        (200, 64)

        # 3. Apply the SAME PCA to a held-out test batch (or single image).
        >>> X_test = np.random.rand(50, 4096).astype(np.float32)
        >>> X_test_reduced = reduce_dimensions(X_test, reducer=reducer)
        >>> X_test_reduced.shape
        (50, 64)

        # 4. Johnson–Lindenstrauss random projection.
        >>> X_jl = reduce_dimensions(X_train, method='johnson_lindenstrauss',
        ...                          n_components=256)
        >>> X_jl.shape
        (200, 256)
    """
    # --- Fast paths -------------------------------------------------------

    # If a pre-fit reducer was supplied, the method argument is effectively
    # implied by the reducer itself — just apply transform() and return.
    if reducer is not None:
        # We still need to handle 1D vs 2D shape so that callers get back
        # what they passed in.
        was_1d = features.ndim == 1
        features_2d = _ensure_2d(features)
        reduced = reducer.transform(features_2d).astype(np.float32, copy=False)
        if was_1d:
            reduced = reduced[0]
        return (reduced, reducer) if return_reducer else reduced

    # Bypass / None — no sklearn dependency, no work performed.
    canonical = _normalize_method(method)
    if canonical is None:
        return (features, None) if return_reducer else features

    # --- Reduction paths --------------------------------------------------
    # Both PCA and JL need a 2D matrix; promote 1D inputs and remember so we
    # can return the same shape the caller passed in.
    was_1d = features.ndim == 1
    features_2d = _ensure_2d(features)
    n_samples, n_features = features_2d.shape

    if was_1d:
        # Per-image call inside ImagePipeline without a pre-fit reducer:
        # this is the case where PCA/JL is mathematically undefined.
        raise ValueError(
            f"reduce_dimensions(method={method!r}) cannot be fit on a single "
            f"1D sample (n_samples=1). To use this pipeline step:\n"
            f"  • supply a pre-fit `reducer` (recommended at inference time), or\n"
            f"  • call batch_process(images, pipeline) — it applies 'reduce' "
            f"at the batch level after running per-image steps."
        )

    if canonical == 'pca':
        # sklearn is heavy; only import when actually needed.
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "PCA reduction requires scikit-learn. "
                "Install with: pip install scikit-learn"
            ) from exc

        # PCA cannot extract more components than min(n_samples, n_features).
        # Clamp the requested value so misconfigured pipelines fail gracefully
        # instead of with an opaque sklearn error.
        max_components = min(n_samples, n_features)
        target = n_components if isinstance(n_components, int) else max_components
        target = max(1, min(target, max_components))

        fitted = PCA(n_components=target, random_state=random_state)
        reduced = fitted.fit_transform(features_2d).astype(np.float32, copy=False)

    else:  # canonical == 'johnson_lindenstrauss'
        try:
            from sklearn.random_projection import GaussianRandomProjection
        except ImportError as exc:
            raise ImportError(
                "Johnson-Lindenstrauss reduction requires scikit-learn. "
                "Install with: pip install scikit-learn"
            ) from exc

        # 'auto' lets sklearn choose a dimension satisfying the JL bound for
        # the given sample count and a default eps=0.1. This is the standard
        # data-independent choice when the user doesn't have a fixed target.
        target: Union[int, str] = n_components if n_components != 'auto' else 'auto'
        if isinstance(target, int):
            # Gaussian projection can in principle go to any dimension, but
            # going *up* (target > n_features) is wasteful — clamp to make
            # the pipeline robust to misconfiguration.
            target = max(1, min(target, n_features))

        fitted = GaussianRandomProjection(
            n_components=target, random_state=random_state
        )
        reduced = fitted.fit_transform(features_2d).astype(np.float32, copy=False)

    if was_1d:
        reduced = reduced[0]

    return (reduced, fitted) if return_reducer else reduced
