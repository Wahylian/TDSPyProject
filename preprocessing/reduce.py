"""
Dimensionality reduction for image features.

This module provides the optional ``'reduce'`` step of the preprocessing
pipeline. It compresses the feature representation produced by the earlier
stages so downstream training is cheaper and often generalises better when the
raw representation is very high-dimensional (e.g. 150 528 raw RGB pixels).

Two subgroups of reduction are exposed, selected via the ``method`` argument.
They differ in the *shape* of data they consume, which lets the same pipeline
serve both classical vector models and matrix-native models (eg: CNNs, ViTs):

Vector reduction — operates on flat feature vectors
    Input is a ``(n_samples, n_features)`` matrix (one flat vector per image,
    as produced by the ``'vectorize'`` step).

    - ``'vec-pca'`` — Principal Component Analysis
      (``sklearn.decomposition.PCA``). A linear, *data-dependent* projection
      that maximises preserved variance. Needs ``n_samples >= n_components``.
    - ``'vec-jl'`` — Johnson–Lindenstrauss random projection
      (``sklearn.random_projection.GaussianRandomProjection``). A
      *data-independent* projection whose distortion is bounded by the
      Johnson–Lindenstrauss lemma; trivially cheap to fit.

Matrix reduction — operates on image matrices, preserving spatial + channel structure
    Input is a stack of image matrices, either single-channel (grayscale) or
    multi-channel (BGR/RGB) — the natural output when ``'vectorize'`` is
    omitted:

        grayscale  ``(n_samples, height, width)``
        colour     ``(n_samples, height, width, channels)``

    Only the width (column) axis is projected, so each image collapses from
    ``(height, width[, channels])`` to ``(height, k[, channels])``. The row
    axis CNNs/ViTs rely on **and** the colour channels are both left intact.

    - ``'mat-pca'`` — two-dimensional PCA. Builds a single column-covariance
      matrix (pooled across the batch and, for colour input, across channels)
      and projects the column axis onto its leading eigenvectors, so one shared
      basis is applied to every channel. Pure NumPy; no scikit-learn required.
    - ``'mat-jl'`` — two-dimensional Gaussian random projection of the column
      axis, with the same projection shared across channels. Data-independent
      and pure NumPy.

``None`` (or the string ``'none'``) is a **bypass**: the input is returned
unchanged, so dropping ``('reduce', {'method': None})`` into any pipeline is a
no-op slot kept available for easy A/B comparison.

Shape conventions
-----------------
The chosen ``method`` disambiguates how the input's dimensionality is read:

    method group   single sample            batch
    -------------   ----------------------   ---------------------------------
    vector          1D (n_features,)         2D (n_samples, n_features)
    matrix (gray)   2D (height, width)       3D (n_samples, height, width)
    matrix (colour) 3D (height, width, ch)   4D (n_samples, height, width, ch)

Fitting any reducer needs the *batch* form (multiple samples). A single sample
is only valid together with a pre-fit ``reducer`` (the inference-time path);
the fitted reducer remembers whether it was trained on grayscale or colour and
reads single-sample rank accordingly.

Lazy imports
------------
scikit-learn is imported only inside the ``vec-*`` branches, so the bypass and
matrix paths carry no scikit-learn dependency.
"""

from typing import Optional, Union

import numpy as np


# Canonical method names grouped by the data shape they consume.
_VECTOR_METHODS = frozenset({'vec-pca', 'vec-jl'})
_MATRIX_METHODS = frozenset({'mat-pca', 'mat-jl'})

# User-facing spellings mapped to a canonical name. Hyphen/underscore and case
# are normalised before lookup, so 'Vec_PCA', 'pca' and 'johnson_lindenstrauss'
# all resolve here. The short aliases keep configuration terse.
_METHOD_ALIASES = {
    'pca': 'vec-pca',
    'vec-pca': 'vec-pca',
    'jl': 'vec-jl',
    'vec-jl': 'vec-jl',
    'johnson-lindenstrauss': 'vec-jl',
    'mat-pca': 'mat-pca',
    'mat-jl': 'mat-jl',
}


def _normalize_method(method: Optional[str]) -> Optional[str]:
    """Map a user-facing ``method`` value to its canonical internal name."""
    if method is None:
        return None
    if not isinstance(method, str):
        raise TypeError(
            f"reduce_dimensions: method must be None or str, got {type(method).__name__}"
        )
    key = method.strip().lower().replace('_', '-')
    if key == 'none':
        return None
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"Unsupported reduction method: {method!r}. Choose from: "
            f"None, 'vec-pca', 'vec-jl', 'mat-pca', 'mat-jl'."
        )
    return _METHOD_ALIASES[key]


# ---------------------------------------------------------------------------
# Matrix reducers — pure NumPy, reusable across train/test like sklearn objects
# ---------------------------------------------------------------------------

def _project_width(X: np.ndarray, components: np.ndarray, has_channels: bool) -> np.ndarray:
    """
    Project the width (column) axis of ``X`` onto ``components`` (width, k).

    Handles a single image or a batch, grayscale or multi-channel, by routing on
    where the width axis sits:

    - grayscale: width is the last axis, ``(..., width)`` — a plain matmul
      ``(..., width) @ (width, k)`` broadcasts over any leading sample/row axes.
    - multi-channel: width is the second-to-last axis, ``(..., width, channels)``
      — an einsum projects width while leaving the trailing channel axis in
      place, giving ``(..., k, channels)``.
    """
    if has_channels:
        # 'w' = width (contracted), 'k' = reduced width, 'c' = channel (kept);
        # leading '...' absorbs the optional sample axis and the row axis.
        return np.einsum('...wc,wk->...kc', X, components).astype(np.float32, copy=False)
    return (X @ components).astype(np.float32, copy=False)


class _Matrix2DPCA:
    """
    Two-dimensional PCA over a batch of image matrices (grayscale or colour).

    Projects the width (column) axis of each image onto the leading eigenvectors
    of the batch column-covariance matrix. A grayscale image ``(height, width)``
    maps to ``(height, n_components)``; a colour image ``(height, width,
    channels)`` maps to ``(height, n_components, channels)``. The row axis is
    left untouched and, for colour, the single shared basis is applied to every
    channel (the covariance is pooled across channels at fit time).

    Exposes the small ``fit`` / ``transform`` / ``fit_transform`` surface that
    :func:`reduce_dimensions` relies on, so a fitted instance can be reused on
    held-out data exactly like a scikit-learn reducer.
    """

    is_matrix = True

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None  # (width, n_components)
        # Set at fit time; lets transform/inference read single-sample rank.
        self.expects_channels_ = False

    def fit(self, X: np.ndarray) -> "_Matrix2DPCA":
        # X: (n_samples, height, width) grayscale or (n_samples, height, width,
        # channels) colour. Width is always axis 2 in both layouts.
        self.expects_channels_ = X.ndim == 4
        n_samples = X.shape[0]
        width = X.shape[2]
        centered = X.astype(np.float64) - X.mean(axis=0, dtype=np.float64)
        # Column-covariance matrix (width, width): average of Aᵀ·A over the
        # batch. For colour we contract the channel axis 'c' as well, pooling
        # all channels into one shared basis. The 1/n_samples scaling does not
        # affect eigenvector directions; it only keeps magnitudes sane.
        if self.expects_channels_:
            covariance = np.einsum('nhic,nhjc->ij', centered, centered) / n_samples
        else:
            covariance = np.einsum('nhi,nhj->ij', centered, centered) / n_samples
        # eigh returns eigenvalues ascending; take the top-k columns.
        _, eigenvectors = np.linalg.eigh(covariance)
        k = max(1, min(self.n_components, width))
        self.components_ = eigenvectors[:, ::-1][:, :k].astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # (..., height, width[, channels]) -> (..., height, k[, channels]).
        return _project_width(X, self.components_, self.expects_channels_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class _Matrix2DRandomProjection:
    """
    Two-dimensional Gaussian random projection of image matrices.

    Multiplies the width axis of each image by a fixed Gaussian matrix
    ``(width, n_components)``. A grayscale image ``(height, width)`` becomes
    ``(height, n_components)``; a colour image ``(height, width, channels)``
    becomes ``(height, n_components, channels)`` with the same projection shared
    across channels. The projection is data-independent (drawn once at ``fit``
    time from ``random_state``) so the same matrix transforms any later batch.
    """

    is_matrix = True

    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.components_: Optional[np.ndarray] = None  # (width, n_components)
        # Set at fit time; lets transform/inference read single-sample rank.
        self.expects_channels_ = False

    def fit(self, X: np.ndarray) -> "_Matrix2DRandomProjection":
        # Width is axis 2 for both (n, h, w) and (n, h, w, c) layouts.
        self.expects_channels_ = X.ndim == 4
        width = X.shape[2]
        k = max(1, min(self.n_components, width))
        rng = np.random.default_rng(self.random_state)
        # 1/sqrt(k) scaling keeps projected norms in expectation comparable to
        # the input — the standard Gaussian random-projection normalisation.
        self.components_ = (
            rng.standard_normal((width, k)) / np.sqrt(k)
        ).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # (..., height, width[, channels]) -> (..., height, k[, channels]).
        return _project_width(X, self.components_, self.expects_channels_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Fitting helpers (one per subgroup)
# ---------------------------------------------------------------------------

def _fit_vector(features, canonical, n_components, random_state):
    """Fit + apply a vector reducer to a ``(n_samples, n_features)`` matrix."""
    if features.ndim == 1:
        raise ValueError(
            f"reduce_dimensions(method={canonical!r}) cannot be fit on a single "
            f"1D sample (n_samples=1). To use this step:\n"
            f"  • supply a pre-fit `reducer` (recommended at inference time), or\n"
            f"  • call batch_process(images, pipeline) — it applies 'reduce' "
            f"at the batch level after running the per-image steps."
        )
    if features.ndim != 2:
        raise ValueError(
            f"Vector reduction ({canonical!r}) expects a 2D "
            f"(n_samples, n_features) matrix, got shape {features.shape}."
        )

    n_samples, n_features = features.shape

    if canonical == 'vec-pca':
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "vec-pca reduction requires scikit-learn. "
                "Install with: pip install scikit-learn"
            ) from exc
        # PCA cannot extract more components than min(n_samples, n_features);
        # clamp so a misconfigured request fails gracefully, not opaquely.
        max_components = min(n_samples, n_features)
        target = n_components if isinstance(n_components, int) else max_components
        target = max(1, min(target, max_components))
        fitted = PCA(n_components=target, random_state=random_state)
        reduced = fitted.fit_transform(features).astype(np.float32, copy=False)
        return reduced, fitted

    # canonical == 'vec-jl'
    try:
        from sklearn.random_projection import GaussianRandomProjection
    except ImportError as exc:
        raise ImportError(
            "vec-jl reduction requires scikit-learn. "
            "Install with: pip install scikit-learn"
        ) from exc
    # 'auto' lets sklearn pick a width satisfying the JL bound for the sample
    # count; an explicit int is clamped to n_features (projecting up is wasteful).
    target: Union[int, str] = n_components if n_components != 'auto' else 'auto'
    if isinstance(target, int):
        target = max(1, min(target, n_features))
    fitted = GaussianRandomProjection(n_components=target, random_state=random_state)
    reduced = fitted.fit_transform(features).astype(np.float32, copy=False)
    return reduced, fitted


def _fit_matrix(features, canonical, n_components, random_state):
    """Fit + apply a matrix reducer to a grayscale or colour image stack.

    Accepts a 3D ``(n_samples, height, width)`` grayscale stack or a 4D
    ``(n_samples, height, width, channels)`` colour stack.
    """
    if features.ndim == 2:
        raise ValueError(
            f"reduce_dimensions(method={canonical!r}) cannot be fit on a single "
            f"image matrix (n_samples=1). To use this step:\n"
            f"  • supply a pre-fit `reducer` (recommended at inference time), or\n"
            f"  • call batch_process(images, pipeline) — it applies 'reduce' "
            f"at the batch level after running the per-image steps."
        )
    if features.ndim not in (3, 4):
        raise ValueError(
            f"Matrix reduction ({canonical!r}) expects a 3D "
            f"(n_samples, height, width) grayscale stack or a 4D "
            f"(n_samples, height, width, channels) colour stack, got shape "
            f"{features.shape}. Omit 'vectorize' so each image stays a matrix."
        )

    if canonical == 'mat-pca':
        fitted = _Matrix2DPCA(n_components).fit(features)
    else:  # canonical == 'mat-jl'
        fitted = _Matrix2DRandomProjection(n_components, random_state).fit(features)

    return fitted.transform(features), fitted


def _apply_reducer(features, reducer, return_reducer):
    """Apply an already-fitted reducer, preserving single-vs-batch shape."""
    # Single-sample rank depends on the reducer kind:
    #   vector          -> 1D (n_features,)
    #   matrix grayscale -> 2D (height, width)
    #   matrix colour    -> 3D (height, width, channels)
    # The matrix reducer recorded `expects_channels_` at fit time, so a colour
    # reducer correctly treats a 3D array as one image, not a grayscale batch.
    if getattr(reducer, 'is_matrix', False):
        single_ndim = 3 if getattr(reducer, 'expects_channels_', False) else 2
    else:
        single_ndim = 1
    was_single = features.ndim == single_ndim
    batch = features[np.newaxis, ...] if was_single else features
    reduced = reducer.transform(batch).astype(np.float32, copy=False)
    if was_single:
        reduced = reduced[0]
    return (reduced, reducer) if return_reducer else reduced


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def reduce_dimensions(
    features: np.ndarray,
    method: Optional[str] = None,
    n_components: Union[int, str] = 128,
    random_state: int = 42,
    reducer: Optional[object] = None,
    return_reducer: bool = False,
):
    """
    Reduce feature dimensionality with a vector, matrix, or bypass method.

    This is the final ``'reduce'`` step of an image preprocessing pipeline. The
    ``method`` selects both *which* reducer runs and *what shape* the input is
    read as (see the module docstring's shape table):

    - Vector methods (``'vec-pca'``, ``'vec-jl'``) take a flat feature matrix
      ``(n_samples, n_features)`` and return ``(n_samples, n_components)``.
    - Matrix methods (``'mat-pca'``, ``'mat-jl'``) take an image stack —
      ``(n_samples, height, width)`` grayscale or ``(n_samples, height, width,
      channels)`` colour — and return ``(n_samples, height, n_components)`` or
      ``(n_samples, height, n_components, channels)`` respectively, preserving
      the row axis (and colour channels) for CNN/ViT inputs.
    - ``None`` returns the input unchanged.

    Args:
        features: The data to reduce. For vector methods, a 2D
            ``(n_samples, n_features)`` matrix (or a single 1D vector). For
            matrix methods, a 3D ``(n_samples, height, width)`` grayscale stack
            or a 4D ``(n_samples, height, width, channels)`` colour stack (or a
            single ``(height, width)`` / ``(height, width, channels)`` image).
        method: One of ``None``, ``'vec-pca'``, ``'vec-jl'``, ``'mat-pca'``,
            ``'mat-jl'``. Matching is case-insensitive and hyphen/underscore
            agnostic; the aliases ``'none'``, ``'pca'`` (→ ``'vec-pca'``),
            ``'jl'`` / ``'johnson-lindenstrauss'`` (→ ``'vec-jl'``) are accepted.
        n_components: Target reduced width. For PCA it is clamped to the rank
            limit ``min(n_samples, n_features)``; for the random projections it
            is clamped to the input width. ``vec-jl`` also accepts ``'auto'``.
            Default: 128.
        random_state: Seed forwarded to PCA / the random projections so results
            are reproducible. Default: 42.
        reducer: A pre-fit reducer (e.g. one returned with
            ``return_reducer=True``). When given, ``method``/``n_components`` are
            ignored and only ``reducer.transform(features)`` runs — the path for
            applying a training-time projection to new data (test split or a
            single image at inference time).
        return_reducer: If True, return ``(reduced_features, fitted_reducer)``
            so the reducer can be reused; the reducer is ``None`` for the bypass
            method. If False (default), return only the reduced features.

    Returns:
        ``reduced_features`` (or the ``(reduced_features, reducer)`` tuple when
        ``return_reducer=True``). A single sample passed in (1D for vector, 2D
        for matrix) is returned with its sample axis dropped.

    Raises:
        TypeError: If ``method`` is neither ``None`` nor a string.
        ValueError: If ``method`` is unsupported, the input rank does not match
            the chosen method, or a reducer is fit on a single sample.
        ImportError: If a ``vec-*`` method is requested without scikit-learn.

    Examples:
        # Bypass — identity passthrough, no scikit-learn needed.
        >>> X = np.random.rand(100, 50_000).astype(np.float32)
        >>> reduce_dimensions(X, method=None) is X
        True

        # Vector PCA on a batch, reused on a held-out split.
        >>> X_train = np.random.rand(200, 4096).astype(np.float32)
        >>> X_train_r, reducer = reduce_dimensions(
        ...     X_train, method='vec-pca', n_components=64, return_reducer=True)
        >>> X_train_r.shape
        (200, 64)
        >>> X_test = np.random.rand(50, 4096).astype(np.float32)
        >>> reduce_dimensions(X_test, reducer=reducer).shape
        (50, 64)

        # Matrix 2D-PCA on a stack of grayscale images.
        >>> imgs = np.random.rand(80, 128, 128).astype(np.float32)
        >>> reduce_dimensions(imgs, method='mat-pca', n_components=32).shape
        (80, 128, 32)

        # Matrix 2D-PCA on a stack of colour (BGR/RGB) images — channels kept.
        >>> rgb = np.random.rand(80, 128, 128, 3).astype(np.float32)
        >>> reduce_dimensions(rgb, method='mat-pca', n_components=32).shape
        (80, 128, 32, 3)
    """
    # Pre-fit reducer: transform-only, regardless of method.
    if reducer is not None:
        return _apply_reducer(features, reducer, return_reducer)

    canonical = _normalize_method(method)

    # Bypass — no work, no dependency.
    if canonical is None:
        return (features, None) if return_reducer else features

    if canonical in _VECTOR_METHODS:
        reduced, fitted = _fit_vector(features, canonical, n_components, random_state)
    else:
        reduced, fitted = _fit_matrix(features, canonical, n_components, random_state)

    return (reduced, fitted) if return_reducer else reduced
