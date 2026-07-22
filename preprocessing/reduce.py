"""Optional 'reduce' step: compress image features so training is cheaper.

The method selects both the reducer and how the input shape is read:

- Vector methods ('vec-pca', 'vec-jl') take a flat (n_samples, n_features)
  matrix. vec-pca is a data-dependent PCA (needs n_samples >= n_components);
  vec-jl is a data-independent Johnson-Lindenstrauss random projection.
- Matrix methods ('mat-pca', 'mat-jl') take an image stack, grayscale
  (n_samples, height, width) or colour (..., channels), and project only the
  width axis so the row axis and channels stay intact for CNNs/ViTs. Both are
  pure NumPy with one basis shared across channels.
- None (or 'none') is a bypass returning the input unchanged.

Fitting needs a batch; a single sample is valid only with a pre-fit reducer,
which records whether it was trained on grayscale or colour. scikit-learn is
imported only in the vec-* branches, so the bypass and matrix paths avoid it.
"""

from typing import Optional, Union

import numpy as np


# Canonical method names grouped by the data shape they consume.
_VECTOR_METHODS = frozenset({'vec-pca', 'vec-jl'})
_MATRIX_METHODS = frozenset({'mat-pca', 'mat-jl'})

# User-facing spellings to canonical names. Case and hyphen/underscore are
# normalized before lookup, so 'Vec_PCA', 'pca', 'johnson_lindenstrauss' all resolve.
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


# Matrix reducers: pure NumPy, reusable across train/test like sklearn objects.

def _project_width(X: np.ndarray, components: np.ndarray, has_channels: bool) -> np.ndarray:
    """Project the width axis of X onto components (width, k), grayscale or colour.

    Grayscale has width last, so a plain matmul broadcasts over leading axes;
    colour has width second-to-last, so an einsum keeps the trailing channel axis.
    """
    if has_channels:
        # w=width (contracted), k=reduced width, c=channel (kept); '...' absorbs sample/row axes.
        return np.einsum('...wc,wk->...kc', X, components).astype(np.float32, copy=False)
    return (X @ components).astype(np.float32, copy=False)


class _Matrix2DPCA:
    """2D PCA over a batch of image matrices, projecting the width axis.

    Projects width onto the leading eigenvectors of the batch column-covariance,
    leaving the row axis untouched. For colour the covariance is pooled across
    channels so one shared basis applies to all. Exposes the fit/transform/
    fit_transform surface reduce_dimensions reuses on held-out data.
    """

    is_matrix = True

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None  # (width, n_components)
        # Set at fit time; lets transform/inference read single-sample rank.
        self.expects_channels_ = False

    def fit(self, X: np.ndarray) -> "_Matrix2DPCA":
        # Width is axis 2 for both (n, h, w) and (n, h, w, c) layouts.
        self.expects_channels_ = X.ndim == 4
        n_samples = X.shape[0]
        width = X.shape[2]
        centered = X.astype(np.float64) - X.mean(axis=0, dtype=np.float64)
        # Column-covariance (width, width): mean of Aᵀ·A over the batch, contracting
        # the channel axis too for colour. The 1/n_samples factor only tames magnitudes.
        if self.expects_channels_:
            covariance = np.einsum('nhic,nhjc->ij', centered, centered) / n_samples
        else:
            covariance = np.einsum('nhi,nhj->ij', centered, centered) / n_samples
        # eigh returns eigenvalues ascending; reverse to take the top-k.
        _, eigenvectors = np.linalg.eigh(covariance)
        k = max(1, min(self.n_components, width))
        self.components_ = eigenvectors[:, ::-1][:, :k].astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _project_width(X, self.components_, self.expects_channels_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class _Matrix2DRandomProjection:
    """2D Gaussian random projection of the width axis of image matrices.

    Multiplies width by a fixed Gaussian matrix drawn once at fit time from
    random_state, so the same data-independent projection transforms any later
    batch. Colour shares one projection across channels.
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
        # 1/sqrt(k) is the standard normalization keeping projected norms comparable.
        self.components_ = (
            rng.standard_normal((width, k)) / np.sqrt(k)
        ).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _project_width(X, self.components_, self.expects_channels_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# Fitting helpers, one per subgroup.

def _fit_vector(features, canonical, n_components, random_state):
    """Fit and apply a vector reducer to a (n_samples, n_features) matrix."""
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
        # PCA rank limit is min(n_samples, n_features); clamp to fail gracefully.
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
    # 'auto' lets sklearn pick a JL-bound width; an int is clamped to n_features.
    target: Union[int, str] = n_components if n_components != 'auto' else 'auto'
    if isinstance(target, int):
        target = max(1, min(target, n_features))
    fitted = GaussianRandomProjection(n_components=target, random_state=random_state)
    reduced = fitted.fit_transform(features).astype(np.float32, copy=False)
    return reduced, fitted


def _fit_matrix(features, canonical, n_components, random_state):
    """Fit and apply a matrix reducer to a 3D grayscale or 4D colour image stack."""
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
    """Apply a fitted reducer, keeping single-vs-batch shape."""
    # Single-sample rank: 1D vector, 2D grayscale, 3D colour. The reducer's
    # recorded expects_channels_ lets a colour reducer read a 3D array as one image.
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


def reduce_dimensions(
    features: np.ndarray,
    method: Optional[str] = None,
    n_components: Union[int, str] = 128,
    random_state: int = 42,
    reducer: Optional[object] = None,
    return_reducer: bool = False,
):
    """Reduce feature dimensionality via a vector, matrix, or bypass method.

    method picks both the reducer and how the input rank is read (see module
    docstring): vec-* on a flat (n_samples, n_features) matrix, mat-* on a
    grayscale/colour image stack (row axis and channels preserved), None as a
    passthrough. n_components is clamped to the rank/width limit; vec-jl also
    accepts 'auto'. Pass a pre-fit reducer to transform new data with a
    training-time projection (method/n_components then ignored), or
    return_reducer=True to get the fitted reducer back for reuse. Fitting
    requires a batch; a single sample is returned with its sample axis dropped.
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
