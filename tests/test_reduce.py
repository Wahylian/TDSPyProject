"""
Tests for batch-level dimensionality reduction in ``preprocessing/reduce.py``.

``reduce_dimensions`` selects both *which* reducer runs and *what shape* the
input is read as. These tests are organised by subgroup:

* :class:`TestVectorReduce` — the flat ``(n_samples, n_features)`` methods
  (``None`` bypass, ``vec-pca``, ``vec-jl``) plus method normalization.
* :class:`TestMatrixReduce` — the grayscale image-stack methods
  (``mat-pca`` / ``mat-jl``).
* :class:`TestMatrixReduceMultiChannel` — the colour image-stack path.

The fit-once/reuse "reducer" pattern (train then apply to held-out data) is
covered for both vector and matrix subgroups.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import reduce_dimensions


class TestVectorReduce:
    """Vector dimensionality reduction (None / ``vec-pca`` / ``vec-jl``).

    Covers the bypass no-op, both vector reduction methods, component clamping,
    method-name normalization, and the fit-once/reuse "reducer" pattern that
    keeps train and inference data projected into the same space.
    """

    def test_none_is_identity_passthrough(self, feature_matrix):
        """``method=None`` returns the exact same array object (no copy).

        Documented no-op: the bypass path must not allocate or transform, so
        identity (``is``) — not just equality — is asserted.
        """
        # Act + Assert
        assert reduce_dimensions(feature_matrix, method=None) is feature_matrix

    def test_none_string_is_identity_passthrough(self, feature_matrix):
        """The string ``'none'`` is treated as the bypass, same as ``None``.

        Ease-of-use: config files often carry the literal string ``'none'``;
        it must normalize to the no-op rather than being read as an unknown
        method name.
        """
        # Act + Assert
        assert reduce_dimensions(feature_matrix, method="none") is feature_matrix

    def test_none_with_return_reducer_yields_none_reducer(self, feature_matrix):
        """The bypass with ``return_reducer=True`` returns ``(features, None)``.

        The bypass fits nothing, so the reducer slot of the returned tuple must
        be ``None`` while the features pass through unchanged.
        """
        # Act
        reduced, reducer = reduce_dimensions(
            feature_matrix, method=None, return_reducer=True
        )
        # Assert: features untouched, no reducer produced.
        assert reduced is feature_matrix
        assert reducer is None

    def test_pca_reduces_to_requested_components(self, feature_matrix):
        """PCA reduces the feature axis to the requested component count."""
        # Act
        reduced = reduce_dimensions(feature_matrix, method="vec-pca", n_components=16)
        # Assert: sample count preserved, feature axis narrowed to 16, float32.
        assert reduced.shape == (feature_matrix.shape[0], 16)
        assert reduced.dtype == np.float32

    def test_jl_reduces_to_requested_components(self, feature_matrix):
        """Johnson-Lindenstrauss projection hits the requested width.

        The JL method is an alternative to PCA; this confirms it honours
        ``n_components`` and preserves the sample count.
        """
        # Act
        reduced = reduce_dimensions(
            feature_matrix, method="vec-jl", n_components=64
        )
        # Assert
        assert reduced.shape == (feature_matrix.shape[0], 64)

    def test_jl_auto_components_picks_a_valid_width(self, rng):
        """``vec-jl`` with ``n_components='auto'`` lets sklearn size the output.

        The JL lemma fixes a safe target width from the sample count and ``eps``;
        for that width to be valid it must not exceed the input feature count, so
        this uses a wide, few-sample matrix. The result keeps all samples and
        lands on a positive width below the original feature dimension.
        """
        # Arrange: few samples, many features so the JL min-dim fits within width.
        wide = rng.random((8, 3000)).astype(np.float32)
        # Act
        reduced = reduce_dimensions(wide, method="vec-jl", n_components="auto")
        # Assert: rows preserved and sklearn chose a sensible reduced width.
        assert reduced.shape[0] == 8
        assert 1 <= reduced.shape[1] <= 3000

    def test_pca_components_clamped_to_matrix_rank(self, rng):
        """Over-asking for components clamps to ``min(n_samples, n_features)``.

        Edge case: requesting more components than the matrix rank supports
        must clamp gracefully rather than surface an opaque sklearn error.
        """
        # Arrange: a 5x12 matrix whose rank caps PCA at 5 components.
        small = rng.random((5, 12)).astype(np.float32)
        # Act: deliberately request far more components than possible.
        reduced = reduce_dimensions(small, method="vec-pca", n_components=999)
        # Assert: clamped to the rank limit.
        assert reduced.shape[1] == min(5, 12)

    def test_shared_reducer_applies_same_projection(self, rng):
        """A fitted reducer projects held-out data into the training space.

        Validates the recommended train/test pattern: fit the reducer once on
        training data (``return_reducer=True``), then reuse it on test data so
        both end up in the same reduced subspace.
        """
        # Arrange: separate train and test matrices sharing the same feature width.
        train = rng.random((30, 100)).astype(np.float32)
        test = rng.random((7, 100)).astype(np.float32)
        # Act: fit on train (capturing the reducer), then apply to test.
        train_reduced, reducer = reduce_dimensions(
            train, method="vec-pca", n_components=8, return_reducer=True
        )
        test_reduced = reduce_dimensions(test, reducer=reducer)
        # Assert: both projected to width 8, sample counts preserved.
        assert train_reduced.shape == (30, 8)
        assert test_reduced.shape == (7, 8)

    def test_prefit_reducer_projects_single_vector(self, rng):
        """A pre-fit reducer projects a single 1D sample at inference time.

        Inference-time path: a fitted reducer applied to one 1D vector must
        return a 1D vector (not a (1, k) matrix), matching how single-image
        embeddings are reduced in production.
        """
        # Arrange: fit a reducer on a training batch.
        train = rng.random((30, 100)).astype(np.float32)
        _, reducer = reduce_dimensions(
            train, method="vec-pca", n_components=8, return_reducer=True
        )
        # Act: project a lone 1D sample through the fitted reducer.
        one = rng.random(100).astype(np.float32)
        projected = reduce_dimensions(one, reducer=reducer)
        # Assert: 1D in -> 1D out, width 8.
        assert projected.shape == (8,)  # 1D in -> 1D out

    def test_single_sample_without_reducer_raises(self, rng):
        """Fitting PCA on a single 1D vector raises.

        Edge case: PCA needs multiple samples to estimate variance, so fitting
        it on one lone vector (with no pre-fit reducer supplied) is
        mathematically undefined and must raise.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_dimensions(rng.random(100).astype(np.float32), method="vec-pca")

    def test_unknown_method_raises(self, feature_matrix):
        """An unsupported reduction method raises.

        Edge case: methods outside {None, "vec-pca", "vec-jl", "mat-pca",
        "mat-jl"} (e.g. "umap") are unimplemented and must fail loudly.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_dimensions(feature_matrix, method="umap")

    def test_non_string_method_raises_typeerror(self, feature_matrix):
        """A non-string, non-None ``method`` raises ``TypeError``.

        Edge case: ``method`` must be either ``None`` or a string; passing an
        int (e.g. a mistaken ``n_components`` in the wrong slot) is a type error
        and must be reported as such, not silently coerced.
        """
        # Act + Assert
        with pytest.raises(TypeError):
            reduce_dimensions(feature_matrix, method=123)

    def test_legacy_aliases_resolve_to_vector_methods(self, feature_matrix):
        """The terse aliases ``'pca'`` / ``'johnson_lindenstrauss'`` still work.

        Ease-of-use: ``'pca'`` resolves to ``'vec-pca'`` and the hyphen/underscore
        spellings of Johnson-Lindenstrauss resolve to ``'vec-jl'``, so existing
        configs keep working without spelling out the ``vec-`` prefix.
        """
        # Act: request reduction via the short aliases.
        via_pca = reduce_dimensions(feature_matrix, method="pca", n_components=8)
        via_jl = reduce_dimensions(
            feature_matrix, method="johnson_lindenstrauss", n_components=8
        )
        # Assert: both behave exactly like their canonical vec-* names.
        assert via_pca.shape == (feature_matrix.shape[0], 8)
        assert via_jl.shape == (feature_matrix.shape[0], 8)


class TestMatrixReduce:
    """Matrix dimensionality reduction (``mat-pca`` / ``mat-jl``).

    The matrix subgroup consumes a ``(n_samples, height, width)`` grayscale stack
    and reduces only the width axis, mapping each image to ``(height,
    n_components)`` so the row structure CNNs/ViTs rely on is preserved. These
    tests mirror the vector cases (shapes, clamping, the fit-once/reuse pattern,
    single-sample handling) for the grayscale matrix path. The multi-channel
    cases live in :class:`TestMatrixReduceMultiChannel`.
    """

    @pytest.mark.parametrize("method", ["mat-pca", "mat-jl"])
    def test_matrix_reduce_preserves_rows_and_narrows_columns(self, matrix_stack, method):
        """Both matrix methods keep (n_samples, height) and narrow the width.

        Args:
            matrix_stack: (24, 32, 32) float32 image stack fixture.
            method: matrix reduction method supplied by the parametrize sweep.
        """
        # Act
        reduced = reduce_dimensions(matrix_stack, method=method, n_components=8)
        # Assert: sample + row axes intact, column axis collapsed to 8, float32.
        assert reduced.shape == (matrix_stack.shape[0], matrix_stack.shape[1], 8)
        assert reduced.dtype == np.float32

    def test_none_bypass_on_matrix_stack_is_identity(self, matrix_stack):
        """``method=None`` returns the same 3D stack object untouched.

        The bypass must be shape-agnostic — it leaves an image stack exactly as
        it leaves a flat matrix, with no copy.
        """
        # Act + Assert
        assert reduce_dimensions(matrix_stack, method=None) is matrix_stack

    def test_mat_pca_components_clamped_to_width(self, matrix_stack):
        """Over-asking for components clamps to the image width.

        Edge case: requesting more columns than the image has must clamp to the
        width rather than surface an indexing error.
        """
        # Act: request far more components than the 32-wide images allow.
        reduced = reduce_dimensions(matrix_stack, method="mat-pca", n_components=999)
        # Assert: clamped to the width (32).
        assert reduced.shape[2] == matrix_stack.shape[2]

    def test_shared_matrix_reducer_applies_same_projection(self, rng):
        """A fitted matrix reducer projects held-out images into the same space.

        Validates the train/test pattern for the matrix path: fit once on a
        training stack, then reuse the reducer on a held-out stack of the same
        width so both land in the same reduced column space.
        """
        # Arrange: train/test stacks sharing height and width.
        train = rng.random((20, 16, 24)).astype(np.float32)
        test = rng.random((5, 16, 24)).astype(np.float32)
        # Act: fit on train (capturing the reducer), then transform test.
        train_reduced, reducer = reduce_dimensions(
            train, method="mat-pca", n_components=6, return_reducer=True
        )
        test_reduced = reduce_dimensions(test, reducer=reducer)
        # Assert: both reduced to width 6, sample + row axes preserved.
        assert train_reduced.shape == (20, 16, 6)
        assert test_reduced.shape == (5, 16, 6)

    def test_prefit_matrix_reducer_projects_single_image(self, rng):
        """A pre-fit matrix reducer projects a single 2D image at inference time.

        Inference-time path: a fitted matrix reducer applied to one 2D image
        must return a 2D ``(height, k)`` matrix (sample axis dropped), matching
        how a single CNN/ViT input is reduced in production.
        """
        # Arrange: fit a reducer on a training stack.
        train = rng.random((20, 16, 24)).astype(np.float32)
        _, reducer = reduce_dimensions(
            train, method="mat-jl", n_components=6, return_reducer=True
        )
        # Act: project one lone 2D image through the fitted reducer.
        one = rng.random((16, 24)).astype(np.float32)
        projected = reduce_dimensions(one, reducer=reducer)
        # Assert: 2D in -> 2D out, width reduced to 6.
        assert projected.shape == (16, 6)

    def test_single_matrix_without_reducer_raises(self, rng):
        """Fitting a matrix reducer on a single 2D image raises.

        Edge case: like its vector counterpart, a matrix reducer needs multiple
        samples to fit, so a lone 2D image (no pre-fit reducer) is undefined and
        must raise rather than silently misread the rows as a batch.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_dimensions(rng.random((16, 24)).astype(np.float32), method="mat-pca")

    def test_matrix_method_on_vector_matrix_raises(self, feature_matrix):
        """A matrix method given a flat 2D feature matrix raises.

        Edge case: a matrix method needs a 3D image stack; handing it the 2D
        ``(n_samples, n_features)`` output of ``vectorize`` is a pipeline
        misconfiguration and must fail loudly. (A bare 2D array is read as a
        single image sample, which cannot be fit without a reducer.)
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_dimensions(feature_matrix, method="mat-pca", n_components=4)


class TestMatrixReduceMultiChannel:
    """Matrix reduction on multi-channel (BGR/RGB) image stacks.

    CNNs and ViTs typically ingest colour tensors, so the matrix subgroup must
    accept a ``(n_samples, height, width, channels)`` stack and reduce only the
    width axis — collapsing each image to ``(height, n_components, channels)``
    while leaving the row axis and every colour channel intact. These tests pin
    that contract: output shapes, channel preservation, clamping, and the
    fit-once/reuse + single-image inference paths for colour input.
    """

    @pytest.mark.parametrize("method", ["mat-pca", "mat-jl"])
    def test_colour_reduce_narrows_width_keeps_channels(self, color_matrix_stack, method):
        """Both matrix methods narrow width and preserve (rows, channels).

        Args:
            color_matrix_stack: (24, 32, 32, 3) float32 colour image stack.
            method: matrix reduction method supplied by the parametrize sweep.
        """
        n, h, _, c = color_matrix_stack.shape
        # Act
        reduced = reduce_dimensions(color_matrix_stack, method=method, n_components=8)
        # Assert: sample/row/channel axes intact, width collapsed to 8, float32.
        assert reduced.shape == (n, h, 8, c)
        assert reduced.dtype == np.float32
        assert np.isfinite(reduced).all()

    def test_colour_channels_reduced_independently_of_each_other(self, rng):
        """A constant channel stays constant — width mixing never crosses channels.

        Builds a stack whose channel 0 is identically zero. Because the shared
        column basis is applied per channel (no cross-channel mixing), channel 0
        of the output must remain all zeros, proving channels are not blended.
        """
        # Arrange: 3-channel stack with channel 0 forced to zero.
        stack = rng.random((20, 12, 16, 3)).astype(np.float32)
        stack[..., 0] = 0.0
        # Act
        reduced = reduce_dimensions(stack, method="mat-jl", n_components=5)
        # Assert: the zero channel survives reduction untouched.
        assert np.allclose(reduced[..., 0], 0.0)
        assert reduced.shape == (20, 12, 5, 3)

    def test_colour_none_bypass_is_identity(self, color_matrix_stack):
        """``method=None`` returns the same 4D colour stack object untouched."""
        # Act + Assert
        assert reduce_dimensions(color_matrix_stack, method=None) is color_matrix_stack

    def test_colour_mat_pca_clamps_components_to_width(self, color_matrix_stack):
        """Over-asking for components clamps to the image width, channels intact."""
        n, h, w, c = color_matrix_stack.shape
        # Act: request more components than the 32-wide images allow.
        reduced = reduce_dimensions(color_matrix_stack, method="mat-pca", n_components=999)
        # Assert: width clamped to 32, channel axis preserved.
        assert reduced.shape == (n, h, w, c)

    def test_shared_colour_reducer_projects_single_colour_image(self, rng):
        """A reducer fit on a colour batch projects one lone colour image.

        Inference-time path for colour: a fitted reducer applied to a single
        ``(height, width, channels)`` image must return ``(height, k, channels)``
        (sample axis dropped) and agree with the batched result. The reducer must
        read the 3D input as one colour image, not as a grayscale batch.
        """
        # Arrange: fit on a colour training stack.
        train = rng.random((20, 16, 24, 3)).astype(np.float32)
        batch_reduced, reducer = reduce_dimensions(
            train, method="mat-pca", n_components=6, return_reducer=True
        )
        # Act: project a single colour image through the fitted reducer.
        one = train[0]                       # (16, 24, 3)
        projected = reduce_dimensions(one, reducer=reducer)
        # Assert: 3D in -> 3D out, width reduced to 6, matches the batch result.
        assert batch_reduced.shape == (20, 16, 6, 3)
        assert projected.shape == (16, 6, 3)
        assert np.allclose(projected, batch_reduced[0], atol=1e-5)

    def test_grayscale_reducer_rejects_colour_image(self, rng):
        """A grayscale-fit reducer applied to a colour image fails on the mismatch.

        A reducer fit on ``(n, h, w)`` learns a width-only basis; feeding it a
        ``(h, w, c)`` colour image is a shape mismatch and must raise rather than
        silently produce a wrong-shaped result.
        """
        # Arrange: reducer fit on a grayscale stack.
        gray = rng.random((20, 16, 24)).astype(np.float32)
        _, reducer = reduce_dimensions(
            gray, method="mat-pca", n_components=6, return_reducer=True
        )
        # Act + Assert: a colour image has the wrong width axis for this reducer.
        with pytest.raises(Exception):
            reduce_dimensions(rng.random((16, 24, 3)).astype(np.float32), reducer=reducer)
