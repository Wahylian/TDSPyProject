"""Tests for batch-level dimensionality reduction in preprocessing/reduce.py.

Organised by subgroup: vector methods (None, vec-pca, vec-jl) plus method
normalization, grayscale matrix methods (mat-pca/mat-jl), and the colour
matrix path. The fit-once/reuse reducer pattern is covered for each.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import reduce_dimensions


class TestVectorReduce:
    """Vector reduction (None / vec-pca / vec-jl), clamping, aliases, reuse."""

    def test_none_is_identity_passthrough(self, feature_matrix):
        """method=None returns the exact same array object (no copy)."""
        assert reduce_dimensions(feature_matrix, method=None) is feature_matrix

    def test_none_string_is_identity_passthrough(self, feature_matrix):
        """The string 'none' is treated as the bypass, same as None."""
        assert reduce_dimensions(feature_matrix, method="none") is feature_matrix

    def test_none_with_return_reducer_yields_none_reducer(self, feature_matrix):
        """The bypass with return_reducer=True returns (features, None)."""
        reduced, reducer = reduce_dimensions(
            feature_matrix, method=None, return_reducer=True
        )
        assert reduced is feature_matrix
        assert reducer is None

    def test_pca_reduces_to_requested_components(self, feature_matrix):
        """PCA reduces the feature axis to the requested component count."""
        reduced = reduce_dimensions(feature_matrix, method="vec-pca", n_components=16)
        assert reduced.shape == (feature_matrix.shape[0], 16)
        assert reduced.dtype == np.float32

    def test_jl_reduces_to_requested_components(self, feature_matrix):
        """Johnson-Lindenstrauss projection hits the requested width."""
        reduced = reduce_dimensions(
            feature_matrix, method="vec-jl", n_components=64
        )
        assert reduced.shape == (feature_matrix.shape[0], 64)

    def test_jl_auto_components_picks_a_valid_width(self, rng):
        """vec-jl with n_components='auto' lets sklearn size the output."""
        wide = rng.random((8, 3000)).astype(np.float32)
        reduced = reduce_dimensions(wide, method="vec-jl", n_components="auto")
        assert reduced.shape[0] == 8
        assert 1 <= reduced.shape[1] <= 3000

    def test_pca_components_clamped_to_matrix_rank(self, rng):
        """Over-asking clamps to min(n_samples, n_features)."""
        small = rng.random((5, 12)).astype(np.float32)
        reduced = reduce_dimensions(small, method="vec-pca", n_components=999)
        assert reduced.shape[1] == min(5, 12)

    def test_shared_reducer_applies_same_projection(self, rng):
        """A fitted reducer projects held-out data into the training space."""
        train = rng.random((30, 100)).astype(np.float32)
        test = rng.random((7, 100)).astype(np.float32)
        train_reduced, reducer = reduce_dimensions(
            train, method="vec-pca", n_components=8, return_reducer=True
        )
        test_reduced = reduce_dimensions(test, reducer=reducer)
        assert train_reduced.shape == (30, 8)
        assert test_reduced.shape == (7, 8)

    def test_prefit_reducer_projects_single_vector(self, rng):
        """A pre-fit reducer projects a single 1D sample, returning 1D."""
        train = rng.random((30, 100)).astype(np.float32)
        _, reducer = reduce_dimensions(
            train, method="vec-pca", n_components=8, return_reducer=True
        )
        one = rng.random(100).astype(np.float32)
        projected = reduce_dimensions(one, reducer=reducer)
        assert projected.shape == (8,)

    def test_single_sample_without_reducer_raises(self, rng):
        """Fitting PCA on a single 1D vector raises (variance undefined)."""
        with pytest.raises(ValueError):
            reduce_dimensions(rng.random(100).astype(np.float32), method="vec-pca")

    def test_unknown_method_raises(self, feature_matrix):
        """An unsupported reduction method raises."""
        with pytest.raises(ValueError):
            reduce_dimensions(feature_matrix, method="umap")

    def test_non_string_method_raises_typeerror(self, feature_matrix):
        """A non-string, non-None method raises TypeError."""
        with pytest.raises(TypeError):
            reduce_dimensions(feature_matrix, method=123)

    def test_legacy_aliases_resolve_to_vector_methods(self, feature_matrix):
        """The aliases 'pca' / 'johnson_lindenstrauss' resolve to the vec-* names."""
        via_pca = reduce_dimensions(feature_matrix, method="pca", n_components=8)
        via_jl = reduce_dimensions(
            feature_matrix, method="johnson_lindenstrauss", n_components=8
        )
        assert via_pca.shape == (feature_matrix.shape[0], 8)
        assert via_jl.shape == (feature_matrix.shape[0], 8)


class TestMatrixReduce:
    """Grayscale matrix reduction (mat-pca / mat-jl), reducing only the width axis."""

    @pytest.mark.parametrize("method", ["mat-pca", "mat-jl"])
    def test_matrix_reduce_preserves_rows_and_narrows_columns(self, matrix_stack, method):
        """Both matrix methods keep (n_samples, height) and narrow the width."""
        reduced = reduce_dimensions(matrix_stack, method=method, n_components=8)
        assert reduced.shape == (matrix_stack.shape[0], matrix_stack.shape[1], 8)
        assert reduced.dtype == np.float32

    def test_none_bypass_on_matrix_stack_is_identity(self, matrix_stack):
        """method=None returns the same 3D stack object untouched."""
        assert reduce_dimensions(matrix_stack, method=None) is matrix_stack

    def test_mat_pca_components_clamped_to_width(self, matrix_stack):
        """Over-asking for components clamps to the image width."""
        reduced = reduce_dimensions(matrix_stack, method="mat-pca", n_components=999)
        assert reduced.shape[2] == matrix_stack.shape[2]

    def test_shared_matrix_reducer_applies_same_projection(self, rng):
        """A fitted matrix reducer projects held-out images into the same space."""
        train = rng.random((20, 16, 24)).astype(np.float32)
        test = rng.random((5, 16, 24)).astype(np.float32)
        train_reduced, reducer = reduce_dimensions(
            train, method="mat-pca", n_components=6, return_reducer=True
        )
        test_reduced = reduce_dimensions(test, reducer=reducer)
        assert train_reduced.shape == (20, 16, 6)
        assert test_reduced.shape == (5, 16, 6)

    def test_prefit_matrix_reducer_projects_single_image(self, rng):
        """A pre-fit matrix reducer projects a single 2D image to (height, k)."""
        train = rng.random((20, 16, 24)).astype(np.float32)
        _, reducer = reduce_dimensions(
            train, method="mat-jl", n_components=6, return_reducer=True
        )
        one = rng.random((16, 24)).astype(np.float32)
        projected = reduce_dimensions(one, reducer=reducer)
        assert projected.shape == (16, 6)

    def test_single_matrix_without_reducer_raises(self, rng):
        """Fitting a matrix reducer on a single 2D image raises."""
        with pytest.raises(ValueError):
            reduce_dimensions(rng.random((16, 24)).astype(np.float32), method="mat-pca")

    def test_matrix_method_on_vector_matrix_raises(self, feature_matrix):
        """A matrix method given a flat 2D feature matrix raises."""
        with pytest.raises(ValueError):
            reduce_dimensions(feature_matrix, method="mat-pca", n_components=4)


class TestMatrixReduceMultiChannel:
    """Matrix reduction on colour (BGR/RGB) stacks: width narrows, channels kept."""

    @pytest.mark.parametrize("method", ["mat-pca", "mat-jl"])
    def test_colour_reduce_narrows_width_keeps_channels(self, color_matrix_stack, method):
        """Both matrix methods narrow width and preserve (rows, channels)."""
        n, h, _, c = color_matrix_stack.shape
        reduced = reduce_dimensions(color_matrix_stack, method=method, n_components=8)
        assert reduced.shape == (n, h, 8, c)
        assert reduced.dtype == np.float32
        assert np.isfinite(reduced).all()

    def test_colour_channels_reduced_independently_of_each_other(self, rng):
        """A constant channel stays constant; width mixing never crosses channels."""
        stack = rng.random((20, 12, 16, 3)).astype(np.float32)
        stack[..., 0] = 0.0
        reduced = reduce_dimensions(stack, method="mat-jl", n_components=5)
        assert np.allclose(reduced[..., 0], 0.0)
        assert reduced.shape == (20, 12, 5, 3)

    def test_colour_none_bypass_is_identity(self, color_matrix_stack):
        """method=None returns the same 4D colour stack object untouched."""
        assert reduce_dimensions(color_matrix_stack, method=None) is color_matrix_stack

    def test_colour_mat_pca_clamps_components_to_width(self, color_matrix_stack):
        """Over-asking for components clamps to the image width, channels intact."""
        n, h, w, c = color_matrix_stack.shape
        reduced = reduce_dimensions(color_matrix_stack, method="mat-pca", n_components=999)
        assert reduced.shape == (n, h, w, c)

    def test_shared_colour_reducer_projects_single_colour_image(self, rng):
        """A reducer fit on a colour batch projects one lone colour image.

        The single image must be read as (h, w, c), not a grayscale batch, and
        agree with the batched result.
        """
        train = rng.random((20, 16, 24, 3)).astype(np.float32)
        batch_reduced, reducer = reduce_dimensions(
            train, method="mat-pca", n_components=6, return_reducer=True
        )
        one = train[0]                       # (16, 24, 3)
        projected = reduce_dimensions(one, reducer=reducer)
        assert batch_reduced.shape == (20, 16, 6, 3)
        assert projected.shape == (16, 6, 3)
        assert np.allclose(projected, batch_reduced[0], atol=1e-5)

    def test_grayscale_reducer_rejects_colour_image(self, rng):
        """A grayscale-fit reducer applied to a colour image fails on the mismatch."""
        gray = rng.random((20, 16, 24)).astype(np.float32)
        _, reducer = reduce_dimensions(
            gray, method="mat-pca", n_components=6, return_reducer=True
        )
        with pytest.raises(Exception):
            reduce_dimensions(rng.random((16, 24, 3)).astype(np.float32), reducer=reducer)
