"""Tests for batch-level feature standardization in preprocessing/scale.py.

Pins the zero-mean/unit-variance contract, shape/dtype, the zero-variance guard,
the with_mean/with_std switches, and the fit-once/reuse + single-sample paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import standardize_features


class TestStandardizeFeatures:
    """Per-feature standardization (fit, reuse, options, edge cases)."""

    def test_columns_become_zero_mean_unit_variance(self, feature_matrix):
        """Standardizing the fit batch yields ~zero-mean, ~unit-variance columns."""
        scaled = standardize_features(feature_matrix)
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-4)
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-4)

    def test_output_shape_and_dtype_preserved(self, feature_matrix):
        """Standardization keeps the matrix shape and returns float32."""
        scaled = standardize_features(feature_matrix)
        assert scaled.shape == feature_matrix.shape
        assert scaled.dtype == np.float32

    def test_zero_variance_column_is_centered_not_blown_up(self, rng):
        """A constant column stays finite (centered to 0), never inf/nan."""
        X = rng.random((30, 5)).astype(np.float32)
        X[:, 2] = 7.0
        scaled = standardize_features(X)
        assert np.isfinite(scaled).all()
        assert np.allclose(scaled[:, 2], 0.0)

    def test_with_mean_false_skips_centering(self, rng):
        """with_mean=False leaves the mean in place, only scaling by std."""
        X = (rng.random((40, 6)).astype(np.float32) + 5.0)
        scaled = standardize_features(X, with_mean=False)
        assert (scaled.mean(axis=0) > 1.0).all()
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-4)

    def test_with_std_false_skips_scaling(self, rng):
        """with_std=False centers each column but leaves its spread intact."""
        X = (rng.random((40, 6)).astype(np.float32) * 10.0)
        scaled = standardize_features(X, with_std=False)
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-4)
        assert not np.allclose(scaled.std(axis=0), 1.0, atol=1e-2)

    def test_return_reducer_yields_a_reusable_scaler(self, feature_matrix):
        """return_reducer=True hands back the fitted scaler alongside output."""
        scaled, scaler = standardize_features(feature_matrix, return_reducer=True)
        assert scaled.shape == feature_matrix.shape
        assert hasattr(scaler, "transform")

    def test_shared_scaler_applies_training_stats(self, rng):
        """A fitted scaler standardizes held-out data with the training stats."""
        train = (rng.random((50, 20)).astype(np.float32) + 3.0)
        test = (rng.random((8, 20)).astype(np.float32) + 3.0)
        _, scaler = standardize_features(train, return_reducer=True)
        test_scaled = standardize_features(test, reducer=scaler)
        expected = (test.astype(np.float64) - scaler.mean_) / scaler.scale_
        assert test_scaled.shape == (8, 20)
        np.testing.assert_allclose(test_scaled, expected.astype(np.float32), rtol=1e-5)

    def test_prefit_scaler_standardizes_single_vector(self, rng):
        """A pre-fit scaler standardizes a single 1D sample, returning 1D."""
        train = rng.random((50, 20)).astype(np.float32)
        _, scaler = standardize_features(train, return_reducer=True)
        one = rng.random(20).astype(np.float32)
        out = standardize_features(one, reducer=scaler)
        assert out.shape == (20,)

    def test_fit_on_single_sample_without_reducer_raises(self, rng):
        """Fitting on a single 1D vector without a scaler raises."""
        with pytest.raises(ValueError):
            standardize_features(rng.random(20).astype(np.float32))

    def test_three_dimensional_input_raises(self, matrix_stack):
        """A 3D image stack is rejected; 'scale' is for flat features only."""
        with pytest.raises(ValueError):
            standardize_features(matrix_stack)
