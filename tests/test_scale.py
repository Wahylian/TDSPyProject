"""
Tests for batch-level feature standardization in ``preprocessing/scale.py``.

``standardize_features`` learns a per-feature mean/std on a training batch and
reuses them on held-out data, implementing the same fit-once/reuse "reducer"
protocol as :func:`reduce_dimensions`. These tests pin: the zero-mean/unit-
variance contract, shape/dtype, the zero-variance guard, the ``with_mean`` /
``with_std`` switches, and the fit-once/reuse + single-sample inference paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import standardize_features


class TestStandardizeFeatures:
    """Per-feature standardization (fit, reuse, options, edge cases)."""

    def test_columns_become_zero_mean_unit_variance(self, feature_matrix):
        """Standardizing the fit batch yields ~zero-mean, ~unit-variance columns.

        The defining contract: each feature column of the transformed *training*
        batch has mean ≈ 0 and standard deviation ≈ 1.
        """
        # Act
        scaled = standardize_features(feature_matrix)
        # Assert: per-column stats hit the standardized targets (float32 tol).
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-4)
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-4)

    def test_output_shape_and_dtype_preserved(self, feature_matrix):
        """Standardization keeps the matrix shape and returns float32."""
        # Act
        scaled = standardize_features(feature_matrix)
        # Assert
        assert scaled.shape == feature_matrix.shape
        assert scaled.dtype == np.float32

    def test_zero_variance_column_is_centered_not_blown_up(self, rng):
        """A constant feature column stays finite (centered to 0), never inf/nan.

        Edge case: a zero-variance column would divide by a 0 std. The scale
        guard sets that std to 1.0, so the column is merely centered — finite,
        and exactly zero after subtracting its (constant) mean.
        """
        # Arrange: column 2 is constant across all rows.
        X = rng.random((30, 5)).astype(np.float32)
        X[:, 2] = 7.0
        # Act
        scaled = standardize_features(X)
        # Assert: all finite, and the constant column collapsed to zeros.
        assert np.isfinite(scaled).all()
        assert np.allclose(scaled[:, 2], 0.0)

    def test_with_mean_false_skips_centering(self, rng):
        """``with_mean=False`` leaves the mean in place, only scaling by std."""
        # Arrange: a strictly positive, shifted batch so a non-zero mean shows.
        X = (rng.random((40, 6)).astype(np.float32) + 5.0)
        # Act
        scaled = standardize_features(X, with_mean=False)
        # Assert: not centered (column means stay well above zero), but unit std.
        assert (scaled.mean(axis=0) > 1.0).all()
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-4)

    def test_with_std_false_skips_scaling(self, rng):
        """``with_std=False`` centers each column but leaves its spread intact."""
        # Arrange
        X = (rng.random((40, 6)).astype(np.float32) * 10.0)
        # Act
        scaled = standardize_features(X, with_std=False)
        # Assert: centered (≈0 mean) but std differs from 1 (unscaled spread).
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-4)
        assert not np.allclose(scaled.std(axis=0), 1.0, atol=1e-2)

    def test_return_reducer_yields_a_reusable_scaler(self, feature_matrix):
        """``return_reducer=True`` hands back the fitted scaler alongside output."""
        # Act
        scaled, scaler = standardize_features(feature_matrix, return_reducer=True)
        # Assert: features scaled, and a scaler exposing transform() is returned.
        assert scaled.shape == feature_matrix.shape
        assert hasattr(scaler, "transform")

    def test_shared_scaler_applies_training_stats(self, rng):
        """A fitted scaler standardizes held-out data with the *training* stats.

        Validates the train/test pattern: fit on train (``return_reducer=True``),
        then reuse the scaler on test so both are scaled by the same per-feature
        mean/std — not stats refit on the test split.
        """
        # Arrange: train/test sharing feature width.
        train = (rng.random((50, 20)).astype(np.float32) + 3.0)
        test = (rng.random((8, 20)).astype(np.float32) + 3.0)
        # Act: fit on train, capture the scaler, apply to test.
        _, scaler = standardize_features(train, return_reducer=True)
        test_scaled = standardize_features(test, reducer=scaler)
        # Assert: the manual (train-stats) transform matches the reuse path.
        expected = (test.astype(np.float64) - scaler.mean_) / scaler.scale_
        assert test_scaled.shape == (8, 20)
        np.testing.assert_allclose(test_scaled, expected.astype(np.float32), rtol=1e-5)

    def test_prefit_scaler_standardizes_single_vector(self, rng):
        """A pre-fit scaler standardizes a single 1D sample at inference time.

        Inference-time path: a fitted scaler applied to one 1D vector returns a
        1D vector (not a (1, n) matrix), matching single-image embedding scaling.
        """
        # Arrange: fit a scaler on a training batch.
        train = rng.random((50, 20)).astype(np.float32)
        _, scaler = standardize_features(train, return_reducer=True)
        # Act: standardize a lone 1D sample.
        one = rng.random(20).astype(np.float32)
        out = standardize_features(one, reducer=scaler)
        # Assert: 1D in -> 1D out, same width.
        assert out.shape == (20,)

    def test_fit_on_single_sample_without_reducer_raises(self, rng):
        """Fitting on a single 1D vector (no scaler) raises.

        Edge case: per-feature std is undefined for one sample, so fitting on a
        lone vector with no pre-fit scaler must fail loudly.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            standardize_features(rng.random(20).astype(np.float32))

    def test_three_dimensional_input_raises(self, matrix_stack):
        """A 3D image stack is rejected — ``'scale'`` is for flat features only.

        Edge case: standardization runs on the 2D ``(n_samples, n_features)``
        output of vectorize/reduce, so handing it a 3D matrix stack is a
        misconfiguration and must raise.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            standardize_features(matrix_stack)
