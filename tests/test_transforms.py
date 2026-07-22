"""Tests for the per-image transforms in preprocessing/transforms.py.

Covers to_grayscale, resize_image, normalize_image and reduce_noise across
standard workflows, numerical guarantees, and argument-validation failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import (
    normalize_image,
    reduce_noise,
    resize_image,
    to_grayscale,
)


class TestToGrayscale:
    """Behaviour of to_grayscale across input shapes."""

    def test_color_to_grayscale_drops_channel_axis(self, color_image):
        """A colour (H, W, 3) image collapses to 2D, keeping dims and dtype."""
        gray = to_grayscale(color_image)
        assert gray.ndim == 2
        assert gray.shape == color_image.shape[:2]
        assert gray.dtype == color_image.dtype

    def test_grayscale_passthrough_is_noop(self, gray_image):
        """An already-grayscale image is returned unchanged when force is off."""
        result = to_grayscale(gray_image)
        assert result is gray_image or np.array_equal(result, gray_image)

    def test_force_on_grayscale_returns_same_values(self, gray_image):
        """force=True on grayscale input still preserves pixel values."""
        result = to_grayscale(gray_image, force=True)
        assert np.array_equal(result, gray_image)

    def test_invalid_dimensions_raise(self):
        """A 1D array is neither grayscale nor colour and raises."""
        with pytest.raises(ValueError):
            to_grayscale(np.zeros((10,), dtype=np.uint8))


class TestResizeImage:
    """Resizing behaviour, including aspect-preserving letterbox padding."""

    def test_exact_resize_without_aspect(self, color_image):
        """preserve_aspect=False stretches the image to the target."""
        resized = resize_image(color_image, (64, 64), preserve_aspect=False)
        assert resized.shape == (64, 64, 3)

    def test_preserve_aspect_pads_to_exact_target(self, rng):
        """A non-square input is letterboxed to the exact target size.

        Regression: guards the padding bug where the after-pad double-counted
        the before-pad and overshot target_size.
        """
        tall = rng.integers(0, 256, size=(100, 50, 3), dtype=np.uint8)
        resized = resize_image(tall, (64, 64), preserve_aspect=True)
        assert resized.shape == (64, 64, 3)

    @pytest.mark.parametrize("shape", [(100, 50, 3), (50, 100, 3), (200, 100, 3), (480, 640, 3)])
    def test_preserve_aspect_always_hits_target_for_varied_aspects(self, shape):
        """Every aspect ratio pads to exactly (64, 64)."""
        img = np.zeros(shape, dtype=np.uint8)
        assert resize_image(img, (64, 64), preserve_aspect=True).shape == (64, 64, 3)

    def test_grayscale_preserve_aspect_hits_target(self, rng):
        """A 2D grayscale input letterboxes to target without adding a channel axis."""
        tall_gray = rng.integers(0, 256, size=(100, 40), dtype=np.uint8)
        resized = resize_image(tall_gray, (64, 64), preserve_aspect=True)
        assert resized.shape == (64, 64)
        assert resized.ndim == 2

    def test_resize_preserves_dtype(self, color_image):
        """Resizing returns the same dtype as the input."""
        resized = resize_image(color_image, (32, 32), preserve_aspect=False)
        assert resized.dtype == color_image.dtype

    @pytest.mark.parametrize("interpolation", ["nearest", "bilinear", "bicubic", "lanczos"])
    def test_all_interpolation_methods_hit_target(self, color_image, interpolation):
        """Every documented interpolation mode resizes to the requested size."""
        resized = resize_image(
            color_image, (48, 48), preserve_aspect=False, interpolation=interpolation
        )
        assert resized.shape == (48, 48, 3)

    def test_invalid_target_size_raises(self, color_image):
        """A zero or negative target dimension is rejected."""
        with pytest.raises(ValueError):
            resize_image(color_image, (0, 64))

    def test_unknown_interpolation_raises(self, color_image):
        """An unsupported interpolation name raises rather than guessing."""
        with pytest.raises(ValueError):
            resize_image(color_image, (64, 64), interpolation="sinc")


class TestNormalizeImage:
    """The normalize_image methods and their numerical guarantees."""

    def test_minmax_maps_into_unit_range(self, color_image):
        """Min-max normalization maps pixels into [0, 1] as float32."""
        normalized = normalize_image(color_image, method="minmax")
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0 - 1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_minmax_honours_custom_value_range(self, color_image):
        """Min-max with a custom value_range stretches into that interval."""
        normalized = normalize_image(color_image, method="minmax", value_range=(-1.0, 1.0))
        assert normalized.min() >= -1.0 - 1e-6
        assert normalized.max() <= 1.0 + 1e-6
        assert np.isclose(normalized.min(), -1.0, atol=1e-5)
        assert np.isclose(normalized.max(), 1.0, atol=1e-5)

    def test_standard_gives_zero_mean_unit_variance(self, color_image):
        """Standard (z-score) normalization yields ~0 mean and ~1 std."""
        normalized = normalize_image(color_image, method="standard")
        assert abs(float(normalized.mean())) < 1e-4
        assert abs(float(normalized.std()) - 1.0) < 1e-4

    def test_standard_on_constant_image_returns_zeros(self):
        """A constant image standardizes to all-zeros, not NaN (zero-std guard)."""
        flat = np.full((8, 8), 200, dtype=np.uint8)
        out = normalize_image(flat, method="standard")
        assert np.all(np.isfinite(out))
        assert np.all(out == 0.0)

    def test_flat_image_minmax_avoids_divide_by_zero(self):
        """A constant image normalizes to the 0.5 midpoint, not NaN (max==min guard)."""
        flat = np.full((8, 8), 128, dtype=np.uint8)
        out = normalize_image(flat, method="minmax")
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.5)

    def test_histogram_on_grayscale_maps_into_unit_range(self, gray_image):
        """Histogram equalization on grayscale returns float32 in [0, 1]."""
        out = normalize_image(gray_image, method="histogram")
        assert out.dtype == np.float32
        assert out.shape == gray_image.shape
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_histogram_on_color_image_raises(self, color_image):
        """Histogram equalization rejects multi-channel input."""
        with pytest.raises(ValueError):
            normalize_image(color_image, method="histogram")

    def test_unknown_method_raises(self, gray_image):
        """An unrecognised normalization method raises."""
        with pytest.raises(ValueError):
            normalize_image(gray_image, method="zscore-but-typo")


class TestReduceNoise:
    """Denoising preserves image shape and validates its arguments."""

    def test_shape_is_preserved(self, color_image):
        """Bilateral denoising returns an image of the same shape."""
        denoised = reduce_noise(color_image, method="bilateral")
        assert denoised.shape == color_image.shape

    @pytest.mark.parametrize("method", ["bilateral", "gaussian", "morphological", "median"])
    def test_all_methods_return_same_shape(self, small_color_image, method):
        """Every denoise method is shape-preserving."""
        out = reduce_noise(small_color_image, method=method, kernel_size=5)
        assert out.shape == small_color_image.shape

    def test_even_kernel_size_raises(self, color_image):
        """An even kernel size is rejected (kernels need a defined centre)."""
        with pytest.raises(ValueError):
            reduce_noise(color_image, kernel_size=4)

    @pytest.mark.parametrize("bad_kernel", [0, -3])
    def test_non_positive_kernel_size_raises(self, color_image, bad_kernel):
        """A zero or negative kernel size is rejected."""
        with pytest.raises(ValueError):
            reduce_noise(color_image, kernel_size=bad_kernel)

    def test_unknown_method_raises(self, color_image):
        """An unsupported denoise method name raises."""
        with pytest.raises(ValueError):
            reduce_noise(color_image, method="nonexistent")
