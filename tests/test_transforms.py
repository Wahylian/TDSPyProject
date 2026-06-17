"""
Tests for the per-image transforms in ``preprocessing/transforms.py``.

Covers the four building-block transforms — ``to_grayscale``, ``resize_image``,
``normalize_image`` and ``reduce_noise`` — across their standard workflows,
numerical guarantees and argument-validation failure modes. Every test follows
Arrange-Act-Assert; edge-case tests name the specific failure mode they guard.
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
    """Behaviour of the ``to_grayscale`` transform across input shapes."""

    def test_color_to_grayscale_drops_channel_axis(self, color_image):
        """A colour (H, W, 3) image collapses to a 2D (H, W) image.

        Args:
            color_image: 224x224x3 uint8 fixture image.
        """
        # Arrange handled by the color_image fixture.
        # Act
        gray = to_grayscale(color_image)
        # Assert: channel axis removed, spatial dims and dtype preserved.
        assert gray.ndim == 2
        assert gray.shape == color_image.shape[:2]
        assert gray.dtype == color_image.dtype

    def test_grayscale_passthrough_is_noop(self, gray_image):
        """An already-grayscale image is returned unchanged when ``force`` is off.

        Confirms the transform short-circuits on 2D input instead of erroring
        or needlessly copying.
        """
        # Act
        result = to_grayscale(gray_image)
        # Assert: same object or at least value-identical (a no-op).
        assert result is gray_image or np.array_equal(result, gray_image)

    def test_force_on_grayscale_returns_same_values(self, gray_image):
        """``force=True`` on grayscale input still preserves pixel values.

        The ``force`` flag re-runs conversion logic; for input that is already
        single-channel this is documented to be value-preserving, so no pixels
        may shift.
        """
        # Act
        result = to_grayscale(gray_image, force=True)
        # Assert: values are byte-for-byte identical.
        assert np.array_equal(result, gray_image)

    def test_invalid_dimensions_raise(self):
        """A 1D array is rejected as neither grayscale nor colour.

        Edge case: only 2D (grayscale) and 3D (colour) inputs are meaningful;
        a 1D array signals a caller error and must raise rather than silently
        misinterpret the data.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            to_grayscale(np.zeros((10,), dtype=np.uint8))


class TestResizeImage:
    """Resizing behaviour, including aspect-preserving letterbox padding."""

    def test_exact_resize_without_aspect(self, color_image):
        """With ``preserve_aspect=False`` the image is stretched to the target.

        Args:
            color_image: 224x224x3 uint8 fixture image.
        """
        # Act
        resized = resize_image(color_image, (64, 64), preserve_aspect=False)
        # Assert: the output matches the requested size exactly.
        assert resized.shape == (64, 64, 3)

    def test_preserve_aspect_pads_to_exact_target(self, rng):
        """A non-square input is letterboxed to the EXACT target size.

        Edge case / regression: this guards the padding bug where the "after"
        pad double-counted the "before" pad and overshot ``target_size`` — a
        (100, 50) image used to yield (64, 80) instead of (64, 64).
        """
        # Arrange: a tall, non-square image that triggers vertical/horizontal padding.
        tall = rng.integers(0, 256, size=(100, 50, 3), dtype=np.uint8)
        # Act
        resized = resize_image(tall, (64, 64), preserve_aspect=True)
        # Assert: padding lands the result on the exact target, not oversized.
        assert resized.shape == (64, 64, 3)

    @pytest.mark.parametrize("shape", [(100, 50, 3), (50, 100, 3), (200, 100, 3), (480, 640, 3)])
    def test_preserve_aspect_always_hits_target_for_varied_aspects(self, shape):
        """Every aspect ratio pads to exactly (64, 64).

        Regression sweep complementing the single-case test above: portrait,
        landscape and common photo ratios must all land on the target so the
        padding maths is correct in both axes.

        Args:
            shape: Input image shape supplied by the parametrize sweep.
        """
        # Arrange / Act / Assert in one line per shape.
        img = np.zeros(shape, dtype=np.uint8)
        assert resize_image(img, (64, 64), preserve_aspect=True).shape == (64, 64, 3)

    def test_grayscale_preserve_aspect_hits_target(self, rng):
        """A 2D grayscale input is letterboxed to the exact target size.

        Exercises the dedicated 2D ``np.pad`` branch (no channel axis), proving
        the padding maths is correct for single-channel images too — not just
        the 3D colour path covered above.
        """
        # Arrange: a tall, non-square grayscale image.
        tall_gray = rng.integers(0, 256, size=(100, 40), dtype=np.uint8)
        # Act
        resized = resize_image(tall_gray, (64, 64), preserve_aspect=True)
        # Assert: 2D output padded to the exact target, channel axis never added.
        assert resized.shape == (64, 64)
        assert resized.ndim == 2

    def test_resize_preserves_dtype(self, color_image):
        """Resizing returns the same dtype as the input.

        Downstream stages key off dtype (e.g. uint8 vs float), so a resize must
        not silently promote or demote the pixel type.
        """
        # Act
        resized = resize_image(color_image, (32, 32), preserve_aspect=False)
        # Assert
        assert resized.dtype == color_image.dtype

    @pytest.mark.parametrize("interpolation", ["nearest", "bilinear", "bicubic", "lanczos"])
    def test_all_interpolation_methods_hit_target(self, color_image, interpolation):
        """Every documented interpolation mode resizes to the requested size.

        Args:
            color_image: 224x224x3 uint8 fixture image.
            interpolation: Interpolation name supplied by the parametrize sweep.
        """
        # Act
        resized = resize_image(
            color_image, (48, 48), preserve_aspect=False, interpolation=interpolation
        )
        # Assert
        assert resized.shape == (48, 48, 3)

    def test_invalid_target_size_raises(self, color_image):
        """A zero (or negative) target dimension is rejected.

        Edge case: a 0-sized axis is nonsensical and would otherwise produce
        an empty or crashing resize, so it must fail fast.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            resize_image(color_image, (0, 64))

    def test_unknown_interpolation_raises(self, color_image):
        """An unsupported interpolation name raises rather than guessing.

        Edge case: only the documented OpenCV interpolation modes are valid;
        a typo / unsupported mode must be reported, not silently defaulted.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            resize_image(color_image, (64, 64), interpolation="sinc")


class TestNormalizeImage:
    """The ``normalize_image`` methods and their numerical guarantees."""

    def test_minmax_maps_into_unit_range(self, color_image):
        """Min-max normalization maps pixels into [0, 1] as float32.

        The tiny epsilon tolerance absorbs floating-point rounding at the
        range endpoints.
        """
        # Act
        normalized = normalize_image(color_image, method="minmax")
        # Assert: float32 output bounded within the unit interval.
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0 - 1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_minmax_honours_custom_value_range(self, color_image):
        """Min-max with a custom ``value_range`` stretches into that interval.

        The default maps to [0, 1]; supplying e.g. (-1, 1) must rescale the
        endpoints accordingly so the transform is usable for models expecting
        a different input range.
        """
        # Act
        normalized = normalize_image(color_image, method="minmax", value_range=(-1.0, 1.0))
        # Assert: the extremes land on the requested bounds (a random image
        # spans the full pixel range, so min->-1 and max->+1).
        assert normalized.min() >= -1.0 - 1e-6
        assert normalized.max() <= 1.0 + 1e-6
        assert np.isclose(normalized.min(), -1.0, atol=1e-5)
        assert np.isclose(normalized.max(), 1.0, atol=1e-5)

    def test_standard_gives_zero_mean_unit_variance(self, color_image):
        """Standard (z-score) normalization yields ~0 mean and ~1 std.

        Validates the statistical contract of the "standard" method within a
        small numerical tolerance.
        """
        # Act
        normalized = normalize_image(color_image, method="standard")
        # Assert: mean collapses to ~0 and standard deviation to ~1.
        assert abs(float(normalized.mean())) < 1e-4
        assert abs(float(normalized.std()) - 1.0) < 1e-4

    def test_standard_on_constant_image_returns_zeros(self):
        """A constant image standardizes to all-zeros, not NaN.

        Edge case: a flat image has zero standard deviation, so the naive
        ``(x - mean) / std`` divides by zero. The guard must return a zero
        array instead of propagating NaN/inf.
        """
        # Arrange: a perfectly flat image (std == 0).
        flat = np.full((8, 8), 200, dtype=np.uint8)
        # Act
        out = normalize_image(flat, method="standard")
        # Assert: finite and identically zero.
        assert np.all(np.isfinite(out))
        assert np.all(out == 0.0)

    def test_flat_image_minmax_avoids_divide_by_zero(self):
        """A constant image normalizes to the midpoint, not NaN/inf.

        Edge case: a constant image has ``max == min``, so the naive
        ``(x - min) / (max - min)`` is a 0/0 division. The guard must return
        the 0.5 midpoint instead of propagating NaN/inf.
        """
        # Arrange: a perfectly flat (single-value) image.
        flat = np.full((8, 8), 128, dtype=np.uint8)
        # Act
        out = normalize_image(flat, method="minmax")
        # Assert: finite everywhere and pinned to the midpoint.
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.5)

    def test_histogram_on_grayscale_maps_into_unit_range(self, gray_image):
        """Histogram equalization on a grayscale image returns float32 in [0, 1].

        The success path of the histogram method: a single-channel image is
        equalized then rescaled back to the unit interval for consistency with
        the other normalization methods.
        """
        # Act
        out = normalize_image(gray_image, method="histogram")
        # Assert: float32, 2D shape preserved, bounded in [0, 1].
        assert out.dtype == np.float32
        assert out.shape == gray_image.shape
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_histogram_on_color_image_raises(self, color_image):
        """Histogram equalization rejects multi-channel input.

        Edge case: histogram equalization is only defined for a single
        channel, so passing a colour image must raise rather than silently
        equalizing per-channel and distorting colour.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            normalize_image(color_image, method="histogram")

    def test_unknown_method_raises(self, gray_image):
        """An unrecognised normalization method raises.

        Edge case: a typo'd / unsupported method name must fail fast instead
        of falling through to a silent no-op.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            normalize_image(gray_image, method="zscore-but-typo")


class TestReduceNoise:
    """Denoising preserves image shape and validates its arguments."""

    def test_shape_is_preserved(self, color_image):
        """Bilateral denoising returns an image of the same shape.

        Denoising must never change dimensionality — only pixel values — so
        it can be dropped into a pipeline without breaking later stages.
        """
        # Act
        denoised = reduce_noise(color_image, method="bilateral")
        # Assert
        assert denoised.shape == color_image.shape

    @pytest.mark.parametrize("method", ["bilateral", "gaussian", "morphological", "median"])
    def test_all_methods_return_same_shape(self, small_color_image, method):
        """Every denoise method is shape-preserving.

        Sweeps all supported methods to ensure none of them resize or drop a
        channel, regardless of the underlying OpenCV op.

        Args:
            small_color_image: 32x32x3 fixture (kept tiny for speed).
            method: Denoise method supplied by the parametrize sweep.
        """
        # Act
        out = reduce_noise(small_color_image, method=method, kernel_size=5)
        # Assert
        assert out.shape == small_color_image.shape

    def test_even_kernel_size_raises(self, color_image):
        """An even kernel size is rejected.

        Edge case: OpenCV convolution kernels must be odd-sized so they have a
        defined centre pixel; an even size is a common caller bug and must
        raise rather than crash deep inside OpenCV.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_noise(color_image, kernel_size=4)

    @pytest.mark.parametrize("bad_kernel", [0, -3])
    def test_non_positive_kernel_size_raises(self, color_image, bad_kernel):
        """A zero or negative kernel size is rejected.

        Edge case: kernel sizes must be strictly positive; the guard covers the
        ``kernel_size <= 0`` branch alongside the even-size check.

        Args:
            color_image: 224x224x3 uint8 fixture image.
            bad_kernel: A non-positive kernel size from the parametrize sweep.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_noise(color_image, kernel_size=bad_kernel)

    def test_unknown_method_raises(self, color_image):
        """An unsupported denoise method name raises.

        Edge case: guards against typo'd / unsupported method strings.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_noise(color_image, method="nonexistent")
