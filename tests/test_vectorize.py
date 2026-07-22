"""Tests for image vectorization in preprocessing/vectorize.py.

Covers the 'flat' method (colour, grayscale, channel-preserving) and the
'vgg16' path, the latter via the fake_vgg16 stub so no keras/weights are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import vectorize_image


class TestVectorizeFlat:
    """Flat pixel vectorization into 1D feature vectors."""

    def test_flat_color_shape_and_dtype(self, color_image):
        """Flat vectorization of a colour image gives a float32 H*W*C vector."""
        vec = vectorize_image(color_image, method="flat")
        assert vec.shape == (224 * 224 * 3,)
        assert vec.dtype == np.float32
        assert vec.ndim == 1

    def test_flat_grayscale_shape(self, gray_image):
        """Flat vectorization of a grayscale image gives an H*W vector."""
        vec = vectorize_image(gray_image, method="flat")
        assert vec.shape == (224 * 224,)

    def test_preserve_structure_orders_channels_blockwise(self, color_image):
        """preserve_structure lays channels out block-wise, not interleaved.

        The leading H*W block must equal channel 0 flattened.
        """
        vec = vectorize_image(color_image, method="flat", preserve_structure=True)
        expected_ch0 = color_image[:, :, 0].flatten().astype(np.float32)
        assert np.array_equal(vec[: 224 * 224], expected_ch0)

    def test_non_array_input_raises_typeerror(self):
        """A non-ndarray input raises TypeError."""
        with pytest.raises(TypeError):
            vectorize_image([[1, 2], [3, 4]], method="flat")

    def test_bad_dimensions_raise(self, rng):
        """A 4D tensor (an accidental batch) is rejected."""
        with pytest.raises(ValueError):
            vectorize_image(rng.random((2, 8, 8, 3)), method="flat")

    def test_unknown_method_raises(self, color_image):
        """An unsupported vectorization method raises."""
        with pytest.raises(ValueError):
            vectorize_image(color_image, method="resnet")


@pytest.mark.vgg16
class TestVectorizeVGG16:
    """The VGG16 embedding path, exercised with the fake_vgg16 stub."""

    def test_color_path_returns_embedding_length(self, color_image, fake_vgg16):
        """A 224x224x3 image yields the stub's embedding length, float32."""
        vec = vectorize_image(color_image, method="vgg16", input_size=(224, 224))
        assert vec.shape == (fake_vgg16,)  # 25,088 for 224x224
        assert vec.dtype == np.float32

    def test_grayscale_input_is_promoted_to_three_channels(self, gray_image, fake_vgg16):
        """A 2D grayscale image is replicated to 3 channels and still embeds."""
        vec = vectorize_image(gray_image, method="vgg16", input_size=(224, 224))
        assert vec.shape == (fake_vgg16,)
        assert vec.dtype == np.float32

    def test_mismatched_input_is_resized_before_embedding(self, small_color_image, fake_vgg16):
        """An image smaller than input_size is resized, then embedded."""
        vec = vectorize_image(small_color_image, method="vgg16", input_size=(224, 224))
        assert vec.shape == (fake_vgg16,)
        assert vec.dtype == np.float32
