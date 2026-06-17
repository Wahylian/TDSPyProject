"""
Tests for image vectorization in ``preprocessing/vectorize.py``.

Covers the ``'flat'`` pixel-flattening method (colour, grayscale and
channel-preserving layouts) plus the ``'vgg16'`` embedding path. The VGG16
branch is exercised through the ``fake_vgg16`` fixture, which seeds the
module-level model cache with a lightweight stub so no Keras import or ~500 MB
ImageNet weight download is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import vectorize_image


class TestVectorizeFlat:
    """Flat pixel vectorization into 1D feature vectors."""

    def test_flat_color_shape_and_dtype(self, color_image):
        """Flat vectorization of a colour image gives a float32 H*W*C vector."""
        # Act
        vec = vectorize_image(color_image, method="flat")
        # Assert: a 1D float32 vector of length 224*224*3.
        assert vec.shape == (224 * 224 * 3,)
        assert vec.dtype == np.float32
        assert vec.ndim == 1

    def test_flat_grayscale_shape(self, gray_image):
        """Flat vectorization of a grayscale image gives an H*W vector.

        Confirms the channel count is respected — single-channel input has no
        channel factor in the output length.
        """
        # Act
        vec = vectorize_image(gray_image, method="flat")
        # Assert
        assert vec.shape == (224 * 224,)

    def test_preserve_structure_orders_channels_blockwise(self, color_image):
        """``preserve_structure`` concatenates channels block-wise, not interleaved.

        Validates the memory layout: with preserve_structure the first H*W
        slots must be channel 0 flattened, proving channels are laid out in
        contiguous blocks rather than pixel-interleaved (B,G,R,B,G,R,...).
        """
        # Act
        vec = vectorize_image(color_image, method="flat", preserve_structure=True)
        # Assert: the leading H*W block equals channel 0 flattened.
        expected_ch0 = color_image[:, :, 0].flatten().astype(np.float32)
        assert np.array_equal(vec[: 224 * 224], expected_ch0)

    def test_non_array_input_raises_typeerror(self):
        """A non-ndarray input raises ``TypeError``.

        Edge case: passing a nested Python list instead of a numpy array is a
        frequent misuse and must be rejected with a clear type error.
        """
        # Act + Assert
        with pytest.raises(TypeError):
            vectorize_image([[1, 2], [3, 4]], method="flat")

    def test_bad_dimensions_raise(self, rng):
        """A 4D tensor is rejected.

        Edge case: a 4D array usually means an accidental batch was passed to
        the single-image vectorizer; this is unsupported and must raise.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            vectorize_image(rng.random((2, 8, 8, 3)), method="flat")

    def test_unknown_method_raises(self, color_image):
        """An unsupported vectorization method raises.

        Edge case: only "flat" and "vgg16" are defined; anything else (e.g.
        "resnet") must fail rather than silently no-op.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            vectorize_image(color_image, method="resnet")


@pytest.mark.vgg16
class TestVectorizeVGG16:
    """The VGG16 embedding path, exercised with a stubbed model.

    The ``fake_vgg16`` fixture pre-seeds the module-level cache for the default
    (224, 224) input size, so every test here verifies the branch's shape/dtype
    contract WITHOUT importing keras or downloading ImageNet weights.
    """

    def test_color_path_returns_embedding_length(self, color_image, fake_vgg16):
        """A 224x224x3 image yields the stub's known embedding length, float32.

        Args:
            color_image: 224x224x3 fixture image (already the model's size).
            fake_vgg16: Fixture seeding the stub and returning the expected
                flattened output length (25 088 for 224x224).
        """
        # Act
        vec = vectorize_image(color_image, method="vgg16", input_size=(224, 224))
        # Assert: shape matches the stub's output size and dtype is float32.
        assert vec.shape == (fake_vgg16,)  # 25 088 for 224x224
        assert vec.dtype == np.float32

    def test_grayscale_input_is_promoted_to_three_channels(self, gray_image, fake_vgg16):
        """A 2D grayscale image is replicated to 3 channels and still embeds.

        VGG16 requires 3-channel input; the vectorizer must stack a grayscale
        image across channels rather than rejecting it. The stubbed model then
        returns the standard embedding length.

        Args:
            gray_image: 224x224 single-channel fixture image.
            fake_vgg16: Fixture seeding the stub model.
        """
        # Act
        vec = vectorize_image(gray_image, method="vgg16", input_size=(224, 224))
        # Assert: the grayscale image was accepted and embedded to the right width.
        assert vec.shape == (fake_vgg16,)
        assert vec.dtype == np.float32

    def test_mismatched_input_is_resized_before_embedding(self, small_color_image, fake_vgg16):
        """An image smaller than ``input_size`` is resized, then embedded.

        Exercises the internal resize branch: a 32x32 image must be brought up
        to the model's 224x224 input before the forward pass, producing the
        same embedding length without error.

        Args:
            small_color_image: 32x32x3 fixture (deliberately not 224x224).
            fake_vgg16: Fixture seeding the stub for the (224, 224) input size.
        """
        # Act
        vec = vectorize_image(small_color_image, method="vgg16", input_size=(224, 224))
        # Assert: resize-then-embed produced the standard-length float32 vector.
        assert vec.shape == (fake_vgg16,)
        assert vec.dtype == np.float32
