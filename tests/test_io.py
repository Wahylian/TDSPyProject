"""
Tests for the image-loading helpers in ``preprocessing/io.py``.

Covers the three loaders — ``load_image_from_file`` (OpenCV/BGR),
``load_image_from_bytes`` and ``load_image_from_pil`` (PIL/RGB) — across their
success paths and the failure modes a loader must surface clearly (missing
file, undecodable bytes).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image as PILImage

from preprocessing import (
    load_image_from_bytes,
    load_image_from_file,
    load_image_from_pil,
)


class TestLoadFromFile:
    """The OpenCV file-loading path and its failure modes."""

    def test_load_image_from_file_roundtrip(self, tmp_image_file, color_image):
        """An image written to disk loads back with the same shape and dtype.

        Exercises the real OpenCV file-decoding path against an actual PNG
        written by the ``tmp_image_file`` fixture (a round-trip, not a mock).

        Args:
            tmp_image_file: Path to a PNG written from ``color_image``.
            color_image: The original 224x224x3 array, for shape comparison.
        """
        # Act
        loaded = load_image_from_file(tmp_image_file)
        # Assert: shape and dtype survive the write/read round-trip.
        assert loaded.shape == color_image.shape
        assert loaded.dtype == np.uint8

    def test_load_missing_file_raises(self):
        """A non-existent path raises ``FileNotFoundError``.

        Edge case: a missing file must raise a clear, specific error rather
        than returning ``None`` (OpenCV's default) and crashing later.
        """
        # Act + Assert
        with pytest.raises(FileNotFoundError):
            load_image_from_file("definitely_not_here_12345.png")


class TestLoadFromBytes:
    """Decoding raw image bytes via PIL."""

    def test_load_valid_bytes_returns_rgb_array(self):
        """Valid PNG bytes decode to an ndarray of the encoded size.

        Success path: the loader must turn a real PNG byte string into a
        3-channel uint8 array matching the source image's dimensions.
        """
        # Arrange: encode a small solid-colour RGB image to PNG bytes.
        buf = io.BytesIO()
        PILImage.new("RGB", (10, 6), (10, 20, 30)).save(buf, format="PNG")
        # Act
        arr = load_image_from_bytes(buf.getvalue())
        # Assert: PIL decodes (height, width, channels) as uint8 RGB.
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (6, 10, 3)
        assert arr.dtype == np.uint8

    def test_load_corrupted_bytes_raises(self):
        """Undecodable byte content raises ``ValueError``.

        Edge case: random/garbage bytes are not a valid image; decoding must
        fail loudly instead of yielding a malformed array.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            load_image_from_bytes(b"\x00\x01\x02 this is not an image \xff")


class TestLoadFromPIL:
    """Converting a PIL Image object into a NumPy array."""

    def test_pil_image_becomes_ndarray(self):
        """A PIL Image is converted to an ndarray of matching dimensions.

        The helper is a thin ``np.array`` wrapper, so the result must mirror the
        PIL image's (height, width, channels) layout as uint8.
        """
        # Arrange: a 8x4 RGB PIL image.
        pil_image = PILImage.new("RGB", (4, 8), (1, 2, 3))
        # Act
        arr = load_image_from_pil(pil_image)
        # Assert: ndarray with PIL's (height, width, channels) ordering.
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (8, 4, 3)
        assert arr.dtype == np.uint8
