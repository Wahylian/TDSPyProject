"""Tests for the image loaders in preprocessing/io.py.

Covers load_image_from_file (OpenCV/BGR), load_image_from_bytes and
load_image_from_pil (PIL/RGB) across success paths and failure modes.
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
        """An image written to disk loads back with the same shape and dtype."""
        loaded = load_image_from_file(tmp_image_file)
        assert loaded.shape == color_image.shape
        assert loaded.dtype == np.uint8

    def test_load_missing_file_raises(self):
        """A non-existent path raises FileNotFoundError, not a None return."""
        with pytest.raises(FileNotFoundError):
            load_image_from_file("definitely_not_here_12345.png")


class TestLoadFromBytes:
    """Decoding raw image bytes via PIL."""

    def test_load_valid_bytes_returns_rgb_array(self):
        """Valid PNG bytes decode to a 3-channel uint8 array of the encoded size."""
        buf = io.BytesIO()
        PILImage.new("RGB", (10, 6), (10, 20, 30)).save(buf, format="PNG")
        arr = load_image_from_bytes(buf.getvalue())
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (6, 10, 3)
        assert arr.dtype == np.uint8

    def test_load_corrupted_bytes_raises(self):
        """Undecodable byte content raises ValueError."""
        with pytest.raises(ValueError):
            load_image_from_bytes(b"\x00\x01\x02 this is not an image \xff")


class TestLoadFromPIL:
    """Converting a PIL Image object into a NumPy array."""

    def test_pil_image_becomes_ndarray(self):
        """A PIL Image converts to a uint8 ndarray with matching dimensions."""
        pil_image = PILImage.new("RGB", (4, 8), (1, 2, 3))
        arr = load_image_from_pil(pil_image)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (8, 4, 3)
        assert arr.dtype == np.uint8
