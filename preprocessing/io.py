"""
Image I/O helpers.

Thin wrappers around PIL and OpenCV for loading images into NumPy arrays.
Kept in their own module so callers that only need bytes/file loading don't
have to import the heavier ``transforms.py`` / ``vectorize.py`` modules.
"""

import io
import os

import cv2
import numpy as np
from PIL import Image as PILImage


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from raw bytes using PIL.

    Args:
        image_bytes: Raw byte content of image file.

    Returns:
        Image as np.ndarray (RGB format).

    Raises:
        ValueError: If bytes cannot be read as image.
    """
    try:
        pil_image = PILImage.open(io.BytesIO(image_bytes))
        return np.array(pil_image)
    except Exception as e:
        raise ValueError(f"Failed to load image from bytes: {str(e)}")


def load_image_from_file(file_path: str) -> np.ndarray:
    """
    Load image from file using OpenCV.

    Args:
        file_path: Path to image file.

    Returns:
        Image as np.ndarray (BGR format from OpenCV).

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file cannot be read as image.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to read image: {file_path}")
    return image


def load_image_from_pil(pil_image: PILImage.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.

    Args:
        pil_image: PIL Image object.

    Returns:
        Image as np.ndarray.
    """
    return np.array(pil_image)
