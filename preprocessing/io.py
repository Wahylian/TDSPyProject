"""Load images into NumPy arrays via PIL and OpenCV.

Channel order is not consistent across loaders: the PIL-based loaders return
RGB, while ``load_image_from_file`` returns BGR (cv2 convention). Callers mixing
loaders, or feeding a model that expects RGB, must convert explicitly.
"""

import io
import os

import cv2
import numpy as np
from PIL import Image as PILImage


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB array. Raises ValueError on bad data."""
    try:
        pil_image = PILImage.open(io.BytesIO(image_bytes))
        return np.array(pil_image)
    except Exception as e:
        raise ValueError(f"Failed to load image from bytes: {str(e)}")


def load_image_from_file(file_path: str) -> np.ndarray:
    """Read an image file into a BGR array (cv2 convention, not RGB)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    # cv2.imread returns BGR, unlike the PIL loaders.
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to read image: {file_path}")
    return image


def load_image_from_pil(pil_image: PILImage.Image) -> np.ndarray:
    """Convert a PIL Image to an RGB array."""
    return np.array(pil_image)
