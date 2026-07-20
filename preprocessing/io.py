"""
Image I/O helpers.

Thin wrappers around PIL and OpenCV for loading images into NumPy arrays.
Kept in their own module so callers that only need bytes/file loading don't
have to import the heavier ``transforms.py`` / ``vectorize.py`` modules.

Color-space caveat
------------------
These loaders do NOT return a consistent channel order, because their backends
differ: the PIL-based loaders (``load_image_from_bytes``, ``load_image_from_pil``)
return **RGB**, while the OpenCV-based ``load_image_from_file`` returns **BGR**
(``cv2.imread`` convention). Nothing here reconciles the two. Callers that mix
loaders, or that care about channel order (e.g. color-sensitive features or a
pretrained model expecting RGB), must normalize explicitly — typically
``cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`` on the OpenCV path. The grayscale
front-end used by the training pipelines is unaffected.
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
        Image as np.ndarray (BGR format from OpenCV). Note this differs from the
        RGB returned by the PIL-based loaders — see the module docstring.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file cannot be read as image.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    # cv2.imread returns BGR (not RGB); no conversion is applied here, so callers
    # mixing this with the PIL loaders must reconcile the channel order themselves.
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
        Image as np.ndarray (RGB, matching PIL — see the module docstring).
    """
    return np.array(pil_image)
