"""Per-image transforms: the building blocks ImagePipeline chains together.

Each function takes one image array and returns a transformed image.
"""

from typing import Tuple

import numpy as np
import cv2


def normalize_image(
    image_array: np.ndarray,
    method: str = 'minmax',
    value_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """Scale pixel intensities via minmax, standard (z-score), or histogram equalization.

    minmax stretches to value_range; standard gives zero-mean unit-variance;
    histogram equalizes grayscale only. Returns float32.
    """
    if method not in ('minmax', 'standard', 'histogram'):
        raise ValueError(f"Unsupported method: {method}. Choose from minmax, standard, histogram.")

    # Float32 throughout avoids integer overflow and truncation.
    image_array = image_array.astype(np.float32)

    if method == 'minmax':
        min_val = image_array.min()
        max_val = image_array.max()

        # Flat image: return the range midpoint instead of dividing by zero.
        if max_val == min_val:
            return np.full_like(image_array, (value_range[0] + value_range[1]) / 2)

        # Scale to [0, 1], then stretch to value_range.
        normalized = (image_array - min_val) / (max_val - min_val)
        return normalized * (value_range[1] - value_range[0]) + value_range[0]

    elif method == 'standard':
        mean = image_array.mean()
        std = image_array.std()

        # Zero-std image (solid colour): return zeros instead of dividing by zero.
        if std == 0:
            return np.zeros_like(image_array)

        return (image_array - mean) / std

    elif method == 'histogram':
        if image_array.ndim == 3:
            raise ValueError("Histogram equalization requires grayscale image (2D).")

        # equalizeHist needs uint8; rescale back to [0, 1] float32 to match the other methods.
        uint8_image = np.clip(image_array, 0, 255).astype(np.uint8)
        equalized = cv2.equalizeHist(uint8_image)
        return equalized.astype(np.float32) / 255.0


def resize_image(
    image_array: np.ndarray,
    target_size: Tuple[int, int],
    preserve_aspect: bool = True,
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """Resize to target_size (height, width), stretching or padding to keep aspect ratio.

    When preserve_aspect is True the image is fit inside target_size and padded
    with zeros; otherwise it is stretched to exact dimensions. Dtype is preserved.
    """
    if len(target_size) != 2 or any(s <= 0 for s in target_size):
        raise ValueError(f"target_size must be (height, width) with positive values. Got {target_size}")

    # Readable interpolation names to cv2 constants.
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    if interpolation not in interp_map:
        raise ValueError(f"Unsupported interpolation: {interpolation}")

    interp_flag = interp_map[interpolation]
    target_h, target_w = target_size

    # cv2.resize takes (width, height), the reverse of numpy's order.
    if not preserve_aspect:
        return cv2.resize(image_array, (target_w, target_h), interpolation=interp_flag)

    # Fit the image inside target_size, leaving the constraining dimension full.
    h, w = image_array.shape[:2]
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    if aspect_ratio > target_aspect:
        # Wider than target: fit width.
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        # Taller than target: fit height.
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    resized = cv2.resize(image_array, (new_w, new_h), interpolation=interp_flag)

    # Center via symmetric padding. The "extra" side absorbs any odd-pixel
    # remainder, so before + extra sums to exactly (target - new).
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    pad_h_extra = target_h - new_h - pad_h
    pad_w_extra = target_w - new_w - pad_w

    # Zero-pad (black border), leaving any channel axis untouched.
    if image_array.ndim == 2:
        padded = np.pad(
            resized,
            ((pad_h, pad_h_extra), (pad_w, pad_w_extra)),
            mode='constant',
            constant_values=0
        )
    else:
        padded = np.pad(
            resized,
            ((pad_h, pad_h_extra), (pad_w, pad_w_extra), (0, 0)),
            mode='constant',
            constant_values=0
        )

    return padded


def to_grayscale(
    image_array: np.ndarray,
    force: bool = False
) -> np.ndarray:
    """Convert a color image to single-channel grayscale, preserving dtype.

    Already-grayscale input is returned unchanged unless force is True.
    """
    if image_array.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {image_array.shape}")

    if image_array.ndim == 2:
        # Already grayscale; force just yields a same-dtype copy for consistency.
        if force:
            return image_array.astype(image_array.dtype)
        return image_array

    # cv2 treats color input as BGR; preserve dtype for downstream steps.
    grayscale = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    return grayscale.astype(image_array.dtype)


def reduce_noise(
    image_array: np.ndarray,
    method: str = 'bilateral',
    kernel_size: int = 5,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """Denoise via bilateral, gaussian, morphological, or median filtering.

    bilateral and median preserve edges; gaussian blurs them; morphological
    suppresses salt-and-pepper specks. kernel_size must be positive and odd.
    """
    if method not in ('bilateral', 'gaussian', 'morphological', 'median'):
        raise ValueError(f"Unsupported method: {method}")

    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive and odd. Got {kernel_size}")

    if method == 'bilateral':
        # Edge-preserving: sigma_color bounds intensity blending, sigma_space the
        # spatial reach. Needs uint8 input.
        return cv2.bilateralFilter(
            image_array.astype(np.uint8),
            kernel_size,
            sigma_color,
            sigma_space
        )

    elif method == 'gaussian':
        # Fast, but blurs edges along with noise.
        return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)

    elif method == 'morphological':
        # Opening clears bright specks, closing fills dark holes.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(image_array.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    elif method == 'median':
        # Neighbourhood median; strong against salt-and-pepper noise.
        return cv2.medianBlur(image_array.astype(np.uint8), kernel_size)
