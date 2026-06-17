"""
Per-image preprocessing transforms.

Each function takes a single image (`np.ndarray`) and returns a transformed
image. These are the low-level building blocks used by ``ImagePipeline``.

Functions:
    - ``to_grayscale``: BGR/RGB color image -> single-channel grayscale.
    - ``resize_image``: resize with optional aspect-ratio preservation + padding.
    - ``normalize_image``: scale pixel values (minmax / standard / histogram).
    - ``reduce_noise``: denoising filters (bilateral / gaussian / morphological / median).
"""

from typing import Tuple

import numpy as np
import cv2


def normalize_image(
    image_array: np.ndarray,
    method: str = 'minmax',
    value_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Normalize pixel values to a specified range.

    Applies various normalization techniques to scale pixel intensities,
    improving model convergence and performance.

    Args:
        image_array: Input image as np.ndarray (uint8 or float).
        method: Normalization method. Options:
            - 'minmax': Scale to [value_range[0], value_range[1]] (default)
            - 'standard': Zero-mean, unit-variance standardization
            - 'histogram': Histogram equalization (grayscale only)
        value_range: Target range for minmax normalization. Default: (0.0, 1.0).

    Returns:
        Normalized array of dtype float32.

    Raises:
        ValueError: If method is unsupported or histogram equalization used on color image.

    Example:
        >>> image = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        >>> normalized = normalize_image(image, method='minmax')
        >>> normalized.min(), normalized.max()
        (0.0, 1.0)
    """
    if method not in ('minmax', 'standard', 'histogram'):
        raise ValueError(f"Unsupported method: {method}. Choose from minmax, standard, histogram.")

    # Work in float32 throughout to avoid integer overflow/truncation
    image_array = image_array.astype(np.float32)

    if method == 'minmax':
        min_val = image_array.min()
        max_val = image_array.max()

        # Guard against flat images (all pixels identical) to avoid division by zero
        if max_val == min_val:
            return np.full_like(image_array, (value_range[0] + value_range[1]) / 2)

        # Scale to [0, 1] first, then stretch to the requested value_range
        normalized = (image_array - min_val) / (max_val - min_val)
        return normalized * (value_range[1] - value_range[0]) + value_range[0]

    elif method == 'standard':
        mean = image_array.mean()
        std = image_array.std()

        # Guard against zero-std images (e.g. solid-colour patches) to avoid division by zero
        if std == 0:
            return np.zeros_like(image_array)

        # z-score: (x - mean) / std  →  zero mean, unit variance
        return (image_array - mean) / std

    elif method == 'histogram':
        if image_array.ndim == 3:
            raise ValueError("Histogram equalization requires grayscale image (2D).")

        # cv2.equalizeHist requires uint8 input
        uint8_image = np.clip(image_array, 0, 255).astype(np.uint8)
        equalized = cv2.equalizeHist(uint8_image)
        # Rescale back to [0, 1] float32 for consistency with other methods
        return equalized.astype(np.float32) / 255.0


def resize_image(
    image_array: np.ndarray,
    target_size: Tuple[int, int],
    preserve_aspect: bool = True,
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """
    Resize image to target dimensions.

    Supports aspect-ratio-preserving resizing with optional padding,
    multiple interpolation methods, and works with both grayscale and color images.

    Args:
        image_array: Input image as np.ndarray (2D or 3D).
        target_size: Target size as (height, width).
        preserve_aspect: If True, preserves aspect ratio and pads to target_size.
                        If False, stretches to exact target_size. Default: True.
        interpolation: Interpolation method. Options:
            - 'bilinear': Bilinear interpolation (default, good balance)
            - 'nearest': Nearest neighbor (fast, blocky)
            - 'bicubic': Bicubic interpolation (slower, smoother)
            - 'lanczos': Lanczos (high-quality, slowest)

    Returns:
        Resized array of same dtype as input.

    Raises:
        ValueError: If target_size is invalid or interpolation method is unsupported.

    Example:
        >>> image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> resized = resize_image(image, (224, 224), preserve_aspect=True)
        >>> resized.shape[0], resized.shape[1]
        (224, 224)
    """
    if len(target_size) != 2 or any(s <= 0 for s in target_size):
        raise ValueError(f"target_size must be (height, width) with positive values. Got {target_size}")

    # Map readable interpolation names to cv2 constants
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

    # Fast path: stretch directly to exact dimensions, ignoring aspect ratio
    if not preserve_aspect:
        # Note: cv2.resize takes (width, height), opposite of numpy convention
        return cv2.resize(image_array, (target_w, target_h), interpolation=interp_flag)

    # Preserve aspect ratio with padding
    h, w = image_array.shape[:2]
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    # Determine which dimension is the constraining one
    if aspect_ratio > target_aspect:
        # Image is wider than target: fit width, let height shrink
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        # Image is taller than target: fit height, let width shrink
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    resized = cv2.resize(image_array, (new_w, new_h), interpolation=interp_flag)

    # Compute symmetric padding so the resized image is centred in target_size.
    # `pad_*` is the "before" (top/left) amount; `pad_*_extra` is the "after"
    # (bottom/right) amount and already equals target - new - pad_*, so it
    # absorbs any odd-pixel remainder. The two must sum to (target - new) — do
    # NOT add pad_* into the "after" slot or the image overshoots target_size.
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    pad_h_extra = target_h - new_h - pad_h
    pad_w_extra = target_w - new_w - pad_w

    # Pad with zeros (black border); channel axis untouched for color images
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
    """
    Convert image to grayscale.

    Converts color (RGB/BGR) images to single-channel grayscale using standard
    weights. Grayscale images are returned unchanged unless force=True.

    Args:
        image_array: Input image as np.ndarray (2D for grayscale, 3D for color).
        force: If True, applies grayscale conversion even if already grayscale.
               Default: False.

    Returns:
        Grayscale array of shape (height, width) and dtype same as input.

    Raises:
        ValueError: If image has unexpected number of dimensions.

    Example:
        >>> color_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> gray = to_grayscale(color_image)
        >>> gray.shape
        (224, 224)
    """
    if image_array.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {image_array.shape}")

    # Already grayscale (2D)
    if image_array.ndim == 2:
        if force:
            # force=True on an already-grayscale image is a no-op by design;
            # returns a same-dtype copy to satisfy pipeline consistency
            return image_array.astype(image_array.dtype)
        return image_array

    # Convert 3D color (BGR, as loaded by OpenCV) to 2D grayscale.
    # cv2.COLOR_BGR2GRAY applies luminance weights: 0.114*B + 0.587*G + 0.299*R
    grayscale = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    # Preserve the original dtype so downstream steps don't see unexpected type changes
    return grayscale.astype(image_array.dtype)


def reduce_noise(
    image_array: np.ndarray,
    method: str = 'bilateral',
    kernel_size: int = 5,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Reduce noise in image using various filtering techniques.

    Applies filtering to suppress noise while preserving edges and important features.
    Multiple methods available for different noise characteristics.

    Args:
        image_array: Input image as np.ndarray (uint8 preferred for bilateral).
        method: Noise reduction method. Options:
            - 'bilateral': Bilateral filtering (edge-preserving, default)
            - 'gaussian': Gaussian blur (simple but blurs edges)
            - 'morphological': Morphological opening+closing (binary-like noise)
            - 'median': Median filtering (good for salt-and-pepper noise)
        kernel_size: Kernel size for filtering (must be odd). Default: 5.
        sigma_color: Color sigma for bilateral filter. Default: 75.0.
        sigma_space: Spatial sigma for bilateral filter. Default: 75.0.

    Returns:
        Filtered array of same shape and dtype as input.

    Raises:
        ValueError: If method is unsupported or kernel_size is even/invalid.

    Example:
        >>> noisy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> denoised = reduce_noise(noisy_image, method='bilateral')
        >>> denoised.shape == noisy_image.shape
        True
    """
    if method not in ('bilateral', 'gaussian', 'morphological', 'median'):
        raise ValueError(f"Unsupported method: {method}")

    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive and odd. Got {kernel_size}")

    if method == 'bilateral':
        # Edge-preserving filter: smooths regions while keeping sharp boundaries.
        # sigma_color controls how different pixel intensities can be blended;
        # sigma_space controls how far spatially pixels influence each other.
        # Requires uint8 input.
        return cv2.bilateralFilter(
            image_array.astype(np.uint8),
            kernel_size,
            sigma_color,
            sigma_space
        )

    elif method == 'gaussian':
        # Simple Gaussian blur — fast but blurs edges along with noise
        return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)

    elif method == 'morphological':
        # Opening (erosion then dilation) removes small bright specks;
        # closing (dilation then erosion) fills small dark holes.
        # Together they suppress salt-and-pepper style structural noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(image_array.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    elif method == 'median':
        # Replaces each pixel with the median of its neighbourhood —
        # very effective against salt-and-pepper noise while preserving edges
        return cv2.medianBlur(image_array.astype(np.uint8), kernel_size)
