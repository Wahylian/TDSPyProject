"""
Image vectorization.

Converts a 2D/3D image into a 1D feature vector suitable for traditional ML
models (SVM, Random Forest, etc.). Supports two methods:

- ``'flat'``: Raw pixel flattening (optionally channel-preserving).
- ``'vgg16'``: Pre-trained VGG16 feature embedding (block5_pool, 25088-dim).

VGG16 model instances are cached at module level so the ~500 MB ImageNet
weights are only loaded once per ``input_size``.
"""

from typing import Tuple

import numpy as np

from .transforms import resize_image

# Module-level cache: one VGG16 model per (height, width) input size.
# Keeps memory usage predictable across repeated vectorize() calls.
_vgg16_models: dict = {}


def vectorize_image(
    image_array: np.ndarray,
    method: str = 'flat',
    preserve_structure: bool = False,
    input_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Convert a 2D/3D image array into a 1D feature vector for ML models.

    Args:
        image_array: Input image as np.ndarray (2D for grayscale, 3D for color).
                     Expected shape: (height, width) or (height, width, channels).
        method: Vectorization method. Options:
            - 'flat': Flatten raw pixels into a 1D vector (default)
            - 'vgg16': Extract features from pre-trained VGG16 (block5_pool layer)
        preserve_structure: For method='flat', if True, concatenates channels
                          separately. Default: False.
        input_size: For method='vgg16', target (height, width). Default: (224, 224).

    Returns:
        1D feature vector of dtype float32.

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If image has unsupported dimensions or method is unknown.
        ImportError: If method='vgg16' and TensorFlow/Keras is not installed.

    Example:
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> vectorize_image(image, method='flat').shape
        (150528,)
        >>> vectorize_image(image, method='vgg16').shape
        (25088,)
    """
    # --- Input validation ---
    if not isinstance(image_array, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image_array)}")

    if image_array.ndim not in (2, 3):
        raise ValueError(
            f"Image must be 2D (grayscale) or 3D (color). Got shape {image_array.shape}"
        )

    if method not in ('flat', 'vgg16'):
        raise ValueError(f"Unsupported method: {method}. Choose from flat, vgg16.")

    # --- Flat pixel vectorization ---
    if method == 'flat':
        # Cast to float32 so the output dtype is consistent regardless of input dtype
        image_array = image_array.astype(np.float32)

        # Grayscale: single channel, flatten directly
        if image_array.ndim == 2:
            return image_array.flatten()

        # Color with preserve_structure: flatten each channel separately and concatenate.
        # This keeps all R pixels together, then all G, then all B (rather than interleaved),
        # which can benefit models sensitive to channel-wise spatial patterns.
        if preserve_structure:
            return np.concatenate([image_array[:, :, c].flatten() for c in range(image_array.shape[2])])

        # Default: flatten the entire array in row-major (C) order, interleaving channels
        return image_array.flatten()

    # --- VGG16 deep feature extraction ---
    # _vgg16_models is a module-level dict caching one model per input_size.
    # This avoids reloading the ~500MB ImageNet weights on every call.
    global _vgg16_models
    if input_size not in _vgg16_models:
        # Lazy import: only require Keras if vgg16 method is actually used
        try:
            from keras.applications import VGG16
        except ImportError as exc:
            raise ImportError(
                "VGG16 embeddings require Keras. Install with: pip install keras"
            ) from exc

        target_h, target_w = input_size
        # include_top=False removes the final classification layers,
        # returning the block5_pool feature maps instead (25088-dim for 224x224)
        _vgg16_models[input_size] = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(target_h, target_w, 3),
        )

    # Retrieve the cached model for this input size
    model = _vgg16_models[input_size]

    # VGG16 expects 3-channel input; replicate single channel across all 3
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    # Resize to the model's expected spatial dimensions if needed
    target_h, target_w = input_size
    if image_array.shape[:2] != (target_h, target_w):
        image_array = resize_image(
            image_array,
            (target_h, target_w),
            preserve_aspect=False,  # VGG16 requires exact dimensions
            interpolation='bilinear',
        )

    # preprocess_input expects uint8 in [0, 255]; clip guards against out-of-range floats
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    from keras.applications.vgg16 import preprocess_input

    # Add batch dimension (1, H, W, C), run forward pass, flatten to 1D feature vector
    batch = np.expand_dims(preprocess_input(image_array), axis=0)
    features = model.predict(batch, verbose=0)
    return features.reshape(-1).astype(np.float32)
