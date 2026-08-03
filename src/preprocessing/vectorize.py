"""Flatten an image into a 1D feature vector for classical ML models.

'flat' raw-pixel flattening or 'vgg16' embeddings (block5_pool, 25088-dim).
VGG16 models are cached per input_size so the ImageNet weights load once.
"""

from typing import Tuple

import numpy as np

from .transforms import resize_image

# One cached VGG16 model per (height, width) input size.
_vgg16_models: dict = {}


def vectorize_image(
    image_array: np.ndarray,
    method: str = 'flat',
    preserve_structure: bool = False,
    input_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Flatten an image into a 1D float32 vector via 'flat' pixels or 'vgg16' embeddings.

    For 'flat', preserve_structure concatenates channels separately instead of
    interleaving. For 'vgg16', input_size sets the model's expected dimensions.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image_array)}")

    if image_array.ndim not in (2, 3):
        raise ValueError(
            f"Image must be 2D (grayscale) or 3D (color). Got shape {image_array.shape}"
        )

    if method not in ('flat', 'vgg16'):
        raise ValueError(f"Unsupported method: {method}. Choose from flat, vgg16.")

    if method == 'flat':
        # float32 for a consistent output dtype regardless of input.
        image_array = image_array.astype(np.float32)

        if image_array.ndim == 2:
            return image_array.flatten()

        # preserve_structure groups all of each channel together (R, then G, then B)
        # rather than interleaving pixels.
        if preserve_structure:
            return np.concatenate([image_array[:, :, c].flatten() for c in range(image_array.shape[2])])

        return image_array.flatten()

    # Build and cache the model on first use for this input_size.
    global _vgg16_models
    if input_size not in _vgg16_models:
        # Lazy import so Keras is only required when vgg16 is actually used.
        try:
            from keras.applications import VGG16
        except ImportError as exc:
            raise ImportError(
                "VGG16 embeddings require Keras. Install with: pip install keras"
            ) from exc

        target_h, target_w = input_size
        # include_top=False keeps the block5_pool feature maps (25088-dim at 224x224).
        _vgg16_models[input_size] = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(target_h, target_w, 3),
        )

    model = _vgg16_models[input_size]

    # VGG16 needs 3 channels; replicate grayscale across them.
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    # VGG16 requires exact dimensions, so stretch rather than pad.
    target_h, target_w = input_size
    if image_array.shape[:2] != (target_h, target_w):
        image_array = resize_image(
            image_array,
            (target_h, target_w),
            preserve_aspect=False,
            interpolation='bilinear',
        )

    # preprocess_input expects uint8 in [0, 255].
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    from keras.applications.vgg16 import preprocess_input

    # Add a batch axis, run the forward pass, flatten to 1D.
    batch = np.expand_dims(preprocess_input(image_array), axis=0)
    features = model.predict(batch, verbose=0)
    return features.reshape(-1).astype(np.float32)
