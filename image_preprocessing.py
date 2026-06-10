"""
Modular, Composable Image Preprocessing Pipeline (backward-compatible facade).

This module used to contain the entire preprocessing implementation in one
file. As of the 2026-06 refactor, the implementation has been split into the
``preprocessing/`` package for readability and maintainability:

    preprocessing/
        transforms.py  — to_grayscale, resize_image, normalize_image, reduce_noise
        vectorize.py   — vectorize_image (flat + VGG16)
        reduce.py      — reduce_dimensions (None / PCA / Johnson-Lindenstrauss)
        pipeline.py    — ImagePipeline, batch_process, compose, pipeline_decorator
        io.py          — load_image_from_{bytes,file,pil}

This file is preserved as a thin re-export layer so that every previous
import such as::

    from image_preprocessing import ImagePipeline, vectorize_image, batch_process

continues to work unchanged.

New in 2026-06: an optional dimensionality-reduction step is appended to the
pipeline after vectorization. See :func:`reduce_dimensions` and the new
``'reduce'`` pipeline op for usage. With ``method=None`` (the default) the
step is a pure no-op, so existing pipelines produce bit-identical output.

Example Usage:
    >>> from image_preprocessing import ImagePipeline
    >>> pipeline = ImagePipeline([
    ...     ('grayscale', {'force': True}),
    ...     ('resize', {'target_size': (64, 64)}),
    ...     ('normalize', {'method': 'minmax'}),
    ...     ('vectorize', {}),
    ...     ('reduce', {'method': 'pca', 'n_components': 64}),  # new (optional)
    ... ])
    >>> features = pipeline.process(image_array)             # per-image
    >>> X = batch_process([image_array] * 200, pipeline)     # applies PCA on the batch

Requirements:
    - numpy
    - opencv-python (cv2)
    - scikit-image
    - Pillow
    - tensorflow / keras (optional, required for VGG16 embeddings)
    - scikit-learn (optional, required for PCA / Johnson-Lindenstrauss reduction)
"""

# Re-export the entire public API from the preprocessing package.
from preprocessing import (
    BATCH_LEVEL_OPS,
    ImagePipeline,
    batch_process,
    compose,
    load_image_from_bytes,
    load_image_from_file,
    load_image_from_pil,
    normalize_image,
    pipeline_decorator,
    reduce_dimensions,
    reduce_noise,
    resize_image,
    to_grayscale,
    vectorize_image,
)

# The legacy module exposed _vgg16_models as a module-level cache. Some users
# may import it directly; preserve the alias for backward compatibility.
from preprocessing.vectorize import _vgg16_models  # noqa: F401

__all__ = [
    'BATCH_LEVEL_OPS',
    'ImagePipeline',
    'batch_process',
    'compose',
    'load_image_from_bytes',
    'load_image_from_file',
    'load_image_from_pil',
    'normalize_image',
    'pipeline_decorator',
    'reduce_dimensions',
    'reduce_noise',
    'resize_image',
    'to_grayscale',
    'vectorize_image',
]


if __name__ == "__main__":
    # Example usage demonstrating all functions (kept identical to the
    # pre-refactor demo so anyone using `python image_preprocessing.py` for
    # smoke testing sees the same output shape & flow).
    import numpy as np
    from functools import partial

    print("=" * 70)
    print("Image Preprocessing Pipeline - Example Usage")
    print("=" * 70)

    # Create sample image
    sample_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    print(f"\n1. Original image shape: {sample_image.shape}")

    # Individual function examples
    print("\n2. Individual Function Examples:")

    gray = to_grayscale(sample_image)
    print(f"   - Grayscale: {gray.shape}")

    resized = resize_image(sample_image, (64, 64), preserve_aspect=True)
    print(f"   - Resized: {resized.shape}")

    normalized = normalize_image(sample_image, method='minmax')
    print(f"   - Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    denoised = reduce_noise(sample_image, method='bilateral')
    print(f"   - Denoised: {denoised.shape}")

    vectorized = vectorize_image(sample_image, method='flat')
    print(f"   - Vectorized (flat): {vectorized.shape}")

    try:
        vgg_features = vectorize_image(sample_image, method='vgg16')
        print(f"   - Vectorized (vgg16): {vgg_features.shape}")
    except ImportError as exc:
        print(f"   - VGG16 skipped: {exc}")

    # Pipeline class example
    print("\n3. Pipeline Class Example:")
    pipeline = ImagePipeline([
        ('grayscale', {}),
        ('resize', {'target_size': (64, 64), 'preserve_aspect': True}),
        ('denoise', {'method': 'bilateral', 'kernel_size': 5}),
        ('normalize', {'method': 'minmax'}),
        ('vectorize', {})
    ])
    print(f"   Pipeline: {pipeline}")

    features = pipeline.process(sample_image)
    print(f"   Output shape: {features.shape}")
    print(f"   Output dtype: {features.dtype}")
    print(f"   Value range: [{features.min():.3f}, {features.max():.3f}]")

    # Functional composition example
    print("\n4. Functional Composition Example:")

    gray_op = partial(to_grayscale)
    resize_op = partial(resize_image, target_size=(64, 64))
    norm_op = partial(normalize_image, method='minmax')
    vectorize_op = vectorize_image

    composed_pipeline = compose(vectorize_op, norm_op, resize_op, gray_op)
    features_composed = composed_pipeline(sample_image)
    print(f"   Composed pipeline output: {features_composed.shape}")

    # Batch processing example
    print("\n5. Batch Processing Example:")
    batch_images = [
        np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(4)
    ]
    batch_output = batch_process(batch_images, pipeline)
    print(f"   Batch input: {len(batch_images)} images of shape {batch_images[0].shape}")
    print(f"   Batch output shape: {batch_output.shape}")

    # NEW: dimensionality-reduction demo
    print("\n6. NEW: Dimensionality Reduction Demo:")
    big_batch = [
        np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(80)
    ]

    # 6a. None — bypass (backward-compatible default)
    bypass_pipeline = ImagePipeline([
        ('grayscale', {}),
        ('resize', {'target_size': (64, 64)}),
        ('normalize', {'method': 'minmax'}),
        ('vectorize', {}),
        ('reduce', {'method': None}),
    ])
    X_bypass = batch_process(big_batch, bypass_pipeline)
    print(f"   - method=None    shape: {X_bypass.shape}")

    # 6b. PCA — variance-preserving linear projection
    try:
        pca_pipeline = ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (64, 64)}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {'method': 'pca', 'n_components': 32, 'random_state': 42}),
        ])
        X_pca = batch_process(big_batch, pca_pipeline)
        print(f"   - method=pca     shape: {X_pca.shape}")
    except ImportError as exc:
        print(f"   - PCA skipped: {exc}")

    # 6c. Johnson–Lindenstrauss — data-independent random projection
    try:
        jl_pipeline = ImagePipeline([
            ('grayscale', {}),
            ('resize', {'target_size': (64, 64)}),
            ('normalize', {'method': 'minmax'}),
            ('vectorize', {}),
            ('reduce', {'method': 'johnson_lindenstrauss',
                        'n_components': 64, 'random_state': 42}),
        ])
        X_jl = batch_process(big_batch, jl_pipeline)
        print(f"   - method=JL      shape: {X_jl.shape}")
    except ImportError as exc:
        print(f"   - JL skipped: {exc}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
