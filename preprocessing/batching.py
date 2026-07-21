"""Batch execution of an :class:`ImagePipeline` over many images.

Splits a pipeline into its per-image segment and its batch-level segment, runs
the per-image ops on each image, stacks the results, then applies the
batch-level ``'reduce'`` / ``'scale'`` ops once across the full stack.
"""

from typing import List

import numpy as np

from .pipeline import ImagePipeline


def batch_process(
    images: List[np.ndarray],
    pipeline: ImagePipeline,
) -> np.ndarray:
    """
    Apply a pipeline to a batch of images.

    The pipeline is split into two segments:

    1. **Per-image** ops (grayscale, resize, denoise, normalize, vectorize) are
       run independently on each image and the results are stacked. With a
       ``'vectorize'`` step each image is a 1D vector, so the stack is a
       ``(n_images, n_features)`` matrix; without it each image stays a matrix,
       so the stack is ``(n_images, height, width)`` for grayscale or
       ``(n_images, height, width, channels)`` for colour input.
    2. **Batch-level** ops — ``'reduce'`` and/or ``'scale'`` — are then applied
       once to that stack, in order. This is where the reducer / scaler is fit
       (each needs multiple samples to be meaningful). Vector ``'reduce'``
       methods and ``'scale'`` consume the 2D matrix; matrix ``'reduce'`` methods
       consume the 3D/4D image stack.

    For pipelines without any batch-level ops, this is equivalent to calling
    ``pipeline.process`` on each image and stacking the results.

    Args:
        images: List of image arrays.
        pipeline: :class:`ImagePipeline` instance.

    Returns:
        ``np.ndarray`` whose leading axis is ``n_images``. The trailing shape is
        the vectorized width (vector pipelines), the image height/width (matrix
        pipelines without ``'reduce'``), or the post-reduction shape when a
        ``'reduce'`` step is present.

    Notes:
        Each batch-level ``'reduce'`` op fits a *fresh* reducer on the supplied
        images. To reuse the same projection across train/val/test splits, use
        the pipeline's :meth:`ImagePipeline.fit_transform` (on train) and
        :meth:`ImagePipeline.transform` (on val/test) instead — they store and
        reuse the fitted reducer. (At a lower level you can also call
        :func:`reduce_dimensions` directly with ``return_reducer=True`` and pass
        the fitted ``reducer`` back in; see ``integration_example.py``.)
    """
    # ---- Step 1: per-image segment ------------------------------------------
    # Build a temporary pipeline containing only the per-image ops so that we
    # can reuse ImagePipeline.process for its error wrapping.
    per_image_ops = pipeline.per_image_operations()
    batch_ops = pipeline.batch_operations()

    if per_image_ops:
        per_image_pipeline = ImagePipeline(per_image_ops)
        processed = [per_image_pipeline.process(img) for img in images]
    else:
        # Degenerate but legal: pipeline only contains batch-level ops.
        processed = [img for img in images]

    # Stack per-image outputs. All must share a shape: a 1D vector each (after
    # 'vectorize') stacks to 2D; a 2D matrix each (no 'vectorize') stacks to 3D.
    features = np.array(processed)

    # ---- Step 2: batch-level segment ---------------------------------------
    # Apply each batch-level op once to the whole stack. 'reduce' reads the
    # stack rank according to its method: vector methods expect the 2D matrix,
    # matrix methods expect the 3D image stack.
    for op_name, kwargs in batch_ops:
        operation = ImagePipeline.OPERATIONS[op_name]
        try:
            features = operation(features, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Pipeline failed at batch-level operation '{op_name}' with "
                f"kwargs {kwargs}. Error: {str(e)}"
            ) from e

    return features
