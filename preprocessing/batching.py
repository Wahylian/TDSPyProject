"""Run an ImagePipeline over many images.

Runs the per-image ops on each image, stacks the results, then applies the
batch-level 'reduce'/'scale' ops once across the full stack.
"""

from typing import List

import numpy as np

from .pipeline import ImagePipeline


def batch_process(
    images: List[np.ndarray],
    pipeline: ImagePipeline,
) -> np.ndarray:
    """Run a pipeline across a batch of images and return the stacked result.

    Per-image ops run on each image and stack; batch-level 'reduce'/'scale' then
    apply once to the stack, fitting a fresh reducer/scaler. The leading axis is
    n_images; the trailing shape depends on whether 'vectorize' and 'reduce' run.

    Each 'reduce' fits a fresh reducer on these images. To share one projection
    across train/val/test, use ImagePipeline.fit_transform / transform instead.
    """
    # Reuse ImagePipeline.process (and its error wrapping) for the per-image ops.
    per_image_ops = pipeline.per_image_operations()
    batch_ops = pipeline.batch_operations()

    if per_image_ops:
        per_image_pipeline = ImagePipeline(per_image_ops)
        processed = [per_image_pipeline.process(img) for img in images]
    else:
        # Degenerate but legal: batch-level ops only.
        processed = [img for img in images]

    # Vectors stack to 2D, matrices to 3D/4D. All items must share a shape.
    features = np.array(processed)

    # Apply each batch-level op once. Vector methods expect the 2D matrix,
    # matrix methods the 3D/4D stack.
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
