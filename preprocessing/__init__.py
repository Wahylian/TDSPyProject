"""Public API for the image-preprocessing package.

Import everything from here rather than the internal submodules, which may be
reorganized freely as long as these names keep being re-exported. ``__all__``
lists the supported surface; anything else is internal.

Key semantics callers must know:

- ``'vectorize'`` is optional. Include it (with a ``'vec-*'`` reduction) to get
  flat vectors for classical models; omit it (with a ``'mat-*'`` reduction) to
  keep each image a 2D matrix for CNNs/ViTs.
- Every op is per-image except the batch-level ``'reduce'`` and ``'scale'``,
  which fit statistics across the whole batch. Use ``batch_process`` for a
  one-shot transform, or ``fit_transform``/``transform`` to fit once on train
  and reuse the same projection on val/test.

Optional deps: keras (method='vgg16'), scikit-learn (vec-pca/vec-jl reductions).
"""

from preprocessing.transforms import (
    normalize_image,
    reduce_noise,
    resize_image,
    to_grayscale,
)
from preprocessing.vectorize import vectorize_image
from preprocessing.reduce import reduce_dimensions
from preprocessing.scale import standardize_features
from preprocessing.operations import BATCH_LEVEL_OPS
from preprocessing.pipeline import ImagePipeline
from preprocessing.batching import batch_process
from preprocessing.composition import compose, pipeline_decorator
from preprocessing.io import (
    load_image_from_bytes,
    load_image_from_file,
    load_image_from_pil,
)

# Private VGG16 weight cache, re-exported so tests can seed or inspect it.
from preprocessing.vectorize import _vgg16_models  # noqa: F401

__all__ = [
    # transforms (per-image)
    'to_grayscale',
    'resize_image',
    'normalize_image',
    'reduce_noise',
    # vectorize (per-image)
    'vectorize_image',
    # reduce (batch-level)
    'reduce_dimensions',
    # scale (batch-level)
    'standardize_features',
    # pipeline / composition
    'ImagePipeline',
    'batch_process',
    'compose',
    'pipeline_decorator',
    'BATCH_LEVEL_OPS',
    # io
    'load_image_from_bytes',
    'load_image_from_file',
    'load_image_from_pil',
]
