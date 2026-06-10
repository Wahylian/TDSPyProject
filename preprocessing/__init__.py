"""
Modular image preprocessing package.

Public API (re-exported here for convenience):

    Per-image transforms ............... transforms.py
        - to_grayscale
        - resize_image
        - normalize_image
        - reduce_noise

    Vectorization ...................... vectorize.py
        - vectorize_image

    Dimensionality reduction (new) ..... reduce.py
        - reduce_dimensions

    Composition / batching ............. pipeline.py
        - ImagePipeline
        - batch_process
        - compose
        - pipeline_decorator
        - BATCH_LEVEL_OPS

    I/O helpers ........................ io.py
        - load_image_from_bytes
        - load_image_from_file
        - load_image_from_pil

The legacy single-file module ``image_preprocessing.py`` at the project root
re-exports everything below, so existing code that imports from
``image_preprocessing`` continues to work unchanged.
"""

from .transforms import (
    normalize_image,
    reduce_noise,
    resize_image,
    to_grayscale,
)
from .vectorize import vectorize_image
from .reduce import reduce_dimensions
from .pipeline import (
    BATCH_LEVEL_OPS,
    ImagePipeline,
    batch_process,
    compose,
    pipeline_decorator,
)
from .io import (
    load_image_from_bytes,
    load_image_from_file,
    load_image_from_pil,
)

__all__ = [
    # transforms
    'normalize_image',
    'reduce_noise',
    'resize_image',
    'to_grayscale',
    # vectorize
    'vectorize_image',
    # reduce
    'reduce_dimensions',
    # pipeline
    'BATCH_LEVEL_OPS',
    'ImagePipeline',
    'batch_process',
    'compose',
    'pipeline_decorator',
    # io
    'load_image_from_bytes',
    'load_image_from_file',
    'load_image_from_pil',
]
