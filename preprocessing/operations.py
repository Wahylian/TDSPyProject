"""Shared operation registry for the preprocessing pipeline.

Maps operation names to their callables and records which ones act on a batch
rather than a single image, so ImagePipeline and batch_process agree on one
definition.
"""

from typing import Callable, Dict

from .transforms import normalize_image, reduce_noise, resize_image, to_grayscale
from .vectorize import vectorize_image
from .reduce import reduce_dimensions
from .scale import standardize_features

# Ops that fit statistics across the batch ('reduce' learns a projection,
# 'scale' learns per-feature mean/std). Marks where the chain splits.
BATCH_LEVEL_OPS = frozenset({'reduce', 'scale'})

# Operation name to callable. Batch-level ops appear here too so validation accepts them.
OPERATIONS: Dict[str, Callable] = {
    'vectorize': vectorize_image,
    'normalize': normalize_image,
    'resize': resize_image,
    'grayscale': to_grayscale,
    'denoise': reduce_noise,
    'reduce': reduce_dimensions,
    'scale': standardize_features,
}
