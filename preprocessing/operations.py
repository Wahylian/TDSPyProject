"""Operation registry shared by the preprocessing pipeline.

Maps each pipeline operation name to the callable that implements it, and records
which operations act on a *batch* of samples rather than a single image. Kept in
its own module so :class:`~preprocessing.pipeline.ImagePipeline` and
:func:`~preprocessing.batching.batch_process` share one definition instead of
each carrying their own copy.
"""

from typing import Callable, Dict

from .transforms import normalize_image, reduce_noise, resize_image, to_grayscale
from .vectorize import vectorize_image
from .reduce import reduce_dimensions
from .scale import standardize_features

# Pipeline operations that act on a *batch* of feature vectors rather than a
# single image. Used by ImagePipeline / batch_process to know where to split the
# chain. Both learn statistics across samples (a projection for 'reduce',
# per-feature mean/std for 'scale'). Keep this set small and explicit.
BATCH_LEVEL_OPS = frozenset({'reduce', 'scale'})

# Map operation names to functions. 'reduce' is intentionally listed here even
# though it is batch-level so that pipeline validation accepts it.
OPERATIONS: Dict[str, Callable] = {
    'vectorize': vectorize_image,
    'normalize': normalize_image,
    'resize': resize_image,
    'grayscale': to_grayscale,
    'denoise': reduce_noise,
    'reduce': reduce_dimensions,
    'scale': standardize_features,
}
