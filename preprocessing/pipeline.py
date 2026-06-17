"""
Composable preprocessing pipeline.

This module wires the per-image transforms (``transforms.py``),
the vectorization step (``vectorize.py``) and the batch-level dimensionality
reducer (``reduce.py``) into a single configurable pipeline.

Three composition patterns are exported:

- ``ImagePipeline`` — the recommended class-based, config-driven chain.
- ``compose`` — right-to-left functional composition of plain callables.
- ``pipeline_decorator`` — decorator that runs preprocessing before a
  user-supplied function receives the image.

Vectorization is optional
--------------------------
The ``'vectorize'`` step is just one op among many. Include it to produce flat
feature vectors for classical models (SVM, Random Forest). Omit it to keep each
image as a 2D/3D matrix — the form CNNs and ViTs consume directly. The
``'reduce'`` step supports both: its vector methods reduce flat vectors and its
matrix methods reduce image matrices (see :func:`reduce_dimensions`).

Per-image vs batch-level operations
-----------------------------------
All transforms (grayscale, resize, denoise, normalize, vectorize) act on a
single image. The ``'reduce'`` step acts on a *batch* — it needs multiple
samples to fit:

- ``ImagePipeline.process(image)`` runs every per-image op and treats
  ``'reduce'`` with ``method=None`` as a no-op. With a fitting method it raises
  a helpful ``ValueError`` (a reducer cannot be fit on one sample) unless the op
  kwargs include a pre-fit ``reducer``.
- ``batch_process(images, pipeline)`` splits the pipeline into per-image and
  batch-level segments, runs the per-image segment on each image, stacks the
  results — into ``(n_samples, n_features)`` when vectorized, or a matrix stack
  when not (``(n_samples, height, width)`` for grayscale,
  ``(n_samples, height, width, channels)`` for colour) — then applies the
  batch-level reducer once across the full stack.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .transforms import normalize_image, reduce_noise, resize_image, to_grayscale
from .vectorize import vectorize_image
from .reduce import reduce_dimensions

# Pipeline operations that act on a *batch* of feature vectors rather than a
# single image. Used by ImagePipeline / batch_process to know where to split
# the chain. Keep this set small and explicit.
BATCH_LEVEL_OPS = frozenset({'reduce'})


class ImagePipeline:
    """
    Composable image preprocessing pipeline.

    Chains multiple preprocessing operations in sequence, allowing flexible
    configuration and reuse. Each operation is specified as a tuple of
    ``(function_name, kwargs)``.

    Supported operations:
        - ``'grayscale'`` — Convert to grayscale (:func:`to_grayscale`)
        - ``'resize'`` — Resize image (:func:`resize_image`)
        - ``'denoise'`` — Reduce noise (:func:`reduce_noise`)
        - ``'normalize'`` — Normalize pixel values (:func:`normalize_image`)
        - ``'vectorize'`` — Optionally flatten the image to a 1D feature vector
          (:func:`vectorize_image`). Omit it to keep a 2D/3D matrix for
          CNN/ViT inputs.
        - ``'reduce'`` — Optional dimensionality reduction
          (:func:`reduce_dimensions`). Vector methods (``'vec-pca'`` /
          ``'vec-jl'``) reduce flat vectors; matrix methods (``'mat-pca'`` /
          ``'mat-jl'``) reduce image matrices; ``None`` is a bypass. This is a
          *batch-level* op and is only meaningful inside :func:`batch_process`;
          on a single image it is a no-op for ``method=None`` and an error for
          the fitting methods (they require multiple samples to fit).

    Example:
        >>> # Vector pipeline: flatten, then reduce the flat vectors with PCA.
        >>> pipeline = ImagePipeline([
        ...     ('grayscale', {}),
        ...     ('resize', {'target_size': (64, 64)}),
        ...     ('normalize', {'method': 'minmax'}),
        ...     ('vectorize', {}),
        ...     ('reduce', {'method': 'vec-pca', 'n_components': 64}),
        ... ])
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> features = pipeline.process(image)            # per-image only
        >>> batch = batch_process([image]*200, pipeline)  # fits PCA across the batch
        >>> # Matrix pipeline: no 'vectorize', reduce the image matrices directly.
        >>> matrix_pipeline = ImagePipeline([
        ...     ('grayscale', {}),
        ...     ('resize', {'target_size': (64, 64)}),
        ...     ('normalize', {'method': 'minmax'}),
        ...     ('reduce', {'method': 'mat-pca', 'n_components': 16}),
        ... ])
        >>> batch = batch_process([image]*200, matrix_pipeline)  # (200, 64, 16)
    """

    # Map operation names to functions. ``'reduce'`` is intentionally listed
    # here even though it is batch-level so that pipeline validation accepts it.
    OPERATIONS: Dict[str, Callable] = {
        'vectorize': vectorize_image,
        'normalize': normalize_image,
        'resize': resize_image,
        'grayscale': to_grayscale,
        'denoise': reduce_noise,
        'reduce': reduce_dimensions,
    }

    def __init__(self, operations: List[Tuple[str, Dict[str, Any]]]):
        """
        Initialize pipeline with ordered operations.

        Args:
            operations: List of ``(operation_name, kwargs)`` tuples.
                Example: ``[('grayscale', {}), ('resize', {'target_size': (64, 64)})]``

        Raises:
            ValueError: If any operation name is not supported.
        """
        self.operations = operations

        for op_name, _ in operations:
            if op_name not in self.OPERATIONS:
                raise ValueError(
                    f"Unknown operation: {op_name}. "
                    f"Supported: {', '.join(self.OPERATIONS.keys())}"
                )

    # --- New split helpers (used by batch_process) ------------------------

    def per_image_operations(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return only the operations that act on a single image."""
        return [(n, k) for n, k in self.operations if n not in BATCH_LEVEL_OPS]

    def batch_operations(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return only the operations that act on a batch of feature vectors."""
        return [(n, k) for n, k in self.operations if n in BATCH_LEVEL_OPS]

    # --- Existing per-image entry point (backward compatible) -------------

    def process(self, image_array: np.ndarray) -> np.ndarray:
        """
        Apply all pipeline operations in sequence to a single image.

        Args:
            image_array: Input image as np.ndarray.

        Returns:
            Processed image / vector after all operations.

            For a pipeline that ends with ``('reduce', {'method': None, ...})``
            this returns the per-image output unchanged (the reduce step is a
            no-op for ``method=None``). For a fitting method (``vec-*`` /
            ``mat-*``), calling ``process`` on a single image without a pre-fit
            ``reducer`` raises ``ValueError`` — use :func:`batch_process`
            instead, which handles the batch-level reduction.

        Raises:
            RuntimeError: If any operation fails (with context about which
                operation and its parameters).
        """
        result = image_array.copy()

        for op_name, kwargs in self.operations:
            try:
                operation = self.OPERATIONS[op_name]
                result = operation(result, **kwargs)
            except Exception as e:
                # Wrap the original error with pipeline context so it's clear
                # which step failed and with what parameters.
                raise RuntimeError(
                    f"Pipeline failed at operation '{op_name}' with kwargs {kwargs}. "
                    f"Error: {str(e)}"
                ) from e

        return result

    def add_operation(
        self,
        operation_name: str,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an operation to the end of the pipeline.

        Args:
            operation_name: Name of operation to add.
            kwargs: Operation parameters. Default: ``{}``.

        Raises:
            ValueError: If operation name is not supported.
        """
        if operation_name not in self.OPERATIONS:
            raise ValueError(f"Unknown operation: {operation_name}")

        self.operations.append((operation_name, kwargs or {}))

    def __repr__(self) -> str:
        """Return string representation of pipeline."""
        ops_str = " -> ".join(
            f"{name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
            for name, kwargs in self.operations
        )
        return f"ImagePipeline([{ops_str}])"


# ============================================================================
# Functional composition helpers
# ============================================================================

def compose(*functions: Callable) -> Callable:
    """
    Compose functions right-to-left (mathematical composition).

    Creates a single function that applies provided functions in sequence,
    with output of one function fed to the next.

    Args:
        *functions: Variable number of callable functions.

    Returns:
        Composed function that applies all operations in sequence.

    Example:
        >>> from functools import partial
        >>> resize_64 = partial(resize_image, target_size=(64, 64))
        >>> normalize_minmax = partial(normalize_image, method='minmax')
        >>> vectorize = vectorize_image
        >>> pipeline = compose(vectorize, normalize_minmax, resize_64)
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>> features = pipeline(image)
    """
    def composed(arg: Any) -> Any:
        result = arg
        # reversed() because compose() is right-to-left: compose(f, g, h)(x) = f(g(h(x)))
        for func in reversed(functions):
            result = func(result)
        return result

    return composed


def pipeline_decorator(*operations: Tuple[Callable, Dict[str, Any]]) -> Callable:
    """
    Decorator to create preprocessing pipeline from decorated function.

    Allows decorating a custom function with preprocessing operations
    that are automatically applied before the function receives the image.

    Args:
        *operations: Variable number of ``(function, kwargs)`` tuples.

    Returns:
        Decorator that wraps a function with preprocessing.

    Example:
        >>> @pipeline_decorator(
        ...     (to_grayscale, {}),
        ...     (lambda x: resize_image(x, (64, 64)), {}),
        ...     (vectorize_image, {})
        ... )
        >>> def extract_features(image):
        ...     return image  # Already preprocessed
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(image_array: np.ndarray) -> Any:
            result = image_array.copy()
            for operation, kwargs in operations:
                result = operation(result, **kwargs)
            return func(result)
        return wrapper

    return decorator


# ============================================================================
# Batch processing
# ============================================================================

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
    2. **Batch-level** ops — currently only ``'reduce'`` — are then applied
       once to that stack. This is where the reducer is fit (it needs multiple
       samples to be meaningful). Vector ``'reduce'`` methods consume the 2D
       matrix; matrix methods consume the 3D/4D image stack.

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
        images. To reuse the same projection across train/test splits, call
        :func:`reduce_dimensions` directly with ``return_reducer=True`` on the
        training features and apply the fitted ``reducer`` to the test split
        (``reduce_dimensions(test_features, reducer=fitted_reducer)``). See
        ``integration_example.py`` for a worked example.
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
