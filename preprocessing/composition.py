"""Functional pipeline-composition helpers.

Two lightweight alternatives to the class-based :class:`ImagePipeline`: compose
plain callables right-to-left, or decorate a function so preprocessing runs
before it receives the image.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np


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
