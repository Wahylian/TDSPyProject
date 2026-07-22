"""Functional pipeline-composition helpers.

Two lightweight alternatives to the class-based :class:`ImagePipeline`: compose
plain callables right-to-left, or decorate a function so preprocessing runs
before it receives the image.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np


def compose(*functions: Callable) -> Callable:
    """Compose callables right-to-left: compose(f, g, h)(x) == f(g(h(x)))."""
    def composed(arg: Any) -> Any:
        result = arg
        # Right-to-left, so iterate in reverse.
        for func in reversed(functions):
            result = func(result)
        return result

    return composed


def pipeline_decorator(*operations: Tuple[Callable, Dict[str, Any]]) -> Callable:
    """Wrap a function so the given (callable, kwargs) ops run on the image first."""
    def decorator(func: Callable) -> Callable:
        def wrapper(image_array: np.ndarray) -> Any:
            result = image_array.copy()
            for operation, kwargs in operations:
                result = operation(result, **kwargs)
            return func(result)
        return wrapper

    return decorator
