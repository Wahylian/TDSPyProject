"""ImagePipeline: the config-driven chain of preprocessing operations.

Wires the per-image transforms, the optional 'vectorize' step, and the
batch-level 'reduce'/'scale' steps, reading its op registry from operations.py.

'vectorize' is optional: include it for flat vectors (classical models), omit it
to keep 2D/3D matrices (CNNs/ViTs); 'reduce' handles both shapes. Every op is
per-image except 'reduce' and 'scale', which fit across a batch. process() runs
the per-image ops and either reuses a stored reducer or, unfitted, treats a
None-method reduce as a no-op and errors on a fitting one. Use batch_process or
fit_transform/transform for the batch-level steps.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Shared with batch_process via operations.py so both use one definition.
from .operations import BATCH_LEVEL_OPS, OPERATIONS


class ImagePipeline:
    """Chain of preprocessing operations, each a (name, kwargs) tuple.

    Supported ops: 'grayscale', 'resize', 'denoise', 'normalize', 'vectorize'
    (per-image), plus the batch-level 'reduce' (vec-*/mat-* projection or None
    bypass) and 'scale' (per-feature standardization). Batch-level ops only fit
    inside batch_process or fit_transform. For a train/val/test workflow, call
    fit_transform on train then transform on each split so the PCA/JL basis stays
    consistent; batch_process instead re-fits a fresh reducer each call.
    """

    # Exposed as a class attribute so batch_process can reach it.
    OPERATIONS = OPERATIONS

    def __init__(self, operations: List[Tuple[str, Dict[str, Any]]]):
        """Store the ordered (name, kwargs) ops, rejecting any unknown name."""
        self.operations = operations

        for op_name, _ in operations:
            if op_name not in self.OPERATIONS:
                raise ValueError(
                    f"Unknown operation: {op_name}. "
                    f"Supported: {', '.join(self.OPERATIONS.keys())}"
                )

        # Fitted batch-level reducers keyed by op index. Populated by
        # fit()/fit_transform() and reused by transform()/process() so the
        # training-batch basis is applied consistently to val/test and single images.
        self._fitted: Dict[int, Any] = {}

    def per_image_operations(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return only the operations that act on a single image."""
        return [(n, k) for n, k in self.operations if n not in BATCH_LEVEL_OPS]

    def batch_operations(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return only the batch-level operations."""
        return [(n, k) for n, k in self.operations if n in BATCH_LEVEL_OPS]

    def process(self, image_array: np.ndarray) -> np.ndarray:
        """Run every op on a single image, returning the processed image or vector.

        An unfitted None-method 'reduce' is a no-op; an unfitted fitting method
        raises (use batch_process). A fitted pipeline reuses its stored reducer.
        """
        result = image_array.copy()

        for idx, (op_name, kwargs) in enumerate(self.operations):
            try:
                if op_name in BATCH_LEVEL_OPS and idx in self._fitted:
                    # Fitted: reuse the stored reducer instead of erroring on a lone sample.
                    result = self._apply_fitted_batch_op(op_name, idx, kwargs, result)
                else:
                    operation = self.OPERATIONS[op_name]
                    result = operation(result, **kwargs)
            except Exception as e:
                # Add pipeline context to the original error.
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
        """Append an operation to the pipeline, rejecting an unknown name."""
        if operation_name not in self.OPERATIONS:
            raise ValueError(f"Unknown operation: {operation_name}")

        self.operations.append((operation_name, kwargs or {}))

    def _apply_per_image_ops(self, images: List[np.ndarray]) -> np.ndarray:
        """Run the per-image ops on every image and stack the results.

        Mirrors the per-image segment of batch_process; shared by fit_transform
        and transform.
        """
        per_image_ops = self.per_image_operations()
        if per_image_ops:
            sub = ImagePipeline(per_image_ops)
            processed = [sub.process(img) for img in images]
        else:
            # Degenerate but legal: batch-level ops only.
            processed = [img for img in images]
        return np.array(processed)

    def _apply_fitted_batch_op(
        self,
        op_name: str,
        idx: int,
        kwargs: Dict[str, Any],
        data: np.ndarray,
    ) -> np.ndarray:
        """Apply a fitted batch-level op via its stored reducer.

        A bypass reducer is None, so fall back to the kwargs to keep it a no-op.
        Works for both a batch and a single sample; the reducer reports its rank.
        """
        operation = self.OPERATIONS[op_name]
        reducer = self._fitted[idx]
        if reducer is not None:
            return operation(data, reducer=reducer)
        return operation(data, **kwargs)

    def fit_transform(self, images: List[np.ndarray]) -> np.ndarray:
        """Fit the batch-level reducer(s) on images and return the reduced batch.

        Runs the per-image ops, stacks them, then fits each batch-level step once,
        storing the fitted reducer so later transform/process reuse the same basis.
        """
        features = self._apply_per_image_ops(images)
        # Re-fitting resets any previously stored reducers.
        self._fitted = {}
        for idx, (op_name, kwargs) in enumerate(self.operations):
            if op_name not in BATCH_LEVEL_OPS:
                continue
            operation = self.OPERATIONS[op_name]
            try:
                # return_reducer=True hands back the fitted reducer to store.
                features, reducer = operation(features, return_reducer=True, **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Pipeline failed fitting batch-level operation '{op_name}' "
                    f"with kwargs {kwargs}. Error: {str(e)}"
                ) from e
            self._fitted[idx] = reducer
        return features

    def fit(self, images: List[np.ndarray]) -> "ImagePipeline":
        """Fit the batch-level reducer(s) on images and return self."""
        self.fit_transform(images)
        return self

    def transform(self, images: List[np.ndarray]) -> np.ndarray:
        """Apply the already-fitted pipeline to a new batch (val/test split).

        Per-image ops run, then each batch-level step reuses its stored reducer
        so held-out data is projected with the training basis. Raises if unfitted.
        """
        features = self._apply_per_image_ops(images)
        for idx, (op_name, kwargs) in enumerate(self.operations):
            if op_name not in BATCH_LEVEL_OPS:
                continue
            if idx not in self._fitted:
                raise RuntimeError(
                    f"Batch-level operation '{op_name}' (index {idx}) is not "
                    f"fitted. Call fit()/fit_transform() before transform()."
                )
            features = self._apply_fitted_batch_op(op_name, idx, kwargs, features)
        return features

    def __repr__(self) -> str:
        """Render the pipeline as ImagePipeline([op1(...) -> op2(...)])."""
        ops_str = " -> ".join(
            f"{name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
            for name, kwargs in self.operations
        )
        return f"ImagePipeline([{ops_str}])"
