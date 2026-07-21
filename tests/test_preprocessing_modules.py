"""Tests for the post-split ``preprocessing`` submodule layout.

After ``pipeline.py`` was split into cohesive modules (``operations``,
``pipeline``, ``batching``, ``composition``), these tests pin the new structure
*and* the invariant that the split changed nothing observable:

* each moved symbol lives in its intended submodule;
* the package front door re-exports the *same object* (identity, not a copy);
* the shared operation registry is one object across ImagePipeline and the
  ``operations`` module (so batch dispatch and validation agree);
* each module's headline behaviour still works end to end.
"""

from __future__ import annotations

import numpy as np

import preprocessing as ip
from preprocessing import operations, pipeline, batching, composition


class TestSymbolHoming:
    """Each public symbol resolves from its new submodule and the package."""

    def test_operations_module_owns_registry_and_batch_set(self):
        assert hasattr(operations, "OPERATIONS")
        assert operations.BATCH_LEVEL_OPS == frozenset({"reduce", "scale"})

    def test_package_reexports_are_the_same_objects(self):
        """The front door forwards the identical objects, not copies."""
        assert ip.ImagePipeline is pipeline.ImagePipeline
        assert ip.batch_process is batching.batch_process
        assert ip.compose is composition.compose
        assert ip.pipeline_decorator is composition.pipeline_decorator
        assert ip.BATCH_LEVEL_OPS is operations.BATCH_LEVEL_OPS

    def test_pipeline_still_exposes_batch_level_ops_alias(self):
        """``preprocessing.pipeline.BATCH_LEVEL_OPS`` keeps resolving (re-import)."""
        assert pipeline.BATCH_LEVEL_OPS is operations.BATCH_LEVEL_OPS

    def test_single_shared_operation_registry(self):
        """ImagePipeline.OPERATIONS is the one registry defined in operations.py."""
        assert ip.ImagePipeline.OPERATIONS is operations.OPERATIONS


class TestModuleBehaviourUnchanged:
    """Each split-out module still performs its job."""

    def test_batch_process_runs_a_pipeline(self, image_batch):
        """``batching.batch_process`` stacks per-image features across the batch."""
        pipe = ip.ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
        ])
        X = batching.batch_process(image_batch, pipe)
        assert X.shape == (len(image_batch), 16 * 16)

    def test_compose_applies_right_to_left(self):
        """``composition.compose`` feeds each output into the next, right-to-left."""
        f = composition.compose(lambda x: x + 1, lambda x: x * 2)
        # compose(add1, mul2)(3) = add1(mul2(3)) = 7
        assert f(3) == 7

    def test_pipeline_decorator_preprocesses_before_call(self):
        """``composition.pipeline_decorator`` runs ops before the wrapped fn."""
        @composition.pipeline_decorator((lambda a: a + 1, {}))
        def identity(image):
            return image

        out = identity(np.zeros((2, 2), dtype=np.int64))
        assert np.array_equal(out, np.ones((2, 2), dtype=np.int64))

    def test_imagepipeline_fit_transform_matches_batch_process(self, image_batch):
        """The split modules interoperate: a no-batch-op pipeline agrees both ways."""
        pipe = ip.ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
        ])
        np.testing.assert_array_equal(
            pipe.fit_transform(image_batch),
            batching.batch_process(image_batch, pipe),
        )
