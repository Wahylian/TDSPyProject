"""
Tests for the ready-made pipeline factories in ``prebuilt_pipelines.py``.

``PrebuiltPipelines`` is a library of named factory methods, each returning a
fresh :class:`ImagePipeline`. These tests pin three contracts:

* every factory returns a valid ``ImagePipeline``;
* each call returns an independent instance (mutating one must not affect
  another, since the module promises fresh, safely-mutable results);
* the per-image stages and trailing reduce configuration of each named pipeline
  match its documented intent, and the reduce variants run end-to-end through
  ``batch_process`` to the expected output shapes.

The VGG16-based ``fast_embedding_pipeline`` is checked structurally only, so no
Keras import or weight download is triggered.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import ImagePipeline, batch_process
from prebuilt_pipelines import PrebuiltPipelines


# Every zero-argument factory on PrebuiltPipelines, named for parametrize ids.
NULLARY_FACTORIES = [
    "svm_pipeline",
    "fast_pipeline",
    "hq_pipeline",
    "no_denoise_pipeline",
    "fast_embedding_pipeline",
    "reduction_bypass_pipeline",
    "vec_pca_pipeline",
    "vec_jl_pipeline",
    "mat_pca_pipeline",
    "mat_jl_pipeline",
]


def _op_names(pipeline: ImagePipeline) -> list[str]:
    """Return the ordered operation names of a pipeline (kwargs dropped)."""
    return [name for name, _ in pipeline.operations]


class TestFactoryContracts:
    """Construction-level guarantees shared by every factory."""

    @pytest.mark.parametrize("factory_name", NULLARY_FACTORIES)
    def test_factory_returns_image_pipeline(self, factory_name):
        """Each factory returns a populated ``ImagePipeline``.

        Args:
            factory_name: Name of a ``PrebuiltPipelines`` factory from the sweep.
        """
        # Act
        pipeline = getattr(PrebuiltPipelines, factory_name)()
        # Assert: a real pipeline with at least one configured stage.
        assert isinstance(pipeline, ImagePipeline)
        assert len(pipeline.operations) >= 1

    @pytest.mark.parametrize("factory_name", NULLARY_FACTORIES)
    def test_factory_returns_fresh_independent_instance(self, factory_name):
        """Two calls yield distinct objects, and mutating one leaves the other intact.

        The module documents that each factory returns a fresh pipeline so
        callers can mutate the result freely; this guards against an accidental
        shared/module-level instance leaking mutations between callers.

        Args:
            factory_name: Name of a ``PrebuiltPipelines`` factory from the sweep.
        """
        # Arrange: build two pipelines from the same factory.
        first = getattr(PrebuiltPipelines, factory_name)()
        second = getattr(PrebuiltPipelines, factory_name)()
        original_len = len(second.operations)
        # Act: mutate only the first instance.
        first.add_operation("vectorize", {})
        # Assert: distinct objects; the second is untouched by the mutation.
        assert first is not second
        assert len(second.operations) == original_len


class TestPipelineStructure:
    """Each named pipeline's stage sequence matches its documented intent."""

    @pytest.mark.parametrize(
        "factory_name",
        ["svm_pipeline", "fast_pipeline", "hq_pipeline", "no_denoise_pipeline"],
    )
    def test_vector_pipelines_flatten_then_reduce_and_scale(self, factory_name):
        """The registry vector pipelines flatten, PCA-reduce, then standardize.

        These four are the self-contained classifier front-ends exposed through
        the training registry: each starts by going to grayscale, flattens with
        ``vectorize``, compresses the flat vector with a ``vec-pca`` reduce, and
        ends with a ``scale`` standardization so the features are ready for a
        scale-sensitive model with nothing appended downstream.

        Args:
            factory_name: A vectorizing factory name from the sweep.
        """
        # Act
        ops = getattr(PrebuiltPipelines, factory_name)().operations
        names = [n for n, _ in ops]
        # Assert: grayscale first; flatten then reduce then standardize, in order.
        assert names[0] == "grayscale"
        assert names[-1] == "scale"
        assert names.index("vectorize") < names.index("reduce") < names.index("scale")
        # The reduce stage is a vector PCA compression.
        reduce_kwargs = ops[names.index("reduce")][1]
        assert reduce_kwargs.get("method") == "vec-pca"

    def test_fast_embedding_pipeline_uses_vgg16_vectorize(self):
        """The embedding pipeline's vectorize stage requests the VGG16 method.

        Structural-only check (no execution) so the test never imports Keras or
        downloads ImageNet weights.
        """
        # Act
        ops = PrebuiltPipelines.fast_embedding_pipeline().operations
        # Assert: the final stage is a VGG16 vectorize.
        name, kwargs = ops[-1]
        assert name == "vectorize"
        assert kwargs.get("method") == "vgg16"

    def test_reduction_bypass_pipeline_has_none_reduce_tail(self):
        """The bypass pipeline vectorizes then carries a ``reduce(method=None)`` tail."""
        # Act
        ops = PrebuiltPipelines.reduction_bypass_pipeline().operations
        names = [n for n, _ in ops]
        # Assert: vectorize present and the trailing reduce is the no-op bypass.
        assert "vectorize" in names
        assert ops[-1][0] == "reduce"
        assert ops[-1][1].get("method") is None

    @pytest.mark.parametrize(
        "factory_name,expected_method,has_vectorize",
        [
            ("vec_pca_pipeline", "vec-pca", True),
            ("vec_jl_pipeline", "vec-jl", True),
            ("mat_pca_pipeline", "mat-pca", False),
            ("mat_jl_pipeline", "mat-jl", False),
        ],
    )
    def test_reduce_pipelines_carry_expected_method_and_vectorize(
        self, factory_name, expected_method, has_vectorize
    ):
        """Reduce pipelines end with the right method; matrix ones omit vectorize.

        Vector reducers operate on flat vectors (so they include ``vectorize``),
        while matrix reducers operate on image matrices (so they omit it). The
        trailing reduce stage must name the expected method.

        Args:
            factory_name: A reducing factory name from the sweep.
            expected_method: The reduce method the factory should configure.
            has_vectorize: Whether the pipeline should include a vectorize stage.
        """
        # Act
        ops = getattr(PrebuiltPipelines, factory_name)().operations
        names = [n for n, _ in ops]
        # Assert: vectorize presence matches the subgroup, and the reduce tail
        # names the expected method.
        assert ("vectorize" in names) is has_vectorize
        assert ops[-1][0] == "reduce"
        assert ops[-1][1].get("method") == expected_method

    @pytest.mark.parametrize(
        "factory_name",
        ["vec_pca_pipeline", "vec_jl_pipeline", "mat_pca_pipeline", "mat_jl_pipeline"],
    )
    def test_n_components_argument_propagates_to_reduce_stage(self, factory_name):
        """A custom ``n_components`` argument reaches the trailing reduce kwargs.

        Args:
            factory_name: A reducing factory name from the sweep.
        """
        # Act: build with a distinctive component count.
        ops = getattr(PrebuiltPipelines, factory_name)(n_components=17).operations
        # Assert: the reduce stage carries the requested width.
        assert ops[-1][1].get("n_components") == 17


class TestPipelineExecution:
    """A representative subset run end-to-end through ``batch_process``."""

    def test_fast_pipeline_produces_reduced_feature_matrix(self, image_batch):
        """``fast_pipeline`` yields one reduced, standardized row per image.

        The pipeline now ends with a ``vec-pca`` reduce (requested 150) and a
        ``scale``. PCA cannot keep more components than the batch rank, so on
        this small batch the width is clamped to ``n_images``; the row count and
        float32 dtype are the stable contract.

        Args:
            image_batch: list of 6 same-shaped colour images (fixture).
        """
        # Act
        out = batch_process(image_batch, PrebuiltPipelines.fast_pipeline())
        # Assert: one row per image, width reduced (clamped to the batch rank).
        assert out.shape[0] == len(image_batch)
        assert out.shape[1] <= min(len(image_batch), 64 * 64)
        assert out.dtype == np.float32

    def test_vec_pca_pipeline_reduces_width_across_batch(self, image_batch):
        """``vec_pca_pipeline`` collapses the flat vectors to ``n_components``.

        Args:
            image_batch: list of 6 same-shaped colour images (fixture).
        """
        # Arrange: keep n_components below the batch size so PCA can fit it.
        pipeline = PrebuiltPipelines.vec_pca_pipeline(n_components=4)
        # Act
        out = batch_process(image_batch, pipeline)
        # Assert: one reduced row per image, width clamped to the request.
        assert out.shape == (len(image_batch), 4)

    def test_mat_pca_pipeline_preserves_rows_and_narrows_width(self, image_batch):
        """``mat_pca_pipeline`` keeps the 128 rows and narrows the width axis.

        Args:
            image_batch: list of 6 same-shaped colour images (fixture).
        """
        # Arrange
        pipeline = PrebuiltPipelines.mat_pca_pipeline(n_components=8)
        # Act
        out = batch_process(image_batch, pipeline)
        # Assert: (n_images, 128 rows, 8 reduced columns) — rows preserved.
        assert out.shape == (len(image_batch), 128, 8)
