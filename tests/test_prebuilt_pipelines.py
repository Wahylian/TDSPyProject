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
    "svm_jl_pipeline",
    "fast_pipeline",
    "hq_pipeline",
    "no_denoise_pipeline",
    "fast_embedding_pipeline",
    "embedding_pca_pipeline",
    "embedding_jl_pipeline",
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

    def test_svm_jl_pipeline_mirrors_svm_but_uses_jl_reduce(self):
        """``svm_jl_pipeline`` matches ``svm_pipeline`` stage-for-stage but reduces with JL.

        The two share the same per-image front-end and trailing ``scale`` so an
        A/B run isolates the reduction method; only the reduce stage differs
        (``vec-jl`` here vs ``vec-pca`` in ``svm_pipeline``).
        """
        # Act
        jl_ops = PrebuiltPipelines.svm_jl_pipeline().operations
        svm_ops = PrebuiltPipelines.svm_pipeline().operations
        jl_names = [n for n, _ in jl_ops]
        # Assert: identical stage sequence, ending in scale after the reduce.
        assert jl_names == [n for n, _ in svm_ops]
        assert jl_names[-1] == "scale"
        assert jl_names.index("vectorize") < jl_names.index("reduce") < jl_names.index("scale")
        # The reduce stage is a JL random projection, not PCA.
        reduce_kwargs = jl_ops[jl_names.index("reduce")][1]
        assert reduce_kwargs.get("method") == "vec-jl"

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

    @pytest.mark.parametrize(
        "factory_name,expected_method",
        [("embedding_pca_pipeline", "vec-pca"), ("embedding_jl_pipeline", "vec-jl")],
    )
    def test_embedding_pipelines_vgg16_then_reduce_then_scale(
        self, factory_name, expected_method
    ):
        """Embedding pipelines embed with VGG16, then reduce (PCA/JL) then scale.

        Both variants share the same VGG16 front-end and trailing ``scale`` and
        differ only in the reduce method, so an A/B run isolates the reduction.

        Args:
            factory_name: An embedding factory name from the sweep.
            expected_method: The reduce method that factory should configure.
        """
        # Act
        ops = getattr(PrebuiltPipelines, factory_name)().operations
        names = [n for n, _ in ops]
        # Assert: vgg16 vectorize, then the reduce, then scale, in that order.
        assert ops[names.index("vectorize")][1].get("method") == "vgg16"
        assert names.index("vectorize") < names.index("reduce") < names.index("scale")
        assert names[-1] == "scale"
        assert ops[names.index("reduce")][1].get("method") == expected_method

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


class TestPixelPipelines:
    """No-PCA pixel pipelines that feed the raw-image torch models."""

    def test_pixels_pipeline_emits_flat_4096_no_reduce(self, image_batch):
        from prebuilt_pipelines import PrebuiltPipelines
        pipe = PrebuiltPipelines.pixels_pipeline()
        X = pipe.fit_transform(image_batch)
        assert X.ndim == 2 and X.shape[1] == 64 * 64
        assert 0.0 <= float(X.min()) and float(X.max()) <= 1.0
        ops = [name for name, _ in pipe.operations]
        assert "reduce" not in ops and "scale" not in ops

    def test_pixels_hq_pipeline_emits_flat_16384(self, image_batch):
        from prebuilt_pipelines import PrebuiltPipelines
        pipe = PrebuiltPipelines.pixels_hq_pipeline()
        X = pipe.fit_transform(image_batch)
        assert X.ndim == 2 and X.shape[1] == 128 * 128

    def test_both_registered(self):
        from trainbase.pipeline_registry import PIPELINE_REGISTRY
        assert "pixels" in PIPELINE_REGISTRY
        assert "pixels_hq" in PIPELINE_REGISTRY


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
