"""Tests for the ready-made pipeline factories in trainbase/prebuilt_pipelines.py.

Pins that every factory returns a valid, independent ImagePipeline, that each
named pipeline's stages match its documented intent, and that the reduce
variants run end-to-end to the expected shapes. The VGG16 embedding pipeline is
checked structurally only, so no Keras import or weight download is triggered.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing import ImagePipeline, batch_process
from trainbase import PrebuiltPipelines


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
        """Each factory returns a populated ImagePipeline."""
        pipeline = getattr(PrebuiltPipelines, factory_name)()
        assert isinstance(pipeline, ImagePipeline)
        assert len(pipeline.operations) >= 1

    @pytest.mark.parametrize("factory_name", NULLARY_FACTORIES)
    def test_factory_returns_fresh_independent_instance(self, factory_name):
        """Two calls yield distinct objects; mutating one leaves the other intact."""
        first = getattr(PrebuiltPipelines, factory_name)()
        second = getattr(PrebuiltPipelines, factory_name)()
        original_len = len(second.operations)
        first.add_operation("vectorize", {})
        assert first is not second
        assert len(second.operations) == original_len


class TestPipelineStructure:
    """Each named pipeline's stage sequence matches its documented intent."""

    @pytest.mark.parametrize(
        "factory_name",
        ["svm_pipeline", "fast_pipeline", "hq_pipeline", "no_denoise_pipeline"],
    )
    def test_vector_pipelines_flatten_then_reduce_and_scale(self, factory_name):
        """The registry vector pipelines flatten, PCA-reduce, then standardize."""
        ops = getattr(PrebuiltPipelines, factory_name)().operations
        names = [n for n, _ in ops]
        assert names[0] == "grayscale"
        assert names[-1] == "scale"
        assert names.index("vectorize") < names.index("reduce") < names.index("scale")
        reduce_kwargs = ops[names.index("reduce")][1]
        assert reduce_kwargs.get("method") == "vec-pca"

    def test_svm_jl_pipeline_mirrors_svm_but_uses_jl_reduce(self):
        """svm_jl_pipeline matches svm_pipeline stage-for-stage but reduces with JL."""
        jl_ops = PrebuiltPipelines.svm_jl_pipeline().operations
        svm_ops = PrebuiltPipelines.svm_pipeline().operations
        jl_names = [n for n, _ in jl_ops]
        assert jl_names == [n for n, _ in svm_ops]
        assert jl_names[-1] == "scale"
        assert jl_names.index("vectorize") < jl_names.index("reduce") < jl_names.index("scale")
        reduce_kwargs = jl_ops[jl_names.index("reduce")][1]
        assert reduce_kwargs.get("method") == "vec-jl"

    def test_fast_embedding_pipeline_uses_vgg16_vectorize(self):
        """The embedding pipeline's vectorize stage requests the VGG16 method."""
        ops = PrebuiltPipelines.fast_embedding_pipeline().operations
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
        """Embedding pipelines embed with VGG16, then reduce (PCA/JL), then scale."""
        ops = getattr(PrebuiltPipelines, factory_name)().operations
        names = [n for n, _ in ops]
        assert ops[names.index("vectorize")][1].get("method") == "vgg16"
        assert names.index("vectorize") < names.index("reduce") < names.index("scale")
        assert names[-1] == "scale"
        assert ops[names.index("reduce")][1].get("method") == expected_method

    def test_reduction_bypass_pipeline_has_none_reduce_tail(self):
        """The bypass pipeline vectorizes then carries a reduce(method=None) tail."""
        ops = PrebuiltPipelines.reduction_bypass_pipeline().operations
        names = [n for n, _ in ops]
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
        """Reduce pipelines end with the right method; matrix ones omit vectorize."""
        ops = getattr(PrebuiltPipelines, factory_name)().operations
        names = [n for n, _ in ops]
        assert ("vectorize" in names) is has_vectorize
        assert ops[-1][0] == "reduce"
        assert ops[-1][1].get("method") == expected_method

    @pytest.mark.parametrize(
        "factory_name",
        ["vec_pca_pipeline", "vec_jl_pipeline", "mat_pca_pipeline", "mat_jl_pipeline"],
    )
    def test_n_components_argument_propagates_to_reduce_stage(self, factory_name):
        """A custom n_components argument reaches the trailing reduce kwargs."""
        ops = getattr(PrebuiltPipelines, factory_name)(n_components=17).operations
        assert ops[-1][1].get("n_components") == 17


class TestPixelPipelines:
    """No-PCA pixel pipelines that feed the raw-image torch models."""

    def test_pixels_pipeline_emits_flat_4096_no_reduce(self, image_batch):
        pipe = PrebuiltPipelines.pixels_pipeline()
        X = pipe.fit_transform(image_batch)
        assert X.ndim == 2 and X.shape[1] == 64 * 64
        assert 0.0 <= float(X.min()) and float(X.max()) <= 1.0
        ops = [name for name, _ in pipe.operations]
        assert "reduce" not in ops and "scale" not in ops

    def test_pixels_hq_pipeline_emits_flat_16384(self, image_batch):
        pipe = PrebuiltPipelines.pixels_hq_pipeline()
        X = pipe.fit_transform(image_batch)
        assert X.ndim == 2 and X.shape[1] == 128 * 128

    def test_both_registered(self):
        from trainbase.pipeline_registry import PIPELINE_REGISTRY
        assert "pixels" in PIPELINE_REGISTRY
        assert "pixels_hq" in PIPELINE_REGISTRY

    def test_pixels_pretrained_pipeline_emits_flat_rgb_224(self, image_batch):
        """224x224 RGB, channel-major, no grayscale/reduce/scale step."""
        pipe = PrebuiltPipelines.pixels_pretrained_pipeline()
        X = pipe.fit_transform(image_batch)
        assert X.ndim == 2 and X.shape[1] == 3 * 224 * 224
        assert 0.0 <= float(X.min()) and float(X.max()) <= 1.0
        ops = [name for name, _ in pipe.operations]
        assert "grayscale" not in ops
        assert "reduce" not in ops and "scale" not in ops
        assert pipe.operations[-1][0] == "vectorize"
        assert pipe.operations[-1][1].get("preserve_structure") is True

    def test_pixels_pretrained_registered(self):
        from trainbase.pipeline_registry import PIPELINE_REGISTRY
        assert "pixels_pretrained" in PIPELINE_REGISTRY


class TestPipelineExecution:
    """A representative subset run end-to-end through batch_process."""

    def test_fast_pipeline_produces_reduced_feature_matrix(self, image_batch):
        """fast_pipeline yields one reduced, standardized row per image.

        PCA can't keep more components than the batch rank, so on this small
        batch the width is clamped; the row count and float32 dtype are the contract.
        """
        out = batch_process(image_batch, PrebuiltPipelines.fast_pipeline())
        assert out.shape[0] == len(image_batch)
        assert out.shape[1] <= min(len(image_batch), 64 * 64)
        assert out.dtype == np.float32

    def test_vec_pca_pipeline_reduces_width_across_batch(self, image_batch):
        """vec_pca_pipeline collapses the flat vectors to n_components."""
        pipeline = PrebuiltPipelines.vec_pca_pipeline(n_components=4)
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), 4)

    def test_mat_pca_pipeline_preserves_rows_and_narrows_width(self, image_batch):
        """mat_pca_pipeline keeps the 128 rows and narrows the width axis."""
        pipeline = PrebuiltPipelines.mat_pca_pipeline(n_components=8)
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), 128, 8)
