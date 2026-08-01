"""Tests for trainbase/pipeline_registry.py, the preprocessing front-end catalogue.

Pins that the catalogue is non-empty with each name mapped to a callable, that
each factory returns a populated independent ImagePipeline, and that every entry
routes to a genuine PrebuiltPipelines factory. Construction only; nothing runs.
"""

from __future__ import annotations

import inspect

import pytest

from preprocessing import ImagePipeline
from trainbase import PrebuiltPipelines
from trainbase.pipeline_registry import PIPELINE_REGISTRY

# Genuine factories on PrebuiltPipelines, found by introspection so adding one
# never requires editing this test; the registry may only route to these.
PREBUILT_FACTORIES = {
    fn for _, fn in inspect.getmembers(PrebuiltPipelines, predicate=inspect.isfunction)
}


class TestRegistryContents:
    """Top-level shape of the registry."""

    def test_registry_is_non_empty(self):
        """The catalogue ships at least one pipeline (guards vacuous sweeps below)."""
        assert PIPELINE_REGISTRY

    @pytest.mark.parametrize("name", sorted(PIPELINE_REGISTRY))
    def test_entry_is_callable_returning_image_pipeline(self, name):
        """Each entry is a factory that builds a populated ImagePipeline."""
        factory = PIPELINE_REGISTRY[name]
        assert callable(factory)
        pipeline = factory()
        assert isinstance(pipeline, ImagePipeline)
        assert len(pipeline.operations) >= 1


class TestFactoryBehaviour:
    """Freshness and routing of the registered factories."""

    @pytest.mark.parametrize("name", sorted(PIPELINE_REGISTRY))
    def test_factory_returns_fresh_independent_instance(self, name):
        """Two calls yield distinct pipelines; mutating one leaves the other intact."""
        first = PIPELINE_REGISTRY[name]()
        second = PIPELINE_REGISTRY[name]()
        original_len = len(second.operations)

        first.add_operation("vectorize", {})

        assert first is not second
        assert len(second.operations) == original_len

    @pytest.mark.parametrize("name", sorted(PIPELINE_REGISTRY))
    def test_entry_routes_to_a_prebuilt_factory(self, name):
        """Each registry entry is one of the genuine PrebuiltPipelines factories."""
        assert PIPELINE_REGISTRY[name] in PREBUILT_FACTORIES


class TestSvmJlRegistration:
    """The JL-reduction pipeline is registered and routes to its factory."""

    def test_svm_jl_registered_and_routes_to_factory(self):
        assert "svm_jl" in PIPELINE_REGISTRY
        assert PIPELINE_REGISTRY["svm_jl"] is PrebuiltPipelines.svm_jl_pipeline


class TestConditionalEmbeddingRegistration:
    """The VGG16 embedding pipelines register only when keras is importable."""

    def test_embedding_present_iff_keras(self):
        keras_installed = True
        try:
            import keras  # noqa: F401
        except ImportError:
            keras_installed = False
        keys = {"embedding_pca", "embedding_jl"}
        present = keys & set(PIPELINE_REGISTRY)
        if keras_installed:
            assert present == keys
            assert PIPELINE_REGISTRY["embedding_pca"] is PrebuiltPipelines.embedding_pca_pipeline
            assert PIPELINE_REGISTRY["embedding_jl"] is PrebuiltPipelines.embedding_jl_pipeline
        else:
            assert present == set()
