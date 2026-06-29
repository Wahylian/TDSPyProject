"""
Tests for ``trainbase/pipeline_registry.py`` — the preprocessing front-end catalogue.

:data:`PIPELINE_REGISTRY` maps a CLI-friendly name to a zero-argument factory
returning a fresh :class:`ImagePipeline`. These tests pin:

* the catalogue is non-empty, each name mapped to a callable;
* each factory returns a populated, independent ``ImagePipeline`` (so a caller
  mutating one result never affects another or a later run);
* each registry entry routes to a genuine :class:`PrebuiltPipelines` factory
  (verified by introspection, not a hand-maintained name->factory map).

Construction only — no pipeline is executed, so nothing is decoded or fitted.
"""

from __future__ import annotations

import inspect

import pytest

from preprocessing import ImagePipeline
from prebuilt_pipelines import PrebuiltPipelines
from trainbase.pipeline_registry import PIPELINE_REGISTRY

# The set of genuine pipeline factories defined on PrebuiltPipelines. Derived by
# introspection so adding a factory never requires editing this test — the
# registry is only allowed to route to one of these, not to an ad-hoc lambda.
PREBUILT_FACTORIES = {
    fn for _, fn in inspect.getmembers(PrebuiltPipelines, predicate=inspect.isfunction)
}


class TestRegistryContents:
    """Top-level shape of the registry."""

    def test_registry_is_non_empty(self):
        """The catalogue ships at least one pipeline.

        Sweeps below parametrise over the live registry, so a broken or empty
        import would silently collect *zero* cases and pass vacuously; this
        guards that ``--pipeline`` always has something to select.
        """
        assert PIPELINE_REGISTRY

    @pytest.mark.parametrize("name", sorted(PIPELINE_REGISTRY))
    def test_entry_is_callable_returning_image_pipeline(self, name):
        """Each entry is a factory that builds a populated ``ImagePipeline``.

        Args:
            name: A registered pipeline name from the sweep.
        """
        factory = PIPELINE_REGISTRY[name]
        assert callable(factory)
        pipeline = factory()
        assert isinstance(pipeline, ImagePipeline)
        assert len(pipeline.operations) >= 1


class TestFactoryBehaviour:
    """Freshness and routing of the registered factories."""

    @pytest.mark.parametrize("name", sorted(PIPELINE_REGISTRY))
    def test_factory_returns_fresh_independent_instance(self, name):
        """Two calls yield distinct pipelines; mutating one leaves the other intact.

        The factories promise fresh, safely-mutable results — this guards against
        a shared module-level instance leaking mutations between training runs.

        Args:
            name: A registered pipeline name from the sweep.
        """
        first = PIPELINE_REGISTRY[name]()
        second = PIPELINE_REGISTRY[name]()
        original_len = len(second.operations)

        first.add_operation("vectorize", {})

        assert first is not second
        assert len(second.operations) == original_len

    @pytest.mark.parametrize("name", sorted(PIPELINE_REGISTRY))
    def test_entry_routes_to_a_prebuilt_factory(self, name):
        """Each registry entry is one of the genuine ``PrebuiltPipelines`` factories.

        Rather than re-stating which name maps to which factory (a copy of the
        registry that tests nothing independent), this asserts the registry only
        routes to a real prebuilt front-end — never to an ad-hoc lambda that
        would dodge the maintained ``PrebuiltPipelines`` catalogue.

        Args:
            name: A registered pipeline name from the sweep.
        """
        assert PIPELINE_REGISTRY[name] in PREBUILT_FACTORIES
