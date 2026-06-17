"""
Tests for the ``preprocessing`` package's public API surface.

Pins the contract advertised by ``preprocessing.__all__``: exactly which names
are public, that each resolves to a real attribute, and that the private VGG16
weight cache stays reachable for advanced callers without leaking into the
public surface.
"""

from __future__ import annotations

import preprocessing as ip


class TestPublicApiSurface:
    """The package must expose exactly its advertised public API."""

    def test_all_symbols_are_importable(self):
        """``__all__`` matches the expected set and every name actually resolves.

        Guards against two drifts at once: the advertised API changing
        unintentionally, and a name being listed in ``__all__`` without a
        backing attribute (which would break ``from preprocessing import *``).
        """
        # Arrange: the 14 names the package promises in __all__.
        expected = {
            "to_grayscale", "resize_image", "normalize_image", "reduce_noise",
            "vectorize_image", "reduce_dimensions", "ImagePipeline",
            "batch_process", "compose", "pipeline_decorator", "BATCH_LEVEL_OPS",
            "load_image_from_bytes", "load_image_from_file", "load_image_from_pil",
        }
        # Act
        published = set(ip.__all__)
        # Assert: surface matches and every name resolves to a real attribute.
        assert published == expected
        for name in expected:
            assert hasattr(ip, name), f"package missing advertised symbol {name!r}"

    def test_private_vgg_cache_alias_present_but_not_public(self):
        """The private VGG16 cache stays reachable but out of the public API.

        Edge case / backward-compat: tests and legacy callers reach into
        ``_vgg16_models`` to seed a stub model, so the alias must keep
        existing — yet it must not leak into ``__all__`` as public surface.
        """
        # Assert: the cache is a dict that exists, but is not advertised publicly.
        assert isinstance(ip._vgg16_models, dict)
        assert "_vgg16_models" not in ip.__all__
