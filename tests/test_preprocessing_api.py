"""Tests for the preprocessing package's public API surface.

Pins preprocessing.__all__: which names are public, that each resolves, and that
the private VGG16 cache stays reachable without leaking into the public surface.
"""

from __future__ import annotations

import preprocessing as ip


class TestPublicApiSurface:
    """The package must expose exactly its advertised public API."""

    def test_all_symbols_are_importable(self):
        """__all__ matches the expected set and every name resolves."""
        expected = {
            "to_grayscale", "resize_image", "normalize_image", "reduce_noise",
            "vectorize_image", "reduce_dimensions", "standardize_features",
            "ImagePipeline", "batch_process", "compose", "pipeline_decorator",
            "BATCH_LEVEL_OPS", "load_image_from_bytes", "load_image_from_file",
            "load_image_from_pil",
        }
        published = set(ip.__all__)
        assert published == expected
        for name in expected:
            assert hasattr(ip, name), f"package missing advertised symbol {name!r}"

    def test_private_vgg_cache_alias_present_but_not_public(self):
        """The private VGG16 cache stays reachable but out of the public API."""
        assert isinstance(ip._vgg16_models, dict)
        assert "_vgg16_models" not in ip.__all__
