"""End-to-end smoke tests for the public preprocessing API.

Each part asserts one documented workflow (standalone transforms, the pipeline
class, functional composition, batch processing, dimensionality reduction) so the
demo stays a living, executable contract.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from preprocessing import (
    ImagePipeline,
    batch_process,
    compose,
    normalize_image,
    reduce_noise,
    resize_image,
    to_grayscale,
    vectorize_image,
)


class TestExampleUsage:
    """The full public-API workflow, asserted end to end."""

    def test_individual_function_examples(self, color_image):
        """Each standalone transform behaves as advertised on one image."""
        assert to_grayscale(color_image).shape == (224, 224)
        assert resize_image(color_image, (64, 64), preserve_aspect=True).shape == (64, 64, 3)

        normalized = normalize_image(color_image, method="minmax")
        assert 0.0 <= normalized.min() and normalized.max() <= 1.0

        assert reduce_noise(color_image, method="bilateral").shape == color_image.shape
        assert vectorize_image(color_image, method="flat").shape == (224 * 224 * 3,)

    def test_pipeline_class_example(self, color_image):
        """The full ImagePipeline flow yields a float32 feature vector."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (64, 64), "preserve_aspect": True}),
            ("denoise", {"method": "bilateral", "kernel_size": 5}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ])
        features = pipeline.process(color_image)
        assert features.shape == (64 * 64,)
        assert features.dtype == np.float32

    def test_functional_composition_example(self, color_image):
        """The functional compose flow matches the pipeline-class output shape."""
        composed = compose(
            vectorize_image,
            partial(normalize_image, method="minmax"),
            partial(resize_image, target_size=(64, 64)),
            partial(to_grayscale),
        )
        assert composed(color_image).shape == (64 * 64,)

    def test_batch_processing_example(self, rng):
        """Batch processing stacks per-image vectors into a feature matrix."""
        batch = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (32, 32)}),
            ("vectorize", {}),
        ])
        out = batch_process(batch, pipeline)
        assert out.shape == (4, 32 * 32)

    def test_dimensionality_reduction_demo(self, rng):
        """Compare None / PCA / JL reduction tails on one shared batch."""
        batch = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        base = [
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ]
        bypass = batch_process(batch, ImagePipeline(base + [("reduce", {"method": None})]))
        pca = batch_process(
            batch, ImagePipeline(base + [("reduce", {"method": "vec-pca", "n_components": 8})])
        )
        jl = batch_process(
            batch,
            ImagePipeline(base + [("reduce", {"method": "vec-jl", "n_components": 12})]),
        )
        # Bypass keeps full width; PCA and JL collapse to their component counts.
        assert bypass.shape == (20, 16 * 16)
        assert pca.shape == (20, 8)
        assert jl.shape == (20, 12)
