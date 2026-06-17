"""
End-to-end "example usage" smoke tests for the public ``preprocessing`` API.

These mirror the six-part demo that the package once printed from a
``__main__`` block. Each part is asserted here instead of printed, so the
documented end-to-end workflows (standalone transforms, the pipeline class,
functional composition, batch processing and dimensionality reduction) stay a
living, executable contract of the public API.
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
        """Each standalone transform behaves as advertised on one image.

        Smoke-asserts grayscale, aspect-preserving resize, min-max normalize,
        bilateral denoise and flat vectorize on a single image.
        """
        # Act + Assert: grayscale collapses channels; resize hits target.
        assert to_grayscale(color_image).shape == (224, 224)
        assert resize_image(color_image, (64, 64), preserve_aspect=True).shape == (64, 64, 3)

        # normalize maps into the unit range.
        normalized = normalize_image(color_image, method="minmax")
        assert 0.0 <= normalized.min() and normalized.max() <= 1.0

        # denoise preserves shape; flat vectorize gives the full H*W*C vector.
        assert reduce_noise(color_image, method="bilateral").shape == color_image.shape
        assert vectorize_image(color_image, method="flat").shape == (224 * 224 * 3,)

    def test_pipeline_class_example(self, color_image):
        """The full ``ImagePipeline`` class flow yields a float32 feature vector.

        Builds the demo's five-stage pipeline (including denoise) and confirms
        it produces the expected resized feature vector.
        """
        # Arrange
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (64, 64), "preserve_aspect": True}),
            ("denoise", {"method": "bilateral", "kernel_size": 5}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ])
        # Act
        features = pipeline.process(color_image)
        # Assert
        assert features.shape == (64 * 64,)
        assert features.dtype == np.float32

    def test_functional_composition_example(self, color_image):
        """The functional ``compose`` flow matches the pipeline-class output shape.

        Mirrors the pipeline-class example using ``compose`` instead, proving
        the two styles produce the same shaped output.
        """
        # Arrange / Act
        composed = compose(
            vectorize_image,
            partial(normalize_image, method="minmax"),
            partial(resize_image, target_size=(64, 64)),
            partial(to_grayscale),
        )
        # Assert
        assert composed(color_image).shape == (64 * 64,)

    def test_batch_processing_example(self, rng):
        """Batch processing stacks per-image vectors into a feature matrix.

        Args:
            rng: Seeded generator used to synthesise a 4-image batch.
        """
        # Arrange
        batch = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (32, 32)}),
            ("vectorize", {}),
        ])
        # Act
        out = batch_process(batch, pipeline)
        # Assert: 4 rows, each the resized H*W width.
        assert out.shape == (4, 32 * 32)

    def test_dimensionality_reduction_demo(self, rng):
        """Compare None / PCA / JL reduction tails on one shared batch.

        Runs the same per-image base pipeline with three different trailing
        reduce stages and asserts each produces the expected output width —
        bypass keeps full width, PCA and JL collapse to their component counts.
        """
        # Arrange: a 20-image batch and a shared per-image base pipeline.
        batch = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        base = [
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ]
        # Act: run the same base pipeline with three different reduce tails.
        bypass = batch_process(batch, ImagePipeline(base + [("reduce", {"method": None})]))
        pca = batch_process(
            batch, ImagePipeline(base + [("reduce", {"method": "vec-pca", "n_components": 8})])
        )
        jl = batch_process(
            batch,
            ImagePipeline(base + [("reduce", {"method": "vec-jl", "n_components": 12})]),
        )
        # Assert: bypass keeps full width; PCA and JL collapse to their counts.
        assert bypass.shape == (20, 16 * 16)
        assert pca.shape == (20, 8)
        assert jl.shape == (20, 12)
