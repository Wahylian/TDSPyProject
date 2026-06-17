"""
Tests for the composable pipeline in ``preprocessing/pipeline.py``.

Covers the three composition patterns and the batch runner:

* :class:`TestImagePipeline` — the class-based, config-driven chain.
* :class:`TestFunctionalComposition` — ``compose`` and ``pipeline_decorator``.
* :class:`TestBatchProcess` — ``batch_process`` splitting per-image vs
  batch-level (``reduce``) stages.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from preprocessing import (
    ImagePipeline,
    batch_process,
    compose,
    normalize_image,
    pipeline_decorator,
    resize_image,
    to_grayscale,
    vectorize_image,
)


class TestImagePipeline:
    """The ``ImagePipeline`` class: construction, introspection, execution.

    Covers building a multi-stage pipeline, fail-fast validation of operation
    names, the per-image vs batch-level operation split, and how a single-image
    ``process`` call interacts with batch-only stages like PCA.
    """

    def test_process_produces_1d_feature_vector(self, small_color_image):
        """A full grayscale->resize->normalize->vectorize pipeline yields a vector.

        End-to-end happy path: confirms stage outputs chain correctly and the
        final result is a 1D float32 feature vector of the resized H*W length.
        """
        # Arrange: a four-stage per-image pipeline.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ])
        # Act
        features = pipeline.process(small_color_image)
        # Assert
        assert features.shape == (16 * 16,)
        assert features.dtype == np.float32

    def test_unknown_operation_rejected_at_construction(self):
        """An unknown operation name fails at construction, not at runtime.

        Edge case: a typo'd op ("grayscaale") should be caught when the
        pipeline is built so the error surfaces near its cause rather than
        deep inside a later ``process`` call.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            ImagePipeline([("grayscaale", {})])

    def test_add_operation_appends(self):
        """``add_operation`` appends a stage to the end of the pipeline."""
        # Arrange
        pipeline = ImagePipeline([("grayscale", {})])
        # Act
        pipeline.add_operation("vectorize", {})
        # Assert: the new op is now last in the operations list.
        assert pipeline.operations[-1][0] == "vectorize"

    def test_add_unknown_operation_raises(self):
        """``add_operation`` rejects an unsupported operation name.

        Edge case: mutating a pipeline must apply the same fail-fast validation
        as construction, so a typo'd op name raises rather than being appended
        and only failing later inside ``process``.
        """
        # Arrange
        pipeline = ImagePipeline([("grayscale", {})])
        # Act + Assert
        with pytest.raises(ValueError):
            pipeline.add_operation("vectoriize", {})

    def test_repr_lists_operations_in_order(self):
        """``repr`` lists the operations in their execution order.

        The textual order matters for debuggability — it should reflect the
        actual pipeline sequence, so "grayscale" must appear before "vectorize".
        """
        # Arrange
        pipeline = ImagePipeline([("grayscale", {}), ("vectorize", {})])
        # Act
        text = repr(pipeline)
        # Assert: both names present and in the correct order.
        assert "grayscale" in text and "vectorize" in text
        assert text.index("grayscale") < text.index("vectorize")

    def test_per_image_and_batch_split(self):
        """Operations are partitioned into per-image vs batch-level stages.

        The pipeline must classify each op correctly: pixel transforms run
        per image, while ``reduce`` runs once over the whole batch. This split
        is what lets ``batch_process`` fit reducers across all images at once.
        """
        # Arrange: a mix of per-image ops and a batch-level reduce.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("vectorize", {}),
            ("reduce", {"method": "vec-pca", "n_components": 4}),
        ])
        # Act
        per_image = [n for n, _ in pipeline.per_image_operations()]
        batch = [n for n, _ in pipeline.batch_operations()]
        # Assert: grayscale/vectorize are per-image; reduce is batch-level.
        assert per_image == ["grayscale", "vectorize"]
        assert batch == ["reduce"]

    def test_process_with_reduce_none_is_noop(self, small_color_image):
        """A trailing ``reduce(method=None)`` leaves the per-image output intact.

        The bypass reduction must not alter the vector produced by the
        per-image stages — width stays at the resized H*W.
        """
        # Arrange: pipeline ending in a no-op reduce.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": None}),
        ])
        # Act
        features = pipeline.process(small_color_image)
        # Assert: unchanged width.
        assert features.shape == (16 * 16,)

    def test_process_with_pca_on_single_image_raises(self, small_color_image):
        """Running PCA via ``process`` on a single image raises ``RuntimeError``.

        Edge case: PCA is a batch-only operation, so calling ``process`` (the
        single-image entry point) with a PCA stage is invalid. The underlying
        ValueError is re-wrapped as a RuntimeError carrying pipeline context.
        """
        # Arrange: a pipeline whose final stage needs a batch.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": "vec-pca", "n_components": 4}),
        ])
        # Act + Assert
        with pytest.raises(RuntimeError):
            pipeline.process(small_color_image)

    def test_process_without_vectorize_returns_2d_matrix(self, small_color_image):
        """Omitting ``vectorize`` keeps the per-image output a 2D matrix.

        Vectorization is optional: a grayscale->resize->normalize pipeline with
        no ``vectorize`` stage must yield a 2D ``(height, width)`` matrix — the
        form CNNs/ViTs consume — rather than a flattened vector.
        """
        # Arrange: a matrix-output pipeline (no vectorize step).
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
        ])
        # Act
        result = pipeline.process(small_color_image)
        # Assert: a 2D matrix of the resized size, not a 1D vector.
        assert result.shape == (16, 16)
        assert result.ndim == 2

    def test_process_with_matrix_reduce_none_is_noop(self, small_color_image):
        """A trailing ``reduce(method=None)`` leaves a matrix output intact.

        The bypass must be shape-agnostic: on a non-vectorized pipeline it
        returns the 2D matrix unchanged, exactly as it returns a vector
        unchanged in the vectorized case.
        """
        # Arrange: matrix pipeline ending in a no-op reduce.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("reduce", {"method": None}),
        ])
        # Act
        result = pipeline.process(small_color_image)
        # Assert: unchanged 2D matrix.
        assert result.shape == (16, 16)


class TestFunctionalComposition:
    """The functional-style composition helpers (``compose`` / decorator)."""

    def test_compose_runs_right_to_left(self, small_color_image):
        """``compose`` applies functions right-to-left (math convention).

        The rightmost callable (``to_grayscale``) runs first and the leftmost
        (``vectorize_image``) last, mirroring f(g(x)) ordering — verified by
        the final vectorized shape.
        """
        # Arrange: compose grayscale -> resize -> normalize -> vectorize (R-to-L).
        composed = compose(
            vectorize_image,
            partial(normalize_image, method="minmax"),
            partial(resize_image, target_size=(16, 16)),
            to_grayscale,
        )
        # Act
        features = composed(small_color_image)
        # Assert: produces the resized H*W vector, proving correct ordering.
        assert features.shape == (16 * 16,)

    def test_pipeline_decorator_preprocesses_before_call(self, small_color_image):
        """``pipeline_decorator`` runs its stages before the wrapped function.

        The decorated function receives an already-preprocessed image, so
        returning the input verbatim yields the fully transformed vector —
        confirming the decorator applies its stages up front.
        """
        # Arrange: decorate a pass-through function with a 3-stage pipeline.
        @pipeline_decorator(
            (to_grayscale, {}),
            (partial(resize_image, target_size=(16, 16)), {}),
            (vectorize_image, {}),
        )
        def extract(image):
            return image  # already preprocessed

        # Act
        out = extract(small_color_image)
        # Assert: the decorator preprocessed the input to the expected vector.
        assert out.shape == (16 * 16,)


class TestBatchProcess:
    """Batch execution: per-image stages run per image, reduce runs once."""

    def test_batch_without_reduce_stacks_vectors(self, image_batch):
        """Without a reduce stage, each image's vector is stacked into a matrix.

        Args:
            image_batch: list of 6 same-shaped colour images (fixture).
        """
        # Arrange: a per-image-only pipeline.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
        ])
        # Act
        out = batch_process(image_batch, pipeline)
        # Assert: one row per image, each of width H*W.
        assert out.shape == (len(image_batch), 16 * 16)

    def test_batch_with_pca_fits_across_batch(self, image_batch):
        """A PCA stage is fit once across the whole batch matrix.

        Unlike single-image ``process`` (which can't fit PCA), ``batch_process``
        stacks every image first, so PCA has enough samples and the output
        width collapses to ``n_components``.
        """
        # Arrange: per-image stages followed by a batch-level PCA reduce.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": "vec-pca", "n_components": 3}),
        ])
        # Act
        out = batch_process(image_batch, pipeline)
        # Assert: one row per image, width reduced to n_components.
        assert out.shape == (len(image_batch), 3)

    def test_batch_with_only_batch_level_ops_stacks_raw_images(self, image_batch):
        """A pipeline of only batch-level ops stacks the raw images unchanged.

        Degenerate-but-legal path: with no per-image stages, ``batch_process``
        skips per-image processing and stacks the inputs as-is, then applies the
        no-op reduce — yielding a 4D ``(n, H, W, C)`` stack of the originals.

        Args:
            image_batch: list of 6 same-shaped colour images (fixture).
        """
        # Arrange: a pipeline with no per-image ops, only a bypass reduce.
        pipeline = ImagePipeline([("reduce", {"method": None})])
        # Act
        out = batch_process(image_batch, pipeline)
        # Assert: the raw images are stacked untouched.
        assert out.shape == (len(image_batch), *image_batch[0].shape)
        assert np.array_equal(out[0], image_batch[0])

    def test_batch_without_vectorize_stacks_matrices(self, image_batch):
        """Without ``vectorize``, per-image matrices stack into a 3D image stack.

        Optionality at the batch level: a matrix pipeline produces one
        ``(height, width)`` matrix per image, so ``batch_process`` returns a 3D
        ``(n_images, height, width)`` stack instead of a 2D feature matrix.

        Args:
            image_batch: list of 6 same-shaped colour images (fixture).
        """
        # Arrange: a per-image matrix pipeline (no vectorize).
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
        ])
        # Act
        out = batch_process(image_batch, pipeline)
        # Assert: a 3D stack, one matrix per image.
        assert out.shape == (len(image_batch), 16, 16)

    @pytest.mark.parametrize("method,width", [("mat-pca", 4), ("mat-jl", 6)])
    def test_batch_with_matrix_reduce_fits_across_batch(self, image_batch, method, width):
        """A matrix reduce stage is fit once across the whole image stack.

        End-to-end matrix path: per-image stages keep each image a 2D matrix,
        ``batch_process`` stacks them to 3D, and the matrix reducer collapses the
        width axis to ``n_components`` while preserving the sample and row axes.

        Args:
            image_batch: list of 6 colour images (fixture).
            method: matrix reduction method from the parametrize sweep.
            width: requested reduced width from the parametrize sweep.
        """
        # Arrange: per-image matrix stages followed by a batch-level matrix reduce.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("reduce", {"method": method, "n_components": width}),
        ])
        # Act
        out = batch_process(image_batch, pipeline)
        # Assert: (n_images, height, n_components) — rows kept, columns reduced.
        assert out.shape == (len(image_batch), 16, width)

    def test_batch_colour_matrix_reduce_preserves_channels(self, image_batch):
        """A colour matrix pipeline (no grayscale/vectorize) keeps the channel axis.

        Full CNN/ViT-style path: with neither ``grayscale`` nor ``vectorize``,
        each image stays ``(height, width, 3)``; ``batch_process`` stacks them to
        a 4D ``(n, height, width, 3)`` tensor, and the matrix reducer narrows only
        the width axis, yielding ``(n, height, n_components, 3)``.

        Args:
            image_batch: list of 6 same-shaped colour images (fixture).
        """
        # Arrange: keep colour — resize + normalize, then matrix reduce.
        pipeline = ImagePipeline([
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("reduce", {"method": "mat-pca", "n_components": 5}),
        ])
        # Act
        out = batch_process(image_batch, pipeline)
        # Assert: width reduced to 5, the 3 colour channels preserved.
        assert out.shape == (len(image_batch), 16, 5, 3)
