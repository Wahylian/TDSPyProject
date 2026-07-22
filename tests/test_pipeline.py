"""Tests for the composable pipeline in preprocessing/pipeline.py.

Covers ImagePipeline (the config-driven chain), the functional compose /
pipeline_decorator helpers, batch_process, and the stateful fit/transform paths.
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
    reduce_dimensions,
    resize_image,
    standardize_features,
    to_grayscale,
    vectorize_image,
)


class TestImagePipeline:
    """ImagePipeline construction, introspection, and single-image execution."""

    def test_process_produces_1d_feature_vector(self, small_color_image):
        """A grayscale->resize->normalize->vectorize pipeline yields a float32 vector."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ])
        features = pipeline.process(small_color_image)
        assert features.shape == (16 * 16,)
        assert features.dtype == np.float32

    def test_unknown_operation_rejected_at_construction(self):
        """An unknown operation name fails at construction, not at runtime."""
        with pytest.raises(ValueError):
            ImagePipeline([("grayscaale", {})])

    def test_add_operation_appends(self):
        """add_operation appends a stage to the end of the pipeline."""
        pipeline = ImagePipeline([("grayscale", {})])
        pipeline.add_operation("vectorize", {})
        assert pipeline.operations[-1][0] == "vectorize"

    def test_add_unknown_operation_raises(self):
        """add_operation rejects an unsupported operation name."""
        pipeline = ImagePipeline([("grayscale", {})])
        with pytest.raises(ValueError):
            pipeline.add_operation("vectoriize", {})

    def test_repr_lists_operations_in_order(self):
        """repr lists the operations in their execution order."""
        pipeline = ImagePipeline([("grayscale", {}), ("vectorize", {})])
        text = repr(pipeline)
        assert "grayscale" in text and "vectorize" in text
        assert text.index("grayscale") < text.index("vectorize")

    def test_per_image_and_batch_split(self):
        """Ops are partitioned into per-image vs batch-level stages."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("vectorize", {}),
            ("reduce", {"method": "vec-pca", "n_components": 4}),
        ])
        per_image = [n for n, _ in pipeline.per_image_operations()]
        batch = [n for n, _ in pipeline.batch_operations()]
        assert per_image == ["grayscale", "vectorize"]
        assert batch == ["reduce"]

    def test_process_with_reduce_none_is_noop(self, small_color_image):
        """A trailing reduce(method=None) leaves the per-image vector intact."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": None}),
        ])
        features = pipeline.process(small_color_image)
        assert features.shape == (16 * 16,)

    def test_process_with_pca_on_single_image_raises(self, small_color_image):
        """PCA via process on a single image raises RuntimeError (batch-only op)."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": "vec-pca", "n_components": 4}),
        ])
        with pytest.raises(RuntimeError):
            pipeline.process(small_color_image)

    def test_process_without_vectorize_returns_2d_matrix(self, small_color_image):
        """Omitting vectorize keeps the per-image output a 2D matrix."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
        ])
        result = pipeline.process(small_color_image)
        assert result.shape == (16, 16)
        assert result.ndim == 2

    def test_process_with_matrix_reduce_none_is_noop(self, small_color_image):
        """A trailing reduce(method=None) leaves a matrix output intact."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("reduce", {"method": None}),
        ])
        result = pipeline.process(small_color_image)
        assert result.shape == (16, 16)


class TestFunctionalComposition:
    """The functional compose / pipeline_decorator helpers."""

    def test_compose_runs_right_to_left(self, small_color_image):
        """compose applies functions right-to-left (f(g(x)) ordering)."""
        composed = compose(
            vectorize_image,
            partial(normalize_image, method="minmax"),
            partial(resize_image, target_size=(16, 16)),
            to_grayscale,
        )
        features = composed(small_color_image)
        assert features.shape == (16 * 16,)

    def test_pipeline_decorator_preprocesses_before_call(self, small_color_image):
        """pipeline_decorator runs its stages before the wrapped function."""
        @pipeline_decorator(
            (to_grayscale, {}),
            (partial(resize_image, target_size=(16, 16)), {}),
            (vectorize_image, {}),
        )
        def extract(image):
            return image  # already preprocessed

        out = extract(small_color_image)
        assert out.shape == (16 * 16,)


class TestBatchProcess:
    """Batch execution: per-image stages run per image, reduce runs once."""

    def test_batch_without_reduce_stacks_vectors(self, image_batch):
        """Without a reduce stage, each image's vector stacks into a matrix."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
        ])
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), 16 * 16)

    def test_batch_with_pca_fits_across_batch(self, image_batch):
        """A PCA stage is fit once across the whole batch matrix."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": "vec-pca", "n_components": 3}),
        ])
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), 3)

    def test_batch_with_only_batch_level_ops_stacks_raw_images(self, image_batch):
        """A pipeline of only batch-level ops stacks the raw images unchanged."""
        pipeline = ImagePipeline([("reduce", {"method": None})])
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), *image_batch[0].shape)
        assert np.array_equal(out[0], image_batch[0])

    def test_batch_without_vectorize_stacks_matrices(self, image_batch):
        """Without vectorize, per-image matrices stack into a 3D image stack."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
        ])
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), 16, 16)

    @pytest.mark.parametrize("method,width", [("mat-pca", 4), ("mat-jl", 6)])
    def test_batch_with_matrix_reduce_fits_across_batch(self, image_batch, method, width):
        """A matrix reduce stage is fit once across the whole image stack."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("reduce", {"method": method, "n_components": width}),
        ])
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), 16, width)

    def test_batch_colour_matrix_reduce_preserves_channels(self, image_batch):
        """A colour matrix pipeline keeps the channel axis, narrowing only width."""
        pipeline = ImagePipeline([
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("reduce", {"method": "mat-pca", "n_components": 5}),
        ])
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), 16, 5, 3)


class TestFitTransform:
    """Stateful fit/transform: a fitted reducer is stored and reused across splits."""

    def _pca_pipeline(self) -> ImagePipeline:
        """A vectorized pipeline ending in a batch-level PCA reduce (op index 3)."""
        return ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": "vec-pca", "n_components": 3}),
        ])

    def test_fit_transform_then_transform_shapes(self, image_batch):
        """fit_transform reduces the train batch; transform reduces other sizes."""
        pipeline = self._pca_pipeline()
        X_train = pipeline.fit_transform(image_batch)
        X_val = pipeline.transform(image_batch[:4])
        assert X_train.shape == (len(image_batch), 3)
        assert X_val.shape == (4, 3)

    def test_transform_applies_training_basis(self, image_batch):
        """transform reuses the stored training reducer, not a fresh fit."""
        pipeline = self._pca_pipeline()
        pipeline.fit(image_batch)
        held_out = image_batch[:4]
        # The reduce op is index 3; grab the reducer fitted on the train batch.
        fitted_reducer = pipeline._fitted[3]
        got = pipeline.transform(held_out)
        per_image = pipeline._apply_per_image_ops(held_out)
        expected = reduce_dimensions(per_image, reducer=fitted_reducer)
        np.testing.assert_allclose(got, expected)

    def test_transform_before_fit_raises(self, image_batch):
        """Calling transform before fitting a reduce stage raises."""
        pipeline = self._pca_pipeline()
        with pytest.raises(RuntimeError):
            pipeline.transform(image_batch)

    def test_process_single_image_after_fit_uses_reducer(self, image_batch):
        """After fitting, single-image process reuses the reducer and matches transform."""
        pipeline = self._pca_pipeline()
        pipeline.fit(image_batch)
        single = pipeline.process(image_batch[0])
        batched = pipeline.transform([image_batch[0]])
        assert single.shape == (3,)
        np.testing.assert_allclose(single, batched[0])

    def test_fit_transform_without_reduce_returns_stacked(self, image_batch):
        """With no batch-level op, fit_transform just stacks per-image vectors."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
        ])
        out = pipeline.fit_transform(image_batch)
        assert out.shape == (len(image_batch), 16 * 16)

    @pytest.mark.parametrize("method", ["mat-pca", "mat-jl"])
    def test_matrix_reduce_fit_transform_reuse(self, image_batch, method):
        """Matrix reducers fit/transform and reuse the basis, like the vector ones."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("reduce", {"method": method, "n_components": 4}),
        ])
        held_out = image_batch[:3]
        X_train = pipeline.fit_transform(image_batch)
        X_val = pipeline.transform(held_out)
        single = pipeline.process(held_out[0])
        # Reduce is op index 3; the stored reducer must reproduce transform.
        expected = reduce_dimensions(
            pipeline._apply_per_image_ops(held_out), reducer=pipeline._fitted[3]
        )
        assert X_train.shape == (len(image_batch), 16, 4)
        assert X_val.shape == (3, 16, 4)
        assert single.shape == (16, 4)
        np.testing.assert_allclose(X_val, expected)

    def test_colour_matrix_reduce_preserves_channels_on_transform(self, image_batch):
        """A colour matrix pipeline keeps the channel axis through fit/transform/process."""
        pipeline = ImagePipeline([
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("reduce", {"method": "mat-pca", "n_components": 5}),
        ])
        X_train = pipeline.fit_transform(image_batch)
        X_val = pipeline.transform(image_batch[:3])
        single = pipeline.process(image_batch[0])
        assert X_train.shape == (len(image_batch), 16, 5, 3)
        assert X_val.shape == (3, 16, 5, 3)
        assert single.shape == (16, 5, 3)


class TestScale:
    """The batch-level 'scale' step inside a pipeline (fit once, reuse)."""

    def _scale_pipeline(self) -> ImagePipeline:
        """A vectorized pipeline ending in a batch-level scale (op index 3)."""
        return ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("scale", {}),
        ])

    def test_scale_is_classified_as_batch_level(self):
        """'scale' is partitioned as a batch-level op, like 'reduce'."""
        pipeline = self._scale_pipeline()
        per_image = [n for n, _ in pipeline.per_image_operations()]
        batch = [n for n, _ in pipeline.batch_operations()]
        assert per_image == ["grayscale", "resize", "vectorize"]
        assert batch == ["scale"]

    def test_fit_transform_standardizes_train_columns(self, image_batch):
        """fit_transform standardizes the train batch to ~zero-mean/unit-var."""
        pipeline = self._scale_pipeline()
        X_train = pipeline.fit_transform(image_batch)
        assert X_train.shape == (len(image_batch), 16 * 16)
        assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-4)
        # Varying columns become unit-variance; constant columns stay at 0.
        std = X_train.std(axis=0)
        varying = std > 1e-6
        assert np.allclose(std[varying], 1.0, atol=1e-4)

    def test_transform_applies_training_stats(self, image_batch):
        """transform reuses the stored training scaler, not a fresh fit."""
        pipeline = self._scale_pipeline()
        pipeline.fit(image_batch)
        held_out = image_batch[:4]
        # Scale is op index 3.
        scaler = pipeline._fitted[3]
        got = pipeline.transform(held_out)
        per_image = pipeline._apply_per_image_ops(held_out)
        expected = standardize_features(per_image, reducer=scaler)
        np.testing.assert_allclose(got, expected)

    def test_transform_before_fit_raises(self, image_batch):
        """Calling transform before fitting a scale stage raises."""
        pipeline = self._scale_pipeline()
        with pytest.raises(RuntimeError):
            pipeline.transform(image_batch)

    def test_process_single_image_after_fit_uses_stats(self, image_batch):
        """After fitting, single-image process reuses the scaler and matches transform."""
        pipeline = self._scale_pipeline()
        pipeline.fit(image_batch)
        single = pipeline.process(image_batch[0])
        batched = pipeline.transform([image_batch[0]])
        assert single.shape == (16 * 16,)
        np.testing.assert_allclose(single, batched[0])

    def test_reduce_then_scale_chain(self, image_batch):
        """A reduce->scale chain fits both and reuses both across splits."""
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": "vec-pca", "n_components": 3}),
            ("scale", {}),
        ])
        held_out = image_batch[:4]
        X_train = pipeline.fit_transform(image_batch)
        X_val = pipeline.transform(held_out)
        single = pipeline.process(held_out[0])
        assert X_train.shape == (len(image_batch), 3)
        assert X_val.shape == (4, 3)
        assert single.shape == (3,)
        assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-4)
        # Compare against a batch-of-one, not X_val[0]: sklearn PCA's transform
        # takes a batch-size-dependent BLAS path, so 1-vs-4 images differ by ~1e-5
        # in float32, which the trailing scale step amplifies.
        np.testing.assert_allclose(single, pipeline.transform([held_out[0]])[0])
