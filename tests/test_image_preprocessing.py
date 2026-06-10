"""
Unit tests for the ``image_preprocessing`` facade and the submodules it exposes.

Scope
-----
* Facade surface — every symbol promised by ``__all__`` is importable.
* The per-image transforms (grayscale / resize / normalize / denoise).
* Vectorization (flat + a mocked VGG16 path).
* Batch-level dimensionality reduction (None / PCA / Johnson–Lindenstrauss).
* Pipeline composition (``ImagePipeline``, ``compose``, ``pipeline_decorator``,
  ``batch_process``).
* The I/O helpers.
* A faithful migration of the ``__main__`` "example usage" demo that used to
  live at the bottom of ``image_preprocessing.py`` (now ``test_example_usage_*``).
* Edge cases: missing files, corrupted bytes, degenerate dimensions, invalid
  arguments.

Every test follows Arrange-Act-Assert. Edge-case tests carry a comment naming
the specific failure mode they guard against.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import image_preprocessing as ip
from image_preprocessing import (
    ImagePipeline,
    batch_process,
    compose,
    load_image_from_bytes,
    load_image_from_file,
    normalize_image,
    pipeline_decorator,
    reduce_dimensions,
    reduce_noise,
    resize_image,
    to_grayscale,
    vectorize_image,
)


# ===========================================================================
# Facade surface
# ===========================================================================

class TestFacadeSurface:
    """The facade must expose exactly its advertised public API."""

    def test_all_symbols_are_importable(self):
        # Arrange: the 14 names the facade promises in __all__.
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
            assert hasattr(ip, name), f"facade missing advertised symbol {name!r}"

    def test_private_vgg_cache_alias_present_but_not_public(self):
        # Backward-compat alias must survive, yet stay out of the public API.
        assert isinstance(ip._vgg16_models, dict)
        assert "_vgg16_models" not in ip.__all__


# ===========================================================================
# Per-image transforms
# ===========================================================================

class TestToGrayscale:
    def test_color_to_grayscale_drops_channel_axis(self, color_image):
        # Arrange handled by fixture.
        # Act
        gray = to_grayscale(color_image)
        # Assert
        assert gray.ndim == 2
        assert gray.shape == color_image.shape[:2]
        assert gray.dtype == color_image.dtype

    def test_grayscale_passthrough_is_noop(self, gray_image):
        # A 2D image without force must be returned unchanged.
        result = to_grayscale(gray_image)
        assert result is gray_image or np.array_equal(result, gray_image)

    def test_force_on_grayscale_returns_same_values(self, gray_image):
        # force=True on already-grayscale is documented as a value-preserving no-op.
        result = to_grayscale(gray_image, force=True)
        assert np.array_equal(result, gray_image)

    def test_invalid_dimensions_raise(self):
        # Edge case: a 1D array is neither a grayscale nor a colour image.
        with pytest.raises(ValueError):
            to_grayscale(np.zeros((10,), dtype=np.uint8))


class TestResizeImage:
    def test_exact_resize_without_aspect(self, color_image):
        resized = resize_image(color_image, (64, 64), preserve_aspect=False)
        assert resized.shape == (64, 64, 3)

    def test_preserve_aspect_pads_to_exact_target(self, rng):
        # Edge case / regression: a non-square input must be letterboxed to the
        # EXACT target size. This guards the padding bug where the "after" pad
        # double-counted the "before" pad and overshot target_size (a (100,50)
        # image used to yield (64,80) instead of (64,64)).
        tall = rng.integers(0, 256, size=(100, 50, 3), dtype=np.uint8)
        resized = resize_image(tall, (64, 64), preserve_aspect=True)
        assert resized.shape == (64, 64, 3)

    @pytest.mark.parametrize("shape", [(100, 50, 3), (50, 100, 3), (200, 100, 3), (480, 640, 3)])
    def test_preserve_aspect_always_hits_target_for_varied_aspects(self, shape):
        # Regression sweep: every aspect ratio must pad to exactly (64, 64).
        img = np.zeros(shape, dtype=np.uint8)
        assert resize_image(img, (64, 64), preserve_aspect=True).shape == (64, 64, 3)

    def test_invalid_target_size_raises(self, color_image):
        # Edge case: a zero/negative dimension is nonsensical and must be rejected.
        with pytest.raises(ValueError):
            resize_image(color_image, (0, 64))

    def test_unknown_interpolation_raises(self, color_image):
        with pytest.raises(ValueError):
            resize_image(color_image, (64, 64), interpolation="sinc")


class TestNormalizeImage:
    def test_minmax_maps_into_unit_range(self, color_image):
        normalized = normalize_image(color_image, method="minmax")
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0 - 1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_standard_gives_zero_mean_unit_variance(self, color_image):
        normalized = normalize_image(color_image, method="standard")
        assert abs(float(normalized.mean())) < 1e-4
        assert abs(float(normalized.std()) - 1.0) < 1e-4

    def test_flat_image_minmax_avoids_divide_by_zero(self):
        # Edge case: a constant image has max == min; the guard must return the
        # midpoint instead of producing NaN/inf from a 0/0 division.
        flat = np.full((8, 8), 128, dtype=np.uint8)
        out = normalize_image(flat, method="minmax")
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.5)

    def test_histogram_on_color_image_raises(self, color_image):
        # Edge case: histogram equalization is only defined for single-channel.
        with pytest.raises(ValueError):
            normalize_image(color_image, method="histogram")

    def test_unknown_method_raises(self, gray_image):
        with pytest.raises(ValueError):
            normalize_image(gray_image, method="zscore-but-typo")


class TestReduceNoise:
    def test_shape_is_preserved(self, color_image):
        denoised = reduce_noise(color_image, method="bilateral")
        assert denoised.shape == color_image.shape

    @pytest.mark.parametrize("method", ["bilateral", "gaussian", "morphological", "median"])
    def test_all_methods_return_same_shape(self, small_color_image, method):
        out = reduce_noise(small_color_image, method=method, kernel_size=5)
        assert out.shape == small_color_image.shape

    def test_even_kernel_size_raises(self, color_image):
        # Edge case: OpenCV kernels must be odd; an even size is a common bug.
        with pytest.raises(ValueError):
            reduce_noise(color_image, kernel_size=4)

    def test_unknown_method_raises(self, color_image):
        with pytest.raises(ValueError):
            reduce_noise(color_image, method="nonexistent")


# ===========================================================================
# Vectorization
# ===========================================================================

class TestVectorizeImage:
    def test_flat_color_shape_and_dtype(self, color_image):
        vec = vectorize_image(color_image, method="flat")
        assert vec.shape == (224 * 224 * 3,)
        assert vec.dtype == np.float32
        assert vec.ndim == 1

    def test_flat_grayscale_shape(self, gray_image):
        vec = vectorize_image(gray_image, method="flat")
        assert vec.shape == (224 * 224,)

    def test_preserve_structure_orders_channels_blockwise(self, color_image):
        # With preserve_structure, the first H*W block must equal channel 0
        # flattened — verifying channels are concatenated, not interleaved.
        vec = vectorize_image(color_image, method="flat", preserve_structure=True)
        expected_ch0 = color_image[:, :, 0].flatten().astype(np.float32)
        assert np.array_equal(vec[: 224 * 224], expected_ch0)

    def test_non_array_input_raises_typeerror(self):
        # Edge case: passing a Python list (not ndarray) is a frequent misuse.
        with pytest.raises(TypeError):
            vectorize_image([[1, 2], [3, 4]], method="flat")

    def test_bad_dimensions_raise(self, rng):
        # Edge case: a 4D tensor (e.g. an accidental batch) is unsupported.
        with pytest.raises(ValueError):
            vectorize_image(rng.random((2, 8, 8, 3)), method="flat")

    def test_unknown_method_raises(self, color_image):
        with pytest.raises(ValueError):
            vectorize_image(color_image, method="resnet")

    @pytest.mark.vgg16
    def test_vgg16_path_with_mocked_model(self, color_image, fake_vgg16):
        # The fake_vgg16 fixture seeds the model cache, so no weights load.
        vec = vectorize_image(color_image, method="vgg16", input_size=(224, 224))
        assert vec.shape == (fake_vgg16,)  # 25 088 for 224x224
        assert vec.dtype == np.float32


# ===========================================================================
# Dimensionality reduction (batch-level)
# ===========================================================================

class TestReduceDimensions:
    def test_none_is_identity_passthrough(self, feature_matrix):
        # method=None must return the *same* array object (documented no-op).
        assert reduce_dimensions(feature_matrix, method=None) is feature_matrix

    def test_pca_reduces_to_requested_components(self, feature_matrix):
        reduced = reduce_dimensions(feature_matrix, method="pca", n_components=16)
        assert reduced.shape == (feature_matrix.shape[0], 16)
        assert reduced.dtype == np.float32

    def test_jl_reduces_to_requested_components(self, feature_matrix):
        reduced = reduce_dimensions(
            feature_matrix, method="johnson_lindenstrauss", n_components=64
        )
        assert reduced.shape == (feature_matrix.shape[0], 64)

    def test_pca_components_clamped_to_matrix_rank(self, rng):
        # Edge case: asking for more components than min(n_samples, n_features)
        # must clamp gracefully rather than raise an opaque sklearn error.
        small = rng.random((5, 12)).astype(np.float32)
        reduced = reduce_dimensions(small, method="pca", n_components=999)
        assert reduced.shape[1] == min(5, 12)

    def test_shared_reducer_applies_same_projection(self, rng):
        # The recommended train/test pattern: fit once, reuse on held-out data.
        train = rng.random((30, 100)).astype(np.float32)
        test = rng.random((7, 100)).astype(np.float32)
        train_reduced, reducer = reduce_dimensions(
            train, method="pca", n_components=8, return_reducer=True
        )
        test_reduced = reduce_dimensions(test, reducer=reducer)
        assert train_reduced.shape == (30, 8)
        assert test_reduced.shape == (7, 8)

    def test_prefit_reducer_projects_single_vector(self, rng):
        # Inference-time path: apply a fitted reducer to one 1D sample.
        train = rng.random((30, 100)).astype(np.float32)
        _, reducer = reduce_dimensions(
            train, method="pca", n_components=8, return_reducer=True
        )
        one = rng.random(100).astype(np.float32)
        projected = reduce_dimensions(one, reducer=reducer)
        assert projected.shape == (8,)  # 1D in -> 1D out

    def test_single_sample_without_reducer_raises(self, rng):
        # Edge case: fitting PCA on a lone 1D vector is mathematically undefined.
        with pytest.raises(ValueError):
            reduce_dimensions(rng.random(100).astype(np.float32), method="pca")

    def test_unknown_method_raises(self, feature_matrix):
        with pytest.raises(ValueError):
            reduce_dimensions(feature_matrix, method="umap")


# ===========================================================================
# Pipeline composition
# ===========================================================================

class TestImagePipeline:
    def test_process_produces_1d_feature_vector(self, small_color_image):
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
        # Edge case: a typo'd op name should fail fast at build time.
        with pytest.raises(ValueError):
            ImagePipeline([("grayscaale", {})])

    def test_add_operation_appends(self):
        pipeline = ImagePipeline([("grayscale", {})])
        pipeline.add_operation("vectorize", {})
        assert pipeline.operations[-1][0] == "vectorize"

    def test_repr_lists_operations_in_order(self):
        pipeline = ImagePipeline([("grayscale", {}), ("vectorize", {})])
        text = repr(pipeline)
        assert "grayscale" in text and "vectorize" in text
        assert text.index("grayscale") < text.index("vectorize")

    def test_per_image_and_batch_split(self):
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("vectorize", {}),
            ("reduce", {"method": "pca", "n_components": 4}),
        ])
        per_image = [n for n, _ in pipeline.per_image_operations()]
        batch = [n for n, _ in pipeline.batch_operations()]
        assert per_image == ["grayscale", "vectorize"]
        assert batch == ["reduce"]

    def test_process_with_reduce_none_is_noop(self, small_color_image):
        # A trailing reduce(method=None) must not change the per-image output.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": None}),
        ])
        features = pipeline.process(small_color_image)
        assert features.shape == (16 * 16,)

    def test_process_with_pca_on_single_image_raises(self, small_color_image):
        # Edge case: PCA needs a batch; on one image process() must error
        # (wrapped as RuntimeError with pipeline context).
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": "pca", "n_components": 4}),
        ])
        with pytest.raises(RuntimeError):
            pipeline.process(small_color_image)


class TestFunctionalComposition:
    def test_compose_runs_right_to_left(self, small_color_image):
        composed = compose(
            vectorize_image,
            partial(normalize_image, method="minmax"),
            partial(resize_image, target_size=(16, 16)),
            to_grayscale,
        )
        features = composed(small_color_image)
        assert features.shape == (16 * 16,)

    def test_pipeline_decorator_preprocesses_before_call(self, small_color_image):
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
    def test_batch_without_reduce_stacks_vectors(self, image_batch):
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
        ])
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), 16 * 16)

    def test_batch_with_pca_fits_across_batch(self, image_batch):
        # PCA is fit once over the whole stacked matrix; output width == n_components.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("vectorize", {}),
            ("reduce", {"method": "pca", "n_components": 3}),
        ])
        out = batch_process(image_batch, pipeline)
        assert out.shape == (len(image_batch), 3)


# ===========================================================================
# I/O helpers + edge cases
# ===========================================================================

class TestIOHelpers:
    def test_load_image_from_file_roundtrip(self, tmp_image_file, color_image):
        loaded = load_image_from_file(tmp_image_file)
        assert loaded.shape == color_image.shape
        assert loaded.dtype == np.uint8

    def test_load_missing_file_raises(self):
        # Edge case: a non-existent path must raise FileNotFoundError, not crash.
        with pytest.raises(FileNotFoundError):
            load_image_from_file("definitely_not_here_12345.png")

    def test_load_corrupted_bytes_raises(self):
        # Edge case: random/garbage bytes are not a decodable image.
        with pytest.raises(ValueError):
            load_image_from_bytes(b"\x00\x01\x02 this is not an image \xff")


# ===========================================================================
# Migrated "__main__" example usage (was the demo in image_preprocessing.py)
# ===========================================================================

class TestExampleUsageMigration:
    """
    The original ``if __name__ == '__main__':`` block printed a 6-part demo.
    Here that same flow is asserted instead of printed, preserving it as a
    living smoke test of the public API end to end.
    """

    def test_individual_function_examples(self, color_image):
        # Part 2 of the original demo, now with assertions.
        assert to_grayscale(color_image).shape == (224, 224)
        assert resize_image(color_image, (64, 64), preserve_aspect=True).shape == (64, 64, 3)

        normalized = normalize_image(color_image, method="minmax")
        assert 0.0 <= normalized.min() and normalized.max() <= 1.0

        assert reduce_noise(color_image, method="bilateral").shape == color_image.shape
        assert vectorize_image(color_image, method="flat").shape == (224 * 224 * 3,)

    def test_pipeline_class_example(self, color_image):
        # Part 3 of the original demo.
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
        # Part 4 of the original demo.
        composed = compose(
            vectorize_image,
            partial(normalize_image, method="minmax"),
            partial(resize_image, target_size=(64, 64)),
            partial(to_grayscale),
        )
        assert composed(color_image).shape == (64 * 64,)

    def test_batch_processing_example(self, rng):
        # Part 5 of the original demo.
        batch = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (32, 32)}),
            ("vectorize", {}),
        ])
        out = batch_process(batch, pipeline)
        assert out.shape == (4, 32 * 32)

    def test_dimensionality_reduction_demo(self, rng):
        # Part 6 of the original demo: None / PCA / JL over a batch.
        batch = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(20)]
        base = [
            ("grayscale", {}),
            ("resize", {"target_size": (16, 16)}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ]
        bypass = batch_process(batch, ImagePipeline(base + [("reduce", {"method": None})]))
        pca = batch_process(
            batch, ImagePipeline(base + [("reduce", {"method": "pca", "n_components": 8})])
        )
        jl = batch_process(
            batch,
            ImagePipeline(base + [("reduce", {"method": "johnson_lindenstrauss", "n_components": 12})]),
        )
        assert bypass.shape == (20, 16 * 16)
        assert pca.shape == (20, 8)
        assert jl.shape == (20, 12)
