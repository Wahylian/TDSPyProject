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
        """``__all__`` matches the expected set and every name actually resolves.

        Guards against two drifts at once: the advertised API changing
        unintentionally, and a name being listed in ``__all__`` without a
        backing attribute (which would break ``from facade import *``).
        """
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
        """The private VGG16 cache stays reachable but out of the public API.

        Edge case / backward-compat: tests and legacy callers reach into
        ``_vgg16_models`` to seed a stub model, so the alias must keep
        existing — yet it must not leak into ``__all__`` as public surface.
        """
        # Assert: the cache is a dict that exists, but is not advertised publicly.
        assert isinstance(ip._vgg16_models, dict)
        assert "_vgg16_models" not in ip.__all__


# ===========================================================================
# Per-image transforms
# ===========================================================================

class TestToGrayscale:
    """Behaviour of the ``to_grayscale`` transform across input shapes."""

    def test_color_to_grayscale_drops_channel_axis(self, color_image):
        """A colour (H, W, 3) image collapses to a 2D (H, W) image.

        Args:
            color_image: 224x224x3 uint8 fixture image.
        """
        # Arrange handled by the color_image fixture.
        # Act
        gray = to_grayscale(color_image)
        # Assert: channel axis removed, spatial dims and dtype preserved.
        assert gray.ndim == 2
        assert gray.shape == color_image.shape[:2]
        assert gray.dtype == color_image.dtype

    def test_grayscale_passthrough_is_noop(self, gray_image):
        """An already-grayscale image is returned unchanged when ``force`` is off.

        Confirms the transform short-circuits on 2D input instead of erroring
        or needlessly copying.
        """
        # Act
        result = to_grayscale(gray_image)
        # Assert: same object or at least value-identical (a no-op).
        assert result is gray_image or np.array_equal(result, gray_image)

    def test_force_on_grayscale_returns_same_values(self, gray_image):
        """``force=True`` on grayscale input still preserves pixel values.

        The ``force`` flag re-runs conversion logic; for input that is already
        single-channel this is documented to be value-preserving, so no pixels
        may shift.
        """
        # Act
        result = to_grayscale(gray_image, force=True)
        # Assert: values are byte-for-byte identical.
        assert np.array_equal(result, gray_image)

    def test_invalid_dimensions_raise(self):
        """A 1D array is rejected as neither grayscale nor colour.

        Edge case: only 2D (grayscale) and 3D (colour) inputs are meaningful;
        a 1D array signals a caller error and must raise rather than silently
        misinterpret the data.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            to_grayscale(np.zeros((10,), dtype=np.uint8))


class TestResizeImage:
    """Resizing behaviour, including aspect-preserving letterbox padding."""

    def test_exact_resize_without_aspect(self, color_image):
        """With ``preserve_aspect=False`` the image is stretched to the target.

        Args:
            color_image: 224x224x3 uint8 fixture image.
        """
        # Act
        resized = resize_image(color_image, (64, 64), preserve_aspect=False)
        # Assert: the output matches the requested size exactly.
        assert resized.shape == (64, 64, 3)

    def test_preserve_aspect_pads_to_exact_target(self, rng):
        """A non-square input is letterboxed to the EXACT target size.

        Edge case / regression: this guards the padding bug where the "after"
        pad double-counted the "before" pad and overshot ``target_size`` — a
        (100, 50) image used to yield (64, 80) instead of (64, 64).
        """
        # Arrange: a tall, non-square image that triggers vertical/horizontal padding.
        tall = rng.integers(0, 256, size=(100, 50, 3), dtype=np.uint8)
        # Act
        resized = resize_image(tall, (64, 64), preserve_aspect=True)
        # Assert: padding lands the result on the exact target, not oversized.
        assert resized.shape == (64, 64, 3)

    @pytest.mark.parametrize("shape", [(100, 50, 3), (50, 100, 3), (200, 100, 3), (480, 640, 3)])
    def test_preserve_aspect_always_hits_target_for_varied_aspects(self, shape):
        """Every aspect ratio pads to exactly (64, 64).

        Regression sweep complementing the single-case test above: portrait,
        landscape and common photo ratios must all land on the target so the
        padding maths is correct in both axes.

        Args:
            shape: Input image shape supplied by the parametrize sweep.
        """
        # Arrange / Act / Assert in one line per shape.
        img = np.zeros(shape, dtype=np.uint8)
        assert resize_image(img, (64, 64), preserve_aspect=True).shape == (64, 64, 3)

    def test_invalid_target_size_raises(self, color_image):
        """A zero (or negative) target dimension is rejected.

        Edge case: a 0-sized axis is nonsensical and would otherwise produce
        an empty or crashing resize, so it must fail fast.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            resize_image(color_image, (0, 64))

    def test_unknown_interpolation_raises(self, color_image):
        """An unsupported interpolation name raises rather than guessing.

        Edge case: only the documented OpenCV interpolation modes are valid;
        a typo / unsupported mode must be reported, not silently defaulted.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            resize_image(color_image, (64, 64), interpolation="sinc")


class TestNormalizeImage:
    """The ``normalize_image`` methods and their numerical guarantees."""

    def test_minmax_maps_into_unit_range(self, color_image):
        """Min-max normalization maps pixels into [0, 1] as float32.

        The tiny epsilon tolerance absorbs floating-point rounding at the
        range endpoints.
        """
        # Act
        normalized = normalize_image(color_image, method="minmax")
        # Assert: float32 output bounded within the unit interval.
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0 - 1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_standard_gives_zero_mean_unit_variance(self, color_image):
        """Standard (z-score) normalization yields ~0 mean and ~1 std.

        Validates the statistical contract of the "standard" method within a
        small numerical tolerance.
        """
        # Act
        normalized = normalize_image(color_image, method="standard")
        # Assert: mean collapses to ~0 and standard deviation to ~1.
        assert abs(float(normalized.mean())) < 1e-4
        assert abs(float(normalized.std()) - 1.0) < 1e-4

    def test_flat_image_minmax_avoids_divide_by_zero(self):
        """A constant image normalizes to the midpoint, not NaN/inf.

        Edge case: a constant image has ``max == min``, so the naive
        ``(x - min) / (max - min)`` is a 0/0 division. The guard must return
        the 0.5 midpoint instead of propagating NaN/inf.
        """
        # Arrange: a perfectly flat (single-value) image.
        flat = np.full((8, 8), 128, dtype=np.uint8)
        # Act
        out = normalize_image(flat, method="minmax")
        # Assert: finite everywhere and pinned to the midpoint.
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.5)

    def test_histogram_on_color_image_raises(self, color_image):
        """Histogram equalization rejects multi-channel input.

        Edge case: histogram equalization is only defined for a single
        channel, so passing a colour image must raise rather than silently
        equalizing per-channel and distorting colour.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            normalize_image(color_image, method="histogram")

    def test_unknown_method_raises(self, gray_image):
        """An unrecognised normalization method raises.

        Edge case: a typo'd / unsupported method name must fail fast instead
        of falling through to a silent no-op.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            normalize_image(gray_image, method="zscore-but-typo")


class TestReduceNoise:
    """Denoising preserves image shape and validates its arguments."""

    def test_shape_is_preserved(self, color_image):
        """Bilateral denoising returns an image of the same shape.

        Denoising must never change dimensionality — only pixel values — so
        it can be dropped into a pipeline without breaking later stages.
        """
        # Act
        denoised = reduce_noise(color_image, method="bilateral")
        # Assert
        assert denoised.shape == color_image.shape

    @pytest.mark.parametrize("method", ["bilateral", "gaussian", "morphological", "median"])
    def test_all_methods_return_same_shape(self, small_color_image, method):
        """Every denoise method is shape-preserving.

        Sweeps all supported methods to ensure none of them resize or drop a
        channel, regardless of the underlying OpenCV op.

        Args:
            small_color_image: 32x32x3 fixture (kept tiny for speed).
            method: Denoise method supplied by the parametrize sweep.
        """
        # Act
        out = reduce_noise(small_color_image, method=method, kernel_size=5)
        # Assert
        assert out.shape == small_color_image.shape

    def test_even_kernel_size_raises(self, color_image):
        """An even kernel size is rejected.

        Edge case: OpenCV convolution kernels must be odd-sized so they have a
        defined centre pixel; an even size is a common caller bug and must
        raise rather than crash deep inside OpenCV.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_noise(color_image, kernel_size=4)

    def test_unknown_method_raises(self, color_image):
        """An unsupported denoise method name raises.

        Edge case: guards against typo'd / unsupported method strings.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_noise(color_image, method="nonexistent")


# ===========================================================================
# Vectorization
# ===========================================================================

class TestVectorizeImage:
    """Flattening / embedding of images into 1D feature vectors."""

    def test_flat_color_shape_and_dtype(self, color_image):
        """Flat vectorization of a colour image gives a float32 H*W*C vector."""
        # Act
        vec = vectorize_image(color_image, method="flat")
        # Assert: a 1D float32 vector of length 224*224*3.
        assert vec.shape == (224 * 224 * 3,)
        assert vec.dtype == np.float32
        assert vec.ndim == 1

    def test_flat_grayscale_shape(self, gray_image):
        """Flat vectorization of a grayscale image gives an H*W vector.

        Confirms the channel count is respected — single-channel input has no
        channel factor in the output length.
        """
        # Act
        vec = vectorize_image(gray_image, method="flat")
        # Assert
        assert vec.shape == (224 * 224,)

    def test_preserve_structure_orders_channels_blockwise(self, color_image):
        """``preserve_structure`` concatenates channels block-wise, not interleaved.

        Validates the memory layout: with preserve_structure the first H*W
        slots must be channel 0 flattened, proving channels are laid out in
        contiguous blocks rather than pixel-interleaved (B,G,R,B,G,R,...).
        """
        # Act
        vec = vectorize_image(color_image, method="flat", preserve_structure=True)
        # Assert: the leading H*W block equals channel 0 flattened.
        expected_ch0 = color_image[:, :, 0].flatten().astype(np.float32)
        assert np.array_equal(vec[: 224 * 224], expected_ch0)

    def test_non_array_input_raises_typeerror(self):
        """A non-ndarray input raises ``TypeError``.

        Edge case: passing a nested Python list instead of a numpy array is a
        frequent misuse and must be rejected with a clear type error.
        """
        # Act + Assert
        with pytest.raises(TypeError):
            vectorize_image([[1, 2], [3, 4]], method="flat")

    def test_bad_dimensions_raise(self, rng):
        """A 4D tensor is rejected.

        Edge case: a 4D array usually means an accidental batch was passed to
        the single-image vectorizer; this is unsupported and must raise.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            vectorize_image(rng.random((2, 8, 8, 3)), method="flat")

    def test_unknown_method_raises(self, color_image):
        """An unsupported vectorization method raises.

        Edge case: only "flat" and "vgg16" are defined; anything else (e.g.
        "resnet") must fail rather than silently no-op.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            vectorize_image(color_image, method="resnet")

    @pytest.mark.vgg16
    def test_vgg16_path_with_mocked_model(self, color_image, fake_vgg16):
        """The VGG16 path returns the expected embedding length, using a stub.

        The ``fake_vgg16`` fixture pre-seeds the module-level model cache with
        a lightweight stub, so this exercises the vgg16 branch's shape/dtype
        contract WITHOUT importing keras or downloading ~500 MB of weights.

        Args:
            color_image: 224x224x3 fixture image.
            fake_vgg16: Fixture that seeds the stub model and returns the
                expected flattened output length (25 088 for 224x224).
        """
        # Act
        vec = vectorize_image(color_image, method="vgg16", input_size=(224, 224))
        # Assert: shape matches the stub's known output size and dtype is float32.
        assert vec.shape == (fake_vgg16,)  # 25 088 for 224x224
        assert vec.dtype == np.float32


# ===========================================================================
# Dimensionality reduction (batch-level)
# ===========================================================================

class TestReduceDimensions:
    """Batch-level dimensionality reduction (None / PCA / Johnson–Lindenstrauss).

    Covers the bypass no-op, both reduction methods, component clamping, and
    the fit-once/reuse "reducer" pattern that keeps train and inference data
    projected into the same space.
    """

    def test_none_is_identity_passthrough(self, feature_matrix):
        """``method=None`` returns the exact same array object (no copy).

        Documented no-op: the bypass path must not allocate or transform, so
        identity (``is``) — not just equality — is asserted.
        """
        # Act + Assert
        assert reduce_dimensions(feature_matrix, method=None) is feature_matrix

    def test_pca_reduces_to_requested_components(self, feature_matrix):
        """PCA reduces the feature axis to the requested component count."""
        # Act
        reduced = reduce_dimensions(feature_matrix, method="pca", n_components=16)
        # Assert: sample count preserved, feature axis narrowed to 16, float32.
        assert reduced.shape == (feature_matrix.shape[0], 16)
        assert reduced.dtype == np.float32

    def test_jl_reduces_to_requested_components(self, feature_matrix):
        """Johnson–Lindenstrauss projection hits the requested width.

        The JL method is an alternative to PCA; this confirms it honours
        ``n_components`` and preserves the sample count.
        """
        # Act
        reduced = reduce_dimensions(
            feature_matrix, method="johnson_lindenstrauss", n_components=64
        )
        # Assert
        assert reduced.shape == (feature_matrix.shape[0], 64)

    def test_pca_components_clamped_to_matrix_rank(self, rng):
        """Over-asking for components clamps to ``min(n_samples, n_features)``.

        Edge case: requesting more components than the matrix rank supports
        must clamp gracefully rather than surface an opaque sklearn error.
        """
        # Arrange: a 5x12 matrix whose rank caps PCA at 5 components.
        small = rng.random((5, 12)).astype(np.float32)
        # Act: deliberately request far more components than possible.
        reduced = reduce_dimensions(small, method="pca", n_components=999)
        # Assert: clamped to the rank limit.
        assert reduced.shape[1] == min(5, 12)

    def test_shared_reducer_applies_same_projection(self, rng):
        """A fitted reducer projects held-out data into the training space.

        Validates the recommended train/test pattern: fit the reducer once on
        training data (``return_reducer=True``), then reuse it on test data so
        both end up in the same reduced subspace.
        """
        # Arrange: separate train and test matrices sharing the same feature width.
        train = rng.random((30, 100)).astype(np.float32)
        test = rng.random((7, 100)).astype(np.float32)
        # Act: fit on train (capturing the reducer), then apply to test.
        train_reduced, reducer = reduce_dimensions(
            train, method="pca", n_components=8, return_reducer=True
        )
        test_reduced = reduce_dimensions(test, reducer=reducer)
        # Assert: both projected to width 8, sample counts preserved.
        assert train_reduced.shape == (30, 8)
        assert test_reduced.shape == (7, 8)

    def test_prefit_reducer_projects_single_vector(self, rng):
        """A pre-fit reducer projects a single 1D sample at inference time.

        Inference-time path: a fitted reducer applied to one 1D vector must
        return a 1D vector (not a (1, k) matrix), matching how single-image
        embeddings are reduced in production.
        """
        # Arrange: fit a reducer on a training batch.
        train = rng.random((30, 100)).astype(np.float32)
        _, reducer = reduce_dimensions(
            train, method="pca", n_components=8, return_reducer=True
        )
        # Act: project a lone 1D sample through the fitted reducer.
        one = rng.random(100).astype(np.float32)
        projected = reduce_dimensions(one, reducer=reducer)
        # Assert: 1D in -> 1D out, width 8.
        assert projected.shape == (8,)  # 1D in -> 1D out

    def test_single_sample_without_reducer_raises(self, rng):
        """Fitting PCA on a single 1D vector raises.

        Edge case: PCA needs multiple samples to estimate variance, so fitting
        it on one lone vector (with no pre-fit reducer supplied) is
        mathematically undefined and must raise.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_dimensions(rng.random(100).astype(np.float32), method="pca")

    def test_unknown_method_raises(self, feature_matrix):
        """An unsupported reduction method raises.

        Edge case: methods outside {None, "pca", "johnson_lindenstrauss"}
        (e.g. "umap") are unimplemented and must fail loudly.
        """
        # Act + Assert
        with pytest.raises(ValueError):
            reduce_dimensions(feature_matrix, method="umap")


# ===========================================================================
# Pipeline composition
# ===========================================================================

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
            ("reduce", {"method": "pca", "n_components": 4}),
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
            ("reduce", {"method": "pca", "n_components": 4}),
        ])
        # Act + Assert
        with pytest.raises(RuntimeError):
            pipeline.process(small_color_image)


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
            ("reduce", {"method": "pca", "n_components": 3}),
        ])
        # Act
        out = batch_process(image_batch, pipeline)
        # Assert: one row per image, width reduced to n_components.
        assert out.shape == (len(image_batch), 3)


# ===========================================================================
# I/O helpers + edge cases
# ===========================================================================

class TestIOHelpers:
    """The image-loading helpers and their failure modes."""

    def test_load_image_from_file_roundtrip(self, tmp_image_file, color_image):
        """An image written to disk loads back with the same shape and dtype.

        Exercises the real OpenCV file-decoding path against an actual PNG
        written by the ``tmp_image_file`` fixture (a round-trip, not a mock).

        Args:
            tmp_image_file: Path to a PNG written from ``color_image``.
            color_image: The original 224x224x3 array, for shape comparison.
        """
        # Act
        loaded = load_image_from_file(tmp_image_file)
        # Assert: shape and dtype survive the write/read round-trip.
        assert loaded.shape == color_image.shape
        assert loaded.dtype == np.uint8

    def test_load_missing_file_raises(self):
        """A non-existent path raises ``FileNotFoundError``.

        Edge case: a missing file must raise a clear, specific error rather
        than returning ``None`` (OpenCV's default) and crashing later.
        """
        # Act + Assert
        with pytest.raises(FileNotFoundError):
            load_image_from_file("definitely_not_here_12345.png")

    def test_load_corrupted_bytes_raises(self):
        """Undecodable byte content raises ``ValueError``.

        Edge case: random/garbage bytes are not a valid image; decoding must
        fail loudly instead of yielding a malformed array.
        """
        # Act + Assert
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
        """Part 2 of the demo: each standalone transform behaves as advertised.

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
        """Part 3 of the demo: the full ``ImagePipeline`` class flow.

        Builds the original demo's five-stage pipeline (including denoise) and
        confirms it yields the expected float32 feature vector.
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
        """Part 4 of the demo: the functional ``compose`` flow.

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
        """Part 5 of the demo: batch processing stacks per-image vectors.

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
        """Part 6 of the demo: compare None / PCA / JL reduction on one batch.

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
            batch, ImagePipeline(base + [("reduce", {"method": "pca", "n_components": 8})])
        )
        jl = batch_process(
            batch,
            ImagePipeline(base + [("reduce", {"method": "johnson_lindenstrauss", "n_components": 12})]),
        )
        # Assert: bypass keeps full width; PCA and JL collapse to their counts.
        assert bypass.shape == (20, 16 * 16)
        assert pca.shape == (20, 8)
        assert jl.shape == (20, 12)
