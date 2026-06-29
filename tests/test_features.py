"""
Tests for ``trainbase/features.py`` — the feature front-end (build, load, cache).

Four units, each tested in isolation with the heavy collaborators mocked so the
suite never decodes a real image or reads a manifest:

* :func:`build_feature_pipeline` / ``_parse_pipeline_spec`` — pure resolution of
  *which transforms*, from a registry name or a JSON spec (no execution).
* :func:`load_images` — drains a (mocked) ``get_feature_stream`` into memory,
  honouring the ``max_samples`` cap and the empty-stream guard.
* :func:`fit_features` / :func:`transform_features` — fit-on-train /
  transform-on-holdout, with their npz + joblib cache exercised via tmp_path
  (real round-trip on cache *hit*, spied writes on cache *miss*).

The ``ImagePipeline`` is replaced by a light mock whose ``fit_transform`` /
``transform`` return canned matrices, so we test the module's data-flow and
caching logic, not the pipeline internals (covered by the preprocessing suite).
"""

from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pytest

from preprocessing import ImagePipeline
from trainbase.features import (
    _parse_pipeline_spec,
    _train_cache_paths,
    build_feature_pipeline,
    fit_features,
    load_images,
    transform_features,
)


# =============================================================================
# build_feature_pipeline & _parse_pipeline_spec
# =============================================================================
# A minimal but valid custom spec (real op names so ImagePipeline accepts it).
VALID_SPEC = json.dumps([
    ["grayscale", {}],
    ["resize", {"target_size": [32, 32], "preserve_aspect": True}],
    ["vectorize", {}],
])


class TestBuildFeaturePipeline:
    """Resolving the feature pipeline from a registry name or a JSON spec."""

    def test_named_pipeline_resolves_from_registry(self):
        """A known ``pipeline_name`` builds the registered ``ImagePipeline``.

        The ``"fast"`` registry entry should construct a real, populated
        pipeline without touching any data.
        """
        pipeline = build_feature_pipeline(pipeline_name="fast")
        assert isinstance(pipeline, ImagePipeline)
        assert len(pipeline.operations) >= 1

    def test_spec_builds_custom_pipeline_verbatim(self):
        """A JSON ``pipeline_spec`` is parsed into a custom pipeline as written.

        The resulting pipeline's stage names must match the spec exactly — no
        reduce/scale steps are appended by the builder.
        """
        pipeline = build_feature_pipeline(pipeline_spec=VALID_SPEC)
        assert [name for name, _ in pipeline.operations] == [
            "grayscale", "resize", "vectorize",
        ]

    def test_spec_takes_precedence_over_name(self):
        """When both are given, the spec wins (name is ignored).

        Mirrors the documented contract that ``--pipeline-spec`` overrides
        ``--pipeline``.
        """
        pipeline = build_feature_pipeline(
            pipeline_name="fast", pipeline_spec=VALID_SPEC
        )
        # The 3-op spec, not the multi-stage "fast" registry pipeline.
        assert [name for name, _ in pipeline.operations] == [
            "grayscale", "resize", "vectorize",
        ]

    def test_neither_source_raises_value_error(self):
        """Supplying neither a name nor a spec is a ``ValueError``."""
        with pytest.raises(ValueError, match="pipeline_name or a pipeline_spec"):
            build_feature_pipeline()

    def test_unknown_name_raises_key_error(self):
        """An unregistered ``pipeline_name`` surfaces as ``KeyError``."""
        with pytest.raises(KeyError):
            build_feature_pipeline(pipeline_name="does-not-exist")

    def test_spec_with_unknown_op_raises_value_error(self):
        """A spec naming an operation ``ImagePipeline`` rejects is a ``ValueError``.

        Op-name validation is delegated to ``ImagePipeline``'s constructor.
        """
        bad = json.dumps([["not_a_real_op", {}]])
        with pytest.raises(ValueError, match="Unknown operation"):
            build_feature_pipeline(pipeline_spec=bad)


class TestParsePipelineSpec:
    """Structural validation of the custom-pipeline JSON spec."""

    def test_valid_spec_returns_operation_tuples(self):
        """A well-formed spec parses into ``(name, kwargs)`` pairs in order."""
        ops = _parse_pipeline_spec(VALID_SPEC)
        assert ops == [
            ("grayscale", {}),
            ("resize", {"target_size": [32, 32], "preserve_aspect": True}),
            ("vectorize", {}),
        ]

    @pytest.mark.parametrize(
        "spec, match",
        [
            ("{not json", "not valid JSON"),                 # malformed JSON
            (json.dumps({"a": 1}), "non-empty JSON array"),  # object, not array
            (json.dumps([]), "non-empty JSON array"),        # empty array
            (json.dumps([["grayscale"]]), "pair"),           # 1-element entry
            (json.dumps([["a", {}, "b"]]), "pair"),          # 3-element entry
            (json.dumps([[5, {}]]), "must be a"),            # non-string op name
            (json.dumps([["grayscale", 7]]), "kwargs"),      # non-dict kwargs
        ],
    )
    def test_malformed_spec_raises_value_error(self, spec, match):
        """Each ill-formed spec variant raises a descriptive ``ValueError``.

        Args:
            spec: A malformed spec string from the sweep.
            match: A substring expected in the raised error message.
        """
        with pytest.raises(ValueError, match=match):
            _parse_pipeline_spec(spec)


# =============================================================================
# load_images
# =============================================================================
class TestLoadImages:
    """Draining the feature stream into memory, with subsampling."""

    def test_returns_all_images_and_int_label_array(self, image_label_pairs):
        """With ``max_samples=0`` every streamed pair is loaded, labels aligned.

        The returned labels must be an ``int`` ndarray aligned 1:1 with the image
        list — exactly the order the (mocked) stream yields.

        Args:
            image_label_pairs: 7 ``(image, label)`` pairs (fixture).
        """
        with mock.patch(
            "trainbase.features.get_feature_stream", return_value=iter(image_label_pairs)
        ):
            images, y = load_images("train", max_samples=0)

        assert len(images) == len(image_label_pairs)
        assert isinstance(y, np.ndarray) and y.dtype == int
        assert y.tolist() == [label for _, label in image_label_pairs]

    def test_max_samples_caps_the_stream(self, image_label_pairs):
        """``max_samples`` stops the stream early (an unbiased prefix subsample).

        Args:
            image_label_pairs: 7 ``(image, label)`` pairs (fixture).
        """
        with mock.patch(
            "trainbase.features.get_feature_stream", return_value=iter(image_label_pairs)
        ):
            images, y = load_images("train", max_samples=3)

        assert len(images) == 3
        assert len(y) == 3

    def test_empty_stream_raises_runtime_error(self):
        """A split that yields no images is a ``RuntimeError``, not a silent empty.

        Guards downstream code from receiving a zero-row feature matrix.
        """
        with mock.patch(
            "trainbase.features.get_feature_stream", return_value=iter([])
        ):
            with pytest.raises(RuntimeError, match="No usable images"):
                load_images("test", max_samples=0)


# =============================================================================
# fit_features
# =============================================================================
def _stub_pipeline(out: np.ndarray) -> mock.Mock:
    """A mock ImagePipeline whose fit_transform/transform return ``out``."""
    pipe = mock.Mock(spec=ImagePipeline)
    pipe.fit_transform.return_value = out
    pipe.transform.return_value = out
    return pipe


class TestFitFeatures:
    """Fitting the pipeline on train, with optional caching."""

    def test_no_cache_fits_and_returns_features(self, image_label_pairs):
        """With ``cache_dir=None`` the pipeline is fit on train and nothing is written.

        ``fit_transform`` runs exactly once on the loaded train images and the
        produced ``(pipeline, X, y)`` flows straight back to the caller.

        Args:
            image_label_pairs: source pairs for the mocked loader (fixture).
        """
        X_out = np.ones((len(image_label_pairs), 4), dtype=np.float32)
        pipe = _stub_pipeline(X_out)
        images = [img for img, _ in image_label_pairs]
        y = np.array([lbl for _, lbl in image_label_pairs], dtype=int)

        with mock.patch(
            "trainbase.features.load_images", return_value=(images, y)
        ) as load:
            out_pipe, X, out_y = fit_features(
                pipe, max_samples=0, cache_dir=None, cache_prefix="cfg"
            )

        load.assert_called_once_with("train", 0)
        pipe.fit_transform.assert_called_once_with(images)
        assert out_pipe is pipe
        np.testing.assert_array_equal(X, X_out)
        np.testing.assert_array_equal(out_y, y)

    def test_cache_miss_writes_features_and_pipeline(self, tmp_path, image_label_pairs):
        """On a cache miss the train features (npz) and fitted pipeline (joblib) are saved.

        Args:
            tmp_path: pytest temp dir used as the cache (fixture).
            image_label_pairs: source pairs for the mocked loader (fixture).
        """
        X_out = np.arange(8, dtype=np.float32).reshape(4, 2)
        pipe = _stub_pipeline(X_out)
        images = [img for img, _ in image_label_pairs][:4]
        y = np.array([0, 1, 0, 1], dtype=int)

        with mock.patch(
            "trainbase.features.load_images", return_value=(images, y)
        ), mock.patch("trainbase.features.np.savez_compressed") as savez, \
                mock.patch("trainbase.features.joblib.dump") as dump:
            fit_features(pipe, max_samples=4, cache_dir=tmp_path, cache_prefix="cfg")

        # Both artifacts of the train cache are written exactly once.
        savez.assert_called_once()
        dump.assert_called_once()

    def test_cache_hit_loads_without_reloading_images(self, tmp_path):
        """When both train cache files exist they are reused; images are not reloaded.

        Writes a *real* npz + joblib at the paths ``_train_cache_paths`` expects,
        then asserts ``fit_features`` returns the cached pipeline/features and
        never calls ``load_images``.

        Args:
            tmp_path: pytest temp dir used as the cache (fixture).
        """
        import joblib

        cache_prefix, max_samples = "cfg", 5
        feat_path, pipe_path = _train_cache_paths(tmp_path, cache_prefix, max_samples)
        X_cached = np.full((3, 2), 7.0, dtype=np.float32)
        y_cached = np.array([1, 0, 1], dtype=int)
        np.savez_compressed(feat_path, X=X_cached, y=y_cached)
        # A trivial real (picklable) object stands in for the fitted pipeline.
        joblib.dump(ImagePipeline([("grayscale", {})]), pipe_path)

        # A pipeline that would explode if (wrongly) used to recompute.
        live_pipe = _stub_pipeline(np.zeros((1, 1), dtype=np.float32))
        with mock.patch(
            "trainbase.features.load_images",
            side_effect=AssertionError("must not reload images on a cache hit"),
        ):
            out_pipe, X, y = fit_features(
                live_pipe, max_samples, cache_dir=tmp_path, cache_prefix=cache_prefix
            )

        assert isinstance(out_pipe, ImagePipeline)
        np.testing.assert_array_equal(X, X_cached)
        np.testing.assert_array_equal(y, y_cached)
        live_pipe.fit_transform.assert_not_called()


# =============================================================================
# transform_features
# =============================================================================
class TestTransformFeatures:
    """Projecting a held-out split with the already-fitted pipeline."""

    def test_no_cache_transforms_split(self, image_label_pairs):
        """With ``cache_dir=None`` the fitted pipeline's ``transform`` is applied.

        Args:
            image_label_pairs: source pairs for the mocked loader (fixture).
        """
        X_out = np.ones((3, 2), dtype=np.float32)
        pipe = _stub_pipeline(X_out)
        images = [img for img, _ in image_label_pairs][:3]
        y = np.array([0, 1, 0], dtype=int)

        with mock.patch(
            "trainbase.features.load_images", return_value=(images, y)
        ) as load:
            X, out_y = transform_features(
                "val", pipe, max_samples=3, cache_dir=None, cache_prefix="cfg"
            )

        load.assert_called_once_with("val", 3)
        pipe.transform.assert_called_once_with(images)
        np.testing.assert_array_equal(X, X_out)
        np.testing.assert_array_equal(out_y, y)

    def test_cache_miss_writes_split_features(self, tmp_path, image_label_pairs):
        """A cache miss saves the split's features to a per-split npz.

        Args:
            tmp_path: pytest temp dir used as the cache (fixture).
            image_label_pairs: source pairs for the mocked loader (fixture).
        """
        pipe = _stub_pipeline(np.ones((3, 2), dtype=np.float32))
        images = [img for img, _ in image_label_pairs][:3]
        y = np.array([0, 1, 0], dtype=int)

        with mock.patch(
            "trainbase.features.load_images", return_value=(images, y)
        ), mock.patch("trainbase.features.np.savez_compressed") as savez:
            transform_features(
                "test", pipe, max_samples=3, cache_dir=tmp_path, cache_prefix="cfg"
            )

        savez.assert_called_once()

    def test_cache_hit_loads_without_transforming(self, tmp_path):
        """An existing per-split npz is reused; the pipeline is never invoked.

        Args:
            tmp_path: pytest temp dir used as the cache (fixture).
        """
        split, prefix, n = "test", "cfg", 4
        cache_path = tmp_path / f"{prefix}_{split}_n{n}.npz"
        X_cached = np.full((2, 3), 5.0, dtype=np.float32)
        y_cached = np.array([1, 0], dtype=int)
        np.savez_compressed(cache_path, X=X_cached, y=y_cached)

        pipe = _stub_pipeline(np.zeros((1, 1), dtype=np.float32))
        with mock.patch(
            "trainbase.features.load_images",
            side_effect=AssertionError("must not reload images on a cache hit"),
        ):
            X, y = transform_features(
                split, pipe, max_samples=n, cache_dir=tmp_path, cache_prefix=prefix
            )

        np.testing.assert_array_equal(X, X_cached)
        np.testing.assert_array_equal(y, y_cached)
        pipe.transform.assert_not_called()


# =============================================================================
# _train_cache_paths
# =============================================================================
class TestTrainCachePaths:
    """The helper that names the train split's two cache artifacts."""

    def test_none_cache_dir_returns_none_pair(self):
        """No cache directory yields ``(None, None)`` (caching disabled)."""
        assert _train_cache_paths(None, "cfg", 10) == (None, None)

    def test_paths_encode_prefix_and_sample_cap(self, tmp_path):
        """The feature/pipeline paths embed the prefix + sample cap and the right suffixes.

        Args:
            tmp_path: pytest temp dir used as the cache root (fixture).
        """
        feat_path, pipe_path = _train_cache_paths(tmp_path, "myrun", 250)
        assert feat_path.name == "myrun_train_n250.npz"
        assert pipe_path.name == "myrun_train_n250_pipeline.joblib"
        # The directory is created as a side effect so writes can't fail later.
        assert tmp_path.is_dir()
