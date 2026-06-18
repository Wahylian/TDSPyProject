"""
Tests for ``extract_features.py`` — the manifest-driven image streamer.

The streamer reads the split manifest written by ``create_split.py`` (columns
``photo_name``, ``photo_path``, ``label``, ``split``), filters it to one split,
loads each image from local storage, decodes it to a BGR array, and yields
``(image, label)`` pairs in a randomized order. These tests fabricate real,
decodable image files and a manifest CSV under ``tmp_path`` so the load path is
exercised against actual files rather than mocks.

Coverage
--------
* Split validation and the module constants.
* Manifest loading (``_load_entries``): split filtering, int labels, empty-path
  skipping, missing-column and missing-file errors.
* The load path (``_load_image``) for success, a missing file and an
  undecodable file.
* The generator contract of ``get_feature_stream`` (yields decoded BGR arrays).
* The BGR colour convention of the decoded images.
* Feature-extraction integrity: a streamed image pushed through the
  ``preprocessing`` pipeline yields a dense, finite, correctly-shaped vector.
* Randomized ordering: shuffling is reproducible under a seed, loses no
  datapoints, and leaves the global RNG untouched.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest

from extract_features import (
    LABEL_COLUMN,
    PATH_COLUMN,
    SPLIT_COLUMN,
    VALID_SPLITS,
    _load_entries,
    _load_image,
    _validate_split,
    get_feature_stream,
)
from preprocessing import ImagePipeline, resize_image, to_grayscale


# ===========================================================================
# Test helpers
# ===========================================================================

def _write_png(path, color=(40, 120, 200), width: int = 12, height: int = 8) -> str:
    """Write a solid-colour BGR image to *path* as a decodable PNG; return its path.

    ``color`` is given in OpenCV's BGR channel order — exactly what ``cv2.imread``
    returns when the file is read back.
    """
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def _write_manifest(path, rows, header=("photo_name", "photo_path", "label", "split")):
    """Write a small manifest CSV to *path*.

    Each row in *rows* is a tuple matching *header* (default: photo_name,
    photo_path, label, split).
    """
    lines = [",".join(header)]
    lines += [",".join(str(col) for col in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_labeled_manifest(path, image_path, n):
    """Write an ``n``-row train manifest, each row uniquely labeled ``0..n-1``.

    Every row points at the same on-disk image, so the per-row label is the only
    thing distinguishing datapoints — a stable identity tag for tracking exactly
    where each datapoint ends up after the stream's shuffle.
    """
    rows = [(f"img_{i}.png", image_path, i, "train") for i in range(n)]
    _write_manifest(path, rows)


# ===========================================================================
# 1. Split validation + constants
# ===========================================================================

class TestSplitValidationAndConstants:
    """Guard the split allow-list and the module's public constants."""

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_accepts_valid_splits(self, split):
        """Every canonical split name validates without raising."""
        # A recognised split passes silently; the call itself is the assertion.
        _validate_split(split)

    @pytest.mark.parametrize("bad", ["", "training", "TRAIN", "dev", "all"])
    def test_rejects_invalid_splits(self, bad):
        """Unknown / mis-cased / empty split names are rejected."""
        with pytest.raises(ValueError):
            _validate_split(bad)

    def test_module_constants_match_spec(self):
        """The exported constants match the manifest contract."""
        assert VALID_SPLITS == frozenset({"train", "val", "test"})
        assert PATH_COLUMN == "photo_path"
        assert LABEL_COLUMN == "label"
        assert SPLIT_COLUMN == "split"


# ===========================================================================
# 2. Manifest loading
# ===========================================================================

class TestManifestLoading:
    """Resolve ``(path, label)`` entries from the manifest CSV."""

    def test_filters_to_requested_split_with_int_labels(self, tmp_path):
        """Only rows for the requested split are returned, with integer labels.

        Validates the core supervised-learning guarantee: each path is paired
        with the integer label from its own row, and split filtering keeps that
        pairing intact (a "test" row never leaks into the train entries).
        """
        # Arrange: a manifest spanning two splits.
        csv_path = tmp_path / "dataset_split.csv"
        _write_manifest(
            csv_path,
            rows=[
                ("a.jpg", "photos/a.jpg", 0, "train"),
                ("b.jpg", "photos/b.jpg", 1, "train"),
                ("c.jpg", "photos/c.jpg", 1, "test"),
            ],
        )
        # Act
        train_entries = _load_entries("train", str(csv_path))
        test_entries = _load_entries("test", str(csv_path))
        # Assert: labels are ints and stay aligned with their paths after filtering.
        assert train_entries == [("photos/a.jpg", 0), ("photos/b.jpg", 1)]
        assert test_entries == [("photos/c.jpg", 1)]

    def test_empty_path_rows_are_skipped(self, tmp_path):
        """Rows with an empty ``photo_path`` are not loaded.

        Edge case: a row carrying no path has no image to stream, so the loader
        skips it while keeping the surrounding rows.
        """
        # Arrange: the middle row has no path.
        csv_path = tmp_path / "dataset_split.csv"
        _write_manifest(
            csv_path,
            rows=[
                ("a.jpg", "photos/a.jpg", 0, "train"),
                ("b.jpg", "", 1, "train"),
                ("c.jpg", "photos/c.jpg", 0, "train"),
            ],
        )
        # Act
        train_entries = _load_entries("train", str(csv_path))
        # Assert: only the two rows with a path survive.
        assert train_entries == [("photos/a.jpg", 0), ("photos/c.jpg", 0)]

    def test_missing_required_column_raises(self, tmp_path):
        """A manifest missing a required column fails loudly.

        Edge case: without the ``photo_path`` column there is nothing to stream,
        so the loader must raise rather than silently yield an empty result.
        """
        # Arrange: a CSV with no photo_path column.
        bad = tmp_path / "dataset_split.csv"
        _write_manifest(
            bad,
            rows=[("a.jpg", 0, "train")],
            header=("photo_name", "label", "split"),
        )
        # Act + Assert
        with pytest.raises(ValueError):
            _load_entries("train", str(bad))

    def test_missing_manifest_file_raises_filenotfound(self, tmp_path):
        """A manifest path that does not exist raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            _load_entries("train", str(tmp_path / "nope.csv"))


# ===========================================================================
# 3. Load path (local files)
# ===========================================================================

class TestLoadImage:
    """The single-image read/decode path against real local files.

    Covers the happy path plus the two failure modes a resilient loader must
    tolerate: a path that does not exist, and a file that holds undecodable bytes.
    """

    def test_successful_load_returns_bgr_array(self, tmp_path):
        """A valid PNG file decodes to a uint8 BGR array of the right size."""
        img_path = _write_png(tmp_path / "ok.png")
        img = _load_image(img_path)
        assert img is not None
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.uint8
        assert img.shape[:2] == (8, 12)  # (height, width) of _write_png default

    def test_missing_file_returns_none(self, tmp_path):
        """A path that does not exist yields ``None`` after a warning, not a crash.

        Edge case: a recorded path whose file is absent must be swallowed so one
        missing image does not abort the whole stream.
        """
        missing = str(tmp_path / "does_not_exist.png")
        with pytest.warns(UserWarning):
            assert _load_image(missing) is None

    def test_undecodable_file_returns_none(self, tmp_path):
        """A file with undecodable bytes yields ``None`` with a warning."""
        bad = tmp_path / "garbage.png"
        bad.write_bytes(b"not-an-image")
        with pytest.warns(UserWarning):
            assert _load_image(str(bad)) is None


# ===========================================================================
# 4. Error handling (manifest level)
# ===========================================================================

class TestErrorHandling:
    """Split- and file-level failure modes of manifest loading."""

    def test_invalid_split_raises_before_reading(self, tmp_path):
        """An unknown split is rejected by ``get_feature_stream``.

        The generator validates the split eagerly, so draining it on a bad split
        name raises ``ValueError``.
        """
        with pytest.raises(ValueError):
            list(get_feature_stream("bogus", csv_path=str(tmp_path / "any.csv")))


# ===========================================================================
# 5. Generator contract
# ===========================================================================

class TestGeneratorContract:
    """The end-to-end ``get_feature_stream`` generator behaviour."""

    @pytest.fixture
    def streamed_pairs(self, tmp_path):
        """Materialise the full stream for a 3-file, labelled train split.

        Writes three real images and a manifest that points at them with
        alternating labels, then eagerly drains the generator into a list.
        """
        rows = []
        for i in range(3):
            img_path = _write_png(tmp_path / f"{i}.png")
            rows.append((f"{i}.png", img_path, i % 2, "train"))
        _write_manifest(tmp_path / "dataset_split.csv", rows)
        return list(get_feature_stream("train", csv_path=str(tmp_path / "dataset_split.csv")))

    def test_yields_decoded_bgr_images(self, streamed_pairs):
        """The stream yields one well-formed (BGR image, int label) pair per file.

        Order is intentionally not asserted here — ``get_feature_stream``
        randomizes ordering — so labels are compared as an order-independent
        multiset. Ordering is covered in ``TestRandomizedOrdering``.
        """
        assert len(streamed_pairs) == 3
        labels = []
        for img, label in streamed_pairs:
            assert img.dtype == np.uint8
            assert img.ndim == 3 and img.shape[2] == 3
            assert 0 <= img.min() and img.max() <= 255
            assert isinstance(label, int)
            labels.append(label)
        assert sorted(labels) == [0, 0, 1]

    def test_unreadable_paths_are_skipped_not_yielded(self, tmp_path):
        """An unreadable path in the middle is dropped; the stream keeps going.

        Core resilience guarantee: a single missing/undecodable image must not
        truncate the stream or raise — the two good images around it survive.
        """
        # Arrange: a missing-file path sandwiched between two good images.
        good1 = _write_png(tmp_path / "good1.png")
        good2 = _write_png(tmp_path / "good2.png")
        missing = str(tmp_path / "missing.png")
        _write_manifest(
            tmp_path / "dataset_split.csv",
            rows=[
                ("good1.png", good1, 0, "train"),
                ("missing.png", missing, 1, "train"),
                ("good2.png", good2, 0, "train"),
            ],
        )
        # Act: drain the stream, suppressing the expected per-skip UserWarning.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pairs = list(get_feature_stream("train", csv_path=str(tmp_path / "dataset_split.csv")))
        # Assert: only the two decodable images survived.
        assert len(pairs) == 2


# ===========================================================================
# 6. BGR colour convention
# ===========================================================================

class TestBGRConvention:
    """Decoded images must be in OpenCV's BGR channel order."""

    def test_loaded_red_image_is_stored_as_bgr(self, tmp_path):
        """A red image on disk loads with red in BGR channel index 2."""
        red_path = _write_png(tmp_path / "red.png", color=(0, 0, 255))
        img = _load_image(red_path)
        assert img[0, 0, 2] == 255  # red sits in the BGR red channel
        assert img[0, 0, 0] == 0    # blue channel empty


# ===========================================================================
# 7. Feature-extraction integrity
# ===========================================================================

class TestFeatureExtractionIntegrity:
    """Streamed images must survive the preprocessing pipeline intact."""

    def test_streamed_image_through_pipeline_yields_dense_vector(self, tmp_path):
        """A loaded image yields a dense, finite, correctly-shaped vector."""
        img = _load_image(_write_png(tmp_path / "sample.png"))

        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (64, 64), "preserve_aspect": False}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ])
        features = pipeline.process(img)

        assert features.ndim == 1
        assert features.shape[0] == 64 * 64
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))           # no NaN/Inf
        assert features.min() >= 0.0 and features.max() <= 1.0

    def test_individual_transforms_on_streamed_image(self, tmp_path):
        """Individual transforms accept a streamed image and reshape it correctly."""
        img = _load_image(_write_png(tmp_path / "sample.png"))
        gray = to_grayscale(img)
        resized = resize_image(img, (64, 64), preserve_aspect=False)
        assert gray.ndim == 2
        assert resized.shape == (64, 64, 3)


# ===========================================================================
# 8. Randomized ordering (random_seed)
# ===========================================================================

class TestRandomizedOrdering:
    """``get_feature_stream`` randomizes datapoint order, reproducibly when seeded.

    Identity is tracked via the unique per-row label written by
    ``_write_labeled_manifest`` (every row points at the same image file, so the
    label is the only distinguishing tag).
    """

    @pytest.fixture
    def shared_image(self, tmp_path):
        """A single real image file every labelled row in these tests points at."""
        return _write_png(tmp_path / "shared.png")

    def test_same_seed_reproduces_order(self, tmp_path, shared_image):
        """The same ``random_seed`` yields the exact same ordering twice."""
        csv_path = str(tmp_path / "dataset_split.csv")
        _write_labeled_manifest(tmp_path / "dataset_split.csv", shared_image, 25)
        first = [lbl for _, lbl in get_feature_stream("train", csv_path=csv_path, random_seed=123)]
        second = [lbl for _, lbl in get_feature_stream("train", csv_path=csv_path, random_seed=123)]
        assert first == second

    def test_seeded_order_matches_explicit_shuffle(self, tmp_path, shared_image):
        """A seeded stream matches an independent shuffle of the loaded entries.

        Confirms the shuffle is the documented ``random.Random(seed)`` permutation
        of the manifest entries, and that it really moves away from file order.
        """
        csv_path = str(tmp_path / "dataset_split.csv")
        _write_labeled_manifest(tmp_path / "dataset_split.csv", shared_image, 25)
        expected = _load_entries("train", csv_path)
        random.Random(123).shuffle(expected)
        expected_labels = [lbl for _, lbl in expected]
        streamed_labels = [lbl for _, lbl in get_feature_stream("train", csv_path=csv_path, random_seed=123)]
        assert streamed_labels == expected_labels
        assert streamed_labels != list(range(25))

    def test_no_datapoints_lost_or_duplicated(self, tmp_path, shared_image):
        """Shuffling reorders datapoints without dropping or duplicating any."""
        csv_path = str(tmp_path / "dataset_split.csv")
        _write_labeled_manifest(tmp_path / "dataset_split.csv", shared_image, 25)
        labels = [lbl for _, lbl in get_feature_stream("train", csv_path=csv_path, random_seed=7)]
        assert sorted(labels) == list(range(25))

    def test_different_seeds_produce_different_orders(self, tmp_path, shared_image):
        """Different seeds generally yield different orderings."""
        csv_path = str(tmp_path / "dataset_split.csv")
        _write_labeled_manifest(tmp_path / "dataset_split.csv", shared_image, 25)
        a = [lbl for _, lbl in get_feature_stream("train", csv_path=csv_path, random_seed=1)]
        b = [lbl for _, lbl in get_feature_stream("train", csv_path=csv_path, random_seed=2)]
        assert a != b

    def test_does_not_perturb_global_rng(self, tmp_path, shared_image):
        """Seeding the stream leaves the global ``random`` state untouched.

        The implementation uses a local ``random.Random`` instance, so a caller's
        own global RNG sequence must be unaffected by streaming.
        """
        csv_path = str(tmp_path / "dataset_split.csv")
        _write_labeled_manifest(tmp_path / "dataset_split.csv", shared_image, 10)
        # The global-RNG sequence we expect to see, captured before any stream call.
        random.seed(999)
        expected_after = [random.random() for _ in range(3)]
        # Re-seed, drain a seeded stream, then resume drawing from the global RNG.
        random.seed(999)
        list(get_feature_stream("train", csv_path=csv_path, random_seed=123))
        actual_after = [random.random() for _ in range(3)]
        assert actual_after == expected_after
