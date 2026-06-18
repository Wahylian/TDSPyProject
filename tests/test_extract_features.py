"""
Tests for ``extract_features.py`` — the local-path image streamer.

The streamer reads pre-downloaded images from local storage (no network access)
based on path metadata, decodes them to BGR arrays, and yields ``(image, label)``
pairs in a randomized order. These tests fabricate real, decodable image files
under ``tmp_path`` so the load path is exercised against actual files.

Coverage
--------
* Split validation and the module constants.
* Metadata resolution from CSV / JSON (per-split and combined files) using
  ``tmp_path`` instead of a checked-in dataset.
* Error handling for missing directories / files / columns.
* The load path (``_load_image``) for success, a missing file and an
  undecodable file.
* The generator contract of ``get_feature_stream`` (yields decoded BGR arrays).
* The BGR colour convention of the decoded images.
* Feature-extraction integrity: a streamed image pushed through the
  ``preprocessing`` pipeline yields a dense, finite, correctly-shaped
  feature vector (assert shapes, dtypes, embedding integrity).
* Randomized ordering: shuffling is reproducible under a seed, loses no
  datapoints, and leaves the global RNG untouched.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest

import extract_features as ef
from extract_features import (
    PATH_COLUMNS,
    VALID_SPLITS,
    _entries_from_csv,
    _entries_from_json,
    _load_image,
    _load_image_entries,
    _path_column,
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


def _write_split_csv(path, rows, header=("image_path", "data_split")):
    """Write a small split_dataset-style CSV to *path*."""
    lines = [",".join(header)]
    lines += [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_labeled_split_csv(path, image_path, n):
    """Write a combined train CSV with *n* rows, each uniquely labeled ``0..n-1``.

    Every row points at the same on-disk image, so the per-row label is the only
    thing that distinguishes datapoints — a stable identity tag for tracking
    exactly where each datapoint ends up after the stream's shuffle.
    """
    lines = ["image_path,label_numeric,data_split"]
    lines += [f"{image_path},{i},train" for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===========================================================================
# 1. Split validation + constants
# ===========================================================================

class TestSplitValidationAndConstants:
    """Guard the split allow-list and the module's public constants.

    ``_validate_split`` is the gatekeeper every public entry point relies on,
    so these tests pin down exactly which split names are accepted and confirm
    the constants other modules import have not drifted from their contract.
    """

    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_accepts_valid_splits(self, split):
        """Every canonical split name validates without raising.

        Args:
            split: One of the canonical split names, supplied by the
                parametrize sweep over ``VALID_SPLITS``.
        """
        # Act + Assert: a recognised split must pass silently (no return value,
        # no exception). The call itself is the assertion.
        _validate_split(split)

    @pytest.mark.parametrize("bad", ["", "training", "TRAIN", "dev", "all"])
    def test_rejects_invalid_splits(self, bad):
        """Unknown / mis-cased / empty split names are rejected.

        The cases deliberately cover the most likely caller mistakes: an empty
        string, a verbose synonym ("training"), wrong casing ("TRAIN"), an
        unsupported split ("dev"), and the tempting-but-invalid wildcard "all".

        Args:
            bad: An invalid split label supplied by the parametrize sweep.
        """
        # Act + Assert: validation must fail loudly rather than silently
        # accepting an unsupported split and corrupting downstream filtering.
        with pytest.raises(ValueError):
            _validate_split(bad)

    def test_module_constants_match_spec(self):
        """The exported constants match their documented contract.

        Other modules import these values directly, so a change here is a
        breaking API change — this test makes such a change fail fast.
        """
        # Assert: the frozen split set and the recognised path columns are unchanged.
        assert VALID_SPLITS == frozenset({"train", "test", "val"})
        assert PATH_COLUMNS == ("image_path", "path")


# ===========================================================================
# 2. Metadata loading (CSV / JSON) from tmp_path
# ===========================================================================

class TestMetadataLoading:
    """Resolve image-path lists from on-disk metadata.

    These tests use ``tmp_path`` to fabricate the two supported metadata
    layouts — a combined ``split_dataset.csv`` and per-split JSON files — so
    the loader is exercised against real files without shipping a dataset.
    """

    def test_loads_paths_from_combined_split_csv(self, tmp_path):
        """A combined CSV is read and filtered down to the requested split.

        Validates that the loader keys off the ``data_split`` column rather
        than returning every row, so callers asking for "train" never leak
        "test" paths into their stream. With no label column the entries carry
        ``label=None``.
        """
        # Arrange: a combined split_dataset.csv with rows for two splits.
        csv_path = tmp_path / "split_dataset.csv"
        _write_split_csv(
            csv_path,
            rows=[
                ("photos/a.jpg", "train"),
                ("photos/b.jpg", "train"),
                ("photos/c.jpg", "test"),
            ],
        )
        # Act: load each split independently from the same combined file.
        train_entries = _load_image_entries("train", str(tmp_path))
        test_entries = _load_image_entries("test", str(tmp_path))
        # Assert: split filtering keeps only the matching rows; no label column
        # means every entry is unlabeled.
        assert train_entries == [
            ("photos/a.jpg", None),
            ("photos/b.jpg", None),
        ]
        assert test_entries == [("photos/c.jpg", None)]

    def test_loads_path_label_pairs_from_combined_csv(self, tmp_path):
        """A combined CSV with a ``label_numeric`` column yields aligned labels.

        This is the core supervised-learning guarantee: each path is paired with
        the label from its own row, and split filtering keeps that pairing
        intact (the "test"-only row never bleeds into the train entries).
        """
        # Arrange: a combined CSV carrying labels, with rows for two splits.
        csv_path = tmp_path / "split_dataset.csv"
        csv_path.write_text(
            "image_path,label_numeric,data_split\n"
            "photos/a.jpg,0,train\n"
            "photos/b.jpg,1,train\n"
            "photos/c.jpg,1,test\n",
            encoding="utf-8",
        )
        # Act
        train_entries = _load_image_entries("train", str(tmp_path))
        test_entries = _load_image_entries("test", str(tmp_path))
        # Assert: labels stay aligned with their paths after split filtering.
        assert train_entries == [
            ("photos/a.jpg", "0"),
            ("photos/b.jpg", "1"),
        ]
        assert test_entries == [("photos/c.jpg", "1")]

    def test_empty_path_rows_are_skipped(self, tmp_path):
        """Rows whose image could not be downloaded (empty path) are not loaded.

        A failed download is recorded with an empty ``image_path``; the loader
        skips it so it never reaches the stream, while the surrounding rows are
        kept.
        """
        # Arrange: the middle row has no path (a failed download).
        csv_path = tmp_path / "split_dataset.csv"
        csv_path.write_text(
            "image_path,label_numeric,data_split\n"
            "photos/a.jpg,0,train\n"
            ",1,train\n"
            "photos/c.jpg,0,train\n",
            encoding="utf-8",
        )
        # Act
        train_entries = _load_image_entries("train", str(tmp_path))
        # Assert: only the two rows with a path survive.
        assert train_entries == [("photos/a.jpg", "0"), ("photos/c.jpg", "0")]

    def test_loads_paths_from_per_split_json(self, tmp_path):
        """A per-split ``{split}.json`` file is loaded verbatim.

        When a dedicated per-split file exists it already contains only that
        split's paths, so the loader must take it as-is and skip the CSV
        split-filtering path entirely.
        """
        # Arrange: a train.json holding exactly the train paths (no filtering needed).
        (tmp_path / "train.json").write_text(
            '["photos/1.jpg", "photos/2.jpg"]', encoding="utf-8"
        )
        # Act
        entries = _load_image_entries("train", str(tmp_path))
        # Assert: paths are returned unchanged and in order; JSON has no labels.
        assert entries == [("photos/1.jpg", None), ("photos/2.jpg", None)]

    def test_path_column_detection(self):
        """The path column is auto-detected from a header row.

        The loader supports datasets that name the column either "image_path"
        or "path"; this pins down the detection priority for both spellings.
        """
        # Act + Assert: each known column name is recognised regardless of the
        # other (non-path) columns present.
        assert _path_column(["id", "image_path"]) == "image_path"
        assert _path_column(["path", "label"]) == "path"

    def test_missing_path_column_raises(self, tmp_path):
        """A CSV with no recognisable path column fails loudly.

        Edge case: without a path column there is nothing to stream, so the
        loader must raise rather than silently yield an empty result that
        would look like "no images for this split".
        """
        # Arrange: a CSV whose headers ("foo", "bar") match no known path column.
        bad = tmp_path / "split.csv"
        bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
        # Act + Assert
        with pytest.raises(ValueError):
            _entries_from_csv(str(bad), "train", filter_by_split=False)


class TestJsonMetadataStructures:
    """The JSON metadata reader accepts several documented layouts.

    ``_entries_from_json`` supports a bare list, a dict keyed by split name, and a
    dict carrying a generic ``"paths"`` key — and must reject anything else so a
    malformed metadata file fails loudly rather than yielding nothing.
    """

    def test_dict_keyed_by_split_returns_that_splits_paths(self, tmp_path):
        """A dict keyed by split name returns only the requested split's paths."""
        # Arrange: a JSON object with separate lists per split.
        path = tmp_path / "split.json"
        path.write_text(
            '{"train": ["photos/a.jpg"], "test": ["photos/b.jpg"]}',
            encoding="utf-8",
        )
        # Act
        entries = _entries_from_json(str(path), "train")
        # Assert: only the train list is returned; JSON entries are unlabeled.
        assert entries == [("photos/a.jpg", None)]

    def test_dict_with_generic_paths_key_is_used(self, tmp_path):
        """A dict with a generic ``"paths"`` key is read when the split is absent.

        When the object has no per-split entry, the loader falls back to a
        flat ``"paths"`` list rather than failing.
        """
        # Arrange: a JSON object exposing a single shared "paths" list.
        path = tmp_path / "split.json"
        path.write_text('{"paths": ["photos/1.jpg", "photos/2.jpg"]}', encoding="utf-8")
        # Act
        entries = _entries_from_json(str(path), "train")
        # Assert
        assert entries == [("photos/1.jpg", None), ("photos/2.jpg", None)]

    def test_unrecognized_structure_raises(self, tmp_path):
        """A JSON shape that is neither list nor a known dict layout raises.

        Edge case: a dict with no split entry and no ``"paths"`` key is
        unusable, so the loader must raise instead of returning an empty result
        that masquerades as "no images".
        """
        # Arrange: a dict carrying an unrelated key only.
        path = tmp_path / "split.json"
        path.write_text('{"something_else": [1, 2, 3]}', encoding="utf-8")
        # Act + Assert
        with pytest.raises(ValueError):
            _entries_from_json(str(path), "train")


class TestCsvMetadataVariants:
    """CSV column detection and split-filtering edge cases."""

    def test_per_split_csv_loads_all_rows_without_filtering(self, tmp_path):
        """A per-split ``{split}.csv`` returns every row (no split column needed).

        A dedicated per-split file already contains only that split's rows, so
        the loader reads it with ``filter_by_split=False`` and keeps them all,
        even though there is no ``data_split`` column to filter on.
        """
        # Arrange: a train.csv with only a path column (no split column).
        path = tmp_path / "train.csv"
        path.write_text(
            "image_path\nphotos/1.jpg\nphotos/2.jpg\n", encoding="utf-8"
        )
        # Act
        entries = _load_image_entries("train", str(tmp_path))
        # Assert: both rows are returned unfiltered; no label column -> None.
        assert entries == [("photos/1.jpg", None), ("photos/2.jpg", None)]

    def test_path_column_variant_is_detected_end_to_end(self, tmp_path):
        """A combined CSV using the ``path`` column name still streams correctly.

        The loader supports either ``image_path`` or ``path``; this exercises the
        ``path`` spelling through the full combined-file path with split filtering.
        """
        # Arrange: a split_dataset.csv whose path column is named "path".
        path = tmp_path / "split_dataset.csv"
        path.write_text(
            "path,data_split\n"
            "photos/a.jpg,train\n"
            "photos/b.jpg,test\n",
            encoding="utf-8",
        )
        # Act
        train_entries = _load_image_entries("train", str(tmp_path))
        # Assert: the "path" column was detected and split filtering applied.
        assert train_entries == [("photos/a.jpg", None)]

    def test_filter_by_split_without_split_column_raises(self, tmp_path):
        """Requesting split filtering on a CSV lacking a split column raises.

        Edge case: a combined file must carry a ``data_split`` / ``split``
        column to be filtered; without one the loader cannot honour the request
        and must raise rather than silently returning every split's rows.
        """
        # Arrange: a CSV with a path column but no split column.
        path = tmp_path / "split.csv"
        path.write_text("image_path\nphotos/1.jpg\n", encoding="utf-8")
        # Act + Assert
        with pytest.raises(ValueError):
            _entries_from_csv(str(path), "train", filter_by_split=True)


# ===========================================================================
# 3. Error handling
# ===========================================================================

class TestErrorHandling:
    """Filesystem-level failure modes of metadata loading.

    Distinguishes the two distinct "no data" situations a caller can hit —
    a base directory that does not exist versus one that exists but is empty
    — both of which must surface as ``FileNotFoundError``.
    """

    def test_missing_base_dir_raises(self):
        """A non-existent base directory raises ``FileNotFoundError``.

        Edge case: a base_dir that doesn't exist at all (e.g. a typo'd path)
        must fail immediately instead of being treated as "empty dataset".
        """
        # Act + Assert
        with pytest.raises(FileNotFoundError):
            _load_image_entries("train", "./nonexistent_directory_xyz")

    def test_empty_dir_raises_filenotfound(self, tmp_path):
        """An existing-but-empty directory still raises ``FileNotFoundError``.

        Edge case: the directory is present but holds no metadata file, so
        there is no CSV/JSON to read. ``tmp_path`` provides exactly such an
        empty, real directory.
        """
        # Act + Assert
        with pytest.raises(FileNotFoundError):
            _load_image_entries("train", str(tmp_path))


# ===========================================================================
# 4. Load path (local files)
# ===========================================================================

class TestLoadImage:
    """The single-image read/decode path against real local files.

    Covers the happy path plus the two failure modes a resilient loader must
    tolerate: a path that does not exist, and a file that exists but holds
    undecodable bytes.
    """

    def test_successful_load_returns_bgr_array(self, tmp_path):
        """A valid PNG file decodes to a uint8 BGR array of the right size."""
        # Arrange: write a real 12x8 image to disk.
        img_path = _write_png(tmp_path / "ok.png")
        # Act
        img = _load_image(img_path)
        # Assert: decoded to a 3-channel uint8 array of the written size.
        assert img is not None
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.uint8
        assert img.shape[:2] == (8, 12)  # (height, width) of _write_png default

    def test_missing_file_returns_none(self, tmp_path):
        """A path that does not exist yields ``None`` after a warning, not a crash.

        Edge case: a recorded path whose file is absent must be swallowed so one
        missing image does not abort the whole stream — it is skipped (None) and
        the caller is told via a ``UserWarning``.
        """
        # Act + Assert
        missing = str(tmp_path / "does_not_exist.png")
        with pytest.warns(UserWarning):
            assert _load_image(missing) is None

    def test_undecodable_file_returns_none(self, tmp_path):
        """A file with undecodable bytes yields ``None`` with a warning.

        Edge case: the file exists but is not a valid image. Decoding must fail
        gracefully — warn and skip — rather than propagating an OpenCV exception
        up the stream.
        """
        # Arrange: a .png file holding non-image bytes.
        bad = tmp_path / "garbage.png"
        bad.write_bytes(b"not-an-image")
        # Act + Assert
        with pytest.warns(UserWarning):
            assert _load_image(str(bad)) is None


# ===========================================================================
# 5. Generator contract
# ===========================================================================

class TestGeneratorContract:
    """The end-to-end ``get_feature_stream`` generator behaviour.

    Verifies the public streaming entry point both yields well-formed decoded
    images and honours its resilience contract — that unreadable paths are
    skipped rather than aborting iteration.
    """

    @pytest.fixture
    def streamed_pairs(self, tmp_path):
        """Materialise the full stream for a 3-file, labelled train split.

        Writes three real images and a combined CSV that points at them with
        alternating labels, then eagerly drains the generator into a list so
        individual tests can assert on the results.

        Returns:
            list[tuple[np.ndarray, str]]: The (image, label) pairs produced by
            the stream.
        """
        rows = []
        for i in range(3):
            img_path = _write_png(tmp_path / f"{i}.png")
            rows.append((img_path, str(i % 2), "train"))
        _write_split_csv(
            tmp_path / "split_dataset.csv",
            rows=rows,
            header=("image_path", "label_numeric", "data_split"),
        )
        return list(get_feature_stream("train", base_dir=str(tmp_path)))

    def test_yields_decoded_bgr_images(self, streamed_pairs):
        """The stream yields one well-formed (BGR image, label) pair per good file.

        Args:
            streamed_pairs: Pre-drained list of (image, label) pairs from the fixture.
        """
        # Assert: count matches the input files and every item is a valid
        # 3-channel uint8 image paired with its label. Order is intentionally not
        # asserted here — ``get_feature_stream`` randomizes ordering by default —
        # so the labels are compared as an order-independent multiset. Ordering
        # behaviour is covered explicitly in ``TestRandomizedOrdering``.
        assert len(streamed_pairs) == 3
        labels = []
        for img, label in streamed_pairs:
            assert img.dtype == np.uint8
            assert img.ndim == 3 and img.shape[2] == 3
            assert 0 <= img.min() and img.max() <= 255
            labels.append(label)
        # Every datapoint survives the shuffle: same labels, regardless of order.
        assert sorted(labels) == ["0", "0", "1"]

    def test_unreadable_paths_are_skipped_not_yielded(self, tmp_path):
        """An unreadable path in the middle is dropped; the stream keeps going.

        This is the core resilience guarantee: a single missing/undecodable image
        must not truncate the stream or raise — the two good images on either
        side of it are still yielded.
        """
        # Arrange: a split with a missing-file path deliberately sandwiched
        # between two good, real images.
        good1 = _write_png(tmp_path / "good1.png")
        good2 = _write_png(tmp_path / "good2.png")
        missing = str(tmp_path / "missing.png")
        _write_split_csv(
            tmp_path / "split_dataset.csv",
            rows=[
                (good1, "train"),
                (missing, "train"),
                (good2, "train"),
            ],
        )
        # Act: drain the stream, suppressing the expected per-skip UserWarning
        # so it does not clutter the test output (the skip itself is asserted below).
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # unreadable path emits a UserWarning
            pairs = list(get_feature_stream("train", base_dir=str(tmp_path)))
        # Assert: only the two decodable images survived.
        assert len(pairs) == 2  # the single unreadable path was dropped


# ===========================================================================
# 6. BGR colour convention
# ===========================================================================

class TestBGRConvention:
    """Decoded images must be in OpenCV's BGR channel order.

    Downstream OpenCV-based preprocessing assumes BGR. ``cv2.imread`` returns
    BGR directly, so this round-trip test locks in the channel order and catches
    a silent regression (which would swap red/blue everywhere).
    """

    def test_loaded_red_image_is_stored_as_bgr(self, tmp_path):
        """A red image on disk loads with red in BGR channel index 2.

        Writes a pure-red image (BGR ``[0, 0, 255]``) and reads it back through
        the real load path, proving the decoded array keeps red in channel 2 and
        an empty blue channel 0.
        """
        # Arrange: write a solid pure-red image in BGR order.
        red_path = _write_png(tmp_path / "red.png", color=(0, 0, 255))
        # Act
        img = _load_image(red_path)
        # Assert: red sits in the BGR red channel, blue channel is empty.
        assert img[0, 0, 2] == 255  # red sits in the BGR red channel
        assert img[0, 0, 0] == 0    # blue channel empty


# ===========================================================================
# 7. Feature-extraction integrity
# ===========================================================================

class TestFeatureExtractionIntegrity:
    """Streamed images must survive the preprocessing pipeline intact.

    Bridges the streamer and the ``preprocessing`` pipeline: an image
    pulled from the local load path is pushed through a real pipeline and the
    resulting feature vector is checked for shape, dtype and embedding integrity
    (no NaN/Inf, correct width, expected range).
    """

    def test_streamed_image_through_pipeline_yields_dense_vector(self, tmp_path):
        """A loaded image yields a dense, finite, correctly-shaped vector.

        This is the key integration guarantee: the bytes coming out of the
        loader are compatible with the downstream pipeline and produce a
        clean embedding rather than NaNs or a mis-shaped array.
        """
        # Arrange: load one real image straight from the load path.
        img = _load_image(_write_png(tmp_path / "sample.png"))

        # Act: run it through a representative pipeline.
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (64, 64), "preserve_aspect": False}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ])
        features = pipeline.process(img)

        # Assert: a dense 1D embedding of the expected width, type and range.
        assert features.ndim == 1
        assert features.shape[0] == 64 * 64
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))           # no NaN/Inf
        assert features.min() >= 0.0 and features.max() <= 1.0

    def test_individual_transforms_on_streamed_image(self, tmp_path):
        """Individual transforms accept a streamed image and reshape it correctly.

        A finer-grained companion to the full-pipeline test: confirms the two
        foundational transforms (grayscale collapse and resize) behave on real
        loaded data, isolating shape regressions to a single transform.
        """
        # Arrange: load one real image from the load path.
        img = _load_image(_write_png(tmp_path / "sample.png"))
        # Act: apply the two transforms directly.
        gray = to_grayscale(img)
        resized = resize_image(img, (64, 64), preserve_aspect=False)
        # Assert: grayscale drops the channel axis; resize hits the exact target.
        assert gray.ndim == 2
        assert resized.shape == (64, 64, 3)


# ===========================================================================
# 8. Randomized ordering (random_seed)
# ===========================================================================

class TestRandomizedOrdering:
    """``get_feature_stream`` randomizes datapoint order, reproducibly when seeded.

    These tests pin the ordering contract: by default the stream is shuffled
    (so training is not biased by file order), a passed ``random_seed`` makes that
    shuffle exactly reproducible, and — whatever the order — no datapoint is ever
    dropped or duplicated. Identity is tracked via the unique per-row label
    written by ``_write_labeled_split_csv`` (every row points at the same image
    file, so the label is the only distinguishing tag).
    """

    @pytest.fixture
    def shared_image(self, tmp_path):
        """A single real image file every labelled row in these tests points at."""
        return _write_png(tmp_path / "shared.png")

    def test_same_seed_reproduces_order(self, tmp_path, shared_image):
        """The same ``random_seed`` yields the exact same ordering twice.

        This is the core reproducibility guarantee the parameter exists for:
        seeding lets tests (and debugging runs) replay an identical stream.
        """
        # Arrange: a 25-row train split, each row uniquely labeled.
        _write_labeled_split_csv(tmp_path / "split_dataset.csv", shared_image, 25)
        # Act: drain the stream twice with the same seed, tracking label order.
        first = [lbl for _, lbl in get_feature_stream("train", base_dir=str(tmp_path), random_seed=123)]
        second = [lbl for _, lbl in get_feature_stream("train", base_dir=str(tmp_path), random_seed=123)]
        # Assert: identical ordering across the two seeded runs.
        assert first == second

    def test_seeded_order_matches_explicit_shuffle(self, tmp_path, shared_image):
        """A seeded stream matches an independent shuffle of the loaded entries.

        Confirms two things at once: the shuffle is the documented
        ``random.Random(seed)`` permutation of the metadata (so its order is
        predictable from the seed), and the result is a genuine permutation
        rather than the original file order.
        """
        # Arrange
        _write_labeled_split_csv(tmp_path / "split_dataset.csv", shared_image, 25)
        # Arrange: reproduce the expected ordering by applying the same seeded
        # shuffle to an independently loaded copy of the entries.
        expected = _load_image_entries("train", str(tmp_path))
        random.Random(123).shuffle(expected)
        expected_labels = [lbl for _, lbl in expected]
        # Act
        streamed_labels = [lbl for _, lbl in get_feature_stream("train", base_dir=str(tmp_path), random_seed=123)]
        # Assert: stream order matches the explicit shuffle, and it really moved
        # away from the source (0..24) order.
        assert streamed_labels == expected_labels
        assert streamed_labels != [str(i) for i in range(25)]

    def test_no_datapoints_lost_or_duplicated(self, tmp_path, shared_image):
        """Shuffling reorders datapoints without dropping or duplicating any.

        Edge case the randomization must never violate: a permutation preserves
        the full multiset of datapoints — sorting the streamed labels must
        recover exactly the original 0..n-1 set.
        """
        # Arrange
        _write_labeled_split_csv(tmp_path / "split_dataset.csv", shared_image, 25)
        # Act
        labels = [lbl for _, lbl in get_feature_stream("train", base_dir=str(tmp_path), random_seed=7)]
        # Assert: same set of datapoints, just reordered.
        assert sorted(labels, key=int) == [str(i) for i in range(25)]

    def test_different_seeds_produce_different_orders(self, tmp_path, shared_image):
        """Different seeds generally yield different orderings.

        Sanity check that the seed actually drives the permutation (a no-op
        "shuffle" would make every seed identical). With 25 datapoints the odds
        of two distinct seeds colliding are negligible.
        """
        # Arrange
        _write_labeled_split_csv(tmp_path / "split_dataset.csv", shared_image, 25)
        # Act
        a = [lbl for _, lbl in get_feature_stream("train", base_dir=str(tmp_path), random_seed=1)]
        b = [lbl for _, lbl in get_feature_stream("train", base_dir=str(tmp_path), random_seed=2)]
        # Assert
        assert a != b

    def test_does_not_perturb_global_rng(self, tmp_path, shared_image):
        """Seeding the stream leaves the global ``random`` state untouched.

        The implementation uses a local ``random.Random`` instance, so a caller's
        own global RNG sequence must be unaffected by streaming — important for
        test isolation and for any other seeded randomness in a training run.
        """
        # Arrange
        _write_labeled_split_csv(tmp_path / "split_dataset.csv", shared_image, 10)
        # Arrange: the global-RNG sequence we expect to see, captured before any stream call.
        random.seed(999)
        expected_after = [random.random() for _ in range(3)]
        # Act: re-seed, drain a seeded stream, then resume drawing from the global RNG.
        random.seed(999)
        list(get_feature_stream("train", base_dir=str(tmp_path), random_seed=123))
        actual_after = [random.random() for _ in range(3)]
        # Assert: the stream's internal shuffle did not consume the global RNG.
        assert actual_after == expected_after
