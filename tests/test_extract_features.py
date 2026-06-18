"""
Tests for ``extract_features.py`` — the URL-metadata image streamer.

This is the pytest migration of ``verify_extract_features.py``. The original
was a script that downloaded real images over the network and printed PASS/FAIL
lines. Here those checks become isolated pytest functions, and **all network
access is mocked**: ``requests.get`` is replaced with a stub that returns
synthetic PNG bytes, so the tests are fast, deterministic and offline.

Coverage
--------
* Split validation and the module constants.
* Metadata resolution from CSV / JSON (per-split and combined files) using
  ``tmp_path`` instead of a checked-in dataset.
* Error handling for missing directories / files / columns.
* The download path (``_download_image``) for success, HTTP error and broken
  URLs — all mocked.
* The generator contract of ``get_feature_stream`` (yields decoded BGR arrays).
* BGR colour-convention checks (synthetic, no network).
* Feature-extraction integrity: a streamed image pushed through the
  ``preprocessing`` pipeline yields a dense, finite, correctly-shaped
  feature vector (instruction 6 — assert shapes, dtypes, embedding integrity).
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

import extract_features as ef
from extract_features import (
    REQUEST_TIMEOUT,
    VALID_SPLITS,
    _download_image,
    _entries_from_csv,
    _entries_from_json,
    _load_image_entries,
    _url_column,
    _validate_split,
    get_feature_stream,
)
from preprocessing import ImagePipeline, resize_image, to_grayscale


# ===========================================================================
# Test helpers / mocks
# ===========================================================================

def _png_bytes(width: int = 12, height: int = 8, color=(200, 120, 40)) -> bytes:
    """Encode a solid-colour RGB image as PNG bytes (an in-memory fake download)."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeResponse:
    """Minimal stand-in for a ``requests.Response``."""

    def __init__(self, content: bytes, status_ok: bool = True):
        self.content = content
        self._ok = status_ok

    def raise_for_status(self):
        if not self._ok:
            import requests
            raise requests.HTTPError("404")


@pytest.fixture
def mock_download_ok(monkeypatch):
    """Patch requests.get so every download returns the same valid PNG."""
    payload = _png_bytes()

    def fake_get(url, timeout=None):
        return _FakeResponse(payload, status_ok=True)

    monkeypatch.setattr(ef.requests, "get", fake_get)
    return payload


def _write_split_csv(path, rows, header=("image_url", "data_split")):
    """Write a small split_dataset-style CSV to *path*."""
    lines = [",".join(header)]
    lines += [",".join(r) for r in rows]
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
        # Assert: the request timeout and the frozen split set are unchanged.
        assert REQUEST_TIMEOUT == 10
        assert VALID_SPLITS == frozenset({"train", "test", "val"})


# ===========================================================================
# 2. Metadata loading (CSV / JSON) from tmp_path
# ===========================================================================

class TestMetadataLoading:
    """Resolve image-URL lists from on-disk metadata.

    These tests use ``tmp_path`` to fabricate the two supported metadata
    layouts — a combined ``split_dataset.csv`` and per-split JSON files — so
    the loader is exercised against real files without shipping a dataset.
    """

    def test_loads_urls_from_combined_split_csv(self, tmp_path):
        """A combined CSV is read and filtered down to the requested split.

        Validates that the loader keys off the ``data_split`` column rather
        than returning every row, so callers asking for "train" never leak
        "test" URLs into their stream. With no label column the entries carry
        ``label=None``.
        """
        # Arrange: a combined split_dataset.csv with rows for two splits.
        csv_path = tmp_path / "split_dataset.csv"
        _write_split_csv(
            csv_path,
            rows=[
                ("http://example.com/a.jpg", "train"),
                ("http://example.com/b.jpg", "train"),
                ("http://example.com/c.jpg", "test"),
            ],
        )
        # Act: load each split independently from the same combined file.
        train_entries = _load_image_entries("train", str(tmp_path))
        test_entries = _load_image_entries("test", str(tmp_path))
        # Assert: split filtering keeps only the matching rows; no label column
        # means every entry is unlabeled.
        assert train_entries == [
            ("http://example.com/a.jpg", None),
            ("http://example.com/b.jpg", None),
        ]
        assert test_entries == [("http://example.com/c.jpg", None)]

    def test_loads_url_label_pairs_from_combined_csv(self, tmp_path):
        """A combined CSV with a ``label_numeric`` column yields aligned labels.

        This is the core supervised-learning guarantee: each URL is paired with
        the label from its own row, and split filtering keeps that pairing
        intact (the "test"-only row never bleeds into the train entries).
        """
        # Arrange: a combined CSV carrying labels, with rows for two splits.
        csv_path = tmp_path / "split_dataset.csv"
        csv_path.write_text(
            "image_url,label_numeric,data_split\n"
            "http://example.com/a.jpg,0,train\n"
            "http://example.com/b.jpg,1,train\n"
            "http://example.com/c.jpg,1,test\n",
            encoding="utf-8",
        )
        # Act
        train_entries = _load_image_entries("train", str(tmp_path))
        test_entries = _load_image_entries("test", str(tmp_path))
        # Assert: labels stay aligned with their URLs after split filtering.
        assert train_entries == [
            ("http://example.com/a.jpg", "0"),
            ("http://example.com/b.jpg", "1"),
        ]
        assert test_entries == [("http://example.com/c.jpg", "1")]

    def test_loads_urls_from_per_split_json(self, tmp_path):
        """A per-split ``{split}.json`` file is loaded verbatim.

        When a dedicated per-split file exists it already contains only that
        split's URLs, so the loader must take it as-is and skip the CSV
        split-filtering path entirely.
        """
        # Arrange: a train.json holding exactly the train URLs (no filtering needed).
        (tmp_path / "train.json").write_text(
            '["http://x/1.png", "http://x/2.png"]', encoding="utf-8"
        )
        # Act
        entries = _load_image_entries("train", str(tmp_path))
        # Assert: URLs are returned unchanged and in order; JSON has no labels.
        assert entries == [("http://x/1.png", None), ("http://x/2.png", None)]

    def test_url_column_detection(self):
        """The URL column is auto-detected from a header row.

        The loader supports datasets that name the column either "image_url"
        or "url"; this pins down the detection priority for both spellings.
        """
        # Act + Assert: each known column name is recognised regardless of the
        # other (non-URL) columns present.
        assert _url_column(["id", "image_url"]) == "image_url"
        assert _url_column(["url", "label"]) == "url"

    def test_missing_url_column_raises(self, tmp_path):
        """A CSV with no recognisable URL column fails loudly.

        Edge case: without a URL column there is nothing to stream, so the
        loader must raise rather than silently yield an empty result that
        would look like "no images for this split".
        """
        # Arrange: a CSV whose headers ("foo", "bar") match no known URL column.
        bad = tmp_path / "split.csv"
        bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
        # Act + Assert
        with pytest.raises(ValueError):
            _entries_from_csv(str(bad), "train", filter_by_split=False)


class TestJsonMetadataStructures:
    """The JSON metadata reader accepts several documented layouts.

    ``_entries_from_json`` supports a bare list, a dict keyed by split name, and a
    dict carrying a generic ``"urls"`` key — and must reject anything else so a
    malformed metadata file fails loudly rather than yielding nothing.
    """

    def test_dict_keyed_by_split_returns_that_splits_urls(self, tmp_path):
        """A dict keyed by split name returns only the requested split's URLs."""
        # Arrange: a JSON object with separate lists per split.
        path = tmp_path / "split.json"
        path.write_text(
            '{"train": ["http://x/a.png"], "test": ["http://x/b.png"]}',
            encoding="utf-8",
        )
        # Act
        entries = _entries_from_json(str(path), "train")
        # Assert: only the train list is returned; JSON entries are unlabeled.
        assert entries == [("http://x/a.png", None)]

    def test_dict_with_generic_urls_key_is_used(self, tmp_path):
        """A dict with a generic ``"urls"`` key is read when the split is absent.

        When the object has no per-split entry, the loader falls back to a
        flat ``"urls"`` list rather than failing.
        """
        # Arrange: a JSON object exposing a single shared "urls" list.
        path = tmp_path / "split.json"
        path.write_text('{"urls": ["http://x/1.png", "http://x/2.png"]}', encoding="utf-8")
        # Act
        entries = _entries_from_json(str(path), "train")
        # Assert
        assert entries == [("http://x/1.png", None), ("http://x/2.png", None)]

    def test_unrecognized_structure_raises(self, tmp_path):
        """A JSON shape that is neither list nor a known dict layout raises.

        Edge case: a dict with no split entry and no ``"urls"`` key is
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
        # Arrange: a train.csv with only a URL column (no split column).
        path = tmp_path / "train.csv"
        path.write_text(
            "image_url\nhttp://x/1.png\nhttp://x/2.png\n", encoding="utf-8"
        )
        # Act
        entries = _load_image_entries("train", str(tmp_path))
        # Assert: both rows are returned unfiltered; no label column -> None.
        assert entries == [("http://x/1.png", None), ("http://x/2.png", None)]

    def test_url_column_variant_is_detected_end_to_end(self, tmp_path):
        """A combined CSV using the ``url`` column name still streams correctly.

        The loader supports either ``image_url`` or ``url``; this exercises the
        ``url`` spelling through the full combined-file path with split filtering.
        """
        # Arrange: a split_dataset.csv whose URL column is named "url".
        path = tmp_path / "split_dataset.csv"
        path.write_text(
            "url,data_split\n"
            "http://x/a.png,train\n"
            "http://x/b.png,test\n",
            encoding="utf-8",
        )
        # Act
        train_entries = _load_image_entries("train", str(tmp_path))
        # Assert: the "url" column was detected and split filtering applied.
        assert train_entries == [("http://x/a.png", None)]

    def test_filter_by_split_without_split_column_raises(self, tmp_path):
        """Requesting split filtering on a CSV lacking a split column raises.

        Edge case: a combined file must carry a ``data_split`` / ``split``
        column to be filtered; without one the loader cannot honour the request
        and must raise rather than silently returning every split's rows.
        """
        # Arrange: a CSV with a URL column but no split column.
        path = tmp_path / "split.csv"
        path.write_text("image_url\nhttp://x/1.png\n", encoding="utf-8")
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
# 4. Download path (mocked network)
# ===========================================================================

class TestDownloadImage:
    """The single-image download/decode path, with the network mocked.

    ``requests.get`` is replaced (via the ``mock_download_ok`` fixture or an
    inline ``monkeypatch``) so these tests never touch the network. They cover
    the happy path plus the two failure modes a real downloader must tolerate:
    an HTTP error status and a 200 response carrying undecodable bytes.
    """

    def test_successful_download_returns_bgr_array(self, mock_download_ok):
        """A valid PNG download decodes to a uint8 BGR array of the right size.

        Args:
            mock_download_ok: Fixture that stubs ``requests.get`` to return a
                fixed valid PNG payload, so no real request is made.
        """
        # Act
        img = _download_image("http://example.com/whatever.png")
        # Assert: decoded to a 3-channel uint8 BGR array of the fake's size.
        assert img is not None
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.uint8
        assert img.shape[:2] == (8, 12)  # (height, width) of _png_bytes default

    def test_http_error_returns_none(self, monkeypatch):
        """An HTTP error status yields ``None`` after a warning, not a crash.

        Edge case: a 4xx/5xx response must be swallowed so one dead URL does
        not abort the whole stream — the image is skipped (None) and the
        caller is told via a ``UserWarning``.
        """
        # Arrange: stub requests.get with a response whose raise_for_status fails.
        def fake_get(url, timeout=None):
            return _FakeResponse(b"", status_ok=False)

        monkeypatch.setattr(ef.requests, "get", fake_get)
        # Act + Assert: the source warns before swallowing the error; assert both
        # the warning is emitted and None is returned.
        with pytest.warns(UserWarning):
            assert _download_image("http://example.com/404.png") is None

    def test_broken_content_returns_none(self, monkeypatch):
        """A 200 response with undecodable bytes yields ``None`` with a warning.

        Edge case: the request "succeeds" (status OK) but the body is not a
        valid image. Decoding must fail gracefully — warn and skip — rather
        than propagating an OpenCV/PIL exception up the stream.
        """
        # Arrange: stub requests.get to return non-image bytes with an OK status.
        def fake_get(url, timeout=None):
            return _FakeResponse(b"not-an-image", status_ok=True)

        monkeypatch.setattr(ef.requests, "get", fake_get)
        # Act + Assert
        with pytest.warns(UserWarning):
            assert _download_image("http://example.com/garbage") is None


# ===========================================================================
# 5. Generator contract
# ===========================================================================

class TestGeneratorContract:
    """The end-to-end ``get_feature_stream`` generator behaviour.

    Verifies the public streaming entry point both yields well-formed decoded
    images and honours its resilience contract — that undecodable URLs are
    skipped rather than aborting iteration. All downloads are mocked.
    """

    @pytest.fixture
    def streamed_pairs(self, tmp_path, mock_download_ok):
        """Materialise the full stream for a 3-URL, all-mocked train split.

        Arranges a combined CSV with three labelled train URLs and eagerly
        drains the generator into a list so individual tests can assert on the
        results.

        Args:
            tmp_path: Pytest temp directory holding the fabricated CSV.
            mock_download_ok: Stubs every download to return a valid PNG.

        Returns:
            list[tuple[np.ndarray, str]]: The (image, label) pairs produced by
            the stream.
        """
        # Arrange a 3-URL train split with alternating labels; every download is
        # mocked to succeed.
        (tmp_path / "split_dataset.csv").write_text(
            "image_url,label_numeric,data_split\n"
            + "".join(
                f"http://example.com/{i}.png,{i % 2},train\n" for i in range(3)
            ),
            encoding="utf-8",
        )
        return list(get_feature_stream("train", base_dir=str(tmp_path)))

    def test_yields_decoded_bgr_images(self, streamed_pairs):
        """The stream yields one well-formed (BGR image, label) pair per good URL.

        Args:
            streamed_pairs: Pre-drained list of (image, label) pairs from the fixture.
        """
        # Assert: count matches the input URLs and every item is a valid
        # 3-channel uint8 image paired with its label, in CSV order.
        assert len(streamed_pairs) == 3
        for (img, label), expected_label in zip(streamed_pairs, ["0", "1", "0"]):
            assert img.dtype == np.uint8
            assert img.ndim == 3 and img.shape[2] == 3
            assert 0 <= img.min() and img.max() <= 255
            # The label stays aligned with the image yielded alongside it.
            assert label == expected_label

    def test_broken_urls_are_skipped_not_yielded(self, tmp_path, monkeypatch):
        """A broken URL in the middle is dropped; the stream keeps going.

        This is the core resilience guarantee: a single undecodable image
        must not truncate the stream or raise — the two good images on either
        side of it are still yielded.
        """
        # Arrange: a mock that returns garbage only for URLs containing "bad",
        # and a valid PNG for everything else.
        good = _png_bytes()

        def fake_get(url, timeout=None):
            if "bad" in url:
                return _FakeResponse(b"broken", status_ok=True)
            return _FakeResponse(good, status_ok=True)

        monkeypatch.setattr(ef.requests, "get", fake_get)
        # Arrange: a split with the bad URL deliberately sandwiched between two good ones.
        _write_split_csv(
            tmp_path / "split_dataset.csv",
            rows=[
                ("http://example.com/good1.png", "train"),
                ("http://example.com/bad.png", "train"),
                ("http://example.com/good2.png", "train"),
            ],
        )
        # Act: drain the stream, suppressing the expected per-skip UserWarning
        # so it does not clutter the test output (the skip itself is asserted below).
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # broken URL emits a UserWarning
            pairs = list(get_feature_stream("train", base_dir=str(tmp_path)))
        # Assert: only the two decodable images survived (each a (image, label) pair).
        assert len(pairs) == 2  # the single broken URL was dropped


# ===========================================================================
# 6. BGR colour convention (synthetic, no network)
# ===========================================================================

class TestBGRConvention:
    """The downloader must store images in OpenCV's BGR channel order.

    Downstream OpenCV-based preprocessing assumes BGR, but PIL decodes to RGB.
    These synthetic, network-free tests lock in the RGB->BGR conversion so a
    silent channel-order regression (which would swap red/blue everywhere)
    is caught.
    """

    def test_rgb_to_bgr_swaps_red_and_blue(self):
        """``cvtColor`` moves red from RGB index 0 to BGR index 2.

        A unit check of the conversion primitive itself, independent of the
        download path, so a failure here points at the colour conversion
        rather than at I/O.
        """
        # Arrange: a solid pure-red image expressed in RGB order.
        rgb_solid = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb_solid[:, :] = [255, 0, 0]  # pure red in RGB
        # Act: convert to OpenCV's BGR ordering.
        bgr_solid = cv2.cvtColor(rgb_solid, cv2.COLOR_RGB2BGR)
        # Assert: in BGR, red must live in channel index 2, and channel 0 must be 0.
        assert bgr_solid[0, 0, 2] == 255
        assert bgr_solid[0, 0, 0] == 0

    def test_downloaded_red_image_is_stored_as_bgr(self, monkeypatch):
        """End-to-end: a downloaded red image lands with red in BGR channel 2.

        Exercises the conversion *through* the real download path (not just
        the primitive), proving ``_download_image`` applies the RGB->BGR swap.
        """
        # Arrange: stub the network to return a pure-red PNG.
        red_png = _png_bytes(color=(255, 0, 0))

        def fake_get(url, timeout=None):
            return _FakeResponse(red_png, status_ok=True)

        monkeypatch.setattr(ef.requests, "get", fake_get)
        # Act
        img = _download_image("http://example.com/red.png")
        # Assert: red sits in the BGR red channel, blue channel is empty.
        assert img[0, 0, 2] == 255  # red sits in the BGR red channel
        assert img[0, 0, 0] == 0    # blue channel empty


# ===========================================================================
# 7. Feature-extraction integrity (instruction 6)
# ===========================================================================

class TestFeatureExtractionIntegrity:
    """Streamed images must survive the preprocessing pipeline intact.

    Bridges the streamer and the ``preprocessing`` pipeline: an image
    pulled from the (mocked) download path is pushed through a real pipeline
    and the resulting feature vector is checked for shape, dtype and embedding
    integrity (instruction 6 — no NaN/Inf, correct width, expected range).
    """

    def test_streamed_image_through_pipeline_yields_dense_vector(self, mock_download_ok):
        """A streamed image yields a dense, finite, correctly-shaped vector.

        This is the key integration guarantee: the bytes coming out of the
        streamer are compatible with the downstream pipeline and produce a
        clean embedding rather than NaNs or a mis-shaped array.
        """
        # Arrange: pull one mocked image straight from the download path.
        img = _download_image("http://example.com/sample.png")

        # Act: run it through the same pipeline verify_extract_features used.
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

    def test_individual_transforms_on_streamed_image(self, mock_download_ok):
        """Individual transforms accept a streamed image and reshape it correctly.

        A finer-grained companion to the full-pipeline test: confirms the two
        foundational transforms (grayscale collapse and resize) behave on real
        streamed data, isolating shape regressions to a single transform.
        """
        # Arrange: pull one mocked image from the download path.
        img = _download_image("http://example.com/sample.png")
        # Act: apply the two transforms directly.
        gray = to_grayscale(img)
        resized = resize_image(img, (64, 64), preserve_aspect=False)
        # Assert: grayscale drops the channel axis; resize hits the exact target.
        assert gray.ndim == 2
        assert resized.shape == (64, 64, 3)
