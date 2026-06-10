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
  ``image_preprocessing`` pipeline yields a dense, finite, correctly-shaped
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
    _load_image_urls,
    _url_column,
    _urls_from_csv,
    _validate_split,
    get_feature_stream,
)
from image_preprocessing import ImagePipeline, resize_image, to_grayscale


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
    @pytest.mark.parametrize("split", sorted(VALID_SPLITS))
    def test_accepts_valid_splits(self, split):
        # Should not raise.
        _validate_split(split)

    @pytest.mark.parametrize("bad", ["", "training", "TRAIN", "dev", "all"])
    def test_rejects_invalid_splits(self, bad):
        with pytest.raises(ValueError):
            _validate_split(bad)

    def test_module_constants_match_spec(self):
        assert REQUEST_TIMEOUT == 10
        assert VALID_SPLITS == frozenset({"train", "test", "val"})


# ===========================================================================
# 2. Metadata loading (CSV / JSON) from tmp_path
# ===========================================================================

class TestMetadataLoading:
    def test_loads_urls_from_combined_split_csv(self, tmp_path):
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
        # Act
        train_urls = _load_image_urls("train", str(tmp_path))
        test_urls = _load_image_urls("test", str(tmp_path))
        # Assert: split filtering keeps only the matching rows.
        assert train_urls == ["http://example.com/a.jpg", "http://example.com/b.jpg"]
        assert test_urls == ["http://example.com/c.jpg"]

    def test_loads_urls_from_per_split_json(self, tmp_path):
        # Per-split file ({split}.json) takes precedence and needs no filtering.
        (tmp_path / "train.json").write_text(
            '["http://x/1.png", "http://x/2.png"]', encoding="utf-8"
        )
        urls = _load_image_urls("train", str(tmp_path))
        assert urls == ["http://x/1.png", "http://x/2.png"]

    def test_url_column_detection(self):
        assert _url_column(["id", "image_url"]) == "image_url"
        assert _url_column(["url", "label"]) == "url"

    def test_missing_url_column_raises(self, tmp_path):
        # Edge case: a CSV without any recognised URL column is unusable.
        bad = tmp_path / "split.csv"
        bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            _urls_from_csv(str(bad), "train", filter_by_split=False)


# ===========================================================================
# 3. Error handling
# ===========================================================================

class TestErrorHandling:
    def test_missing_base_dir_raises(self):
        # Edge case: a base_dir that doesn't exist at all.
        with pytest.raises(FileNotFoundError):
            _load_image_urls("train", "./nonexistent_directory_xyz")

    def test_empty_dir_raises_filenotfound(self, tmp_path):
        # Edge case: directory exists but contains no metadata file.
        with pytest.raises(FileNotFoundError):
            _load_image_urls("train", str(tmp_path))


# ===========================================================================
# 4. Download path (mocked network)
# ===========================================================================

class TestDownloadImage:
    def test_successful_download_returns_bgr_array(self, mock_download_ok):
        # Act
        img = _download_image("http://example.com/whatever.png")
        # Assert: decoded to a 3-channel uint8 BGR array of the fake's size.
        assert img is not None
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.uint8
        assert img.shape[:2] == (8, 12)  # (height, width) of _png_bytes default

    def test_http_error_returns_none(self, monkeypatch):
        # Edge case: a 4xx/5xx response is swallowed and yields None (skipped).
        def fake_get(url, timeout=None):
            return _FakeResponse(b"", status_ok=False)

        monkeypatch.setattr(ef.requests, "get", fake_get)
        # The source warns before swallowing the error; assert both.
        with pytest.warns(UserWarning):
            assert _download_image("http://example.com/404.png") is None

    def test_broken_content_returns_none(self, monkeypatch):
        # Edge case: a 200 response whose body is not a decodable image.
        def fake_get(url, timeout=None):
            return _FakeResponse(b"not-an-image", status_ok=True)

        monkeypatch.setattr(ef.requests, "get", fake_get)
        with pytest.warns(UserWarning):
            assert _download_image("http://example.com/garbage") is None


# ===========================================================================
# 5. Generator contract
# ===========================================================================

class TestGeneratorContract:
    @pytest.fixture
    def streamed_images(self, tmp_path, mock_download_ok):
        # Arrange a 3-URL train split; every download is mocked to succeed.
        _write_split_csv(
            tmp_path / "split_dataset.csv",
            rows=[(f"http://example.com/{i}.png", "train") for i in range(3)],
        )
        return list(get_feature_stream("train", base_dir=str(tmp_path)))

    def test_yields_decoded_bgr_images(self, streamed_images):
        assert len(streamed_images) == 3
        for img in streamed_images:
            assert img.dtype == np.uint8
            assert img.ndim == 3 and img.shape[2] == 3
            assert 0 <= img.min() and img.max() <= 255

    def test_broken_urls_are_skipped_not_yielded(self, tmp_path, monkeypatch):
        # Mix of good and bad URLs; only the good ones should be yielded.
        good = _png_bytes()

        def fake_get(url, timeout=None):
            if "bad" in url:
                return _FakeResponse(b"broken", status_ok=True)
            return _FakeResponse(good, status_ok=True)

        monkeypatch.setattr(ef.requests, "get", fake_get)
        _write_split_csv(
            tmp_path / "split_dataset.csv",
            rows=[
                ("http://example.com/good1.png", "train"),
                ("http://example.com/bad.png", "train"),
                ("http://example.com/good2.png", "train"),
            ],
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # broken URL emits a UserWarning
            images = list(get_feature_stream("train", base_dir=str(tmp_path)))
        assert len(images) == 2  # the single broken URL was dropped


# ===========================================================================
# 6. BGR colour convention (synthetic, no network)
# ===========================================================================

class TestBGRConvention:
    def test_rgb_to_bgr_swaps_red_and_blue(self):
        # The decoder converts PIL-RGB to OpenCV-BGR; verify the channel swap.
        rgb_solid = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb_solid[:, :] = [255, 0, 0]  # pure red in RGB
        bgr_solid = cv2.cvtColor(rgb_solid, cv2.COLOR_RGB2BGR)
        # In BGR, red must live in channel index 2, and channel 0 must be 0.
        assert bgr_solid[0, 0, 2] == 255
        assert bgr_solid[0, 0, 0] == 0

    def test_downloaded_red_image_is_stored_as_bgr(self, monkeypatch):
        # A red RGB source, once downloaded, must have its red in BGR channel 2.
        red_png = _png_bytes(color=(255, 0, 0))

        def fake_get(url, timeout=None):
            return _FakeResponse(red_png, status_ok=True)

        monkeypatch.setattr(ef.requests, "get", fake_get)
        img = _download_image("http://example.com/red.png")
        assert img[0, 0, 2] == 255  # red sits in the BGR red channel
        assert img[0, 0, 0] == 0    # blue channel empty


# ===========================================================================
# 7. Feature-extraction integrity (instruction 6)
# ===========================================================================

class TestFeatureExtractionIntegrity:
    def test_streamed_image_through_pipeline_yields_dense_vector(self, mock_download_ok):
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
        # Mirrors verify_extract_features' direct to_grayscale / resize checks.
        img = _download_image("http://example.com/sample.png")
        gray = to_grayscale(img)
        resized = resize_image(img, (64, 64), preserve_aspect=False)
        assert gray.ndim == 2
        assert resized.shape == (64, 64, 3)
