"""
Shared pytest fixtures for the preprocessing test suite.

Everything here is deliberately lightweight: synthetic in-memory images and
small batches so the unit tests run in milliseconds without touching disk,
the network, or the ~500 MB VGG16 ImageNet weights. Heavy/optional paths
(VGG16, real downloads) are exercised through mocks defined alongside the
fixtures below.

Fixture cheat-sheet
-------------------
    rng                  -> seeded numpy Generator (reproducible randomness)
    color_image          -> (224, 224, 3) uint8 BGR-style image
    gray_image           -> (224, 224)    uint8 grayscale image
    small_color_image    -> (32, 32, 3)   uint8 image (fast pipeline tests)
    image_batch          -> list[np.ndarray] of color images (default 6)
    feature_matrix       -> (n_samples, n_features) float32 matrix
    tmp_image_file       -> path to a real PNG written to a tmp dir
    fake_vgg16           -> seeds the module-level VGG16 cache with a stub model
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pytest


# Fixed seed so every fixture-derived array is reproducible across runs.
SEED = 20260611


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy random Generator for reproducible test data."""
    return np.random.default_rng(SEED)


@pytest.fixture
def color_image(rng) -> np.ndarray:
    """A standard 224x224x3 uint8 colour image (the canonical pipeline input)."""
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def gray_image(rng) -> np.ndarray:
    """A 224x224 uint8 single-channel image."""
    return rng.integers(0, 256, size=(224, 224), dtype=np.uint8)


@pytest.fixture
def small_color_image(rng) -> np.ndarray:
    """A tiny 32x32x3 image — keeps pipeline/vectorize tests fast."""
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


@pytest.fixture
def image_batch(rng) -> List[np.ndarray]:
    """A list of 6 same-shaped colour images, suitable for batch_process."""
    return [
        rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        for _ in range(6)
    ]


@pytest.fixture
def feature_matrix(rng) -> np.ndarray:
    """A (40, 256) float32 feature matrix for reduce_dimensions tests."""
    return rng.random(size=(40, 256), dtype=np.float64).astype(np.float32)


@pytest.fixture
def tmp_image_file(tmp_path, color_image) -> str:
    """
    Write a real PNG to a temp dir and return its path.

    Used to exercise the OpenCV file-loading path (load_image_from_file)
    against an actual decodable file rather than a mock.
    """
    import cv2

    path = os.path.join(str(tmp_path), "sample.png")
    cv2.imwrite(path, color_image)
    return path


# ---------------------------------------------------------------------------
# VGG16 stub
# ---------------------------------------------------------------------------

class _FakeVGG16:
    """
    Minimal stand-in for a keras VGG16(include_top=False) model.

    The real model's ``predict`` on a 224x224x3 input returns a
    (1, 7, 7, 512) feature map (25 088 values once flattened). We reproduce
    exactly that shape with cheap random data so vectorize_image's vgg16 path
    can be tested for output shape/dtype without loading ImageNet weights.
    """

    output_size = 7 * 7 * 512  # 25 088, matching real block5_pool for 224x224

    def predict(self, batch, verbose=0):  # noqa: D401 - mimics keras signature
        n = batch.shape[0]
        return np.random.default_rng(0).random((n, 7, 7, 512)).astype(np.float32)


@pytest.fixture
def fake_vgg16(monkeypatch):
    """
    Seed the module-level VGG16 cache with a stub so method='vgg16' is fast.

    ``vectorize_image`` looks up ``_vgg16_models[input_size]`` and only imports
    keras + downloads weights on a cache miss. By pre-seeding the cache for the
    default (224, 224) input size we bypass that entirely. Returns the expected
    flattened output length so tests can assert on it.
    """
    from preprocessing import vectorize

    monkeypatch.setitem(vectorize._vgg16_models, (224, 224), _FakeVGG16())
    return _FakeVGG16.output_size
